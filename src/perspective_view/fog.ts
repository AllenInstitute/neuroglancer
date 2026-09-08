/**
 * @license
 * Copyright 2026 Google Inc.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file Parameters for scene-wide depth fog in the 3-d view.
 *
 * Kept apart from the panel so it can be tested without a WebGL context. Two bugs here were
 * invisible on the screen until they were pinned by tests: an earlier version assumed a fixed
 * eye-space distance to the focal point, which was wrong by a factor of two and silently disabled
 * fog everywhere (the falloff clamps at zero below the start depth); and it measured distances in
 * display units, which are anisotropic, so fog thickened and thinned purely from turning the
 * camera. Distances taken here must be in canonical voxels, a uniform physical scale.
 */

import type { mat4 } from "#src/util/geom.js";
import type { GL } from "#src/webgl/context.js";
import type { ShaderBuilder, ShaderProgram } from "#src/webgl/shader.js";

/**
 * Zoom scale, in canonical voxels per eye unit, at which a fog scaling exponent of 1 reproduces
 * the raw fog value. Only affects how fog tracks zoom, never the shape of the falloff.
 */
export const FOG_REFERENCE_DEPTH = 1000.0;

export interface FogParameters {
  /** Extinction per unit of eye-space depth. 0 disables fog. */
  fogDensity: number;
  /** Eye-space depth at which fog begins, i.e. the focal point. */
  fogStartDepth: number;
}

/**
 * Length of one column of `invViewMatrix`, measured in canonical voxels.
 *
 * Display space is expressed in each display dimension's own voxel units, so it is anisotropic
 * whenever the voxels are: with 40nm sections of 4nm pixels, a step of 1 along z covers ten times
 * the physical distance of a step of 1 along x. Weighting each component by
 * `canonicalVoxelFactors` converts to canonical voxels, which are a single physical size in every
 * direction, so lengths taken here are true physical distances up to one uniform scale.
 */
function canonicalVoxelColumnLength(
  invViewMatrix: mat4,
  canonicalVoxelFactors: ArrayLike<number>,
  column: number,
): number {
  const i = column * 4;
  return Math.hypot(
    invViewMatrix[i] * canonicalVoxelFactors[0],
    invViewMatrix[i + 1] * canonicalVoxelFactors[1],
    invViewMatrix[i + 2] * canonicalVoxelFactors[2],
  );
}

/**
 * Derives the fog parameters from the inverse view matrix.
 *
 * `invViewMatrix` maps eye space to display space, and `NavigationState.pose.toMat4` builds it by
 * scaling display row `i` by `zoom / canonicalVoxelFactors[i]`; PerspectivePanel then steps the
 * camera one unit back along the local z axis. Undoing that per-axis weighting -- which is what
 * `canonicalVoxelColumnLength` does -- leaves a pure rotation scaled by `zoom`, so:
 *
 *  - every column has the same length in canonical voxels, namely the zoom scale;
 *  - the third column, divided by that scale, is the eye-space depth of the focal point, always 1.
 *
 * Both are measured rather than assumed, so this keeps working if the panel changes how it
 * composes the matrix. Measuring them in *display* units instead was the anisotropy bug: the
 * columns then have different lengths, and both parameters swung by up to the anisotropy ratio as
 * the camera turned.
 *
 * @param canonicalVoxelFactors `DisplayDimensionRenderInfo.canonicalVoxelFactors`, i.e. the factor
 *     converting each display dimension's voxel units to canonical voxels.
 */
export function computeFogParameters(
  invViewMatrix: mat4,
  canonicalVoxelFactors: ArrayLike<number>,
  fogAmount: number,
  fogScaling: number,
): FogParameters {
  const canonicalVoxelsPerEyeUnit = Math.max(
    canonicalVoxelColumnLength(invViewMatrix, canonicalVoxelFactors, 0),
    1e-6,
  );
  const fogStartDepth = Math.max(
    canonicalVoxelColumnLength(invViewMatrix, canonicalVoxelFactors, 2) /
      canonicalVoxelsPerEyeUnit,
    1e-6,
  );
  if (!(fogAmount > 0)) {
    return { fogDensity: 0, fogStartDepth };
  }
  // Two multiplicative components: the user's amount, and a zoom term that is 1 when fogScaling is
  // 0 (identical falloff at every zoom) and grows as you zoom in when it is positive. Dividing by
  // the focal depth makes a fog value of 1 attenuate to exp(-1) one focal distance beyond the
  // point being looked at, whatever the zoom.
  const zoomTerm = Math.pow(
    FOG_REFERENCE_DEPTH / canonicalVoxelsPerEyeUnit,
    fogScaling,
  );
  return { fogDensity: (fogAmount * zoomTerm) / fogStartDepth, fogStartDepth };
}

/**
 * Scene-wide depth fog for the 3-d view.
 *
 * `uFogDensity` is extinction per unit of eye-space depth, already scaled by the zoom term; 0
 * disables fog. `uFogNearFar` carries the projection's near and far planes so that the nonlinear
 * window depth can be turned back into eye-space depth -- using window depth directly would fog
 * almost everything uniformly, since it is heavily compressed away from the near plane.
 *
 * Depth is measured along the view axis rather than radially. That matches how the volume renderer
 * measures it, so a fogged mesh and the volume behind it agree, and it is independent of view
 * direction.
 */
const glsl_perspectiveFogBody = `
// Attenuation for a sample at eyeDepth along the view axis. Fog begins at the focal point --
// whatever the camera is looking at -- so anything nearer than that is returned untouched and the
// falloff accumulates only beyond it. This is the single definition of the falloff; both geometry
// and the volume renderer go through it so they cannot drift apart.
float perspectiveFogFromEyeDepth(float eyeDepth) {
  if (uFogDensity == 0.0) return 1.0;
  return exp(-uFogDensity * max(0.0, eyeDepth - uFogStartDepth));
}
float perspectiveFogAttenuation() {
  if (uFogDensity == 0.0) return 1.0;
  float near = uFogNearFar.x, far = uFogNearFar.y;
  float ndc = 2.0 * gl_FragCoord.z - 1.0;
  float eyeDepth = 2.0 * near * far / (far + near - ndc * (far - near));
  return perspectiveFogFromEyeDepth(eyeDepth);
}
`;

/**
 * Declares the fog uniforms and `perspectiveFogAttenuation()`.
 *
 * The uniforms go through `addUniform` rather than being written into the code string, because
 * ShaderProgram only records names registered that way -- a uniform declared as raw source is
 * invisible to `shader.uniform()`, which then throws.
 *
 * Called by every perspective emitter, so consumers that read `uFogDensity` directly (the volume
 * renderer) must not declare it again.
 */
export function defineFogSupport(builder: ShaderBuilder) {
  builder.addUniform("highp float", "uFogDensity");
  builder.addUniform("highp vec2", "uFogNearFar");
  // Eye-space depth at which fog starts, i.e. the focal point.
  builder.addUniform("highp float", "uFogStartDepth");
  builder.addFragmentCode(glsl_perspectiveFogBody);
}

/**
 * Uploads the fog uniforms declared by `defineFogSupport`. Safe to call on any shader built with
 * one of the perspective emitters; a shader that does not declare them simply gets no-ops.
 */
export function setPerspectiveFogUniforms(
  gl: GL,
  shader: ShaderProgram,
  renderContext: {
    /** Absent for slice-view contexts, which are never fogged. */
    fogDensity?: number;
    /** Eye-space depth at which fog begins; defaults to the focal point being at depth 1. */
    fogStartDepth?: number;
    projectionParameters: { projectionMat: mat4 };
  },
) {
  // `uniform()` throws for names the program does not declare, so probe the map instead: this
  // helper is called from shaders that are also built for slice views, which have no fog.
  const location = shader.uniforms.get("uFogDensity");
  if (location === undefined || location === null) return;
  gl.uniform1f(location, renderContext.fogDensity ?? 0);
  const { near, far } = nearFarFromProjectionMatrix(
    renderContext.projectionParameters.projectionMat,
  );
  const startLocation = shader.uniforms.get("uFogStartDepth");
  if (startLocation != null) {
    gl.uniform1f(startLocation, renderContext.fogStartDepth ?? 1);
  }
  const nearFarLocation = shader.uniforms.get("uFogNearFar");
  if (nearFarLocation != null) {
    gl.uniform2f(nearFarLocation, Math.abs(near), Math.abs(far));
  }
}

/**
 * Recovers the near and far planes from a projection matrix, so that fog works for both the
 * perspective and orthographic cases without threading them through separately.
 */
export function nearFarFromProjectionMatrix(m: mat4): {
  near: number;
  far: number;
} {
  const c = m[10];
  const d = m[14];
  if (m[11] === 0) {
    // Orthographic: c = -2/(far-near), d = -(far+near)/(far-near).
    return { near: (d + 1) / c, far: (d - 1) / c };
  }
  // Perspective: c = -(far+near)/(far-near), d = -2*far*near/(far-near).
  return { near: d / (c - 1), far: d / (c + 1) };
}
