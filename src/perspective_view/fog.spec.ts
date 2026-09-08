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

import { describe, expect, test } from "vitest";
import {
  computeFogParameters,
  FOG_REFERENCE_DEPTH,
  nearFarFromProjectionMatrix,
} from "#src/perspective_view/fog.js";
import { kAxes, mat4, quat, vec3 } from "#src/util/geom.js";

const ISOTROPIC = new Float64Array([1, 1, 1]);
/** 40nm sections of 4nm pixels, i.e. what `canonicalVoxelFactors` looks like for exaSPIM-like data. */
const ANISOTROPIC = new Float64Array([10, 1, 1]);

/**
 * Builds an inverse view matrix exactly as `NavigationState.pose.toMat4` and PerspectivePanel do.
 *
 * Note that `toMat4` scales the *rows* by `zoom / canonicalVoxelFactors[i]` -- display space is in
 * each dimension's own voxel units, so it is anisotropic whenever the voxels are. The panel then
 * flips y and z and steps the camera one unit back along the local z axis.
 */
function makeInvViewMatrix(
  zoomFactor: number,
  canonicalVoxelFactors: Float64Array = ISOTROPIC,
  orientation = quat.create(),
) {
  const m = mat4.fromQuat(mat4.create(), orientation);
  // An arbitrary focal point; none of the fog parameters may depend on it.
  const focalPoint = [11, -7, 3];
  for (let i = 0; i < 3; ++i) {
    const scale = zoomFactor / canonicalVoxelFactors[i];
    m[i] *= scale;
    m[4 + i] *= scale;
    m[8 + i] *= scale;
    m[12 + i] = focalPoint[i];
  }
  mat4.scale(m, m, vec3.fromValues(1, -1, -1));
  mat4.translate(m, m, kAxes[2]);
  return m;
}

describe("computeFogParameters", () => {
  test("the focal point is one eye unit from the camera", () => {
    // Regression: this was hard-coded to 2, i.e. twice the true distance, which pushed the start
    // of the falloff beyond every sample and silently disabled fog everywhere.
    for (const zoomFactor of [0.5, 1, 37, 309.02, 5000]) {
      const { fogStartDepth } = computeFogParameters(
        makeInvViewMatrix(zoomFactor),
        ISOTROPIC,
        1,
        0,
      );
      expect(fogStartDepth, `zoomFactor=${zoomFactor}`).toBeCloseTo(1, 6);
    }
  });

  test("start depth is independent of orientation and of the focal position", () => {
    const reference = computeFogParameters(
      makeInvViewMatrix(100),
      ISOTROPIC,
      1,
      0,
    );
    for (const axis of [kAxes[0], kAxes[1], kAxes[2]]) {
      for (const angle of [0.3, 1.1, Math.PI / 2, Math.PI]) {
        const orientation = quat.setAxisAngle(quat.create(), axis, angle);
        const { fogStartDepth, fogDensity } = computeFogParameters(
          makeInvViewMatrix(100, ISOTROPIC, orientation),
          ISOTROPIC,
          1,
          0,
        );
        expect(fogStartDepth).toBeCloseTo(reference.fogStartDepth, 5);
        expect(fogDensity).toBeCloseTo(reference.fogDensity, 5);
      }
    }
  });

  test("a fog amount of 0 disables fog", () => {
    for (const amount of [0, -1]) {
      expect(
        computeFogParameters(makeInvViewMatrix(100), ISOTROPIC, amount, 0)
          .fogDensity,
      ).toBe(0);
    }
  });

  test("with no zoom scaling, density is the fog amount over the focal depth", () => {
    // Attenuation at one focal distance beyond the focus is then exp(-amount).
    for (const zoomFactor of [10, 100, 1000]) {
      for (const amount of [0.5, 1, 7]) {
        const { fogDensity, fogStartDepth } = computeFogParameters(
          makeInvViewMatrix(zoomFactor),
          ISOTROPIC,
          amount,
          0,
        );
        expect(fogDensity * fogStartDepth).toBeCloseTo(amount, 5);
      }
    }
  });

  test("positive zoom scaling makes fog denser as the view zooms in", () => {
    const zoomedOut = computeFogParameters(
      makeInvViewMatrix(2000),
      ISOTROPIC,
      1,
      1,
    );
    const zoomedIn = computeFogParameters(
      makeInvViewMatrix(200),
      ISOTROPIC,
      1,
      1,
    );
    expect(zoomedIn.fogDensity).toBeGreaterThan(zoomedOut.fogDensity);
    // The term is a plain ratio of the reference scale to the zoom scale.
    expect(zoomedIn.fogDensity / zoomedOut.fogDensity).toBeCloseTo(10, 4);
  });

  test("negative zoom scaling thins fog as the view zooms in", () => {
    const zoomedOut = computeFogParameters(
      makeInvViewMatrix(2000),
      ISOTROPIC,
      1,
      -1,
    );
    const zoomedIn = computeFogParameters(
      makeInvViewMatrix(200),
      ISOTROPIC,
      1,
      -1,
    );
    expect(zoomedIn.fogDensity).toBeLessThan(zoomedOut.fogDensity);
  });

  test("at the reference zoom scale the scaling exponent has no effect", () => {
    const base = computeFogParameters(
      makeInvViewMatrix(FOG_REFERENCE_DEPTH),
      ISOTROPIC,
      3,
      0,
    );
    for (const scaling of [-2, -0.5, 1, 2]) {
      expect(
        computeFogParameters(
          makeInvViewMatrix(FOG_REFERENCE_DEPTH),
          ISOTROPIC,
          3,
          scaling,
        ).fogDensity,
      ).toBeCloseTo(base.fogDensity, 5);
    }
  });
});

describe("computeFogParameters with anisotropic voxels", () => {
  // Display space is measured in each dimension's own voxels, so with 40nm sections of 4nm pixels
  // a step of 1 along z is ten times the physical distance of a step of 1 along x. Distances must
  // therefore be taken in canonical voxels -- display coordinates weighted by canonicalVoxelFactors
  // -- which is a uniform physical scale. Measuring them in raw display units instead makes both
  // fog parameters depend on which way the camera happens to point.
  const ORIENTATIONS = [
    quat.create(),
    // Look along each of the three display axes in turn, plus something off-axis.
    quat.setAxisAngle(quat.create(), kAxes[0], Math.PI / 2),
    quat.setAxisAngle(quat.create(), kAxes[1], Math.PI / 2),
    quat.setAxisAngle(quat.create(), kAxes[2], Math.PI / 2),
    quat.normalize(quat.create(), quat.fromValues(0.3, -0.5, 0.2, 0.8)),
  ];

  test("the focal point is still one eye unit from the camera", () => {
    for (const orientation of ORIENTATIONS) {
      for (const zoomFactor of [1, 100, 5000]) {
        const { fogStartDepth } = computeFogParameters(
          makeInvViewMatrix(zoomFactor, ANISOTROPIC, orientation),
          ANISOTROPIC,
          1,
          0,
        );
        expect(
          fogStartDepth,
          `zoom=${zoomFactor} orientation=${Array.from(orientation)}`,
        ).toBeCloseTo(1, 5);
      }
    }
  });

  test("density does not change as the camera rotates", () => {
    // The visible symptom of the bug: fog got thicker or thinner purely from turning the camera,
    // by up to the anisotropy ratio.
    for (const fogScaling of [0, 1, -1]) {
      const reference = computeFogParameters(
        makeInvViewMatrix(100, ANISOTROPIC, ORIENTATIONS[0]),
        ANISOTROPIC,
        1,
        fogScaling,
      );
      for (const orientation of ORIENTATIONS) {
        const { fogDensity } = computeFogParameters(
          makeInvViewMatrix(100, ANISOTROPIC, orientation),
          ANISOTROPIC,
          1,
          fogScaling,
        );
        expect(
          fogDensity / reference.fogDensity,
          `fogScaling=${fogScaling} orientation=${Array.from(orientation)}`,
        ).toBeCloseTo(1, 5);
      }
    }
  });

  test("anisotropy alone does not change the fog", () => {
    // Two datasets viewed at the same zoom, one anisotropic and one not, must fog identically:
    // the canonical voxel is the same physical size in both, and it is the unit eye space uses.
    for (const fogScaling of [0, 1, -1]) {
      for (const zoomFactor of [10, 100, 1000]) {
        const isotropic = computeFogParameters(
          makeInvViewMatrix(zoomFactor, ISOTROPIC),
          ISOTROPIC,
          2,
          fogScaling,
        );
        const anisotropic = computeFogParameters(
          makeInvViewMatrix(zoomFactor, ANISOTROPIC),
          ANISOTROPIC,
          2,
          fogScaling,
        );
        expect(anisotropic.fogStartDepth).toBeCloseTo(
          isotropic.fogStartDepth,
          5,
        );
        expect(anisotropic.fogDensity / isotropic.fogDensity).toBeCloseTo(1, 5);
      }
    }
  });

  test("equal physical distances attenuate equally along every axis", () => {
    // End-to-end statement of the property: pick a distance in canonical voxels, convert it to an
    // eye depth, and check the attenuation matches whichever way the camera is turned.
    const zoomFactor = 250;
    const canonicalVoxelsPastFocus = 3000;
    let reference: number | undefined;
    for (const orientation of ORIENTATIONS) {
      const { fogDensity, fogStartDepth } = computeFogParameters(
        makeInvViewMatrix(zoomFactor, ANISOTROPIC, orientation),
        ANISOTROPIC,
        1,
        1,
      );
      // One eye unit is `zoomFactor` canonical voxels; see fog.ts.
      const eyeDepth = fogStartDepth + canonicalVoxelsPastFocus / zoomFactor;
      const attenuation = Math.exp(-fogDensity * (eyeDepth - fogStartDepth));
      expect(attenuation).toBeGreaterThan(0);
      expect(attenuation).toBeLessThan(1);
      if (reference === undefined) {
        reference = attenuation;
      } else {
        expect(
          attenuation,
          `orientation=${Array.from(orientation)}`,
        ).toBeCloseTo(reference, 6);
      }
    }
  });

  test("eye space is isotropic, so a raw eye depth is a physical distance", () => {
    // This is the premise the GLSL side rests on: both perspectiveFogFromEyeDepth callers hand it
    // an unweighted eye-space z (the volume renderer via uModelViewMatrix, everything else by
    // inverting gl_FragCoord.z), which is only a physical distance if eye space is a uniform scale
    // of physical space. It is, because pose.toMat4's per-axis display scaling is exactly the
    // inverse of the anisotropy of the display units -- but nothing else asserts that.
    const zoomFactor = 250;
    const viewMatrix = mat4.invert(
      mat4.create(),
      makeInvViewMatrix(
        zoomFactor,
        ANISOTROPIC,
        quat.normalize(quat.create(), quat.fromValues(0.3, -0.5, 0.2, 0.8)),
      ),
    ) as mat4;
    // Display displacements of one canonical voxel along each display axis, i.e. equal physical
    // lengths despite the very different coordinate steps.
    let reference: number | undefined;
    for (let axis = 0; axis < 3; ++axis) {
      const displayOffset = vec3.create();
      displayOffset[axis] = 1 / ANISOTROPIC[axis];
      // Direction only, so use the linear part and drop the translation.
      const eyeOffset = vec3.transformMat4(
        vec3.create(),
        displayOffset,
        viewMatrix,
      );
      const origin = vec3.transformMat4(
        vec3.create(),
        vec3.create(),
        viewMatrix,
      );
      const length = vec3.distance(eyeOffset, origin);
      if (reference === undefined) {
        reference = length;
        // One canonical voxel is 1/zoomFactor eye units.
        expect(length).toBeCloseTo(1 / zoomFactor, 6);
      } else {
        expect(Math.abs(length / reference - 1), `axis=${axis}`).toBeLessThan(
          1e-4,
        );
      }
    }
  });
});

describe("nearFarFromProjectionMatrix", () => {
  // mat4 is a Float32Array, and recovering the far plane means dividing by (c + 1) where c is
  // close to -1, so compare relatively. The conditioning degrades with the near:far ratio -- at
  // 1e7 the far plane comes back ~16% low -- which is harmless here, since fog density is a
  // smooth function of it and the 3-d view never uses ranges anywhere near that wide.
  const RELATIVE_TOLERANCE = 1e-4;

  function expectPlanes(m: mat4, near: number, far: number) {
    const recovered = nearFarFromProjectionMatrix(m);
    expect(Math.abs(Math.abs(recovered.near) / near - 1)).toBeLessThan(
      RELATIVE_TOLERANCE,
    );
    expect(Math.abs(Math.abs(recovered.far) / far - 1)).toBeLessThan(
      RELATIVE_TOLERANCE,
    );
  }

  test("recovers the planes from a perspective matrix", () => {
    for (const [near, far] of [
      [1, 100],
      [7, 900],
      [0.5, 10],
      [1, 10000],
    ]) {
      expectPlanes(
        mat4.perspective(mat4.create(), 0.9, 1.3, near, far) as mat4,
        near,
        far,
      );
    }
  });

  test("recovers the planes from an orthographic matrix", () => {
    for (const [near, far] of [
      [1, 100],
      [7, 900],
      [0.5, 10],
      [1, 10000],
    ]) {
      expectPlanes(
        mat4.ortho(mat4.create(), -3, 3, -2, 2, near, far) as mat4,
        near,
        far,
      );
    }
  });
});
