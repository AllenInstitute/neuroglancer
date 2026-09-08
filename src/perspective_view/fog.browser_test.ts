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
  defineFogSupport,
  setPerspectiveFogUniforms,
} from "#src/perspective_view/fog.js";
import { mat4 } from "#src/util/geom.js";
import { fragmentShaderTest } from "#src/webgl/shader_testing.js";

/**
 * The tester draws a full-screen quad at gl_Position.z == 0, so every fragment sits at NDC depth 0,
 * i.e. gl_FragCoord.z == 0.5.
 */
const TEST_NDC_DEPTH = 0.0;

function expectedAttenuation(
  eyeDepth: number,
  density: number,
  startDepth: number,
) {
  if (density === 0) return 1;
  return Math.exp(-density * Math.max(0, eyeDepth - startDepth));
}

describe("perspectiveFogFromEyeDepth", () => {
  test("matches exp(-density * (depth - start)) beyond the start depth", () => {
    fragmentShaderTest(
      { eyeDepth: "float" },
      { attenuation: "float" },
      (tester) => {
        const { builder } = tester;
        defineFogSupport(builder);
        builder.setFragmentMain(
          "attenuation = perspectiveFogFromEyeDepth(eyeDepth);",
        );
        const { gl, shader } = tester;
        for (const [density, startDepth] of [
          [0.5, 1],
          [0.02, 10],
          [3, 0.25],
        ]) {
          gl.uniform1f(shader.uniform("uFogDensity"), density);
          gl.uniform1f(shader.uniform("uFogStartDepth"), startDepth);
          for (const eyeDepth of [
            0,
            0.1,
            startDepth * 0.5,
            startDepth,
            startDepth * 1.5,
            startDepth * 2,
            startDepth * 4,
            startDepth * 20,
          ]) {
            tester.execute({ eyeDepth });
            expect(
              tester.values.attenuation,
              `density=${density} start=${startDepth} depth=${eyeDepth}`,
            ).toBeCloseTo(
              expectedAttenuation(eyeDepth, density, startDepth),
              5,
            );
          }
        }
      },
    );
  });

  test("is exactly 1 at and nearer than the start depth", () => {
    fragmentShaderTest(
      { eyeDepth: "float" },
      { attenuation: "float" },
      (tester) => {
        const { builder } = tester;
        defineFogSupport(builder);
        builder.setFragmentMain(
          "attenuation = perspectiveFogFromEyeDepth(eyeDepth);",
        );
        const { gl, shader } = tester;
        // A large density would make any leak past the clamp obvious.
        gl.uniform1f(shader.uniform("uFogDensity"), 50);
        gl.uniform1f(shader.uniform("uFogStartDepth"), 4);
        for (const eyeDepth of [0, 1, 3.9, 4]) {
          tester.execute({ eyeDepth });
          expect(tester.values.attenuation, `depth=${eyeDepth}`).toBe(1);
        }
        tester.execute({ eyeDepth: 4.5 });
        expect(tester.values.attenuation).toBeLessThan(1e-6);
      },
    );
  });

  test("a density of 0 disables fog at every depth", () => {
    fragmentShaderTest(
      { eyeDepth: "float" },
      { attenuation: "float" },
      (tester) => {
        const { builder } = tester;
        defineFogSupport(builder);
        builder.setFragmentMain(
          "attenuation = perspectiveFogFromEyeDepth(eyeDepth);",
        );
        const { gl, shader } = tester;
        gl.uniform1f(shader.uniform("uFogDensity"), 0);
        // A start depth of 0 would otherwise fog everything.
        gl.uniform1f(shader.uniform("uFogStartDepth"), 0);
        for (const eyeDepth of [0, 1, 1000, 1e6]) {
          tester.execute({ eyeDepth });
          expect(tester.values.attenuation, `depth=${eyeDepth}`).toBe(1);
        }
      },
    );
  });
});

describe("perspectiveFogAttenuation", () => {
  test("inverts window depth back to eye depth", () => {
    // This is the path every fogged mesh, skeleton and annotation fragment takes. Using window
    // depth directly instead would fog nearly everything uniformly, so pin the inversion against
    // the eye-space formula that the volume renderer uses.
    fragmentShaderTest({}, { attenuation: "float" }, (tester) => {
      const { builder } = tester;
      defineFogSupport(builder);
      builder.setFragmentMain("attenuation = perspectiveFogAttenuation();");
      const { gl, shader } = tester;
      for (const [near, far] of [
        [1, 100],
        [7, 900],
        [0.5, 10],
      ]) {
        for (const [fogDensity, fogStartDepth] of [
          [0.01, 1],
          [0.2, 3],
        ]) {
          const projectionMat = mat4.perspective(
            mat4.create(),
            0.9,
            1.3,
            near,
            far,
          ) as mat4;
          setPerspectiveFogUniforms(gl, shader, {
            fogDensity,
            fogStartDepth,
            projectionParameters: { projectionMat },
          });
          tester.execute({});
          const eyeDepth =
            (2 * near * far) / (far + near - TEST_NDC_DEPTH * (far - near));
          expect(
            tester.values.attenuation,
            `near=${near} far=${far} density=${fogDensity}`,
          ).toBeCloseTo(
            expectedAttenuation(eyeDepth, fogDensity, fogStartDepth),
            4,
          );
        }
      }
    });
  });

  test("a render context without fog leaves geometry untouched", () => {
    // Slice-view contexts omit fogDensity entirely; the uniform must then default to 0.
    fragmentShaderTest({}, { attenuation: "float" }, (tester) => {
      const { builder } = tester;
      defineFogSupport(builder);
      builder.setFragmentMain("attenuation = perspectiveFogAttenuation();");
      const { gl, shader } = tester;
      setPerspectiveFogUniforms(gl, shader, {
        projectionParameters: {
          projectionMat: mat4.perspective(
            mat4.create(),
            0.9,
            1.3,
            1,
            100,
          ) as mat4,
        },
      });
      tester.execute({});
      expect(tester.values.attenuation).toBe(1);
    });
  });

  test("is a no-op on a shader that declares no fog uniforms", () => {
    // mesh/skeleton/annotation shaders are built for slice views too, where the emitter never
    // calls defineFogSupport; uploading must not throw there.
    fragmentShaderTest({}, { attenuation: "float" }, (tester) => {
      const { builder } = tester;
      builder.setFragmentMain("attenuation = 1.0;");
      const { gl, shader } = tester;
      expect(() =>
        setPerspectiveFogUniforms(gl, shader, {
          fogDensity: 5,
          fogStartDepth: 1,
          projectionParameters: {
            projectionMat: mat4.perspective(
              mat4.create(),
              0.9,
              1.3,
              1,
              100,
            ) as mat4,
          },
        }),
      ).not.toThrow();
    });
  });
});
