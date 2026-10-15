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

import "#src/datasource/precomputed/register_default.js";
import "#src/kvstore/http/register_frontend.js";
import "#src/layer/annotation/index.js";
import "#src/layer/image/index.js";
import "#src/sliceview/uncompressed_chunk_format.js";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DisplayContext } from "#src/display_context.js";
import { Viewer } from "#src/viewer.js";
import { mswFixture } from "#tests/fixtures/msw";

const msw = mswFixture();

const disposers: Array<() => void> = [];

afterEach(() => {
  while (disposers.length > 0) disposers.pop()!();
});

// A 50% white point at the center of the view.
const translucentPoint = {
  type: "annotation",
  name: "point",
  source: "local://annotations",
  annotations: [{ type: "point", id: "a", point: [8, 8, 8] }],
  shader: `void main() {
  setPointMarkerSize(20.0);
  setPointMarkerColor(vec4(1.0, 1.0, 1.0, 0.5));
}`,
};

// A white 16^3 image volume with 50% opacity, centered in the view.
const translucentImage = {
  type: "image",
  name: "image",
  source: `precomputed://${location.origin}/transparent-test`,
  opacity: 0.5,
  shader: "void main() { emitGrayscale(1.0); }",
};

async function serveImage() {
  (await msw()).use(
    http.get("*/transparent-test/info", () =>
      HttpResponse.json({
        type: "image",
        data_type: "uint8",
        num_channels: 1,
        scales: [
          {
            key: "1",
            size: [16, 16, 16],
            resolution: [1, 1, 1],
            voxel_offset: [0, 0, 0],
            chunk_sizes: [[16, 16, 16]],
            encoding: "raw",
          },
        ],
      }),
    ),
    http.get(
      "*/transparent-test/1/*",
      () => new HttpResponse(new Uint8Array(16 ** 3).fill(255)),
    ),
    http.get(
      "*/transparent-test/*",
      () => new HttpResponse(null, { status: 404 }),
    ),
  );
}

// Draws a 128x128 viewer and returns the RGBA value of each canvas pixel.
async function draw(
  layout: string,
  layers: object[],
  transparentBackground: boolean,
  showAxisLines = false,
  beforeRead?: (viewer: Viewer) => void,
) {
  const container = document.createElement("div");
  container.style.width = "128px";
  container.style.height = "128px";
  document.body.appendChild(container);
  disposers.push(() => container.remove());
  const display = new DisplayContext(container);
  disposers.push(() => display.dispose());
  const viewer = new Viewer(display, { showUIControls: false });
  disposers.push(() => viewer.dispose());
  viewer.state.restoreState({
    dimensions: { x: [1e-9, "m"], y: [1e-9, "m"], z: [1e-9, "m"] },
    position: [8, 8, 8],
    crossSectionScale: 0.25,
    layout,
    showAxisLines,
    showScaleBar: false,
    showDefaultAnnotations: false,
    layers,
  });
  viewer.uiConfiguration.transparentBackground.value = transparentBackground;
  await vi.waitFor(
    () => {
      display.draw();
      expect(viewer.isReady()).toBe(true);
    },
    { timeout: 5000 },
  );
  // `isReady` processes pending chunk updates, so draw again to show them.
  display.draw();
  beforeRead?.(viewer);
  display.draw();
  const { gl, canvas } = display;
  const { width, height } = canvas;
  const pixels = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  const at = (x: number, y: number) =>
    Array.from(pixels.subarray((y * width + x) * 4, (y * width + x) * 4 + 4));
  return {
    center: at(width >> 1, height >> 1),
    offCenter: at((width >> 1) - (width >> 3), (height >> 1) - (height >> 3)),
    panelCenters: Array.from(display.panels, (panel) => ({
      type: panel.constructor.name,
      color: at(
        Math.floor(
          panel.canvasRelativeClippedLeft + panel.renderViewport.width / 2,
        ),
        Math.floor(
          height -
            panel.canvasRelativeClippedTop -
            panel.renderViewport.height / 2,
        ),
      ),
    })),
    // Inside the panel, outside the content.
    empty: at(width >> 3, height >> 1),
    // Pixels whose color exceeds their alpha, which a premultiplied canvas cannot show correctly.
    invalidPixels: () => {
      let count = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        const a = pixels[i + 3];
        if (pixels[i] > a || pixels[i + 1] > a || pixels[i + 2] > a) ++count;
      }
      return count;
    },
  };
}

// Allows for rounding of 0.5 to 127 or 128.
const half = () => expect.toBeOneOf([127, 128]);

describe("DisplayContext.transparentBackground", () => {
  it("is off by default, so the canvas is opaque", async () => {
    const { center, empty } = await draw("xy", [translucentPoint], false);
    expect(center[3]).toBe(255);
    expect(empty).toEqual([128, 128, 128, 255]);
  });

  it("leaves empty areas of a cross-section transparent", async () => {
    const result = await draw("xy", [translucentPoint], true);
    expect(result.empty).toEqual([0, 0, 0, 0]);
    expect(result.invalidPixels()).toBe(0);
  });

  it("keeps the alpha of a translucent annotation", async () => {
    const { center } = await draw("xy", [translucentPoint], true);
    expect(center).toEqual([half(), half(), half(), half()]);
  });

  it("keeps the alpha of a translucent image", async () => {
    await serveImage();
    const result = await draw("xy", [translucentImage], true);
    expect(result.center).toEqual([half(), half(), half(), half()]);
    expect(result.invalidPixels()).toBe(0);
  });

  it("composites overlapping translucent images", async () => {
    await serveImage();
    const result = await draw(
      "xy",
      [translucentImage, { ...translucentImage, name: "second image" }],
      true,
    );
    expect(result.center).toEqual([192, 192, 192, 192]);
    expect(result.invalidPixels()).toBe(0);
  });

  it("composites translucent image colors with premultiplied alpha", async () => {
    await serveImage();
    const result = await draw(
      "xy",
      [
        {
          ...translucentImage,
          shader: "void main() { emitRGB(vec3(1.0, 0.0, 0.0)); }",
        },
        {
          ...translucentImage,
          name: "green",
          shader: "void main() { emitRGB(vec3(0.0, 1.0, 0.0)); }",
        },
      ],
      true,
    );
    expect(result.center).toEqual([64, 128, 0, 192]);
    expect(result.invalidPixels()).toBe(0);
  });

  it("keeps an opaque image opaque beneath a translucent image", async () => {
    await serveImage();
    const result = await draw(
      "xy",
      [
        { ...translucentImage, opacity: 1 },
        { ...translucentImage, name: "second image" },
      ],
      true,
    );
    expect(result.center).toEqual([255, 255, 255, 255]);
  });

  it("composites additive image layers without squaring their alpha", async () => {
    await serveImage();
    const result = await draw(
      "xy",
      [
        { ...translucentImage, opacity: 0.25, blend: "additive" },
        {
          ...translucentImage,
          name: "second image",
          opacity: 0.25,
          blend: "additive",
        },
      ],
      true,
    );
    expect(result.center).toEqual([128, 128, 128, 128]);
    expect(result.invalidPixels()).toBe(0);
  });

  it("reuses the transparent composition after toggling transparency", async () => {
    await serveImage();
    const result = await draw(
      "xy",
      [translucentImage, { ...translucentImage, name: "second image" }],
      true,
      false,
      (viewer) => {
        viewer.uiConfiguration.transparentBackground.value = false;
        viewer.display.draw();
        viewer.uiConfiguration.transparentBackground.value = true;
      },
    );
    expect(result.center).toEqual([192, 192, 192, 192]);
  });

  it("invalidates the transparent composition after opaque rendering changes", async () => {
    await serveImage();
    const result = await draw(
      "xy",
      [translucentImage, { ...translucentImage, name: "second image" }],
      true,
      false,
      (viewer) => {
        viewer.uiConfiguration.transparentBackground.value = false;
        viewer.display.draw();
        viewer.layerManager.managedLayers[1].setVisible(false);
        viewer.display.draw();
        viewer.uiConfiguration.transparentBackground.value = true;
      },
    );
    expect(result.center).toEqual([half(), half(), half(), half()]);
  });

  it("preserves legacy image colors with an opaque background", async () => {
    await serveImage();
    const result = await draw(
      "xy",
      [translucentImage, { ...translucentImage, name: "second image" }],
      false,
    );
    expect(result.center).toEqual([255, 255, 255, 255]);
  });

  it("preserves 3D slice colors when sharing slices with transparent panels", async () => {
    await serveImage();
    const layers = [
      {
        ...translucentImage,
        shader: "void main() { emitRGB(vec3(1.0, 0.0, 0.0)); }",
      },
      {
        ...translucentImage,
        name: "green",
        shader: "void main() { emitRGB(vec3(0.0, 1.0, 0.0)); }",
      },
    ];
    const opaque = await draw("4panel", layers, false);
    const transparent = await draw("4panel", layers, true);
    const opaque3D = opaque.panelCenters.find(
      (panel) => panel.type === "PerspectivePanel",
    )!;
    const transparent3D = transparent.panelCenters.find(
      (panel) => panel.type === "PerspectivePanel",
    )!;
    expect(opaque3D.color[2]).toBe(0);
    expect(transparent3D.color).toEqual(opaque3D.color);
  });

  it("preserves image alpha when axis lines are enabled", async () => {
    await serveImage();
    const result = await draw("xy", [translucentImage], true, true);
    expect(result.offCenter).toEqual([half(), half(), half(), half()]);
    expect(result.invalidPixels()).toBe(0);
  });

  it("leaves the 3D panel opaque", async () => {
    const { center, empty } = await draw("3d", [translucentPoint], true);
    expect(center[3]).toBe(255);
    expect(empty[3]).toBe(255);
  });
});
