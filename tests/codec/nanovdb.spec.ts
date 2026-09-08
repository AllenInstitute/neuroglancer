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

import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, test } from "vitest";
import {
  decodeNanoVdbToDense,
  NanoVdbError,
  NanoVdbGrid,
  NanoVdbGridType,
} from "#src/sliceview/nanovdb/index.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dataDir = path.resolve(__dirname, "../../testdata/codec/nanovdb");

interface Fixture {
  file: string;
  gridType: string;
  shape: number[];
  byteLength: number;
  leafCount: number;
  lowerCount: number;
  upperCount: number;
  activeVoxelCount: number;
  indexBBox: number[][];
  background: number;
  samples: { z: number; y: number; x: number; value: number }[];
}

const { fixtures } = JSON.parse(
  readFileSync(path.join(dataDir, "fixtures.json"), "utf8"),
) as { fixtures: Fixture[] };

function loadGrid(f: Fixture) {
  const buf = readFileSync(path.join(dataDir, f.file));
  return new NanoVdbGrid(
    new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength),
  );
}

describe.each(fixtures)("nanovdb fixture $file", (f) => {
  test("header matches the generator", () => {
    const grid = loadGrid(f);
    expect(grid.buffer.byteLength).toBe(f.byteLength);
    expect(grid.gridSize).toBe(f.byteLength);
    expect(grid.version[0]).toBe(32);
    expect(NanoVdbGridType[grid.gridType]).toBe(f.gridType);
    expect(grid.leafCount).toBe(f.leafCount);
    expect(grid.lowerCount).toBe(f.lowerCount);
    expect(grid.upperCount).toBe(f.upperCount);
    expect(grid.activeVoxelCount).toBe(f.activeVoxelCount);
    expect(grid.background).toBe(f.background);
  });

  test("index bounding box matches", () => {
    const grid = loadGrid(f);
    expect(Array.from(grid.indexBBox)).toEqual([
      ...f.indexBBox[0],
      ...f.indexBBox[1],
    ]);
  });

  test("getValue matches the source volume at sampled coordinates", () => {
    const grid = loadGrid(f);
    for (const s of f.samples) {
      // Fixture coordinates are (z, y, x) into the dense array; NanoVDB
      // coordinates are (x, y, z).
      const got = grid.getValue(s.x, s.y, s.z);
      expect(
        got,
        `at dense[z=${s.z}][y=${s.y}][x=${s.x}] -> nanovdb(${s.x},${s.y},${s.z})`,
      ).toBeCloseTo(s.value, 6);
    }
  });

  test("dense decode agrees with getValue on every voxel", () => {
    const grid = loadGrid(f);
    const [nz, ny, nx] = f.shape;
    const dense = new Float32Array(nz * ny * nx);
    decodeNanoVdbToDense(grid, f.shape, dense);
    let active = 0;
    for (let z = 0; z < nz; ++z) {
      for (let y = 0; y < ny; ++y) {
        for (let x = 0; x < nx; ++x) {
          const viaTree = grid.getValue(x, y, z);
          const viaDense = dense[(z * ny + y) * nx + x];
          if (viaDense !== f.background) ++active;
          expect(viaDense).toBe(viaTree);
        }
      }
    }
    // Every active voxel in the grid must appear in the dense decode. The
    // fixtures are generated so no active voxel carries the background value.
    expect(active).toBe(f.activeVoxelCount);
  });
});

describe("nanovdb error handling", () => {
  test("rejects a buffer that is too small", () => {
    expect(() => new NanoVdbGrid(new Uint8Array(16))).toThrow(NanoVdbError);
  });

  test("rejects a bad magic number", () => {
    const buf = new Uint8Array(1024);
    new DataView(buf.buffer).setBigUint64(0, 0x1234567812345678n, true);
    expect(() => new NanoVdbGrid(buf)).toThrow(/not a NanoVDB grid buffer/);
  });

  test("rejects an unsupported major version", () => {
    const f = fixtures[0];
    const src = readFileSync(path.join(dataDir, f.file));
    const buf = new Uint8Array(src);
    // Version is major<<21 | minor<<10 | patch.
    new DataView(buf.buffer).setUint32(16, (31 << 21) | (0 << 10) | 0, true);
    expect(() => new NanoVdbGrid(buf)).toThrow(/unsupported NanoVDB major/);
  });
});
