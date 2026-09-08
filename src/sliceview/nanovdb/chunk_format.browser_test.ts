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
 * Verifies that the in-shader NanoVDB traversal returns the same values as the
 * CPU decoder, which is itself checked against buffers validated by upstream
 * NanoVDB (see tests/codec/nanovdb.spec.ts).
 */

import { describe } from "vitest";
import { chunkFormatTest } from "#src/sliceview/chunk_format_testing.js";
import { ChunkFormat } from "#src/sliceview/nanovdb/chunk_format.js";
import {
  decodeNanoVdbToDense,
  NanoVdbGrid,
} from "#src/sliceview/nanovdb/index.js";
import { DataType } from "#src/util/data_type.js";

declare const TEST_DATA_SERVER: string;

const FIXTURES = [
  { name: "sphere_64", size: 64 },
  { name: "ramp_32", size: 32 },
  { name: "single_8", size: 8 },
];

async function loadFixture(name: string, size: number) {
  const response = await fetch(`${TEST_DATA_SERVER}codec/nanovdb/${name}.nvdb`);
  if (!response.ok) {
    throw new Error(`failed to fetch ${name}.nvdb: ${response.status}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  const grid = new NanoVdbGrid(bytes);
  const dense = new Float32Array(size * size * size);
  decodeNanoVdbToDense(grid, [size, size, size], dense);
  return {
    words: new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength >> 2),
    dense,
  };
}

// Loaded at module scope so the data is available when `chunkFormatTest`
// registers its cases.
const fixtures = await Promise.all(
  FIXTURES.map(async ({ name, size }) => ({
    name,
    size,
    ...(await loadFixture(name, size)),
  })),
);

describe("sliceview/nanovdb/chunk_format", () => {
  for (const { name, size, words, dense } of fixtures) {
    describe(`${name} shader traversal matches CPU decode`, () => {
      chunkFormatTest(
        DataType.FLOAT32,
        Uint32Array.of(size, size, size, 1),
        (gl) => {
          const chunkFormat = ChunkFormat.get(gl, DataType.FLOAT32);
          return [chunkFormat, chunkFormat.getTextureLayout(words.length)];
        },
        // `chunkFormatTest` indexes rawData with Fortran-order strides over
        // [x, y, z]; with a unit-stride x axis that is the same linear index as
        // the C-order [z][y][x] array the decoder produces.
        dense,
        words,
      );
    });
  }
});
