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
 * @file Resolver for the proposed `nanovdb` array -> bytes codec.
 *
 * Unlike every other array -> bytes codec, this one does not decode to a dense
 * array. The encoded chunk is a NanoVDB grid buffer -- already a spatial index
 * -- so it is passed through to the chunk format, which uploads it to the GPU
 * verbatim and traverses it in the shader. `passthroughChunkFormat` is what
 * tells `decodeArray` to stop after the bytes -> bytes stage.
 */

import type {
  CodecArrayInfo,
  CodecArrayLayoutInfo,
} from "#src/datasource/zarr/codec/index.js";
import { CodecKind } from "#src/datasource/zarr/codec/index.js";
import { registerCodec } from "#src/datasource/zarr/codec/resolve.js";
import { DataType } from "#src/util/data_type.js";
import {
  parseArray,
  verifyInt,
  verifyObject,
  verifyObjectProperty,
  verifyOptionalObjectProperty,
  verifyString,
} from "#src/util/json.js";

/** `tree_config`, the only configuration the specification registers. */
const TREE_CONFIG = [5, 4, 3];
/** Voxels along one edge of a NanoVDB leaf node, `1 << tree_config[2]`. */
const LEAF_DIM = 1 << TREE_CONFIG[2];
/**
 * Voxels along one edge of an upper internal node,
 * `1 << sum(tree_config)`. A chunk may not exceed this: requirement 1 of grid
 * coherence gives every grid exactly one upper node, at the origin.
 */
const UPPER_DIM = 1 << (TREE_CONFIG[0] + TREE_CONFIG[1] + TREE_CONFIG[2]);

const ROLES = new Set(["grid", "z", "y", "x", "channel"]);

export interface Configuration {
  dimensionRoles: string[];
  gridType: string;
  stats: string;
}

/**
 * Validates `dimension_roles` against the canonical order
 * `[grid ...] z y x [channel]`.
 */
function parseDimensionRoles(value: unknown): string[] {
  const roles = parseArray(value, verifyString);
  for (const r of roles) {
    if (!ROLES.has(r)) {
      throw new Error(`nanovdb: unknown dimension role ${JSON.stringify(r)}`);
    }
  }
  let nGrid = 0;
  while (nGrid < roles.length && roles[nGrid] === "grid") ++nGrid;
  const rest = roles.slice(nGrid);
  if (rest.includes("grid")) {
    throw new Error(
      "nanovdb: all `grid` dimensions must be outermost in dimension_roles " +
        JSON.stringify(roles),
    );
  }
  if (rest[0] !== "z" || rest[1] !== "y" || rest[2] !== "x") {
    throw new Error(
      "nanovdb: dimension_roles must list `z`, `y`, `x` in that order after " +
        `any \`grid\` dimensions, got ${JSON.stringify(roles)}`,
    );
  }
  if (rest.length > 4 || (rest.length === 4 && rest[3] !== "channel")) {
    throw new Error(
      "nanovdb: dimension_roles permits at most one `channel` dimension, " +
        `innermost, got ${JSON.stringify(roles)}`,
    );
  }
  return roles;
}

const SUPPORTED_GRID_TYPES = new Map<string, DataType>([
  ["Float", DataType.FLOAT32],
  ["Int32", DataType.INT32],
  ["UInt32", DataType.UINT32],
  ["Int16", DataType.INT16],
]);

registerCodec({
  name: "nanovdb",
  kind: CodecKind.arrayToBytes,
  // Chunks reach the chunk format as opaque NanoVDB buffers rather than dense
  // arrays; see the file comment.
  passthroughChunkFormat: "nanovdb",
  resolve(
    configuration: unknown,
    decodedArrayInfo: CodecArrayInfo,
  ): { configuration: Configuration } {
    verifyObject(configuration);
    const dimensionRoles = verifyObjectProperty(
      configuration,
      "dimension_roles",
      parseDimensionRoles,
    );
    const gridType =
      verifyOptionalObjectProperty(configuration, "grid_type", verifyString) ??
      "Float";
    const expectedDataType = SUPPORTED_GRID_TYPES.get(gridType);
    if (expectedDataType === undefined) {
      throw new Error(
        `nanovdb: unsupported grid_type ${JSON.stringify(gridType)}`,
      );
    }
    if (expectedDataType !== decodedArrayInfo.dataType) {
      throw new Error(
        `nanovdb: grid_type ${gridType} implies data type ` +
          `${DataType[expectedDataType]}, but array has ` +
          `${DataType[decodedArrayInfo.dataType]}`,
      );
    }
    const treeConfig =
      verifyOptionalObjectProperty(configuration, "tree_config", (value) =>
        parseArray(value, verifyInt),
      ) ?? TREE_CONFIG;
    if (
      treeConfig.length !== TREE_CONFIG.length ||
      treeConfig.some((v, i) => v !== TREE_CONFIG[i])
    ) {
      throw new Error(
        `nanovdb: unsupported tree_config ${JSON.stringify(treeConfig)}; ` +
          `only ${JSON.stringify(TREE_CONFIG)} is registered`,
      );
    }
    const stats =
      verifyOptionalObjectProperty(configuration, "stats", verifyString) ??
      "none";

    const { chunkShape } = decodedArrayInfo;
    if (chunkShape.length !== dimensionRoles.length) {
      throw new Error(
        `nanovdb: chunk rank ${chunkShape.length} does not match ` +
          `${dimensionRoles.length} dimension_roles ` +
          JSON.stringify(dimensionRoles),
      );
    }
    // The specification allows `grid` and `channel` dimensions, but this
    // reader only traverses a single scalar grid per chunk: a `grid`
    // dimension would need the shader to select among several grids in the
    // buffer, and a `channel` dimension a vector value type.
    if (dimensionRoles.length !== 3) {
      throw new Error(
        "nanovdb: this reader supports only `[\"z\", \"y\", \"x\"]`, got " +
          JSON.stringify(dimensionRoles),
      );
    }
    for (let i = 0; i < 3; ++i) {
      const d = chunkShape[i];
      if (d % LEAF_DIM !== 0) {
        throw new Error(
          `nanovdb: chunk extent ${d} along \`${dimensionRoles[i]}\` must be ` +
            `a multiple of the ${LEAF_DIM}-voxel leaf extent`,
        );
      }
      if (d > UPPER_DIM) {
        throw new Error(
          `nanovdb: chunk extent ${d} along \`${dimensionRoles[i]}\` exceeds ` +
            `the ${UPPER_DIM}-voxel upper node extent; a grid must have ` +
            "exactly one upper node, at the origin",
        );
      }
    }
    return { configuration: { dimensionRoles, gridType, stats } };
  },
  getDecodedArrayLayoutInfo(
    configuration: Configuration,
    decodedArrayInfo: CodecArrayInfo,
  ): CodecArrayLayoutInfo {
    configuration;
    return {
      physicalToLogicalDimension: Array.from(
        decodedArrayInfo.chunkShape,
        (_, i) => i,
      ),
      readChunkShape: decodedArrayInfo.chunkShape,
    };
  },
});
