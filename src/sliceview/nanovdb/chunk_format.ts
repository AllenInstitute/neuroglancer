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
 * @file GPU chunk format that stores each chunk as a NanoVDB grid buffer.
 *
 * The buffer is uploaded verbatim into a `usampler2D` and traversed in the
 * shader: `getDataValueAt` walks root -> upper internal -> lower internal ->
 * leaf, exactly as the CPU reader in ./index.ts does. Because NanoVDB is
 * pointer-free -- every link is a byte offset relative to the node holding it
 * -- no relocation is needed between CPU and GPU.
 *
 * The traversal is fixed-depth and fully unrolled, in the style of
 * gpu_hash/shader.ts; the only loop is the bounded scan of the root tile
 * table, which for chunk-sized grids holds a single entry.
 */

import type {
  ChunkFormatHandler,
  VolumeChunkSource,
} from "#src/sliceview/volume/frontend.js";
import { registerChunkFormatHandler } from "#src/sliceview/volume/frontend.js";
import type { VolumeChunkSpecification } from "#src/sliceview/volume/base.js";
import {
  GRID_DATA_SIZE,
  internalTableOffset,
  LEAF_LOG2DIM,
  leafTableOffset,
  LOWER_LOG2DIM,
  LOWER_TOTAL,
  NanoVdbGrid,
  rootDataSize,
  rootTileSize,
  UPPER_LOG2DIM,
} from "#src/sliceview/nanovdb/index.js";
import {
  SingleTextureChunkFormat,
  SingleTextureVolumeChunk,
} from "#src/sliceview/single_texture_chunk_format.js";
import { DataType } from "#src/util/data_type.js";
import { RefCounted } from "#src/util/disposable.js";
import type { GL } from "#src/webgl/context.js";
import type { ShaderBuilder, ShaderProgram, ShaderSamplerType } from "#src/webgl/shader.js";
import { getShaderType } from "#src/webgl/shader_lib.js";
import {
  computeTextureFormat,
  OneDimensionalTextureAccessHelper,
  setOneDimensionalTextureData,
  TextureFormat,
} from "#src/webgl/texture_access.js";

/**
 * Maximum number of root tiles the shader will scan. A chunk-sized grid has
 * one tile per 4096^3 region it spans, so this is generous; the handler
 * rejects buffers that exceed it rather than silently mis-rendering.
 */
export const MAX_ROOT_TILES = 8;

const textureFormat = computeTextureFormat(new TextureFormat(), DataType.UINT32);

export class TextureLayout extends RefCounted {
  constructor(public numWords: number) {
    super();
  }
}

export class ChunkFormat extends SingleTextureChunkFormat<TextureLayout> {
  static get(gl: GL, dataType: DataType) {
    const shaderKey = `sliceview.NanoVdbChunkFormat:${dataType}`;
    return gl.memoize.get(
      shaderKey,
      () => new ChunkFormat(dataType, shaderKey),
    );
  }

  /** The buffer is self-describing, so the layout only records its length. */
  getTextureLayout(numWords: number) {
    return new TextureLayout(numWords);
  }

  private textureAccessHelper: OneDimensionalTextureAccessHelper;

  get shaderSamplerType(): ShaderSamplerType {
    return "usampler2D";
  }

  constructor(dataType: DataType, key: string) {
    super(key, dataType);
    this.textureAccessHelper = new OneDimensionalTextureAccessHelper(
      "nanovdbData",
    );
  }

  defineShader(
    builder: ShaderBuilder,
    numChannelDimensions: number,
    inVertexShader = false,
  ) {
    super.defineShader(builder, numChannelDimensions);
    const { textureAccessHelper, dataType } = this;
    textureAccessHelper.defineShader(builder);
    const local = (x: string) => `nanovdbChunkFormat_${x}`;
    const addCode = inVertexShader
      ? builder.addVertexCode.bind(builder)
      : builder.addFragmentCode.bind(builder);
    addCode(
      textureAccessHelper.getAccessor(
        local("readWord"),
        "uVolumeChunkSampler",
        DataType.UINT32,
        1,
      ),
    );

    // Layout constants, derived exactly as in ./index.ts.
    const v = valueLayoutFor(dataType);
    const upperMaskBytes = (1 << (3 * UPPER_LOG2DIM)) / 8;
    const lowerMaskBytes = (1 << (3 * LOWER_LOG2DIM)) / 8;
    const upperTable = internalTableOffset(UPPER_LOG2DIM, v);
    const lowerTable = internalTableOffset(LOWER_LOG2DIM, v);
    const leafTable = leafTableOffset(v);
    const glslType = getShaderType(dataType);

    // Converts the raw 32-bit word holding a value into the shader value type.
    const decodeValue =
      dataType === DataType.FLOAT32
        ? "uintBitsToFloat(w)"
        : dataType === DataType.INT32
          ? "int(w)"
          : dataType === DataType.INT16
            ? "int((w & 0xFFFFu) << 16u) >> 16"
            : "w";

    addCode(`
// Reads the 32-bit word at a 4-byte-aligned byte offset.
highp uint ${local("word")}(highp uint byteOffset) {
  return ${local("readWord")}(byteOffset >> 2u).value;
}
// Tests bit ${"`"}i${"`"} of a NanoVDB bit mask beginning at ${"`"}base${"`"}. Masks are arrays of
// little-endian uint64 words, so bit i lives in 32-bit word i>>5 at bit i&31.
bool ${local("maskBit")}(highp uint base, highp uint i) {
  highp uint w = ${local("word")}(base + ((i >> 5u) << 2u));
  return ((w >> (i & 31u)) & 1u) != 0u;
}
${glslType} ${local("value")}(highp uint byteOffset) {
  highp uint w = ${local("word")}(byteOffset);
  return ${glslType}(${decodeValue});
}
`);

    let code = `
${glslType} getDataValueAt(highp ivec3 p`;
    for (let i = 0; i < numChannelDimensions; ++i) {
      code += `, highp int channelIndex${i}`;
    }
    code += `) {
  highp uint x = uint(p.x), y = uint(p.y), z = uint(p.z);

  // TreeData immediately follows GridData; mNodeOffset[3] (root) is at +24.
  // Offsets are int64 but always fit in 32 bits for chunk-sized grids.
  highp uint treeBase = ${GRID_DATA_SIZE}u;
  highp uint rootBase = treeBase + ${local("word")}(treeBase + 24u);
  ${glslType} background = ${local("value")}(rootBase + 28u);

  // Root: locate the tile whose key matches this coordinate.
  highp uint keyLo = (z >> ${UPPER_TOTAL_SHIFT}u) | ((y >> ${UPPER_TOTAL_SHIFT}u) << 21u);
  highp uint keyHi = ((y >> ${UPPER_TOTAL_SHIFT}u) >> 11u) | ((x >> ${UPPER_TOTAL_SHIFT}u) << 10u);
  highp uint tableSize = min(${local("word")}(rootBase + 24u), ${MAX_ROOT_TILES}u);
  highp uint tileBase = rootBase + ${rootDataSize(v)}u;
  highp uint upperBase = 0u;
  for (highp uint i = 0u; i < tableSize; ++i) {
    highp uint t = tileBase + i * ${rootTileSize(v)}u;
    if (${local("word")}(t) == keyLo && ${local("word")}(t + 4u) == keyHi) {
      highp uint child = ${local("word")}(t + 8u);
      if (child == 0u) {
        // Constant tile: active state in +16, value in +20.
        return ${local("word")}(t + 16u) != 0u
            ? ${local("value")}(t + 20u) : background;
      }
      upperBase = rootBase + child;
      break;
    }
  }
  if (upperBase == 0u) return background;

  // Upper internal node (32^3 children, each spanning 128 voxels).
  highp uint upperSlot =
      (((x & ${(1 << UPPER_TOTAL_SHIFT_PLUS) - 1}u) >> ${LOWER_TOTAL}u) << ${2 * UPPER_LOG2DIM}u) |
      (((y & ${(1 << UPPER_TOTAL_SHIFT_PLUS) - 1}u) >> ${LOWER_TOTAL}u) << ${UPPER_LOG2DIM}u) |
       ((z & ${(1 << UPPER_TOTAL_SHIFT_PLUS) - 1}u) >> ${LOWER_TOTAL}u);
  highp uint upperChildMask = upperBase + 32u + ${upperMaskBytes}u;
  if (!${local("maskBit")}(upperChildMask, upperSlot)) {
    // Tile rather than child: active tiles carry a value, inactive ones do not.
    return ${local("maskBit")}(upperBase + 32u, upperSlot)
        ? ${local("value")}(upperBase + ${upperTable}u + upperSlot * ${v.tileSize}u)
        : background;
  }
  highp uint lowerBase = upperBase +
      ${local("word")}(upperBase + ${upperTable}u + upperSlot * ${v.tileSize}u);

  // Lower internal node (16^3 children, each spanning 8 voxels).
  highp uint lowerSlot =
      (((x & ${(1 << LOWER_TOTAL) - 1}u) >> ${LEAF_LOG2DIM}u) << ${2 * LOWER_LOG2DIM}u) |
      (((y & ${(1 << LOWER_TOTAL) - 1}u) >> ${LEAF_LOG2DIM}u) << ${LOWER_LOG2DIM}u) |
       ((z & ${(1 << LOWER_TOTAL) - 1}u) >> ${LEAF_LOG2DIM}u);
  highp uint lowerChildMask = lowerBase + 32u + ${lowerMaskBytes}u;
  if (!${local("maskBit")}(lowerChildMask, lowerSlot)) {
    return ${local("maskBit")}(lowerBase + 32u, lowerSlot)
        ? ${local("value")}(lowerBase + ${lowerTable}u + lowerSlot * ${v.tileSize}u)
        : background;
  }
  highp uint leafBase = lowerBase +
      ${local("word")}(lowerBase + ${lowerTable}u + lowerSlot * ${v.tileSize}u);

  // Leaf (8^3 voxels).
  highp uint n = ((x & 7u) << 6u) | ((y & 7u) << 3u) | (z & 7u);
  if (!${local("maskBit")}(leafBase + 16u, n)) return background;
  return ${local("value")}(leafBase + ${leafTable}u + n * ${v.valueSize}u);
}
`;
    addCode(code);
  }

  /**
   * Emits `getEmptySpaceSkip`, which lets the volume renderer step over regions the tree already
   * knows are background.
   *
   * The descent is the same one `getDataValueAt` performs, but it stops as soon as a level reports
   * no child: an absent root tile means the whole 4096^3 region is background, an absent upper
   * child means 128^3, and an absent lower child means 8^3. Given the size of that region, the
   * number of skippable steps is the distance along the ray to the region's exit face, in units of
   * the step vector.
   *
   * Single inactive voxels inside a populated leaf are deliberately *not* skipped: the ray is
   * already inside signal there, so the test would cost more than it saves.
   */
  defineEmptySpaceSkip(builder: ShaderBuilder) {
    const local = (x: string) => `nanovdbChunkFormat_${x}`;
    const v = valueLayoutFor(this.dataType);
    const upperMaskBytes = (1 << (3 * UPPER_LOG2DIM)) / 8;
    const lowerMaskBytes = (1 << (3 * LOWER_LOG2DIM)) / 8;
    const upperTable = internalTableOffset(UPPER_LOG2DIM, v);

    builder.addFragmentCode(`
// Size in voxels of the background-only cube containing (x, y, z), or 0 if the
// voxel is not known to be background.
int ${local("emptyExtent")}(highp uint x, highp uint y, highp uint z) {
  highp uint treeBase = ${GRID_DATA_SIZE}u;
  highp uint rootBase = treeBase + ${local("word")}(treeBase + 24u);

  highp uint keyLo = (z >> ${UPPER_TOTAL_SHIFT}u) | ((y >> ${UPPER_TOTAL_SHIFT}u) << 21u);
  highp uint keyHi = ((y >> ${UPPER_TOTAL_SHIFT}u) >> 11u) | ((x >> ${UPPER_TOTAL_SHIFT}u) << 10u);
  highp uint tableSize = min(${local("word")}(rootBase + 24u), ${MAX_ROOT_TILES}u);
  highp uint tileBase = rootBase + ${rootDataSize(v)}u;
  highp uint upperBase = 0u;
  for (highp uint i = 0u; i < tableSize; ++i) {
    highp uint t = tileBase + i * ${rootTileSize(v)}u;
    if (${local("word")}(t) == keyLo && ${local("word")}(t + 4u) == keyHi) {
      highp uint child = ${local("word")}(t + 8u);
      // A constant tile is background only when its active-state flag is clear.
      if (child == 0u) return ${local("word")}(t + 16u) != 0u ? 0 : ${1 << UPPER_TOTAL_SHIFT};
      upperBase = rootBase + child;
      break;
    }
  }
  if (upperBase == 0u) return ${1 << UPPER_TOTAL_SHIFT};

  highp uint upperSlot =
      (((x & ${(1 << UPPER_TOTAL_SHIFT) - 1}u) >> ${LOWER_TOTAL}u) << ${2 * UPPER_LOG2DIM}u) |
      (((y & ${(1 << UPPER_TOTAL_SHIFT) - 1}u) >> ${LOWER_TOTAL}u) << ${UPPER_LOG2DIM}u) |
       ((z & ${(1 << UPPER_TOTAL_SHIFT) - 1}u) >> ${LOWER_TOTAL}u);
  if (!${local("maskBit")}(upperBase + 32u + ${upperMaskBytes}u, upperSlot)) {
    return ${local("maskBit")}(upperBase + 32u, upperSlot) ? 0 : ${1 << LOWER_TOTAL};
  }
  highp uint lowerBase = upperBase +
      ${local("word")}(upperBase + ${upperTable}u + upperSlot * ${v.tileSize}u);

  highp uint lowerSlot =
      (((x & ${(1 << LOWER_TOTAL) - 1}u) >> ${LEAF_LOG2DIM}u) << ${2 * LOWER_LOG2DIM}u) |
      (((y & ${(1 << LOWER_TOTAL) - 1}u) >> ${LEAF_LOG2DIM}u) << ${LOWER_LOG2DIM}u) |
       ((z & ${(1 << LOWER_TOTAL) - 1}u) >> ${LEAF_LOG2DIM}u);
  if (!${local("maskBit")}(lowerBase + 32u + ${lowerMaskBytes}u, lowerSlot)) {
    return ${local("maskBit")}(lowerBase + 32u, lowerSlot) ? 0 : ${1 << LEAF_LOG2DIM};
  }
  return 0;
}

int getEmptySpaceSkip(vec3 posInChunk, vec3 stepVector) {
  vec3 clamped = max(vec3(0.0), posInChunk);
  highp ivec3 p = ivec3(floor(clamped));
  int extent = ${local("emptyExtent")}(uint(p.x), uint(p.y), uint(p.z));
  if (extent <= 1) return 0;
  // The background region is aligned to its own size.
  vec3 lo = vec3(p & ivec3(~(extent - 1)));
  vec3 hi = lo + float(extent);
  // Steps until the ray leaves that region.
  float tExit = 1.0e30;
  for (int i = 0; i < 3; ++i) {
    float s = stepVector[i];
    if (abs(s) < 1.0e-20) continue;
    tExit = min(tExit, ((s > 0.0 ? hi[i] : lo[i]) - clamped[i]) / s);
  }
  // Conservative: stay strictly inside, and never skip backwards.
  return max(0, int(floor(tExit - 1.0e-3)));
}
`);
  }

  setupTextureLayout(
    _gl: GL,
    _shader: ShaderProgram,
    _textureLayout: TextureLayout,
  ) {
    // The buffer is self-describing: no per-chunk uniforms are needed.
  }

  setTextureData(gl: GL, _textureLayout: TextureLayout, data: Uint32Array) {
    setOneDimensionalTextureData(gl, textureFormat, data);
  }
}

const UPPER_TOTAL_SHIFT = LOWER_TOTAL + UPPER_LOG2DIM; // 12
const UPPER_TOTAL_SHIFT_PLUS = UPPER_TOTAL_SHIFT; // mask width for upper node

function valueLayoutFor(dataType: DataType) {
  switch (dataType) {
    case DataType.FLOAT32:
    case DataType.INT32:
    case DataType.UINT32:
      return { dataType, numComponents: 1, valueSize: 4, statsSize: 4, tileSize: 8 };
    case DataType.INT16:
      return { dataType, numComponents: 1, valueSize: 2, statsSize: 4, tileSize: 8 };
    default:
      throw new Error(
        `nanovdb chunk format: unsupported data type ${DataType[dataType]}`,
      );
  }
}

export class NanoVdbVolumeChunk extends SingleTextureVolumeChunk<
  Uint32Array,
  TextureLayout
> {
  declare CHUNK_FORMAT_TYPE: ChunkFormat;
  private grid: NanoVdbGrid | null = null;

  constructor(source: VolumeChunkSource, x: any) {
    super(source, x);
    const buffer = x.data as Uint8Array | null;
    if (buffer == null) {
      this.data = null;
      return;
    }
    // Validate up front: a malformed buffer must fail here rather than produce
    // silently wrong pixels after upload.
    const grid = new NanoVdbGrid(buffer);
    if (grid.rootTableSize > MAX_ROOT_TILES) {
      throw new Error(
        `nanovdb chunk has ${grid.rootTableSize} root tiles, more than the ` +
          `${MAX_ROOT_TILES} the shader scans`,
      );
    }
    this.grid = grid;
    this.data = new Uint32Array(
      buffer.buffer,
      buffer.byteOffset,
      buffer.byteLength >> 2,
    );
  }

  setTextureData(gl: GL) {
    const data = this.data!;
    this.textureLayout = new TextureLayout(data.length);
    this.chunkFormat.setTextureData(gl, this.textureLayout, data);
  }

  getValueAt(dataPosition: Uint32Array): number {
    const { grid } = this;
    if (grid === null) return this.source.spec.fillValue as number;
    return grid.getValue(dataPosition[0], dataPosition[1], dataPosition[2]);
  }
}

export class NanoVdbChunkFormatHandler
  extends RefCounted
  implements ChunkFormatHandler
{
  chunkFormat: ChunkFormat;

  constructor(gl: GL, spec: VolumeChunkSpecification) {
    super();
    this.chunkFormat = this.registerDisposer(
      ChunkFormat.get(gl, spec.dataType),
    );
  }

  getChunk(source: VolumeChunkSource, x: any) {
    return new NanoVdbVolumeChunk(source, x);
  }
}

registerChunkFormatHandler((gl: GL, spec: VolumeChunkSpecification) => {
  if (spec.nanovdbEncoding) {
    return new NanoVdbChunkFormatHandler(gl, spec);
  }
  return null;
});
