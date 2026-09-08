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
 * @file Parser for NanoVDB grid buffers (ABI major version 32).
 *
 * NanoVDB is a pointer-free linearization of an OpenVDB tree, which makes it
 * usable directly as GPU memory: every link is a byte offset relative to the
 * structure holding it, so the buffer can be uploaded verbatim and traversed
 * in a shader.
 *
 * Only the default 5-4-3 tree configuration is supported: 8^3 leaves, 16^3
 * children per lower internal node (128^3 voxels), and 32^3 children per upper
 * internal node (4096^3 voxels).
 *
 * Layout (see nanovdb/NanoVDB.h):
 *
 *   [GridData(672B)][TreeData(64B)][RootData][N x Root::Tile]
 *      ---[InternalData<5>...]---[InternalData<4>...]---[LeafData<3>...]
 */

import { DataType } from "#src/util/data_type.js";

export const NANOVDB_MAGIC_GRID = 0x314244566f6e614en;
export const NANOVDB_MAGIC_NUMB = 0x304244566f6e614en;

export const GRID_DATA_SIZE = 672;
export const TREE_DATA_SIZE = 64;
export const DATA_ALIGNMENT = 32;

export const LEAF_LOG2DIM = 3;
export const LOWER_LOG2DIM = 4;
export const UPPER_LOG2DIM = 5;
/** Voxels per leaf edge. */
export const LEAF_DIM = 1 << LEAF_LOG2DIM;
/** Total bit shift covered by a lower internal node (128 voxels). */
export const LOWER_TOTAL = LEAF_LOG2DIM + LOWER_LOG2DIM;
/** Total bit shift covered by an upper internal node (4096 voxels). */
export const UPPER_TOTAL = LOWER_TOTAL + UPPER_LOG2DIM;

/** NanoVDB GridType enum values that this reader understands. */
export enum NanoVdbGridType {
  Float = 1,
  Double = 2,
  Int16 = 3,
  Int32 = 4,
  Int64 = 5,
  Vec3f = 6,
  Vec3d = 7,
  Mask = 8,
  UInt32 = 10,
  Vec4f = 17,
  Vec4d = 18,
}

/** GridFlags bits. */
export const GRID_FLAG_HAS_BBOX = 1 << 1;
export const GRID_FLAG_HAS_MINMAX = 1 << 2;
export const GRID_FLAG_HAS_AVERAGE = 1 << 3;
export const GRID_FLAG_HAS_STDDEV = 1 << 4;

export interface NanoVdbValueLayout {
  dataType: DataType;
  /** Components per voxel: 1 for scalar grids, 3 or 4 for vector grids. */
  numComponents: number;
  /** Bytes per voxel value (all components). */
  valueSize: number;
  /** Bytes of a min/max/average statistic. */
  statsSize: number;
  /** Bytes per InternalData tile: max(valueSize, 8). */
  tileSize: number;
}

function alignUp(n: number, a: number = DATA_ALIGNMENT) {
  return Math.ceil(n / a) * a;
}

const VALUE_LAYOUTS: Partial<Record<NanoVdbGridType, NanoVdbValueLayout>> = {
  [NanoVdbGridType.Float]: layout(DataType.FLOAT32, 1, 4, 4),
  [NanoVdbGridType.Int32]: layout(DataType.INT32, 1, 4, 4),
  [NanoVdbGridType.UInt32]: layout(DataType.UINT32, 1, 4, 4),
  [NanoVdbGridType.Int16]: layout(DataType.INT16, 1, 2, 4),
  [NanoVdbGridType.Vec3f]: layout(DataType.FLOAT32, 3, 12, 4),
  [NanoVdbGridType.Vec4f]: layout(DataType.FLOAT32, 4, 16, 4),
};

function layout(
  dataType: DataType,
  numComponents: number,
  valueSize: number,
  statsSize: number,
): NanoVdbValueLayout {
  return {
    dataType,
    numComponents,
    valueSize,
    statsSize,
    tileSize: Math.max(valueSize, 8),
  };
}

/** Byte offset of the tile table within an InternalData of the given log2dim. */
export function internalTableOffset(
  log2dim: number,
  v: NanoVdbValueLayout,
): number {
  const maskBytes = (1 << (3 * log2dim)) / 8;
  return alignUp(24 + 8 + 2 * maskBytes + 2 * v.valueSize + 2 * v.statsSize);
}

/** Total size in bytes of an InternalData of the given log2dim. */
export function internalNodeSize(
  log2dim: number,
  v: NanoVdbValueLayout,
): number {
  return (
    internalTableOffset(log2dim, v) + (1 << (3 * log2dim)) * v.tileSize
  );
}

/** Byte offset of the value array within a LeafData. */
export function leafTableOffset(v: NanoVdbValueLayout): number {
  return alignUp(12 + 3 + 1 + 64 + 2 * v.valueSize + 2 * v.statsSize);
}

/** Total size in bytes of a LeafData. */
export function leafNodeSize(v: NanoVdbValueLayout): number {
  return leafTableOffset(v) + 512 * v.valueSize;
}

/** Size in bytes of RootData (excluding the tile array that follows). */
export function rootDataSize(v: NanoVdbValueLayout): number {
  return alignUp(24 + 4 + 3 * v.valueSize + 2 * v.statsSize);
}

/** Size in bytes of one RootData::Tile. */
export function rootTileSize(v: NanoVdbValueLayout): number {
  return alignUp(8 + 8 + 4 + v.valueSize);
}

/**
 * LeafNode::CoordToOffset -- note x is the *most* significant axis, which is
 * the opposite of the C-order convention used for Zarr chunk shapes.
 */
export function leafCoordToOffset(x: number, y: number, z: number): number {
  return ((x & 7) << 6) | ((y & 7) << 3) | (z & 7);
}

/** InternalNode::CoordToOffset. */
export function internalCoordToOffset(
  x: number,
  y: number,
  z: number,
  log2dim: number,
  childTotal: number,
): number {
  const mask = (1 << (log2dim + childTotal)) - 1;
  return (
    (((x & mask) >> childTotal) << (2 * log2dim)) |
    (((y & mask) >> childTotal) << log2dim) |
    ((z & mask) >> childTotal)
  );
}

export class NanoVdbError extends Error {}

/**
 * Parsed view over a NanoVDB grid buffer. Holds no copies: all accessors read
 * through to the original buffer.
 */
export class NanoVdbGrid {
  readonly dataView: DataView;
  readonly gridType: NanoVdbGridType;
  readonly valueLayout: NanoVdbValueLayout;
  readonly gridName: string;
  readonly flags: number;
  readonly gridSize: number;
  readonly version: readonly [number, number, number];
  /** Byte offsets, absolute within the buffer. */
  readonly treeBase: number;
  readonly rootOffset: number;
  readonly leafOffset: number;
  readonly lowerOffset: number;
  readonly upperOffset: number;
  readonly leafCount: number;
  readonly lowerCount: number;
  readonly upperCount: number;
  readonly activeVoxelCount: number;
  readonly rootTableSize: number;

  constructor(public readonly buffer: Uint8Array) {
    const dv = (this.dataView = new DataView(
      buffer.buffer,
      buffer.byteOffset,
      buffer.byteLength,
    ));
    if (buffer.byteLength < GRID_DATA_SIZE + TREE_DATA_SIZE) {
      throw new NanoVdbError(
        `buffer too small to be a NanoVDB grid: ${buffer.byteLength} bytes`,
      );
    }
    const magic = dv.getBigUint64(0, /*littleEndian=*/ true);
    if (magic !== NANOVDB_MAGIC_GRID && magic !== NANOVDB_MAGIC_NUMB) {
      throw new NanoVdbError(
        `not a NanoVDB grid buffer (magic 0x${magic.toString(16)})`,
      );
    }
    const rawVersion = dv.getUint32(16, true);
    const major = rawVersion >>> 21;
    this.version = [major, (rawVersion >>> 10) & 0x7ff, rawVersion & 0x3ff];
    if (major !== 32) {
      throw new NanoVdbError(
        `unsupported NanoVDB major version ${major} (expected 32)`,
      );
    }
    this.flags = dv.getUint32(20, true);
    this.gridSize = Number(dv.getBigUint64(32, true));
    let nameEnd = 40;
    while (nameEnd < 40 + 256 && buffer[nameEnd] !== 0) ++nameEnd;
    this.gridName = new TextDecoder().decode(buffer.subarray(40, nameEnd));
    const gridType = dv.getUint32(636, true) as NanoVdbGridType;
    const valueLayout = VALUE_LAYOUTS[gridType];
    if (valueLayout === undefined) {
      throw new NanoVdbError(
        `unsupported NanoVDB GridType ${gridType} (${NanoVdbGridType[gridType] ?? "unknown"})`,
      );
    }
    this.gridType = gridType;
    this.valueLayout = valueLayout;

    const t = (this.treeBase = GRID_DATA_SIZE);
    this.leafOffset = t + Number(dv.getBigInt64(t, true));
    this.lowerOffset = t + Number(dv.getBigInt64(t + 8, true));
    this.upperOffset = t + Number(dv.getBigInt64(t + 16, true));
    this.rootOffset = t + Number(dv.getBigInt64(t + 24, true));
    this.leafCount = dv.getUint32(t + 32, true);
    this.lowerCount = dv.getUint32(t + 36, true);
    this.upperCount = dv.getUint32(t + 40, true);
    this.activeVoxelCount = Number(dv.getBigUint64(t + 56, true));
    this.rootTableSize = dv.getUint32(this.rootOffset + 24, true);
  }

  /** Background (inactive) value, as a number for scalar grids. */
  get background(): number {
    return this.readValue(this.rootOffset + 28);
  }

  /** Index-space bounding box of active values: [minX,minY,minZ,maxX,maxY,maxZ]. */
  get indexBBox(): Int32Array {
    const r = new Int32Array(6);
    for (let i = 0; i < 6; ++i) {
      r[i] = this.dataView.getInt32(this.rootOffset + i * 4, true);
    }
    return r;
  }

  /** Root-level min/max of active values, or undefined if not stored. */
  get valueRange(): [number, number] | undefined {
    if ((this.flags & GRID_FLAG_HAS_MINMAX) === 0) return undefined;
    const { valueSize } = this.valueLayout;
    return [
      this.readValue(this.rootOffset + 28 + valueSize),
      this.readValue(this.rootOffset + 28 + 2 * valueSize),
    ];
  }

  /** Reads the first component of a value at an absolute byte offset. */
  readValue(offset: number): number {
    const dv = this.dataView;
    switch (this.valueLayout.dataType) {
      case DataType.FLOAT32:
        return dv.getFloat32(offset, true);
      case DataType.INT32:
        return dv.getInt32(offset, true);
      case DataType.UINT32:
        return dv.getUint32(offset, true);
      case DataType.INT16:
        return dv.getInt16(offset, true);
      default:
        throw new NanoVdbError(
          `unhandled data type ${DataType[this.valueLayout.dataType]}`,
        );
    }
  }

  /** Absolute byte offset of leaf `i` in the leaf array. */
  leafBase(i: number): number {
    return this.leafOffset + i * leafNodeSize(this.valueLayout);
  }

  /** Origin (x, y, z) of leaf `i`, derived from mBBoxMin masked to the lattice. */
  leafOrigin(i: number, out = new Int32Array(3)): Int32Array {
    const base = this.leafBase(i);
    for (let k = 0; k < 3; ++k) {
      out[k] = this.dataView.getInt32(base + k * 4, true) & ~(LEAF_DIM - 1);
    }
    return out;
  }

  /** True if the voxel at leaf-local offset `n` (0..511) is active. */
  leafIsActive(leafIndex: number, n: number): boolean {
    const wordOffset = this.leafBase(leafIndex) + 16 + (n >> 6) * 8;
    const word = this.dataView.getBigUint64(wordOffset, true);
    return ((word >> BigInt(n & 63)) & 1n) === 1n;
  }

  /** Value at leaf-local offset `n`, regardless of active state. */
  leafValue(leafIndex: number, n: number): number {
    const { valueSize } = this.valueLayout;
    return this.readValue(
      this.leafBase(leafIndex) + leafTableOffset(this.valueLayout) + n * valueSize,
    );
  }

  /**
   * Random-access lookup by global index coordinate, walking root -> upper ->
   * lower -> leaf. Returns the background value for inactive voxels.
   *
   * This mirrors what the fragment shader does, and is the CPU path behind
   * `VolumeChunk.getValueAt`.
   */
  getValue(x: number, y: number, z: number): number {
    const dv = this.dataView;
    const v = this.valueLayout;
    const background = this.background;

    // Root: linear scan of the tile table (tiles are few; for chunk-sized
    // grids there is normally exactly one).
    const key = rootCoordToKey(x, y, z);
    const tileBase = this.rootOffset + rootDataSize(v);
    const tileStride = rootTileSize(v);
    let upperBase = -1;
    for (let i = 0; i < this.rootTableSize; ++i) {
      const t = tileBase + i * tileStride;
      if (dv.getBigUint64(t, true) === key) {
        const child = Number(dv.getBigInt64(t + 8, true));
        if (child === 0) {
          // Constant tile: value lives in the tile itself.
          return dv.getUint32(t + 16, true) !== 0
            ? this.readValue(t + 20)
            : background;
        }
        upperBase = this.rootOffset + child;
        break;
      }
    }
    if (upperBase < 0) return background;

    const upperSlot = internalCoordToOffset(
      x,
      y,
      z,
      UPPER_LOG2DIM,
      LOWER_TOTAL,
    );
    const lowerBase = this.descend(upperBase, upperSlot, UPPER_LOG2DIM);
    if (lowerBase === undefined) {
      return this.internalTileValue(upperBase, upperSlot, UPPER_LOG2DIM);
    }

    const lowerSlot = internalCoordToOffset(x, y, z, LOWER_LOG2DIM, LEAF_LOG2DIM);
    const leafBase = this.descend(lowerBase, lowerSlot, LOWER_LOG2DIM);
    if (leafBase === undefined) {
      return this.internalTileValue(lowerBase, lowerSlot, LOWER_LOG2DIM);
    }

    const n = leafCoordToOffset(x, y, z);
    const word = dv.getBigUint64(leafBase + 16 + (n >> 6) * 8, true);
    if (((word >> BigInt(n & 63)) & 1n) !== 1n) return background;
    return this.readValue(leafBase + leafTableOffset(v) + n * v.valueSize);
  }

  /**
   * Returns the absolute offset of the child node at `slot`, or undefined if
   * the slot holds a tile value rather than a child.
   */
  private descend(
    nodeBase: number,
    slot: number,
    log2dim: number,
  ): number | undefined {
    const maskWords = (1 << (3 * log2dim)) / 64;
    // mChildMask follows mBBox (24B), mFlags (8B) and mValueMask.
    const childMaskBase = nodeBase + 32 + maskWords * 8;
    const word = this.dataView.getBigUint64(
      childMaskBase + (slot >> 6) * 8,
      true,
    );
    if (((word >> BigInt(slot & 63)) & 1n) !== 1n) return undefined;
    const table = nodeBase + internalTableOffset(log2dim, this.valueLayout);
    const rel = Number(
      this.dataView.getBigInt64(table + slot * this.valueLayout.tileSize, true),
    );
    return nodeBase + rel;
  }

  /** Value of a (possibly inactive) tile slot on an internal node. */
  private internalTileValue(
    nodeBase: number,
    slot: number,
    log2dim: number,
  ): number {
    const valueMaskBase = nodeBase + 32;
    const word = this.dataView.getBigUint64(
      valueMaskBase + (slot >> 6) * 8,
      true,
    );
    if (((word >> BigInt(slot & 63)) & 1n) !== 1n) return this.background;
    const table = nodeBase + internalTableOffset(log2dim, this.valueLayout);
    return this.readValue(table + slot * this.valueLayout.tileSize);
  }
}

/** RootData::CoordToKey with NANOVDB_USE_SINGLE_ROOT_KEY. */
export function rootCoordToKey(x: number, y: number, z: number): bigint {
  const zk = BigInt((z >>> 0) >>> UPPER_TOTAL);
  const yk = BigInt((y >>> 0) >>> UPPER_TOTAL);
  const xk = BigInt((x >>> 0) >>> UPPER_TOTAL);
  return zk | (yk << 21n) | (xk << 42n);
}

/**
 * Decodes a NanoVDB grid into a dense array in C order ([z][y][x], x
 * innermost), which is the layout Neuroglancer's uncompressed chunk path
 * expects. `origin` is the grid coordinate of output element 0.
 */
export function decodeNanoVdbToDense(
  grid: NanoVdbGrid,
  shape: readonly number[],
  out: Float32Array,
  origin: readonly number[] = [0, 0, 0],
): Float32Array {
  const [nz, ny, nx] = shape;
  out.fill(grid.background);
  const v = grid.valueLayout;
  const vt = leafTableOffset(v);
  const originArr = new Int32Array(3);
  for (let i = 0; i < grid.leafCount; ++i) {
    const base = grid.leafBase(i);
    grid.leafOrigin(i, originArr);
    for (let w = 0; w < 8; ++w) {
      let word = grid.dataView.getBigUint64(base + 16 + w * 8, true);
      while (word !== 0n) {
        const bit = trailingZeros64(word);
        word &= word - 1n;
        const n = (w << 6) | bit;
        const lx = (n >> 6) & 7;
        const ly = (n >> 3) & 7;
        const lz = n & 7;
        const X = originArr[0] + lx - origin[0];
        const Y = originArr[1] + ly - origin[1];
        const Z = originArr[2] + lz - origin[2];
        if (X < 0 || X >= nx || Y < 0 || Y >= ny || Z < 0 || Z >= nz) continue;
        out[(Z * ny + Y) * nx + X] = grid.readValue(base + vt + n * v.valueSize);
      }
    }
  }
  return out;
}

function trailingZeros64(v: bigint): number {
  let n = 0;
  let x = v;
  while ((x & 1n) === 0n) {
    x >>= 1n;
    ++n;
  }
  return n;
}
