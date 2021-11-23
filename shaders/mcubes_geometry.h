// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef NVPRO_SAMPLES_VK_TIMELINE_SEMAPHORE_MCUBES_GEOMETRY_H_
#define NVPRO_SAMPLES_VK_TIMELINE_SEMAPHORE_MCUBES_GEOMETRY_H_

#include "mcubes_params.h"

#ifdef __cplusplus
#include <nvmath/nvmath_glsltypes.h>  // emulate glsl types in C++
#include <stdint.h>
#define VEC3 nvmath::vec3f
#else
#define VEC3 vec3
#endif

struct McubesCell
{
#ifdef __cplusplus
  using uint = uint32_t;
#endif
  VEC3 offset;
  uint vertexCount;      // 3 times number of triangles generated in cell.
  uint packedVerts[12];  // bitfield, 10 bits for x, y, z -- see unpackMcubesVertex
};

struct McubesGeometry
{
#ifdef __cplusplus
  using uint = uint32_t;
#endif
  // VkDrawIndirectCommand, keep at offset 0.
  uint vertexCount;    // Set to 12 times number of valid cells.
  uint instanceCount;  // Set to 1
  uint firstVertex;    // Set to 0
  uint firstInstance;  // Set to 0

  // Scale factor for packedVerts data -- see unpackMcubesVertex
  VEC3 packedVertScale;

  float      _cellsPadding[1];
  McubesCell cells[MCUBES_CELLS_PER_GEOMETRY];  // Aligned to 16 bytes
};

#ifdef VULKAN
vec3 unpackMcubesVertex(vec3 packedVertScale, vec3 offset, uint packedVert)
{
  uvec3 unpacked;
  unpacked.x = packedVert & 0x3FF;
  unpacked.y = (packedVert >> 10) & 0x3FF;
  unpacked.z = (packedVert >> 20) & 0x3FF;
  return offset + packedVertScale * vec3(unpacked);
}

// Pack the given 3-vector, with the scale being such that 0 indicates 0.0 and denominator indicates 1.0
uint packNormalizedVec3(vec3 v, uint denominator)
{
  ivec3 unpacked = clamp(ivec3(v * vec3(denominator) + 0.5), ivec3(0), ivec3(denominator));
  return uint(unpacked.x | unpacked.y << 10 | unpacked.z << 20);
}
#endif

#undef VEC3
#endif
