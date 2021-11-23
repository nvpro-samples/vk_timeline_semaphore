// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef NVPRO_SAMPLES_VK_TIMELINE_SEMAPHORE_MCUBES_PARAMS_H_
#define NVPRO_SAMPLES_VK_TIMELINE_SEMAPHORE_MCUBES_PARAMS_H_

// clang-format off

// Step 1 of marching cubes is to generate the 3D image of sample values.
// This is the dimensions of the image (it's a cube this many texels long on each edge).
#define MCUBES_CHUNK_EDGE_LENGTH_TEXELS 128  // Keep as power of 2

// Same, but in units of marching cube cells (strictly between samples)
#define MCUBES_CHUNK_EDGE_LENGTH_CELLS (MCUBES_CHUNK_EDGE_LENGTH_TEXELS - 1)

// Step 2 is to extract triangles. We subdivide the above image into a
// grid composed of cubes this many cells long on each edge.  Cells
// exist between cubes of 8 adjacent texels (so there are
// (MCUBES_CHUNK_EDGE_LENGTH - 1) ^ 3 cells per image, not evenly
// divided by this). The extracted triangles from one such subdivision
// becomes a McubesGeometry instance.
#define MCUBES_GEOMETRY_EDGE_LENGTH 16  // Keep as power of 2
#define MCUBES_CELLS_PER_GEOMETRY \
  (MCUBES_GEOMETRY_EDGE_LENGTH \
 * MCUBES_GEOMETRY_EDGE_LENGTH \
 * MCUBES_GEOMETRY_EDGE_LENGTH)

// Number of McubesGeometry instances generated per 3D image.
#define MCUBES_GEOMETRIES_PER_CHUNK \
  ((MCUBES_CHUNK_EDGE_LENGTH_TEXELS / MCUBES_GEOMETRY_EDGE_LENGTH) \
 * (MCUBES_CHUNK_EDGE_LENGTH_TEXELS / MCUBES_GEOMETRY_EDGE_LENGTH) \
 * (MCUBES_CHUNK_EDGE_LENGTH_TEXELS / MCUBES_GEOMETRY_EDGE_LENGTH))

#ifdef __cplusplus
#include <nvmath/nvmath_glsltypes.h>  // emulate glsl types in C++
#define VEC3 nvmath::vec3f
#else
#define VEC3 vec3
#endif

#define MCUBES_GEOMETRY_BINDING 0
#define MCUBES_IMAGE_BINDING 1

struct McubesParams
{
  VEC3  offset;  // world coordinate represented by texel 0,0,0
  float t;
  VEC3  size;  // length/height/width of the cuboid to be filled by this compute dispatch.
               // texel [MCUBES_CHUNK_EDGE_LENGTH_CELLS, "", ""] is at world coordinate offset + size
  float _pad[1];
};

#undef VEC3

#endif
