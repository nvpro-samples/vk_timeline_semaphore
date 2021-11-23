// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
//
// Vertex shader for unpacking geometry described in a McubesGeometry instance.
// Meant to be used with multi draw indirect (vertex count embedded in McubesGeometry),
// each draw handles one element of the McubesGeometry array (passed as storage buffer).

#version 460
#include "mcubes_geometry.h"

#include "camera_transforms.h"
#include "mcubes_params.h"

layout(set = 0, binding = 0) uniform CameraTransformsBuffer
{
  CameraTransforms cameraTransforms;
};

// Somewhat unusually, we are reading what usually would be vertex attributes directly in the vertex shader.
// This is because the compression scheme in McubesCell can't be supported by fixed-function vertex fetch hardware.
// * Be sure to take this into account for synchronization purposes. *
// This sort of "decompress then draw" work is something mesh shaders excel at, but we are using the
// more familiar vertex shader here.
layout(set = 1, binding = MCUBES_GEOMETRY_BINDING) buffer GeometryBuffer
{
  McubesGeometry geometryArray[];
};

layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;


void main()
{
  uint cellIndex = gl_VertexIndex / 12u;
#define CELL geometryArray[gl_DrawID].cells[cellIndex]

  uint vertIndexInCell = gl_VertexIndex % 12u;
  uint packedVert      = CELL.packedVerts[vertIndexInCell];
  vec3 packedVertScale = geometryArray[gl_DrawID].packedVertScale;
  vec3 offset          = CELL.offset;
  bool degenerateVert  = vertIndexInCell >= CELL.vertexCount;
  vec3 worldVert       = degenerateVert ? vec3(0) : unpackMcubesVertex(packedVertScale, offset, packedVert);
  gl_Position          = cameraTransforms.viewProj * vec4(worldVert, 1.0);

  // The McubesCell specifies anywhere from 0 to 4 triangles (0 to 12 vertices) to draw.
  // Since we are using a vertex shader, we use the degenerate triangle trick to cull the extra triangles.
  // If we were using mesh shaders, we could express this more directly.
  if(!degenerateVert)
  {
    // Deduce normal.
    uint baseVert = (vertIndexInCell / 3u) * 3u;
    vec3 tri0     = unpackMcubesVertex(packedVertScale, vec3(0), CELL.packedVerts[baseVert]);
    vec3 tri1     = unpackMcubesVertex(packedVertScale, vec3(0), CELL.packedVerts[baseVert + 1]);
    vec3 tri2     = unpackMcubesVertex(packedVertScale, vec3(0), CELL.packedVerts[baseVert + 2]);
    worldNormal   = normalize(cross(tri1 - tri0, tri2 - tri1));
  }
}
