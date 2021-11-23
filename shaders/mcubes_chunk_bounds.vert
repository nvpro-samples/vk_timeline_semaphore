// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
//
// Debug shader for drawing the boundaries of a chunk as GL_LINES
// This is taken from McubesParams push constant.
// Draw with 24 vertices.
#version 460
#include "camera_transforms.h"
#include "mcubes_params.h"

layout(push_constant) uniform PushConstantBlock
{
  McubesParams pushConstant;
};

layout(set = 0, binding = 0) uniform CameraTransformBuffer
{
  CameraTransforms cameraTransforms;
};

layout(location = 0) out float _0_to_1;

// clang-format off
vec3[] cubeTable = vec3[] (
    vec3(0,0,0), vec3(0,0,1), vec3(0,0,1), vec3(0,1,1),
    vec3(0,1,1), vec3(0,1,0), vec3(0,1,0), vec3(0,0,0),

    vec3(0,0,0), vec3(1,0,0), vec3(0,0,1), vec3(1,0,1),
    vec3(0,1,0), vec3(1,1,0), vec3(0,1,1), vec3(1,1,1),

    vec3(1,0,0), vec3(1,0,1), vec3(1,0,1), vec3(1,1,1),
    vec3(1,1,1), vec3(1,1,0), vec3(1,1,0), vec3(1,0,0));
// clang-format on

void main()
{
  vec3 worldCoord = pushConstant.offset + pushConstant.size * cubeTable[gl_VertexIndex];
  gl_Position     = cameraTransforms.viewProj * vec4(worldCoord, 1.0);
  _0_to_1         = float(gl_VertexIndex & 1u);
}
