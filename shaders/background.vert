// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#version 460
#extension GL_GOOGLE_include_directive : enable

// Basic vertex shader for drawing a full-screen triangle, when drawn
// with 3 vertices as a triangle.
layout(location=0) out vec2 normalizedPixel;
const float MAX_DEPTH = 1.0;

void main()
{
  switch (gl_VertexIndex)
  {
    case 0: gl_Position = vec4(-1, -1, MAX_DEPTH, 1.0); break;
    case 1: gl_Position = vec4(-1, +3, MAX_DEPTH, 1.0); break;
    default:gl_Position = vec4(+3, -1, MAX_DEPTH, 1.0); break;
  }
  normalizedPixel = gl_Position.xy;
}
