// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#version 460
layout(location = 0) in float _0_to_1;
layout(location = 0) out vec4 fragColor;

void main()
{
  float m   = abs(2 * _0_to_1 - 1);
  fragColor = vec4(1.0, 1.0 - 0.5 * m, 1.0 - m, 1.0);
}
