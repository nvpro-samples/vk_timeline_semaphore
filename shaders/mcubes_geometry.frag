// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#version 460
#include "camera_transforms.h"
#include "mcubes_debug_view_push_constant.h"
#include "skybox.glsl"

layout(push_constant) uniform PushConstantBlock
{
  McubesDebugViewPushConstant debugViewPushConstant;
};

layout(set = 0, binding = 0) uniform CameraTransformsBuffer
{
  CameraTransforms cameraTransforms;
};

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;

layout(location = 0) out vec4 fragColor;

void main()
{
  if(debugViewPushConstant.enabled == 0.0)
  {
    vec3 normalizedNormal = normalize(worldNormal);  // Natural language "operator oveloading"
    vec3 cameraOrigin     = (cameraTransforms.viewInverse * vec4(0, 0, 0, 1)).xyz;
    vec3 reflected        = normalize(reflect(worldPosition - cameraOrigin, normalizedNormal));
    vec3 reflectColor     = 0.5 * sampleSkyboxNormalized(reflected);
    vec3 normalColor      = vec3(0.125) + 0.125 * normalizedNormal;  // Any way to present 3D color to dichromats?
    fragColor             = vec4(mix(reflectColor, normalColor, cameraTransforms.colorByNormalAmount), 1.0);
  }
  else
  {
    fragColor = vec4(debugViewPushConstant.red, debugViewPushConstant.green, debugViewPushConstant.blue, 1.0);
  }
}
