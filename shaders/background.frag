// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#version 460
#extension GL_GOOGLE_include_directive : enable

#include "camera_transforms.h"
#include "skybox.glsl"
#include "srgb.h"

layout(set=0, binding=0) uniform CameraTransformsBuffer
{
  CameraTransforms cameraTransforms;
};

layout(location=0) in vec2 normalizedPixel;
layout(location=0) out vec4 color;

// Copied pseudo random number generation code.
// http://www.jcgt.org/published/0009/03/02/
// Hash Functions for GPU Rendering, Mark Jarzynski, Marc Olano, NVIDIA
uvec3 pcg3d(uvec3 v)
{
  v = v*1664525u + uvec3(1013904223u);
  v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
  v ^= v >> uvec3(16u);
  v.x += v.y*v.z; v.y += v.z*v.x; v.z += v.x*v.y;
  return v;
}

// http://www.thetenthplanet.de/archives/5367
// Apply dithering to hide banding artifacts.
vec3 dither(vec3 linearColor)
{
  vec3  noise = (1.0 / 4294967296.0) * vec3(pcg3d(uvec3(gl_FragCoord.xy, 42)));
  uvec3 lowQuant;
  lowQuant.r = srgbFromLinearBias(linearColor.r, 0.0);
  lowQuant.g = srgbFromLinearBias(linearColor.g, 0.0);
  lowQuant.b = srgbFromLinearBias(linearColor.b, 0.0);
  uvec3 highQuant  = lowQuant + uvec3(1);
  vec3  lowLinear  = linearFromSrgbVec(lowQuant);
  vec3  highLinear = linearFromSrgbVec(highQuant);
  vec3  discr      = mix(lowLinear, highLinear, noise);
  return mix(lowLinear, highLinear, lessThan(discr, linearColor));
}

void main()
{
  mat4 invV = cameraTransforms.viewInverse, invP = cameraTransforms.projInverse;
  // mat4 invV = mat4(1), invP = mat4(1);

  // Convert NDC xy coordinate to world direction, and sample skybox.
  vec3 origin    = (invV * vec4(0, 0, 0, 1)).xyz;
  vec3 target    = (invP * vec4(normalizedPixel, 1, 1)).xyz;
  vec3 direction = normalize((invV * vec4(normalize(target), 0)).xyz);
  color          = vec4(sampleSkyboxNormalized(direction), 1.0);

  // Randomly perturb the color a bit to minimize banding artifacts.
  color.rgb = dither(color.rgb);
}
