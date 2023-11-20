// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
// sRGB <-> Linear color utils.
#ifndef NVPRO_SAMPLES_VK_TIMELINE_SEMAPHORE_SHADERS_SRGB_H_
#define NVPRO_SAMPLES_VK_TIMELINE_SEMAPHORE_SHADERS_SRGB_H_

#ifdef __cplusplus
#include <glm/glm.hpp>
#include "nvmath/nvmath_glsltypes.h"
#define uint uint32_t
#define vec4 glm::vec4
#define clamp glm::clamp
#endif

// Convert 8-bit sRGB red/green/blue component value to linear.
float linearFromSrgb(uint arg)
{
  #ifdef __cplusplus
    arg = arg > 255u ? 255u : arg;
  #else
    arg = min(arg, 255u);
  #endif
  float u = arg * (1/255.);
  return u <= 0.04045 ? u * (25 / 323.)
                      : pow((200 * u + 11) * (1/211.), 2.4);
}

uint srgbFromLinearBias(float arg, float bias)
{
  float srgb = arg <= 0.0031308 ? (323/25.) * arg
                                : 1.055 * pow(arg, 1/2.4) - 0.055;
  return uint(clamp(srgb * 255. + bias, 0, 255));
}

// Convert float linear red/green/blue value to 8-bit sRGB component.
uint srgbFromLinear(float arg)
{
  return srgbFromLinearBias(arg, 0.5);
}

#ifdef __cplusplus
#undef uint
#undef vec4
#undef clamp

#else
vec3 linearFromSrgbVec(uvec3 arg)
{
  return vec3(linearFromSrgb(arg.r), linearFromSrgb(arg.g),
              linearFromSrgb(arg.b));
}

vec4 linearFromSrgbVec(uvec4 arg)
{
  return vec4(linearFromSrgb(arg.r), linearFromSrgb(arg.g),
              linearFromSrgb(arg.b), arg.a * (1.0f / 255.0f));
}

uvec4 srgbFromLinearVec(vec4 arg)
{
  uint alpha = clamp(uint(arg.a * 255.0f + 0.5f), 0u, 255u);
  return uvec4(srgbFromLinear(arg.r), srgbFromLinear(arg.g),
               srgbFromLinear(arg.b), alpha);
}

#endif

#endif // Include guard
