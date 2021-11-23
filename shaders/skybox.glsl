// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef NVPRO_SAMPLES_VK_TIMELINE_SEMAPHORE_SHADERS_SKYBOX_GLSL_
#define NVPRO_SAMPLES_VK_TIMELINE_SEMAPHORE_SHADERS_SKYBOX_GLSL_

float logistic(float L, float k, float x, float x0)
{
  return L / (1.0 + exp(-k * (x-x0)));
}


// No skybox texture (for now), just compute a color.
vec3 sampleSkyboxNormalized(vec3 normalizedDirection)
{
  vec3 sunDirection = vec3(0.80, 0.60, 0.0);
  vec3 sunColor     = vec3(0.80, 0.80, 0.6);
  vec3 groundColor  = vec3(0.08, 0.08, 0.0);
  vec3 skyColor     = vec3(0.02, 0.15, 0.5);

  float skyness   = logistic(1.0, 3.5, normalizedDirection.y, 0.0);
  float sunCosine = dot(normalizedDirection, sunDirection);
  float sunnyness = logistic(1.0, 12.0, sunCosine, 0.85);

  return mix(groundColor, skyColor, skyness) + sunnyness * sunColor;
}

vec3 sampleSkybox(vec3 direction)
{
  return sampleSkyboxNormalized(normalize(direction));
}

#endif
