// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
// Polyglot include file (GLSL and C++) for the camera transforms struct.

#ifndef VK_NV_INHERITED_SCISSOR_VIEWPORT_CAMERA_TRANSFORMS_H_
#define VK_NV_INHERITED_SCISSOR_VIEWPORT_CAMERA_TRANSFORMS_H_

#ifdef __cplusplus
#include <stdint.h>
#endif

struct CameraTransforms
{
#ifdef __cplusplus
  using mat4 = glm::mat4;
  using uint = uint32_t;
#endif

  // View and projection matrices, along with their inverses.
  mat4 view;
  mat4 proj;
  mat4 viewProj;  // proj * view
  mat4 viewInverse;
  mat4 projInverse;
  mat4 viewProjInverse;  // (viewProj)^{-1}

  // Extra non-camera parameters that are along for the ride.
  float colorByNormalAmount;
};

#endif /* !VK_NV_INHERITED_SCISSOR_VIEWPORT_CAMERA_TRANSFORMS_H_ */
