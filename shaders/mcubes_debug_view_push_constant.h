// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#ifndef NVPRO_SAMPLES_VK_TIMELINE_SEMAPHORE_MCUBES_DEBUG_VIEW_PUSH_CONSTANT_H_
#define NVPRO_SAMPLES_VK_TIMELINE_SEMAPHORE_MCUBES_DEBUG_VIEW_PUSH_CONSTANT_H_

// Push constant for overriding the fragment color for debug visualization purposes.

struct McubesDebugViewPushConstant
{
  float red;
  float green;
  float blue;
  float enabled;  // Override color if nonzero.
};

#endif
