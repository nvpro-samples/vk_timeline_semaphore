// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vulkan/vulkan.h>

#include "nvvk/resourceallocator_vk.hpp"

// Maximum number of McubesChunk structs to compute or draw per command buffer.
#define MCUBES_MAX_CHUNKS_PER_BATCH 6

// Number of McubesChunk allocated.
// Balance between avoiding synchronization stalls (if too low) and VRAM exhaustion (if too high).
#define MCUBES_CHUNK_COUNT 12

// Bundle of data passed between the marching cubes compute pipeline and the graphics pipeline.
struct McubesChunk
{
  nvvk::Image     image;  // 3D 1-component float32 image
  VkImageView     imageView;
  nvvk::Buffer    geometryArrayBuffer;  // Array of MCUBES_GEOMETRIES_PER_IMAGE McubesGeometry
  VkDescriptorSet set;                  // Using mcubesChunkDescriptorSetLayout

  // Graphics queue waits for this timeline semaphore value:
  // indicates that compute is done filling geometryArrayBuffer (resolve RAW hazard)
  // Compute queue waits for this same timeline semaphore value (on a different semaphore): indicates that
  // graphics is done reading (drawing) geometryArrayBuffer and this McubesChunk can be recycled (resolve WAR hazard)
  uint64_t timelineValue = 0;
};

extern McubesChunk g_mcubesChunkArray[MCUBES_CHUNK_COUNT];

// binding = MCUBES_GEOMETRY_BINDING refers to McubesChunk::geometryArrayBuffer as storage buffer
// binding = MCUBES_IMAGE_BINDING refers to McubesChunk::image as storage image
extern VkDescriptorSetLayout g_mcubesChunkDescriptorSetLayout;

void setupMcubesChunks();
void shutdownMcubesChunks();
