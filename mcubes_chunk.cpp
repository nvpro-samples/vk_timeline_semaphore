// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#include "mcubes_chunk.hpp"

#include <cassert>

#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/error_vk.hpp"

#include "timeline_semaphore_main.hpp"

#include "shaders/mcubes_geometry.h"
#include "shaders/mcubes_params.h"

McubesChunk           g_mcubesChunkArray[MCUBES_CHUNK_COUNT];
VkDescriptorSetLayout g_mcubesChunkDescriptorSetLayout;

static nvvk::DescriptorSetContainer s_descriptorSetContainer;
static uint32_t                     s_queueFamilies[2];  // To be filled in.

// Structs used to create McubesChunk::image and McubesChunk::geometryArrayBuffer.
static const VkImageCreateInfo  mcubesImageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                               nullptr,
                                               0,
                                               VK_IMAGE_TYPE_3D,
                                               VK_FORMAT_R32_SFLOAT,
                                               {MCUBES_CHUNK_EDGE_LENGTH_TEXELS, MCUBES_CHUNK_EDGE_LENGTH_TEXELS,
                                                MCUBES_CHUNK_EDGE_LENGTH_TEXELS},
                                               1,
                                               1,
                                               VK_SAMPLE_COUNT_1_BIT,
                                               VK_IMAGE_TILING_OPTIMAL,
                                               VK_IMAGE_USAGE_STORAGE_BIT,
                                               VK_SHARING_MODE_EXCLUSIVE,
                                               0,
                                               0,
                                               VK_IMAGE_LAYOUT_UNDEFINED};
static const VkBufferCreateInfo mcubesBufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                                 nullptr,
                                                 0,
                                                 MCUBES_GEOMETRIES_PER_CHUNK * sizeof(McubesGeometry),
                                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                                     | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                                 VK_SHARING_MODE_CONCURRENT,
                                                 2,
                                                 s_queueFamilies};

void setupMcubesChunks()
{
  // Set up descriptor set layout.
  s_descriptorSetContainer.init(g_ctx);
  s_descriptorSetContainer.addBinding(MCUBES_IMAGE_BINDING, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  s_descriptorSetContainer.addBinding(MCUBES_GEOMETRY_BINDING, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                      VK_SHADER_STAGE_ALL);
  s_descriptorSetContainer.initLayout();
  g_mcubesChunkDescriptorSetLayout = s_descriptorSetContainer.getLayout();

  // Allocate images and buffers. The buffers need to be shared between graphics and compute queues.
  s_queueFamilies[0] = g_ctx.m_queueGCT.familyIndex;
  s_queueFamilies[1] = g_ctx.m_queueC.familyIndex;
  for(uint32_t i = 0; i < MCUBES_CHUNK_COUNT; ++i)
  {
    g_mcubesChunkArray[i].image               = g_allocator.createImage(mcubesImageInfo);
    g_mcubesChunkArray[i].geometryArrayBuffer = g_allocator.createBuffer(mcubesBufferInfo);
  }

  // Allocate image views and descriptor sets.
  s_descriptorSetContainer.initPool(MCUBES_CHUNK_COUNT);
  VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                 nullptr,
                                 0,
                                 VK_NULL_HANDLE,
                                 VK_IMAGE_VIEW_TYPE_3D,
                                 mcubesImageInfo.format,
                                 {},  // Identity rgba swizzle
                                 VK_IMAGE_ASPECT_COLOR_BIT,
                                 0,
                                 1,
                                 0,
                                 1};
  for(uint32_t i = 0; i < MCUBES_CHUNK_COUNT; ++i)
  {
    VkWriteDescriptorSet writes[2];

    // Image View + descriptor
    viewInfo.image = g_mcubesChunkArray[i].image.image;
    NVVK_CHECK(vkCreateImageView(g_ctx, &viewInfo, nullptr, &g_mcubesChunkArray[i].imageView));
    VkDescriptorImageInfo imageRef{VK_NULL_HANDLE, g_mcubesChunkArray[i].imageView, VK_IMAGE_LAYOUT_GENERAL};
    writes[0] = s_descriptorSetContainer.makeWrite(i, MCUBES_IMAGE_BINDING, &imageRef);

    // McubesGeometry Buffer
    VkDescriptorBufferInfo bufferRef{g_mcubesChunkArray[i].geometryArrayBuffer.buffer, 0, mcubesBufferInfo.size};
    writes[1] = s_descriptorSetContainer.makeWrite(i, MCUBES_GEOMETRY_BINDING, &bufferRef);

    // Get descriptor set
    vkUpdateDescriptorSets(g_ctx, 2, writes, 0, nullptr);
    g_mcubesChunkArray[i].set = s_descriptorSetContainer.getSet(i);
    assert(g_mcubesChunkArray[i].set);
  }
}

void shutdownMcubesChunks()
{
  for(uint32_t i = 0; i < MCUBES_CHUNK_COUNT; ++i)
  {
    vkDestroyImageView(g_ctx, g_mcubesChunkArray[i].imageView, nullptr);
    g_allocator.destroy(g_mcubesChunkArray[i].image);
    g_allocator.destroy(g_mcubesChunkArray[i].geometryArrayBuffer);
  }
  s_descriptorSetContainer.deinit();
}
