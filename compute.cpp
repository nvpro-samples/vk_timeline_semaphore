// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#include "compute.hpp"

#include <atomic>
#include <cassert>
#include <string>
#include <thread>
#include <vulkan/vulkan.h>

#include "nvh/fileoperations.hpp"
#include "nvvk/error_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"

#include "mcubes_chunk.hpp"
#include "timeline_semaphore_main.hpp"

#include "shaders/mcubes_geometry.h"
#include "shaders/mcubes_params.h"

bool g_computeReadyFlag = false;

// Shared pipeline layout for two pipelines.
static VkPipelineLayout s_mcubesPipelineLayout;
static VkPipeline       s_mcubesImagePipeline;
static VkPipeline       s_mcubesGeometryPipeline;

static void setupMcubesPipelineLayout();
static bool setupMcubesImagePipeline(std::string prepend);
static void setupMcubesGeometryPipeline();


void setupCompute(const char* pEquation)
{
  std::string prepend = std::string("#define EQUATION(x, y, z, t) ") + pEquation;
  setupMcubesPipelineLayout();
  bool success = setupMcubesImagePipeline(std::move(prepend));
  assert(success);
  setupMcubesGeometryPipeline();
}

void shutdownCompute()
{
  vkDestroyPipeline(g_ctx, s_mcubesGeometryPipeline, nullptr);
  vkDestroyPipeline(g_ctx, s_mcubesImagePipeline, nullptr);
  vkDestroyPipelineLayout(g_ctx, s_mcubesPipelineLayout, nullptr);
  g_computeReadyFlag = false;
}

// Create a compute pipeline from the given pipeline layout and
// compute shader module. "main" is the entrypoint function.
inline void makeComputePipeline(VkShaderModule   shaderModule,
                                bool             dumpPipelineStats,
                                VkPipelineLayout layout,
                                VkPipeline*      outPipeline,
                                const char*      pShaderName = "<generated shader>")
{
  // Shader module must then get packaged into a <shader stage>
  // This is just an ordinary struct, not a Vulkan object.
  VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                            nullptr,
                                            0,                            // * Must be 0 by Vulkan spec
                                            VK_SHADER_STAGE_COMPUTE_BIT,  // * Type of shader (compute shader)
                                            shaderModule,                 // * Shader module
                                            "main",                       // * Name of function to call
                                            nullptr};                     // * I don't use this

  // Create the compute pipeline. Note that the create struct is
  // typed for different pipeline types (compute, rasterization, ray
  // trace, etc.), yet the VkPipeline output type is the same for all.
  VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                           nullptr,
                                           0,
                                           stageInfo,  // * The compute shader to use
                                           layout,     // * Pipeline Layout
                                           VK_NULL_HANDLE,
                                           0};  // * Unused advanced feature (pipeline caching)
  if(dumpPipelineStats)
  {
    pipelineInfo.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
  }
  NVVK_CHECK(vkCreateComputePipelines(g_ctx,
                                      VK_NULL_HANDLE,    // * Unused (pipeline caching)
                                      1, &pipelineInfo,  // * Array of pipelines to create
                                      nullptr,           // * Default host memory allocator
                                      outPipeline));     // * Pipeline output (array)

  if(dumpPipelineStats)
  {
    nvvk::nvprintPipelineStats(g_ctx, *outPipeline, pShaderName, false);
  }
}

// McubesParams push constant, 1 descriptor set refering to McubesChunk
static void setupMcubesPipelineLayout()
{
  VkPushConstantRange        pushConstant{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(McubesParams)};
  VkPipelineLayoutCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                  nullptr,
                                  0,
                                  1,
                                  &g_mcubesChunkDescriptorSetLayout,
                                  1,
                                  &pushConstant};
  NVVK_CHECK(vkCreatePipelineLayout(g_ctx, &info, nullptr, &s_mcubesPipelineLayout));
}

static bool setupMcubesImagePipeline(std::string prepend)
{
  for(char& c : prepend)
  {
    if(c == '\n')
      c = ' ';
  }
  prepend.push_back('\n');
  auto module_id = g_pShaderCompiler->createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "./shaders/mcubes_image.comp",
                                                         std::move(prepend));
  VkShaderModule module = g_pShaderCompiler->get(module_id);
  if(!module)
  {
    return false;
  }
  vkDestroyPipeline(g_ctx, s_mcubesImagePipeline, nullptr);
  makeComputePipeline(module, false, s_mcubesPipelineLayout, &s_mcubesImagePipeline, "mcubes_image.comp");
  return true;
}

static void setupMcubesGeometryPipeline()
{
  auto module_id = g_pShaderCompiler->createShaderModule(VK_SHADER_STAGE_COMPUTE_BIT, "./shaders/mcubes_geometry.comp");
  makeComputePipeline(g_pShaderCompiler->get(module_id), false, s_mcubesPipelineLayout, &s_mcubesGeometryPipeline,
                      "mcubes_geometry.comp");
}

void computeCmdFillChunkBatch(VkCommandBuffer           cmdBuf,
                              uint32_t                  count,
                              const McubesChunk* const* ppChunks,
                              const McubesParams*       pParams)
{
  // Transition images to general layout, without inserting any execution dependency.
  VkImageMemoryBarrier toGeneralBarriers[MCUBES_MAX_CHUNKS_PER_BATCH];
  assert(count <= MCUBES_MAX_CHUNKS_PER_BATCH);
  for(uint32_t i = 0; i < count; ++i)
  {
    toGeneralBarriers[i].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneralBarriers[i].pNext               = nullptr;
    toGeneralBarriers[i].srcAccessMask       = 0;
    toGeneralBarriers[i].dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    toGeneralBarriers[i].oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
    toGeneralBarriers[i].newLayout           = VK_IMAGE_LAYOUT_GENERAL;
    toGeneralBarriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneralBarriers[i].image               = ppChunks[i]->image.image;
    toGeneralBarriers[i].subresourceRange    = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  }
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, 0, 0, 0,
                       count, toGeneralBarriers);

  // Dispatch fill image shaders.
  for(uint32_t i = 0; i < count; ++i)
  {
    const McubesChunk&  chunk  = *ppChunks[i];
    const McubesParams& params = pParams[i];
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, s_mcubesPipelineLayout, 0, 1, &chunk.set, 0, 0);
    vkCmdPushConstants(cmdBuf, s_mcubesPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, s_mcubesImagePipeline);
    vkCmdDispatch(cmdBuf, MCUBES_CHUNK_EDGE_LENGTH_TEXELS, MCUBES_CHUNK_EDGE_LENGTH_TEXELS, 1);
  }

  // Wait for images to be filled.
  VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT};
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,  //
                       1, &barrier, 0, nullptr, 0, nullptr);

  // Dispatch fill McubesGeometry shaders.
  for(uint32_t i = 0; i < count; ++i)
  {
    const McubesChunk&  chunk  = *ppChunks[i];
    const McubesParams& params = pParams[i];
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, s_mcubesPipelineLayout, 0, 1, &chunk.set, 0, 0);
    vkCmdPushConstants(cmdBuf, s_mcubesPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, s_mcubesGeometryPipeline);
    vkCmdDispatch(cmdBuf, MCUBES_GEOMETRIES_PER_CHUNK, 1, 1);
  }
}

bool computeReplaceEquation(const char* pEquation)
{
  printf("\x1b[34m\x1b[1mEquation:\x1b[0m '%s'\n", pEquation);
  return setupMcubesImagePipeline(std::string("#define EQUATION(x, y, z, t) ") + pEquation);
}
