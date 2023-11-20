// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

#include <glm/glm.hpp>

#include "timeline_semaphore_main.hpp"

#include <cassert>
#include <math.h>
#include <future>
#include <stdexcept>
#include <utility>
#include <vector>

// nvpro_core headers
#include "nvvk/error_vk.hpp"
#include "nvvk/images_vk.hpp"

// Header files for this project
#include "compute.hpp"
#include "graphics.hpp"
#include "gui.hpp"
#include "mcubes_chunk.hpp"
#include "search_paths.hpp"

// GLSL/C++ shared header files
#include "shaders/mcubes_debug_view_push_constant.h"
#include "shaders/mcubes_geometry.h"

GLFWwindow*                      g_window;
nvvk::Context                    g_ctx;
nvvk::ResourceAllocatorDedicated g_allocator;
VkSurfaceKHR                     g_surface;
nvvk::SwapChain                  g_swapChain;
VkQueue                          g_gctQueue, g_computeQueue;
VkCommandPool                    g_gctPool, g_computePool;
nvvk::ShaderModuleManager*       g_pShaderCompiler;
uint64_t                         g_frameNumber = 0;  // First frame is number 1.

static VkFence         s_submitFrameFences[2];
static VkCommandBuffer s_submitFrameCommandBuffers[2];
static uint32_t        s_windowWidth, s_windowHeight;

// Command pools for the "main" compute and drawing commands.
// Alternate usage per frame.
// When using timeline semaphores, we need to wait for a timeline semaphore value to know when it's safe to reset.
// When using one queue, the fences serve this purpose.
static VkCommandPool s_frameComputePools[2];   // For g_computeQueue
static VkCommandPool s_frameGraphicsPools[2];  // For g_gctQueue
static uint64_t      s_frameComputePoolWaitTimelineValues[2]  = {0, 0};
static uint64_t      s_frameGraphicsPoolWaitTimelineValues[2] = {0, 0};
static VkFence       s_frameComputePoolFences[2];
static VkFence       s_frameGraphicsPoolFences[2];

// Command buffers allocated from the above pools.
static std::vector<VkCommandBuffer> s_frameComputeCmdBufs[2];
static std::vector<VkCommandBuffer> s_frameGraphicsCmdBufs[2];

// Timeline semaphores
// Graphics queue waits on this semaphore to know when an McubesGeometry is fully ready to draw (resolve RAW hazard)
static VkSemaphore s_computeDoneTimelineSemaphore;
// Compute queue waits on this semaphore to know when an McubesGeometry has already been read from, and therefore
// can safely be filled with new, different data (WAR hazard).
static VkSemaphore s_graphicsDoneTimelineSemaphore;
// This is incremented upon each submit that signals (increments) the above semaphores, and indicates the
// value that the semaphore will have upon the submitted work being COMPLETED.
static uint64_t s_upcomingTimelineValue = 1;
// We are using the array of McubesChunk (g_mcubesChunkArray) as a ring buffer for communication between
// compute and graphics queues; this is the cycling index into that array.
static uint32_t s_mcubesChunkIndex = 0;

static bool s_useComputeQueue;



static void setupGlobals()
{
  // * Create GLFW window.
  glfwInit();
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  s_windowWidth  = 1920;
  s_windowHeight = 1080;
  g_window = glfwCreateWindow(s_windowWidth, s_windowHeight, "nvpro Vulkan Timeline Semaphores", nullptr, nullptr);
  if(g_window == nullptr)
  {
    throw std::runtime_error("GLFW window failed to create");
  }

  // * Init Vulkan 1.1 device with needed extensions.
  nvvk::ContextCreateInfo deviceInfo;
  // GLFW (window) extensions.
  const char** glfwExtensions;
  uint32_t     glfwExtensionCount = 0;
  glfwExtensions                  = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
  if(glfwExtensions == nullptr)
  {
    throw std::runtime_error("GLFW Vulkan extension failed");
  }
  deviceInfo.apiMajor = 1;
  deviceInfo.apiMinor = 1;
  for(uint32_t i = 0; i < glfwExtensionCount; ++i)
  {
    deviceInfo.addInstanceExtension(glfwExtensions[i]);
  }
  deviceInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  // Timeline semaphore extension (core in Vulkan 1.2, but still need to enable the feature later).
  VkPhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
  deviceInfo.addDeviceExtension(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, false,  // not optional
                                &timelineSemaphoreFeatures);
  // Initialize device
  g_ctx.init(deviceInfo);
  g_ctx.ignoreDebugMessage(1303270965);  // Bogus "general layout" perf warning.
  // Check needed feature.
  if(!timelineSemaphoreFeatures.timelineSemaphore)
  {
    throw std::runtime_error("Missing timelineSemaphore feature");
  }
  // NOTE For Vulkan 1.2, you must instead enable this feature in VkPhysicalDeviceVulkan12Features::timelineSemaphore.
  // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkPhysicalDeviceVulkan12Features.html

  // * Init memory allocator helper.
  g_allocator.init(g_ctx, g_ctx.m_physicalDevice);

  // * Init swap chain.
  g_surface = VK_NULL_HANDLE;
  glfwCreateWindowSurface(g_ctx.m_instance, g_window, nullptr, &g_surface);
  if(!g_surface)
  {
    throw std::runtime_error("Failed to extract VkSurfaceKHR from GLFW window");
  }
  g_ctx.setGCTQueueWithPresent(g_surface);
  const auto format = VK_FORMAT_B8G8R8A8_SRGB;
  const auto usage  = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  if(!g_swapChain.init(g_ctx, g_ctx.m_physicalDevice, g_ctx.m_queueGCT, g_ctx.m_queueGCT, g_surface, format, usage))
  {
    throw std::runtime_error("Swap chain failed to initialize");
  }
  g_swapChain.setWaitQueue(g_ctx.m_queueGCT);
  g_swapChain.update(s_windowWidth, s_windowHeight, false);

  // * Check needed queues and create corresponding command pools.
  g_gctQueue     = g_ctx.m_queueGCT;
  g_computeQueue = g_ctx.m_queueC;
  if(!g_gctQueue)
    throw std::runtime_error("Missing needed graphics/compute VkQueue");
  if(!g_computeQueue)
    throw std::runtime_error("Missing needed dedicated compute VkQueue");
  VkCommandPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                         VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
  // Graphics command pool
  poolCreateInfo.queueFamilyIndex = g_ctx.m_queueGCT.familyIndex;
  NVVK_CHECK(vkCreateCommandPool(g_ctx, &poolCreateInfo, nullptr, &g_gctPool));
  // Compute command pool
  poolCreateInfo.queueFamilyIndex = g_ctx.m_queueC.familyIndex;
  NVVK_CHECK(vkCreateCommandPool(g_ctx, &poolCreateInfo, nullptr, &g_computePool));

  // * Set up shader compiler, with search directories.
  g_pShaderCompiler = new nvvk::ShaderModuleManager(g_ctx);
  for(const std::string& path : searchPaths)
  {
    g_pShaderCompiler->addDirectory(path);
  }
}

static void shutdownGlobals()
{
  // * Shut down shader compiler
  delete g_pShaderCompiler;
  g_pShaderCompiler = nullptr;

  // * Clean up command pools.
  vkDestroyCommandPool(g_ctx, g_gctPool, nullptr);
  vkDestroyCommandPool(g_ctx, g_computePool, nullptr);

  // * Shut down swap chain.
  g_swapChain.deinit();
  vkDestroySurfaceKHR(g_ctx.m_instance, g_surface, nullptr);

  // * Shut down memory allocator.
  g_allocator.deinit();

  // * Shut down Vulkan device
  g_ctx.deinit();

  // * Shut down GLFW
  glfwDestroyWindow(g_window);
  glfwTerminate();
}

static void setupStatics()
{
  // Fences
  VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, nullptr, VK_FENCE_CREATE_SIGNALED_BIT};
  NVVK_CHECK(vkCreateFence(g_ctx, &fenceInfo, nullptr, &s_submitFrameFences[0]));
  NVVK_CHECK(vkCreateFence(g_ctx, &fenceInfo, nullptr, &s_submitFrameFences[1]));
  NVVK_CHECK(vkCreateFence(g_ctx, &fenceInfo, nullptr, &s_frameComputePoolFences[0]));
  NVVK_CHECK(vkCreateFence(g_ctx, &fenceInfo, nullptr, &s_frameComputePoolFences[1]));
  NVVK_CHECK(vkCreateFence(g_ctx, &fenceInfo, nullptr, &s_frameGraphicsPoolFences[0]));
  NVVK_CHECK(vkCreateFence(g_ctx, &fenceInfo, nullptr, &s_frameGraphicsPoolFences[1]));

  // Allocate graphics command buffers for "submit frame" commands.
  VkCommandBufferAllocateInfo cmdBufInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, g_gctPool,
                                            VK_COMMAND_BUFFER_LEVEL_PRIMARY, 2};
  NVVK_CHECK(vkAllocateCommandBuffers(g_ctx, &cmdBufInfo, s_submitFrameCommandBuffers));

  // Allocate command pools for frame drawing and compute commands.
  VkCommandPoolCreateInfo poolCreateInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                         VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
  poolCreateInfo.queueFamilyIndex = g_ctx.m_queueGCT.familyIndex;
  NVVK_CHECK(vkCreateCommandPool(g_ctx, &poolCreateInfo, nullptr, &s_frameGraphicsPools[0]));
  NVVK_CHECK(vkCreateCommandPool(g_ctx, &poolCreateInfo, nullptr, &s_frameGraphicsPools[1]));
  poolCreateInfo.queueFamilyIndex = g_ctx.m_queueC.familyIndex;
  NVVK_CHECK(vkCreateCommandPool(g_ctx, &poolCreateInfo, nullptr, &s_frameComputePools[0]));
  NVVK_CHECK(vkCreateCommandPool(g_ctx, &poolCreateInfo, nullptr, &s_frameComputePools[1]));

  // Allocate the timeline semaphores; initial value 0. Need extension struct for this.
  VkSemaphoreTypeCreateInfo timelineSemaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, nullptr,
                                                     VK_SEMAPHORE_TYPE_TIMELINE, 0};
  VkSemaphoreCreateInfo     semaphoreInfo         = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, &timelineSemaphoreInfo};
  NVVK_CHECK(vkCreateSemaphore(g_ctx, &semaphoreInfo, nullptr, &s_computeDoneTimelineSemaphore));
  NVVK_CHECK(vkCreateSemaphore(g_ctx, &semaphoreInfo, nullptr, &s_graphicsDoneTimelineSemaphore));
}

static void shutdownStatics()
{
  vkDestroyFence(g_ctx, s_submitFrameFences[0], nullptr);
  vkDestroyFence(g_ctx, s_submitFrameFences[1], nullptr);
  vkDestroyFence(g_ctx, s_frameGraphicsPoolFences[0], nullptr);
  vkDestroyFence(g_ctx, s_frameGraphicsPoolFences[1], nullptr);
  vkDestroyFence(g_ctx, s_frameComputePoolFences[0], nullptr);
  vkDestroyFence(g_ctx, s_frameComputePoolFences[1], nullptr);
  vkDestroySemaphore(g_ctx, s_computeDoneTimelineSemaphore, nullptr);
  vkDestroySemaphore(g_ctx, s_graphicsDoneTimelineSemaphore, nullptr);
  vkDestroyCommandPool(g_ctx, s_frameGraphicsPools[0], nullptr);
  vkDestroyCommandPool(g_ctx, s_frameGraphicsPools[1], nullptr);
  vkDestroyCommandPool(g_ctx, s_frameComputePools[0], nullptr);
  vkDestroyCommandPool(g_ctx, s_frameComputePools[1], nullptr);
}

// Update the framebuffer size for the glfw window; suspend until the
// glfw window has nonzero size (i.e. not minimized).
static void waitNonzeroFramebufferSize()
{
  int width, height;
  glfwGetFramebufferSize(g_window, &width, &height);
  while(width == 0 || height == 0)
  {
    glfwWaitEvents();
    glfwGetFramebufferSize(g_window, &width, &height);
  }
  s_windowWidth  = uint32_t(width);
  s_windowHeight = uint32_t(height);
}

// Return a list of 3D marching cubes images to fill and draw.
static std::vector<McubesParams> getMcubesParamsList(float t)
{
  std::vector<McubesParams> result;
  for(int32_t z = -2; z < 2; ++z)
  {
    for(int32_t y = -2; y < 2; ++y)
    {
      for(int32_t x = -2; x < 2; ++x)
      {
        result.push_back({{x, y, z}, t, {1.0f, 1.0f, 1.0f}});
      }
    }
  }
  return result;
}

// Helper for getting the list of colors to draw each chunk when using debug visualization modes.
// Returns empty vector if no such mode is enabled.
static std::vector<McubesDebugViewPushConstant> makeDebugColors(int                 chunkDebugViewMode,
                                                                uint32_t            batchNumber,
                                                                uint32_t            firstChunkUsed,
                                                                uint32_t            chunkCount,
                                                                McubesChunk* const* chunkPointerArray)
{
  std::vector<McubesDebugViewPushConstant> result;
  McubesDebugViewPushConstant              pc;
  float                                    tmp, rb, g;  // magenta-green is clear to all major forms of colorblindness.
  switch(chunkDebugViewMode)
  {
    case chunkDebugViewBatch:
      tmp        = float(batchNumber % 5u);
      rb         = tmp == 0 ? 0.0f : tmp == 1 ? 0.75f : 1.0f;
      g          = tmp == 2 ? 0.0f : tmp == 3 ? 0.75f : 1.0f;
      tmp        = powf(0.75f, float((batchNumber / 5u) % 8u));
      pc.red     = rb * tmp;
      pc.green   = g * tmp;
      pc.blue    = rb * tmp;
      pc.enabled = 1;
      for(uint32_t i = 0; i < chunkCount; ++i)
        result.push_back(pc);
      return result;
    case chunkDebugViewChunkIndex:
      for(uint32_t i = 0; i < chunkCount; ++i)
      {
        // Color based on relative index from first chunk used in frame, otherwise, we get massive flickering.
        ptrdiff_t chunkIndex = chunkPointerArray[i] - g_mcubesChunkArray;
        assert(chunkIndex < MCUBES_CHUNK_COUNT);
        int32_t relativeChunkIndex = int32_t(chunkIndex) - int32_t(firstChunkUsed);
        relativeChunkIndex += relativeChunkIndex < 0 ? MCUBES_CHUNK_COUNT : 0;
        static_assert(MCUBES_CHUNK_COUNT <= 25, "Colors not guaranteed to be unique");
        rb         = powf(0.5f, float(relativeChunkIndex % 5));
        g          = powf(0.5f, float(relativeChunkIndex / 5));
        pc.red     = rb;
        pc.green   = g;
        pc.blue    = rb;
        pc.enabled = 1;
        result.push_back(pc);
      }
      return result;
    default:
      return result;
  }
}

// clang-format off

// Submit compute and graphics commands for generating marching cubes geometry
// and drawing it to the offscreen framebuffer.
// THIS is the main point of the sample.
static void computeDrawCommandsTwoQueues(const Gui* pGui)
{
  // Pick and reset the command pools (graphics and compute) for this frame.
  // Need to wait on the timeline semaphores to know when all command buffers in the pool to reset have retired.
  VkSemaphore         waitSemaphores[2] = {s_computeDoneTimelineSemaphore,
                                           s_graphicsDoneTimelineSemaphore};
  uint64_t            waitValues[2]     = {s_frameComputePoolWaitTimelineValues[g_frameNumber & 1u],
                                           s_frameGraphicsPoolWaitTimelineValues[g_frameNumber & 1u]};
  VkSemaphoreWaitInfo waitInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, nullptr, 0, // default -- wait for all
                                  2, waitSemaphores, waitValues};
  vkWaitSemaphoresKHR(g_ctx, &waitInfo, ~uint64_t(0));  // or vkWaitSemaphores in Vulkan 1.2
  // clang-format on

  // Reset command pools.
  VkCommandPool ourComputePool  = s_frameComputePools[g_frameNumber & 1u];
  VkCommandPool ourGraphicsPool = s_frameGraphicsPools[g_frameNumber & 1u];
  NVVK_CHECK(vkResetCommandPool(g_ctx, ourComputePool, 0));  // 0 = don't release resources; we'll still need them soon.
  NVVK_CHECK(vkResetCommandPool(g_ctx, ourGraphicsPool, 0));
  std::vector<VkCommandBuffer>& ourComputeCmdBufs  = s_frameComputeCmdBufs[g_frameNumber & 1u];
  std::vector<VkCommandBuffer>& ourGraphicsCmdBufs = s_frameGraphicsCmdBufs[g_frameNumber & 1u];

  // List of compute and graphics jobs to run.
  std::vector<McubesParams> paramsList = pGui->getMcubesJobs();

  // Structs for allocating or recycling command buffers.
  // Note that we need to recycle command buffers, because command pool resets only reset the command buffers,
  // not actually destroy them.
  uint32_t                 nextCmdBufIndex       = 0;
  VkCommandBufferBeginInfo oneTimeBeginInfo      = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  oneTimeBeginInfo.flags                         = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VkCommandBufferAllocateInfo computeCmdBufInfo  = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
                                                   ourComputePool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
  VkCommandBufferAllocateInfo graphicsCmdBufInfo = computeCmdBufInfo;
  graphicsCmdBufInfo.commandPool                 = ourGraphicsPool;
  VkCommandBuffer batchComputeCmdBuf, batchGraphicsCmdBuf;

  // Set up queue submission structs ahead-of-time.
  // Because timeline semaphores are a later addition to Vulkan, WHICH semaphore to wait/signal on
  // is in a separate struct from WHAT value to wait/set the timeline semaphore to.
  uint64_t                      computeWaitTimelineValue = 0, computeSignalTimelineValue = 0;
  uint64_t                      graphicsWaitTimelineValue = 0, graphicsSignalTimelineValue = 0;
  VkTimelineSemaphoreSubmitInfo computeTimelineInfo = {
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      nullptr,
      1,
      &computeWaitTimelineValue,  // Compute queue waits for /at least/ this timeline semaphore value of
      1,                          // s_graphicsDoneTimelineSemaphore (semaphore set below).
      &computeSignalTimelineValue};
  VkTimelineSemaphoreSubmitInfo graphicsTimelineInfo = {
      VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
      nullptr,
      1,
      &graphicsWaitTimelineValue,  // Graphic queue waits for /at least/ this timeline semaphore value of
      1,                           // s_computeDoneTimelineSemaphore (semaphore set below).
      &graphicsSignalTimelineValue};

  VkPipelineStageFlags computeStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  VkPipelineStageFlags readGeometryArrayStage =
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
  // See NOTE -- readGeometryArrayStage

  VkSubmitInfo computeSubmitInfo  = {VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                    &computeTimelineInfo,  // Extension struct
                                    1,
                                    &s_graphicsDoneTimelineSemaphore,  // Compute waits for graphics queue
                                    &computeStage,                     // Waits for semaphore before starting compute
                                    1,
                                    &batchComputeCmdBuf,
                                    1,
                                    &s_computeDoneTimelineSemaphore};
  VkSubmitInfo graphicsSubmitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                     &graphicsTimelineInfo,  // Extension struct
                                     1,
                                     &s_computeDoneTimelineSemaphore,  // Wait for compute queue
                                     &readGeometryArrayStage,          // the stage that reads McubesChunk
                                     1,
                                     &batchGraphicsCmdBuf,
                                     1,
                                     &s_graphicsDoneTimelineSemaphore};

  // Split the list of jobs into batches of up to batchSize McubesChunk jobs.
  uint32_t batchSize  = glm::clamp<uint32_t>(pGui->m_batchSize, 1u, MCUBES_MAX_CHUNKS_PER_BATCH);
  uint32_t batchCount = uint32_t(paramsList.size() + batchSize - 1u) / batchSize;
  uint32_t firstChunkUsed;

  // Record and submit fill and draw McubesChunk commands.
  for(uint32_t batch = 0; batch < batchCount; ++batch)
  {
    // Allocate or recycle command buffers for batch.
    if(nextCmdBufIndex < ourComputeCmdBufs.size())
    {
      batchComputeCmdBuf = ourComputeCmdBufs[nextCmdBufIndex];  // recycle
    }
    else
    {
      NVVK_CHECK(vkAllocateCommandBuffers(g_ctx, &computeCmdBufInfo, &batchComputeCmdBuf));
      ourComputeCmdBufs.push_back(batchComputeCmdBuf);  // allocate new, and save for future recycling.
    }
    if(nextCmdBufIndex < ourGraphicsCmdBufs.size())
    {
      batchGraphicsCmdBuf = ourGraphicsCmdBufs[nextCmdBufIndex];
    }
    else
    {
      NVVK_CHECK(vkAllocateCommandBuffers(g_ctx, &graphicsCmdBufInfo, &batchGraphicsCmdBuf));
      ourGraphicsCmdBufs.push_back(batchGraphicsCmdBuf);
    }
    NVVK_CHECK(vkBeginCommandBuffer(batchComputeCmdBuf, &oneTimeBeginInfo));
    NVVK_CHECK(vkBeginCommandBuffer(batchGraphicsCmdBuf, &oneTimeBeginInfo));
    nextCmdBufIndex++;

    // Start-of-frame commands (clear depth buffer, etc.)
    if(batch == 0)
    {
      CameraTransforms cameraTransforms = pGui->getTransforms(s_windowWidth, s_windowHeight);
      graphicsCmdPrepareFrame(batchGraphicsCmdBuf, &cameraTransforms);
    }

    // Record compute and draw commands for batch.
    uint32_t batchStart = batch * batchSize, batchEnd = std::min(batchStart + batchSize, uint32_t(paramsList.size()));
    // List of McubesChunk objects to use for compute->graphics communication in this batch.
    McubesChunk* chunkPointerArray[MCUBES_MAX_CHUNKS_PER_BATCH];
    for(uint32_t localIndex = 0, paramIndex = batchStart; paramIndex < batchEnd; ++paramIndex, ++localIndex)
    {
      // Select next McubesChunk in ringbuffer array.
      ++s_mcubesChunkIndex;
      if(s_mcubesChunkIndex >= MCUBES_CHUNK_COUNT)
      {
        s_mcubesChunkIndex = 0;
      }
      chunkPointerArray[localIndex] = &g_mcubesChunkArray[s_mcubesChunkIndex];
      if(localIndex == 0 && batch == 0)
        firstChunkUsed = s_mcubesChunkIndex;  // Just for debug color view
    }

    // Record compute commands.
    // We also keep track of the s_graphicsDoneTimelineSemaphore value that these compute commands
    // need to wait on (to safely recycle the McubesChunk).
    for(uint32_t localIndex = 0, paramIndex = batchStart; paramIndex < batchEnd; ++paramIndex, ++localIndex)
    {
      computeWaitTimelineValue = std::max(computeWaitTimelineValue, chunkPointerArray[localIndex]->timelineValue);
    }
    computeCmdFillChunkBatch(batchComputeCmdBuf, batchEnd - batchStart, chunkPointerArray, &paramsList[batchStart]);

    // Ensure memory dependency resolved between upcoming compute command and upcoming graphics commands.
    // This is separate from (and an additional requirement on top of) the execution dependency
    // handled by the timeline semaphore.
    // No queue ownership transfer -- using VK_SHARING_MODE_CONCURRENT.
    VkMemoryBarrier computeToGraphicsBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT,
                                                VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(batchGraphicsCmdBuf, computeStage, readGeometryArrayStage, 0, 1, &computeToGraphicsBarrier, 0,
                         0, 0, 0);

    // Graphics commands.
    for(uint32_t localIndex = 0, paramIndex = batchStart; paramIndex < batchEnd; ++paramIndex, ++localIndex)
    {
      // Record the s_graphicsDoneTimelineSemaphore value for this McubesChunk that indicates readiness for recycling.
      chunkPointerArray[localIndex]->timelineValue = s_upcomingTimelineValue;
    }
    std::vector<McubesDebugViewPushConstant> debugColors =
        makeDebugColors(pGui->m_chunkDebugViewMode, batch, firstChunkUsed, batchEnd - batchStart, chunkPointerArray);
    const bool          drawChunkBounds = pGui->m_chunkDebugViewMode != chunkDebugViewOff;
    const McubesParams* pDebugBoxes     = drawChunkBounds ? &paramsList[batchStart] : nullptr;
    graphicsCmdDrawMcubesGeometryBatch(batchGraphicsCmdBuf, batchEnd - batchStart, chunkPointerArray, pDebugBoxes,
                                       debugColors.empty() ? nullptr : debugColors.data());

    if(batch == batchCount - 1u)
    {
      // Include ImGui commands on last batch.
      graphicsCmdDrawImGui(batchGraphicsCmdBuf);
    }

    assert(computeWaitTimelineValue < s_upcomingTimelineValue);  // Circular dependency check.

    // Compute submit, waits for s_graphicsDoneTimelineSemaphore's value == computeWaitTimelineValue
    // and, upon completion, sets s_computeDoneTimelineSemaphore's value := s_upcomingTimelineValue
    NVVK_CHECK(vkEndCommandBuffer(batchComputeCmdBuf));
    // computeWaitTimelineValue will be deduced concurrent with command recording.
    computeSignalTimelineValue = s_upcomingTimelineValue;
    NVVK_CHECK(vkQueueSubmit(g_computeQueue, 1, &computeSubmitInfo, VkFence{}));

    // Graphics submit -- wait for the above just-submitted command to finish by waiting for
    // s_computeDoneTimelineSemaphore's value == s_upcomingTimelineValue and also set
    // s_graphicsDoneTimelineSemaphore's value := s_upcomingTimelineValue
    NVVK_CHECK(vkEndCommandBuffer(batchGraphicsCmdBuf));
    graphicsWaitTimelineValue   = s_upcomingTimelineValue;
    graphicsSignalTimelineValue = s_upcomingTimelineValue;
    NVVK_CHECK(vkQueueSubmit(g_gctQueue, 1, &graphicsSubmitInfo, VkFence{}));
    // We could have just set the VkTimelineSemaphoreSubmitInfo pointers directly, but we do it this way for teaching.

    // For the last batch, remember the timeline semaphore values that
    // lets future us know when we can reset the command pools.
    if(batch == batchCount - 1u)
    {
      s_frameComputePoolWaitTimelineValues[g_frameNumber & 1u]  = computeSignalTimelineValue;
      s_frameGraphicsPoolWaitTimelineValues[g_frameNumber & 1u] = graphicsSignalTimelineValue;
    }

    ++s_upcomingTimelineValue;
  }  // End for each batch
}

// NOTE -- readGeometryArrayStage
//
// Typically, vertex data is consumed in the VK_PIPELINE_STAGE_VERTEX_INPUT_BIT stage, which corresponds to
// fixed-function vertex attribute reads; however, mcubes_geometry.vert does the reads in the vertex shader
// itself, so we are using VK_PIPELINE_STAGE_VERTEX_SHADER_BIT instead.
//
// We also need the draw indirect bit, since indirect commands are read from the buffer.


// For comparison purposes, submit the compute and draw McubesChunk commands using only the GCT queue.
static void computeDrawCommandsGctOnly(const Gui* pGui)
{
  // Pick and reset the graphics (gct) command pool for this frame; wait on protecting fence.
  VkFence ourGraphicsFence = s_frameGraphicsPoolFences[g_frameNumber & 1u];
  NVVK_CHECK(vkWaitForFences(g_ctx, 1, &ourGraphicsFence, VK_TRUE, ~uint64_t(0)));
  NVVK_CHECK(vkResetFences(g_ctx, 1, &ourGraphicsFence));
  VkCommandPool ourGraphicsPool = s_frameGraphicsPools[g_frameNumber & 1u];
  NVVK_CHECK(vkResetCommandPool(g_ctx, ourGraphicsPool, 0));
  std::vector<VkCommandBuffer>& ourGraphicsCmdBufs = s_frameGraphicsCmdBufs[g_frameNumber & 1u];

  // List of compute and graphics jobs to run.
  std::vector<McubesParams> paramsList = pGui->getMcubesJobs();

  // Structs for allocating or recycling command buffers.
  uint32_t                 nextCmdBufIndex       = 0;
  VkCommandBufferBeginInfo oneTimeBeginInfo      = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  oneTimeBeginInfo.flags                         = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VkCommandBufferAllocateInfo graphicsCmdBufInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr,
                                                    ourGraphicsPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};

  VkPipelineStageFlags computeStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  VkPipelineStageFlags readGeometryArrayStage =
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
  // See NOTE -- readGeometryArrayStage

  // Set up queue submission struct ahead-of-time.
  VkCommandBuffer gctBatchCmdBuf;
  VkSubmitInfo    submitInfo = {
      VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &gctBatchCmdBuf, 0, nullptr};

  // Split the list of jobs into batches of up to batchSize McubesChunk jobs.
  uint32_t batchSize  = glm::clamp<uint32_t>(pGui->m_batchSize, 1u, MCUBES_MAX_CHUNKS_PER_BATCH);
  uint32_t batchCount = uint32_t(paramsList.size() + batchSize - 1u) / batchSize;
  uint32_t firstChunkUsed;

  // Record and submit fill and draw McubesChunk commands.
  for(uint32_t batch = 0; batch < batchCount; ++batch)
  {
    // Allocate or recycle new command buffer.
    if(nextCmdBufIndex < ourGraphicsCmdBufs.size())
    {
      gctBatchCmdBuf = ourGraphicsCmdBufs[nextCmdBufIndex];
    }
    else
    {
      NVVK_CHECK(vkAllocateCommandBuffers(g_ctx, &graphicsCmdBufInfo, &gctBatchCmdBuf));
      ourGraphicsCmdBufs.push_back(gctBatchCmdBuf);
    }
    NVVK_CHECK(vkBeginCommandBuffer(gctBatchCmdBuf, &oneTimeBeginInfo));
    nextCmdBufIndex++;

    if(batch == 0)
    {
      // Start-of-frame commands (clear depth buffer, etc.)
      CameraTransforms cameraTransforms = pGui->getTransforms(s_windowWidth, s_windowHeight);
      graphicsCmdPrepareFrame(gctBatchCmdBuf, &cameraTransforms);
    }

    // Record compute and draw commands for batch.
    uint32_t batchStart = batch * batchSize, batchEnd = std::min(batchStart + batchSize, uint32_t(paramsList.size()));
    // List of McubesChunk objects to use for compute->graphics communication in this batch.
    McubesChunk* chunkPointerArray[MCUBES_MAX_CHUNKS_PER_BATCH];
    for(uint32_t localIndex = 0, paramIndex = batchStart; paramIndex < batchEnd; ++paramIndex, ++localIndex)
    {
      // Select next McubesChunk in ringbuffer array.
      ++s_mcubesChunkIndex;
      if(s_mcubesChunkIndex >= MCUBES_CHUNK_COUNT)
      {
        s_mcubesChunkIndex = 0;
      }
      chunkPointerArray[localIndex] = &g_mcubesChunkArray[s_mcubesChunkIndex];
      if(localIndex == 0 && batch == 0)
        firstChunkUsed = s_mcubesChunkIndex;  // Just for debug color view
    }

    // Record compute commands.
    computeCmdFillChunkBatch(gctBatchCmdBuf, batchEnd - batchStart, chunkPointerArray, &paramsList[batchStart]);

    // Barrier. Handles both execution and memory dependency as we are using only one queue.
    // It may seem odd that we are specifying both graphics and compute in src and dst, but this
    // is needed to safely recycle McubesChunk.
    VkMemoryBarrier barrier = {
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT};
    vkCmdPipelineBarrier(gctBatchCmdBuf, computeStage | readGeometryArrayStage, computeStage | readGeometryArrayStage,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);

    // Graphics commands.
    std::vector<McubesDebugViewPushConstant> debugColors =
        makeDebugColors(pGui->m_chunkDebugViewMode, batch, firstChunkUsed, batchEnd - batchStart, chunkPointerArray);
    const bool          drawChunkBounds = pGui->m_chunkDebugViewMode != chunkDebugViewOff;
    const McubesParams* pDebugBoxes     = drawChunkBounds ? &paramsList[batchStart] : nullptr;
    graphicsCmdDrawMcubesGeometryBatch(gctBatchCmdBuf, batchEnd - batchStart, chunkPointerArray, pDebugBoxes,
                                       debugColors.empty() ? nullptr : debugColors.data());

    // NOTE: There is no barrier between this graphics command, and the next iteration's compute commands.
    // This is why we need to ensure any McubesChunk filled in this batch is not recycled for the next batch
    // (only 2 batches later is okay).
    // Otherwise, the next compute command might overwrite the geometry before it's done drawing.
    assert(batchSize <= MCUBES_CHUNK_COUNT / 2u);

    if(batch == batchCount - 1u)
    {
      // Include ImGui commands on last batch.
      graphicsCmdDrawImGui(gctBatchCmdBuf);
    }

    // Last command buffer signals the fence that lets future us know when we can reset the command pool.
    VkFence gctSignalFence = VK_NULL_HANDLE;
    if(batch == batchCount - 1u)
    {
      gctSignalFence = ourGraphicsFence;
    }

    // Submit command buffer.
    NVVK_CHECK(vkEndCommandBuffer(gctBatchCmdBuf));
    NVVK_CHECK(vkQueueSubmit(g_gctQueue, 1, &submitInfo, gctSignalFence));
  }  // End for each batch
}

// Submit end-of-frame commands; acquire/present swap image, and copy from offscreen framebuffer.
static void submitFrame()
{
  // Wait for 2 frames ago to finish, recycle its command buffer.
  VkFence frameFence = s_submitFrameFences[g_frameNumber & 1u];
  NVVK_CHECK(vkWaitForFences(g_ctx, 1, &frameFence, VK_TRUE, ~uint64_t(0)));
  NVVK_CHECK(vkResetFences(g_ctx, 1, &frameFence));
  VkCommandBuffer          cmdBuf    = s_submitFrameCommandBuffers[g_frameNumber & 1u];
  VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
  NVVK_CHECK(vkBeginCommandBuffer(cmdBuf, &beginInfo));

  // Acquire swap image.
  bool                        swapChainRecreated;
  nvvk::SwapChainAcquireState acquired;
  g_swapChain.acquireAutoResize(s_windowWidth, s_windowHeight, &swapChainRecreated, &acquired);

  VkMemoryBarrier copyBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_MEMORY_WRITE_BIT,
                                 VK_ACCESS_TRANSFER_READ_BIT};
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &copyBarrier,
                       0, nullptr, 0, nullptr);

  // Copy offscreen framebuffer color image to swap image.
  nvvk::cmdBarrierImageLayout(cmdBuf, acquired.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_ASPECT_COLOR_BIT);
  VkExtent3D copyExtent;
  copyExtent.width          = std::min(g_swapChain.getWidth(), s_windowWidth);
  copyExtent.height         = std::min(g_swapChain.getHeight(), s_windowHeight);
  copyExtent.depth          = 1u;
  VkImageCopy imageCopyInfo = {
      {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, copyExtent};
  vkCmdCopyImage(cmdBuf, g_drawImage, VK_IMAGE_LAYOUT_GENERAL, acquired.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                 &imageCopyInfo);

  // Present, schedule signalling same fence that we waited on.
  nvvk::cmdBarrierImageLayout(cmdBuf, acquired.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                              VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_ASPECT_COLOR_BIT);
  NVVK_CHECK(vkEndCommandBuffer(cmdBuf));
  VkPipelineStageFlags allCommands = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
  VkSubmitInfo         submitInfo{
      VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 1, &acquired.waitSem, &allCommands, 1, &cmdBuf, 1, &acquired.signalSem};
  NVVK_CHECK(vkQueueSubmit(g_gctQueue, 1, &submitInfo, frameFence));
  g_swapChain.present();
}

int main()
{
  setupGlobals();
  setupStatics();
  setupMcubesChunks();
  setupGraphics();
  Gui* pGui = new Gui;

  setupCompute(pGui->m_equationInput.data());
  s_useComputeQueue = pGui->m_wantComputeQueue;

  VkCommandBuffer             initGuiCmdBuf;
  VkCommandBufferAllocateInfo initGuiCmdBufInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, g_gctPool,
                                                   VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
  NVVK_CHECK(vkAllocateCommandBuffers(g_ctx, &initGuiCmdBufInfo, &initGuiCmdBuf));
  VkCommandBufferBeginInfo initGuiCmdBufBeginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                                     VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, nullptr};
  NVVK_CHECK(vkBeginCommandBuffer(initGuiCmdBuf, &initGuiCmdBufBeginInfo));
  graphicsCmdGuiFirstTimeSetup(initGuiCmdBuf, pGui);
  NVVK_CHECK(vkEndCommandBuffer(initGuiCmdBuf));
  VkSubmitInfo initGuiSubmitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, 0, 0, 1, &initGuiCmdBuf, 0, 0};
  NVVK_CHECK(vkQueueSubmit(g_gctQueue, 1, &initGuiSubmitInfo, VK_NULL_HANDLE));
  vkQueueWaitIdle(g_gctQueue);

  while(!glfwWindowShouldClose(g_window))
  {
    glfwPollEvents();
    ++g_frameNumber;
    waitNonzeroFramebufferSize();
    graphicsWaitResizeFramebufferIfNeeded(s_windowWidth, s_windowHeight);
    pGui->doFrame();

    // Respond to GUI events
    if(pGui->m_vsync != g_swapChain.getVsync())
    {
      g_swapChain.update(s_windowWidth, s_windowHeight, pGui->m_vsync);
    }
    if(pGui->m_wantComputeQueue != s_useComputeQueue)
    {
      vkDeviceWaitIdle(g_ctx);
      s_useComputeQueue = pGui->m_wantComputeQueue;
    }
    if(pGui->m_wantSetEquation)
    {
      vkDeviceWaitIdle(g_ctx);
      pGui->m_compileFailure  = !computeReplaceEquation(pGui->m_equationInput.data());
      pGui->m_wantSetEquation = false;
    }

    if(s_useComputeQueue)
      computeDrawCommandsTwoQueues(pGui);
    else
      computeDrawCommandsGctOnly(pGui);

    submitFrame();
  }
  vkDeviceWaitIdle(g_ctx);
  delete pGui;
  shutdownCompute();
  shutdownGraphics();
  shutdownMcubesChunks();
  shutdownStatics();
  shutdownGlobals();
  return 0;
}
