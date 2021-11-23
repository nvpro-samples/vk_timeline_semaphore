// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#include "graphics.hpp"

#include <cassert>

#include "backends/imgui_impl_vulkan.h"

#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/error_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"

#include "mcubes_chunk.hpp"
#include "timeline_semaphore_main.hpp"

#include "shaders/camera_transforms.h"
#include "shaders/mcubes_debug_view_push_constant.h"
#include "shaders/mcubes_geometry.h"
#include "shaders/mcubes_params.h"

VkImage g_drawImage;

static VkRenderPass                 s_renderPass;
static nvvk::Buffer                 s_cameraTransformsBufferObject;
static nvvk::DescriptorSetContainer s_cameraTransformsDescriptorSetContainer;
static VkPipelineLayout             s_backgroundPipelineLayout;
static VkPipeline                   s_backgroundPipeline;
static VkPipelineLayout             s_mcubesGeometryPipelineLayout;
static VkPipeline                   s_mcubesGeometryPipeline;
static VkPipelineLayout             s_mcubesChunkBoundsPipelineLayout;
static VkPipeline                   s_mcubesChunkBoundsPipeline;

static nvvk::Image   s_colorImageObject;  // Always set g_drawImage to s_colorImageObject.image
static nvvk::Image   s_depthImageObject;
static VkImageView   s_framebufferAttachments[2];
static VkFramebuffer s_framebuffer;
static uint32_t      s_framebufferWidth, s_framebufferHeight;

static const VkFormat colorFormat = VK_FORMAT_B8G8R8A8_SRGB;
static const VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
const VkDeviceSize    zero        = 0;

static void setupRenderPass()
{
  VkAttachmentDescription colorAttachment{};
  colorAttachment.format         = colorFormat;
  colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
  colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
  colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
  colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_GENERAL;
  colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_GENERAL;

  VkAttachmentReference colorAttachmentRef{};
  colorAttachmentRef.attachment = 0;
  colorAttachmentRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

  VkAttachmentDescription depthAttachment{};
  depthAttachment.format         = depthFormat;
  depthAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
  depthAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
  depthAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
  depthAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  depthAttachment.initialLayout  = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
  depthAttachment.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkAttachmentReference depthAttachmentRef{};
  depthAttachmentRef.attachment = 1;
  depthAttachmentRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

  VkSubpassDescription subpass{};
  subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
  subpass.colorAttachmentCount    = 1;
  subpass.pColorAttachments       = &colorAttachmentRef;
  subpass.pDepthStencilAttachment = &depthAttachmentRef;

  VkAttachmentDescription attachments[2] = {colorAttachment, depthAttachment};
  VkRenderPassCreateInfo  renderPassInfo{};
  renderPassInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  renderPassInfo.attachmentCount = 2;
  renderPassInfo.pAttachments    = attachments;
  renderPassInfo.subpassCount    = 1;
  renderPassInfo.pSubpasses      = &subpass;

  NVVK_CHECK(vkCreateRenderPass(g_ctx, &renderPassInfo, nullptr, &s_renderPass));
}

static void setupCameraTransformsBuffer()
{
  // Allocate UBO for holding CameraTransforms struct.
  const auto         usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, sizeof(CameraTransforms), usage};
  s_cameraTransformsBufferObject = g_allocator.createBuffer(bufferInfo);

  // Create 1-binding descriptor set that always points to this buffer.
  s_cameraTransformsDescriptorSetContainer.init(g_ctx);
  s_cameraTransformsDescriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                                      VK_SHADER_STAGE_ALL_GRAPHICS, nullptr);
  s_cameraTransformsDescriptorSetContainer.initLayout();
  s_cameraTransformsDescriptorSetContainer.initPool(1);
  VkDescriptorBufferInfo descriptorInfo{s_cameraTransformsBufferObject.buffer, 0, sizeof(CameraTransforms)};
  VkWriteDescriptorSet   write = s_cameraTransformsDescriptorSetContainer.makeWrite(0, 0, &descriptorInfo, 0);
  vkUpdateDescriptorSets(g_ctx, 1, &write, 0, nullptr);
}

static void setupBackgroundPipeline()
{
  // Set up pipeline layout, one CameraTransforms UBO input.
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 1;
  VkDescriptorSetLayout layout      = s_cameraTransformsDescriptorSetContainer.getLayout();
  pipelineLayoutInfo.pSetLayouts    = &layout;
  NVVK_CHECK(vkCreatePipelineLayout(g_ctx, &pipelineLayoutInfo, nullptr, &s_backgroundPipelineLayout));

  // Hides all the graphics pipeline boilerplate (in particular
  // enabling dynamic viewport and scissor). We just have to
  // disable the depth test and write.
  nvvk::GraphicsPipelineState pipelineState;
  pipelineState.depthStencilState.depthTestEnable  = false;
  pipelineState.depthStencilState.depthWriteEnable = false;

  // Compile and load shaders.
  VkShaderModule vs_module = g_pShaderCompiler->get(
      g_pShaderCompiler->createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "./shaders/background.vert"));
  VkShaderModule fs_module = g_pShaderCompiler->get(
      g_pShaderCompiler->createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "./shaders/background.frag"));
  nvvk::GraphicsPipelineGenerator generator(g_ctx, s_backgroundPipelineLayout, s_renderPass, pipelineState);
  generator.addShader(vs_module, VK_SHADER_STAGE_VERTEX_BIT);
  generator.addShader(fs_module, VK_SHADER_STAGE_FRAGMENT_BIT);
  s_backgroundPipeline = generator.createPipeline();
}

static void setupMcubesGeometryPipeline()
{
  // Set up pipeline layout, McubesDebugViewPushConstant push constant,
  // one CameraTransforms UBO input, one McubesGeometry buffer input.
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  VkDescriptorSetLayout layouts[2]  = {s_cameraTransformsDescriptorSetContainer.getLayout(),
                                      g_mcubesChunkDescriptorSetLayout};
  pipelineLayoutInfo.setLayoutCount = 2;
  pipelineLayoutInfo.pSetLayouts    = layouts;

  VkPushConstantRange pushConstantRange     = {VK_SHADER_STAGE_ALL, 0, sizeof(McubesDebugViewPushConstant)};
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges    = &pushConstantRange;
  NVVK_CHECK(vkCreatePipelineLayout(g_ctx, &pipelineLayoutInfo, nullptr, &s_mcubesGeometryPipelineLayout));

  // Hides all the graphics pipeline boilerplate (in particular enabling dynamic viewport and scissor).
  nvvk::GraphicsPipelineState pipelineState;
  pipelineState.depthStencilState.depthCompareOp = VK_COMPARE_OP_GREATER;  // Reversed Z

  // No vertex input bindings (manual fetch from storage buffer).

  // Compile and load shaders.
  VkShaderModule vs_module = g_pShaderCompiler->get(
      g_pShaderCompiler->createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "./shaders/mcubes_geometry.vert"));
  VkShaderModule fs_module = g_pShaderCompiler->get(
      g_pShaderCompiler->createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "./shaders/mcubes_geometry.frag"));

  // Create pipeline.
  nvvk::GraphicsPipelineGenerator generator(g_ctx, s_mcubesGeometryPipelineLayout, s_renderPass, pipelineState);
  generator.addShader(vs_module, VK_SHADER_STAGE_VERTEX_BIT);
  generator.addShader(fs_module, VK_SHADER_STAGE_FRAGMENT_BIT);
  s_mcubesGeometryPipeline = generator.createPipeline();
}

static void setupMcubesChunkBoundsPipeline()
{
  // Set up pipeline layout, one McubesParams push constant, one CameraTransforms UBO input.
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  VkDescriptorSetLayout layouts[1]          = {s_cameraTransformsDescriptorSetContainer.getLayout()};
  pipelineLayoutInfo.setLayoutCount         = 1;
  pipelineLayoutInfo.pSetLayouts            = layouts;
  VkPushConstantRange pushConstantRange     = {VK_SHADER_STAGE_ALL, 0, sizeof(McubesParams)};
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges    = &pushConstantRange;
  NVVK_CHECK(vkCreatePipelineLayout(g_ctx, &pipelineLayoutInfo, nullptr, &s_mcubesChunkBoundsPipelineLayout));

  // Hides all the graphics pipeline boilerplate (in particular enabling dynamic viewport and scissor).
  nvvk::GraphicsPipelineState pipelineState;
  pipelineState.inputAssemblyState.topology      = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  pipelineState.depthStencilState.depthCompareOp = VK_COMPARE_OP_GREATER;  // Reversed Z

  // Compile and load shaders.
  VkShaderModule vs_module = g_pShaderCompiler->get(
      g_pShaderCompiler->createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "./shaders/mcubes_chunk_bounds.vert"));
  VkShaderModule fs_module = g_pShaderCompiler->get(
      g_pShaderCompiler->createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "./shaders/mcubes_chunk_bounds.frag"));

  // Create pipeline.
  nvvk::GraphicsPipelineGenerator generator(g_ctx, s_mcubesChunkBoundsPipelineLayout, s_renderPass, pipelineState);
  generator.addShader(vs_module, VK_SHADER_STAGE_VERTEX_BIT);
  generator.addShader(fs_module, VK_SHADER_STAGE_FRAGMENT_BIT);
  s_mcubesChunkBoundsPipeline = generator.createPipeline();
}

void setupGraphics()
{
  setupRenderPass();
  setupCameraTransformsBuffer();
  setupBackgroundPipeline();
  setupMcubesGeometryPipeline();
  setupMcubesChunkBoundsPipeline();
}

void graphicsCmdGuiFirstTimeSetup(VkCommandBuffer cmdBuf, Gui* pGui)
{
  pGui->cmdInit(cmdBuf, s_renderPass, 0);
}

static void shutdownFramebuffer();

void shutdownGraphics()
{
  shutdownFramebuffer();
  vkDestroyPipeline(g_ctx, s_backgroundPipeline, nullptr);
  vkDestroyPipelineLayout(g_ctx, s_backgroundPipelineLayout, nullptr);
  vkDestroyPipeline(g_ctx, s_mcubesGeometryPipeline, nullptr);
  vkDestroyPipelineLayout(g_ctx, s_mcubesGeometryPipelineLayout, nullptr);
  vkDestroyPipeline(g_ctx, s_mcubesChunkBoundsPipeline, nullptr);
  vkDestroyPipelineLayout(g_ctx, s_mcubesChunkBoundsPipelineLayout, nullptr);
  s_cameraTransformsDescriptorSetContainer.deinit();
  g_allocator.destroy(s_cameraTransformsBufferObject);
  vkDestroyRenderPass(g_ctx, s_renderPass, nullptr);
}

void graphicsWaitResizeFramebufferIfNeeded(uint32_t width, uint32_t height)
{
  assert(g_drawImage == s_colorImageObject.image);
  if(!s_framebuffer || s_framebufferWidth != width || s_framebufferHeight != height)
  {
    vkQueueWaitIdle(g_gctQueue);
    shutdownFramebuffer();

    // Create new color image.
    VkImageCreateInfo colorImageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                     nullptr,
                                     0,
                                     VK_IMAGE_TYPE_2D,
                                     colorFormat,
                                     {width, height, 1u},
                                     1,
                                     1,
                                     VK_SAMPLE_COUNT_1_BIT,
                                     VK_IMAGE_TILING_OPTIMAL,
                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                                     VK_SHARING_MODE_EXCLUSIVE,
                                     0,
                                     nullptr,
                                     VK_IMAGE_LAYOUT_UNDEFINED};
    s_colorImageObject = g_allocator.createImage(colorImageInfo);
    g_drawImage        = s_colorImageObject.image;
    VkImageViewCreateInfo colorViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                        nullptr,
                                        0,
                                        s_colorImageObject.image,
                                        VK_IMAGE_VIEW_TYPE_2D,
                                        colorFormat,
                                        {},
                                        {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    NVVK_CHECK(vkCreateImageView(g_ctx, &colorViewInfo, nullptr, &s_framebufferAttachments[0]));

    // Create new depth image.
    VkImageCreateInfo depthImageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                     nullptr,
                                     0,
                                     VK_IMAGE_TYPE_2D,
                                     depthFormat,
                                     {width, height, 1u},
                                     1,
                                     1,
                                     VK_SAMPLE_COUNT_1_BIT,
                                     VK_IMAGE_TILING_OPTIMAL,
                                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                     VK_SHARING_MODE_EXCLUSIVE,
                                     0,
                                     nullptr,
                                     VK_IMAGE_LAYOUT_UNDEFINED};
    s_depthImageObject = g_allocator.createImage(depthImageInfo);
    VkImageViewCreateInfo depthViewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                                        nullptr,
                                        0,
                                        s_depthImageObject.image,
                                        VK_IMAGE_VIEW_TYPE_2D,
                                        depthFormat,
                                        {},
                                        {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}};
    NVVK_CHECK(vkCreateImageView(g_ctx, &depthViewInfo, nullptr, &s_framebufferAttachments[1]));

    // Create framebuffer.
    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass      = s_renderPass;
    framebufferInfo.attachmentCount = 2;
    framebufferInfo.pAttachments    = s_framebufferAttachments;
    framebufferInfo.width           = width;
    framebufferInfo.height          = height;
    framebufferInfo.layers          = 1;
    NVVK_CHECK(vkCreateFramebuffer(g_ctx, &framebufferInfo, nullptr, &s_framebuffer));

    // Record new size.
    s_framebufferWidth  = width;
    s_framebufferHeight = height;
  }
}

static void shutdownFramebuffer()
{
  if(s_colorImageObject.image)
  {
    g_allocator.destroy(s_colorImageObject);
    s_colorImageObject.image = VK_NULL_HANDLE;
    g_drawImage              = VK_NULL_HANDLE;
    vkDestroyImageView(g_ctx, s_framebufferAttachments[0], nullptr);
  }
  if(s_depthImageObject.image)
  {
    g_allocator.destroy(s_depthImageObject);
    s_depthImageObject.image = VK_NULL_HANDLE;
    vkDestroyImageView(g_ctx, s_framebufferAttachments[1], nullptr);
  }
  if(s_framebuffer)
  {
    vkDestroyFramebuffer(g_ctx, s_framebuffer, nullptr);
    s_framebuffer = VK_NULL_HANDLE;
  }
  s_framebufferWidth  = 0;
  s_framebufferHeight = 0;
}

static void cmdBeginDynamicViewportScissorRenderPass(VkCommandBuffer cmdBuf)
{
  // Begin render pass
  VkRenderPassBeginInfo beginInfo = {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                                     nullptr,
                                     s_renderPass,
                                     s_framebuffer,
                                     {{0, 0}, {s_framebufferWidth, s_framebufferHeight}},
                                     0,
                                     nullptr};
  vkCmdBeginRenderPass(cmdBuf, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);

  // Set dynamic viewport/scissor.
  VkViewport viewport;
  viewport.x        = 0.0f;
  viewport.y        = 0.0f;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  viewport.width    = s_framebufferWidth;
  viewport.height   = s_framebufferHeight;
  auto     ix       = int32_t(viewport.x);
  auto     iy       = int32_t(viewport.y);
  auto     iw       = uint32_t(viewport.width);
  auto     ih       = uint32_t(viewport.height);
  VkRect2D scissor{{ix, iy}, {iw, ih}};
  vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
  vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
}

void graphicsCmdPrepareFrame(VkCommandBuffer cmdBuf, const CameraTransforms* pCameraTransforms)
{
  // Transition framebuffer attachments to defined layout.
  nvvk::cmdBarrierImageLayout(cmdBuf, g_drawImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                              VK_IMAGE_ASPECT_COLOR_BIT);
  nvvk::cmdBarrierImageLayout(cmdBuf, s_depthImageObject.image, VK_IMAGE_LAYOUT_UNDEFINED,
                              VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
  // Update UBO data.
  VkMemoryBarrier uboBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  uboBarrier.srcAccessMask   = VK_ACCESS_UNIFORM_READ_BIT;
  uboBarrier.dstAccessMask   = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &uboBarrier, 0,
                       nullptr, 0, nullptr);
  vkCmdUpdateBuffer(cmdBuf, s_cameraTransformsBufferObject.buffer, 0, sizeof(CameraTransforms), pCameraTransforms);
  uboBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  uboBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT, 0, 1, &uboBarrier, 0,
                       nullptr, 0, nullptr);

  cmdBeginDynamicViewportScissorRenderPass(cmdBuf);

  // Clear depth buffer.
  VkClearValue clearDepthValue;
  clearDepthValue.depthStencil.depth = 0.0;  // Reversed Z
  VkClearAttachment clearDepth       = {VK_IMAGE_ASPECT_DEPTH_BIT, 1, clearDepthValue};
  VkClearRect       clearRect        = {{{0u, 0u}, {s_framebufferWidth, s_framebufferHeight}}, 0, 1};
  vkCmdClearAttachments(cmdBuf, 1, &clearDepth, 1, &clearRect);

  // Draw background.
  VkDescriptorSet cameraTransformsDescriptorSet = s_cameraTransformsDescriptorSetContainer.getSet(0);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, s_backgroundPipelineLayout, 0,  //
                          1, &cameraTransformsDescriptorSet, 0, 0);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, s_backgroundPipeline);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);

  vkCmdEndRenderPass(cmdBuf);
}

void graphicsCmdDrawMcubesGeometryBatch(VkCommandBuffer                    cmdBuf,
                                        uint32_t                           count,
                                        const McubesChunk* const*          ppChunks,
                                        const McubesParams*                pDebugChunkBounds,
                                        const McubesDebugViewPushConstant* pDebugViewColors)
{
  cmdBeginDynamicViewportScissorRenderPass(cmdBuf);

  // Bind pipeline
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, s_mcubesGeometryPipeline);

  // Bind camera UBO descriptor set (0).
  VkDescriptorSet uboSet = s_cameraTransformsDescriptorSetContainer.getSet(0);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, s_mcubesGeometryPipelineLayout, 0, 1, &uboSet, 0, 0);

  for(uint32_t i = 0; i < count; ++i)
  {
    // Bind McubesGeometry descriptor set (1).
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, s_mcubesGeometryPipelineLayout,  //
                            1, 1, &ppChunks[i]->set, 0, 0);

    // Set debug override color.
    static McubesDebugViewPushConstant disabledDebugColor{};
    vkCmdPushConstants(cmdBuf, s_mcubesGeometryPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof disabledDebugColor,
                       pDebugViewColors != nullptr ? &pDebugViewColors[i] : &disabledDebugColor);

    // Draw
    vkCmdDrawIndirect(cmdBuf, ppChunks[i]->geometryArrayBuffer.buffer, 0, MCUBES_GEOMETRIES_PER_CHUNK,
                      sizeof(McubesGeometry));

    if(pDebugChunkBounds != nullptr)
    {
      vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, s_mcubesChunkBoundsPipeline);
      vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, s_mcubesChunkBoundsPipelineLayout, 0, 1, &uboSet,
                              0, 0);
      vkCmdPushConstants(cmdBuf, s_mcubesChunkBoundsPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof pDebugChunkBounds[i],
                         &pDebugChunkBounds[i]);
      vkCmdDraw(cmdBuf, 24, 1, 0, 0);
      vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, s_mcubesGeometryPipeline);
      vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, s_mcubesGeometryPipelineLayout, 0, 1, &uboSet, 0,
                              0);
    }
  }
  vkCmdEndRenderPass(cmdBuf);
}

void graphicsCmdDrawImGui(VkCommandBuffer cmdBuf)
{
  cmdBeginDynamicViewportScissorRenderPass(cmdBuf);
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuf);
  vkCmdEndRenderPass(cmdBuf);
}
