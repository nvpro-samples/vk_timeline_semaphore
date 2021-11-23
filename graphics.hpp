// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <vulkan/vulkan.h>

#include "gui.hpp"

#include "shaders/mcubes_params.h"

// Image that is drawn to via framebuffer.
extern VkImage g_drawImage;

// Initialize and de-initialize graphics state.
void setupGraphics();
void graphicsCmdGuiFirstTimeSetup(VkCommandBuffer cmdBuf, Gui* p_gui);
void shutdownGraphics();

// Deallocate and resize the framebuffer if needed to match.
// If resizing is needed, we wait for g_gctQueue to idle first.
void graphicsWaitResizeFramebufferIfNeeded(uint32_t width, uint32_t height);

// First command for drawing new frame.
struct CameraTransforms;
void graphicsCmdPrepareFrame(VkCommandBuffer cmdBuf, const CameraTransforms* pCameraTransforms);

// Record commands to draw the McubesGeometry instances in the array of McubesChunk to g_drawImage.
// Debug features: if pDebugChunkBounds != nullptr, we also draw the bounding boxes for each chunk drawn,
//   if pDebugViewColors != nullptr, selectively (with `enabled` attribute) override the color used to draw each chunk.
struct McubesChunk;
struct McubesParams;
struct McubesDebugViewPushConstant;
void graphicsCmdDrawMcubesGeometryBatch(VkCommandBuffer                    cmdBuf,
                                        uint32_t                           count,
                                        const McubesChunk* const*          ppChunks,
                                        const McubesParams*                pDebugChunkBounds,
                                        const McubesDebugViewPushConstant* pDebugViewColors = nullptr);

// Wrapper around ImGui Vulkan commands, draw to g_drawImage.
void graphicsCmdDrawImGui(VkCommandBuffer cmdBuf);
