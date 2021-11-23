// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>
#include <vulkan/vulkan.h>

#include "GLFW/glfw3.h"
#include "nvvk/context_vk.hpp"
#include "nvvk/shadermodulemanager_vk.hpp"
#include "nvvk/swapchain_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"

// Foundational Vulkan items used throughout the program.
extern GLFWwindow*                      g_window;
extern nvvk::Context                    g_ctx;
extern nvvk::ResourceAllocatorDedicated g_allocator;
extern VkSurfaceKHR                     g_surface;
extern nvvk::SwapChain                  g_swapChain;
extern VkQueue                          g_gctQueue, g_computeQueue;
extern VkCommandPool                    g_gctPool, g_computePool;
extern nvvk::ShaderModuleManager*       g_pShaderCompiler;
extern uint64_t                         g_frameNumber;
