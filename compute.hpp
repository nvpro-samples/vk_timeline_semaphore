// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vulkan/vulkan.h>

// Initialize and de-initialize compute state.
void setupCompute(const char* pEquation);
void shutdownCompute();

extern bool g_computeReadyFlag;
bool        computeUpdateComputeReadyFlag();

// Record commands to fill the given array of McubesChunk (image and geometry array buffer),
// using the corresponding array of parameters. No implied barriers before or after.
struct McubesChunk;
struct McubesParams;
void computeCmdFillChunkBatch(VkCommandBuffer           cmdBuf,
                              uint32_t                  count,
                              const McubesChunk* const* pChunks,
                              const McubesParams*       pParams);

// Replace the equation being used to generate the marching cubes 3D input image. Returns success flag.
// Ensure that no computeCmdFillChunk commands are running when this function is called.
bool computeReplaceEquation(const char* pEquation);
