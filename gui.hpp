// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vulkan/vulkan.h>
#include "GLFW/glfw3.h"

#include <string>
#include <vector>

#include "nvh/cameramanipulator.hpp"
#include "nvvk/context_vk.hpp"
// #include "nvvk/profiler_vk.hpp"

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include "imgui/imgui_helper.h"

#include "shaders/camera_transforms.h"
#include "shaders/mcubes_params.h"

// This is the data stored behind the GLFW window's user pointer.
// Simple container for ImGui stuff, useful only for my basic needs.
// Unfortunately I couldn't initialize everything in a constructor for
// this class, you have to call cmdInit to initialize the state.
class Gui
{
  static const long s_magicNumber = 0x697547;  // "Gui"
  long              m_magicNumber = s_magicNumber;
  GLFWwindow*       m_pWindow{};
  VkDevice          m_device{};
  VkDescriptorPool  m_pool{};
  ImGuiContext*     m_guiContext{};
  bool              m_firstTime = true;

  // For fps counter, updated once per second.
  float   m_displayedFPS         = 0;
  float   m_displayedFrameTime   = 0;
  float   m_frameCountThisSecond = 1;
  float   m_frameTimeThisSecond  = 0;
  int64_t m_thisSecond           = 0;
  double  m_lastUpdateTime       = 0;

  float m_colorByNormalAmount = 0.5f;
  float m_t                   = 0.0f;
  float m_tSliderMin          = 0.0f;
  float m_tSliderMax          = 1.0f;
  int   m_tMode;

  bool m_wantOpenEquationHeader = false;
  bool m_wantFocusEquation      = false;
  bool m_wantFocusT             = false;
  bool m_wantFocusBoundingBox   = false;

  // worldspace bounding box of region to perform marching cubes on.
  glm::vec3 m_bboxLow{-2, -2, -2}, m_bboxHigh{+2, +2, +2};

  // Number of marching cubes cells along each axis (will be rounded due to granularity of McubesGeometry[]).
  glm::ivec3   m_targetCellCounts{508, 508, 508};
  mutable bool m_didTargetCellCountWarning = false;

public:
  // These are parameters set modified by the gui controls and used by
  // the App class.

  // Internal state of 3D camera.
  nvh::CameraManipulator m_cameraManipulator;

  // Used by input callbacks.
  float m_mouseX = 0, m_mouseY = 0;
  float m_zoomMouseX = 0, m_zoomMouseY = 0;  // For centering zoom.
  bool  m_rmb = false, m_mmb = false, m_lmb = false;
  int   m_glfwMods{};

  // Other Controls
  bool              m_vsync            = false;
  bool              m_guiVisible       = true;
  bool              m_wantComputeQueue = true;
  bool              m_compileFailure   = false;
  bool              m_wantSetEquation  = false;
  std::vector<char> m_equationInput;
  int               m_batchSize;
  int               m_chunkDebugViewMode = 0;

  // Do initialization that cannot be done in constructor, especially
  // recording commands for later execution.
  void cmdInit(VkCommandBuffer cmdBuf, VkRenderPass renderPass, uint32_t subpass);

  Gui();
  Gui(Gui&&) = delete;
  ~Gui();

  // Per-frame ImGui code, except for actual Vulkan draw commands.
  void doFrame();

  // Get camera transform matrices.
  CameraTransforms getTransforms(uint32_t windowWidth, uint32_t windowHeight) const;

  // Get value for t (animation parameter)
  float getT() const;

  // Get list of marching cubes jobs to run.
  std::vector<McubesParams> getMcubesJobs() const;

  // Reset camera position to defaults, sized for current bbox.
  void resetCamera();

private:
  void focusIfFlag(bool* pFlag);
  void doEquationUI();
  void setEquation(const char* pEquation);
#ifdef __linux__
  void equationPastePrimarySelection();
#endif
  void updateT();
  void updateCamera();
  void updateFpsSample();
  void zoomCallback3d(double dy);
  void mouseMoveCallback3d(float dx, float dy);

  static Gui& getData(GLFWwindow* pWindow);
  static void scrollCallback(GLFWwindow* pWindow, double x, double y);
  static void mouseCallback(GLFWwindow* pWindow, int, int, int);
  static void cursorPositionCallback(GLFWwindow* pWindow, double x, double y);
  void        charCallbackImpl(unsigned chr);
  static void charCallback(GLFWwindow* pWindow, unsigned chr);
  static void keyCallback(GLFWwindow*, int, int, int, int);
  static void addCallbacks(GLFWwindow* pWindow);
};

// Values for m_chunkDebugViewMode. See also chunkDebugViewLabels[] in gui.cpp
static constexpr int chunkDebugViewOff        = 0;
static constexpr int chunkDebugViewBounds     = 1;
static constexpr int chunkDebugViewBatch      = 2;
static constexpr int chunkDebugViewChunkIndex = 3;
static constexpr int chunkDebugViewModeCount  = 4;
