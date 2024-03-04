// Copyright 2021 NVIDIA CORPORATION
// SPDX-License-Identifier: Apache-2.0

#include <glm/glm.hpp>

#include "gui.hpp"

#include <math.h>
#include <string.h>

#include "nvh/cameramanipulator.hpp"
#include "nvh/container_utils.hpp"
#include "nvvk/error_vk.hpp"

#include "mcubes_chunk.hpp"
#include "timeline_semaphore_main.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include "nvmath/nvmath.h"


static const float nearPlane = 65536.0f, farPlane = 1.0f / 65536.0f;  // Reversed Z

static const char* const pDefaultEquation =
    "sqrt(square(fract(y) - 0.5) + square(abs(r - 1))) - 0.15 - 0.25*square(cos(t+(floor(y) + 3)*theta))";

static const int tModeManual = 0, tModeSawtooth = 1, tModeTriangle = 2, tModeSin = 3, tMode_0_to_2pi = 4,
                 tModeCount = 5;

const char* tModeLabels[tModeCount] = {"manual", "sawtooth", "triangle", "sine", "0 to 2pi"};

static const char* chunkDebugViewLabels[chunkDebugViewModeCount] = {"off", "draw bounds", "color by batch",
                                                                    "color by McubesChunk used"};

Gui::Gui()
    : m_cameraManipulator(CameraManip)
{
  m_tMode = tMode_0_to_2pi;
  m_equationInput.resize(strlen(pDefaultEquation) + 1u);
  strcpy(m_equationInput.data(), pDefaultEquation);
  m_batchSize = MCUBES_MAX_CHUNKS_PER_BATCH;
}

void Gui::cmdInit(VkCommandBuffer cmdBuf, VkRenderPass renderPass, uint32_t subpass)
{
  m_pWindow = g_window;
  m_device  = g_ctx.m_device;
  assert(m_device != nullptr);

  void* oldUserPointer = glfwGetWindowUserPointer(g_window);
  assert(oldUserPointer == nullptr);
  glfwSetWindowUserPointer(g_window, this);  // Class must be non-moveable
  addCallbacks(g_window);

  resetCamera();

  m_guiContext = ImGui::CreateContext(nullptr);
  assert(m_guiContext != nullptr);
  ImGui::SetCurrentContext(m_guiContext);

  ImGuiH::Init(1920, 1080, nullptr, ImGuiH::FONT_PROPORTIONAL_SCALED);
  ImGuiH::setFonts(ImGuiH::FONT_PROPORTIONAL_SCALED);
  ImGuiH::setStyle(true);

  VkDescriptorPoolSize       poolSizes[] = {VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 1},
                                            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
                                            VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1}};
  VkDescriptorPoolCreateInfo poolInfo    = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                            nullptr,
                                            VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                            arraySize(poolSizes),
                                            arraySize(poolSizes),
                                            poolSizes};
  assert(m_pool == VK_NULL_HANDLE);
  NVVK_CHECK(vkCreateDescriptorPool(g_ctx, &poolInfo, nullptr, &m_pool));

  ImGui_ImplVulkan_InitInfo info{};
  info.Instance            = g_ctx.m_instance;
  info.PhysicalDevice      = g_ctx.m_physicalDevice;
  info.Device              = g_ctx.m_device;
  info.QueueFamily         = g_ctx.m_queueGCT.familyIndex;
  info.Queue               = g_ctx.m_queueGCT.queue;
  info.DescriptorPool      = m_pool;
  info.RenderPass          = renderPass;
  info.Subpass             = subpass;
  info.MinImageCount       = g_swapChain.getImageCount();
  info.ImageCount          = g_swapChain.getImageCount();
  info.MSAASamples         = VK_SAMPLE_COUNT_1_BIT;
  info.UseDynamicRendering = false;
  info.Allocator           = nullptr;
  info.CheckVkResultFn     = [](VkResult err) { NVVK_CHECK(err); };

  ImGui_ImplVulkan_Init(&info);
  ImGui_ImplVulkan_CreateFontsTexture();

  ImGui_ImplGlfw_InitForVulkan(g_window, false);
}

Gui::~Gui()
{
  if(m_device != nullptr)
  {
    vkDestroyDescriptorPool(m_device, m_pool, nullptr);
    ImGui_ImplVulkan_DestroyFontsTexture();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
  }
  if(m_pWindow != nullptr)
  {
    glfwSetWindowUserPointer(m_pWindow, nullptr);
  }
}

void Gui::doFrame()
{
  updateFpsSample();
  updateCamera();
  ImGui::NewFrame();
  ImGui_ImplGlfw_NewFrame();
  float dpiScale = float(ImGuiH::getDPIScale());

  if(m_guiVisible)
  {
    if(m_firstTime)
    {
      ImGui::SetNextWindowPos({0, 0});
      ImGui::SetNextWindowSize({dpiScale * 300, dpiScale * 800});
      ImGui::SetNextItemOpen(true);
    }
    ImGui::Begin("Toggle UI [u]");
    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);
    if(m_compileFailure)
      ImGui::Text("Shader compiler error -- see console");
    else
      ImGui::Text("--");

    if(m_wantOpenEquationHeader)
    {
      ImGui::SetNextItemOpen(true);
      m_wantOpenEquationHeader = false;
    }
    if(ImGui::CollapsingHeader("Equation [e]"))
    {
      doEquationUI();
    }

    ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.5f);
    ImGui::Text("FPS: %.0f", m_displayedFPS);
    ImGui::Text("Max Frame Time: %7.4f ms", m_displayedFrameTime * 1000.);
    ImGui::Checkbox("vsync [v] (may reduce timing accuracy)", &m_vsync);
    ImGui::Checkbox("Use compute-only queue [c]", &m_wantComputeQueue);
    ImGui::SliderFloat("Color by normal [n]", &m_colorByNormalAmount, 0.0f, 1.0f);
    ImGui::SliderInt("Chunks/Batch [-+]", &m_batchSize, 1, MCUBES_MAX_CHUNKS_PER_BATCH);
    ImGui::Combo("Chunk debug view [d]", &m_chunkDebugViewMode, chunkDebugViewLabels, chunkDebugViewModeCount);
    if(ImGui::Button("Reset camera [r]"))
      resetCamera();
    ImGui::End();
  }
  ImGui::Render();
  updateT();

  m_firstTime = false;
}

CameraTransforms Gui::getTransforms(uint32_t windowWidth, uint32_t windowHeight) const
{
  float aspectRatio = float(windowWidth) / float(windowHeight);

  auto      camera = m_cameraManipulator.getCamera();
  glm::mat4 view   = glm::lookAt(camera.eye, camera.ctr, camera.up);
  glm::mat4 proj   = glm::perspectiveRH_ZO(glm::radians(camera.fov), aspectRatio, nearPlane, farPlane);
  proj[1][1] *= -1;

  CameraTransforms transforms;
  transforms.view            = view;
  transforms.proj            = proj;
  transforms.viewProj        = proj * view;
  transforms.viewInverse     = glm::inverse(view);
  transforms.projInverse     = glm::inverse(proj);
  transforms.viewProjInverse = glm::inverse(transforms.viewProj);

  transforms.colorByNormalAmount = m_colorByNormalAmount;
  return transforms;
}

float Gui::getT() const
{
  return m_t;
}

std::vector<McubesParams> Gui::getMcubesJobs() const
{
  std::vector<McubesParams> jobs;

  glm::ivec3 targetCellCounts = glm::clamp(m_targetCellCounts, 0, 1024);
  if(targetCellCounts != m_targetCellCounts && !m_didTargetCellCountWarning)
  {
    fprintf(stderr,
            "%s:%i \x1b[35m\x1b[1mWARNING:\x1b[0m Ignoring unexpectedly high target cell counts (limit set in 2021)\n",
            __FILE__, __LINE__);
    m_didTargetCellCountWarning = true;
  }

  // Convert target cell counts to job count.
  int xJobs = std::max(int(roundf(float(targetCellCounts.x) / MCUBES_CHUNK_EDGE_LENGTH_CELLS)), 1);
  int yJobs = std::max(int(roundf(float(targetCellCounts.y) / MCUBES_CHUNK_EDGE_LENGTH_CELLS)), 1);
  int zJobs = std::max(int(roundf(float(targetCellCounts.z) / MCUBES_CHUNK_EDGE_LENGTH_CELLS)), 1);

  glm::vec3 wholeSize = m_bboxHigh - m_bboxLow;
  glm::vec3 jobCounts = glm::vec3(xJobs, yJobs, zJobs);

  // Trying to be careful to be watertight.
  for(int z = 0; z < zJobs; ++z)
  {
    for(int y = 0; y < yJobs; ++y)
    {
      for(int x = 0; x < xJobs; ++x)
      {
        McubesParams params;
        glm::vec3    low  = m_bboxLow + wholeSize * (glm::vec3(x, y, z) / jobCounts);
        glm::vec3    high = m_bboxLow + wholeSize * (glm::vec3(x + 1, y + 1, z + 1) / jobCounts);
        params.offset     = low;
        params.t          = m_t;
        params.size       = high - low;
        jobs.push_back(params);
      }
    }
  }
  return jobs;
}

void Gui::resetCamera()
{
  glm::vec3 bboxHigh = glm::make_vec3(&m_bboxHigh.x);
  glm::vec3 bboxLow  = glm::make_vec3(&m_bboxLow.x);

  m_cameraManipulator.setLookat(bboxHigh, (bboxLow + bboxHigh) * 0.5f, {0, 1, 0});
}

void Gui::focusIfFlag(bool* pFlag)
{
  if(*pFlag)
  {
    ImGui::SetKeyboardFocusHere();
    *pFlag = false;
  }
}

void Gui::doEquationUI()
{
  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 1.0f);
  focusIfFlag(&m_wantFocusEquation);
  m_wantSetEquation |= ImGui::InputText("##Equation", m_equationInput.data(), m_equationInput.size(),
                                        ImGuiInputTextFlags_EnterReturnsTrue);
  if(strlen(m_equationInput.data()) + 100u > m_equationInput.size())
  {
    m_equationInput.resize(m_equationInput.size() + 100u);
  }
  ImGui::PopItemWidth();

  if(ImGui::Button("Paste Equation [p]"))
    setEquation(glfwGetClipboardString(m_pWindow));

  ImGui::Combo("t mode [m]", &m_tMode, tModeLabels, tModeCount);
  focusIfFlag(&m_wantFocusT);
  float oldTValue = m_t;
  ImGui::SliderFloat("t [t]", &m_t, m_tSliderMin, m_tSliderMax);  // It's fine if user exceeds bounds
  if(m_t != oldTValue)
    m_tMode = tModeManual;

  ImGui::PushItemWidth(ImGui::GetWindowWidth() * 1.0f);
  focusIfFlag(&m_wantFocusBoundingBox);
  ImGui::Text("Bounding Box [b]");
  ImGui::InputFloat3("##low", &m_bboxLow.x);
  ImGui::InputFloat3("##high", &m_bboxHigh.x);
  ImGui::Text("Target Cell Counts [XxYyZz]");
  static_assert(sizeof(int) == 4, "Assumed 32 bit ints here");
  ImGui::InputInt3("##cellCounts", &m_targetCellCounts.x);
  ImGui::PopItemWidth();
  ImGui::Separator();
}

void Gui::setEquation(const char* pEquation)
{
  size_t bytes = strlen(pEquation) + 1;
  if(bytes >= m_equationInput.size())
    m_equationInput.resize(bytes);
  memcpy(m_equationInput.data(), pEquation, bytes);
  m_wantSetEquation = true;
}

#ifdef __linux__
void Gui::equationPastePrimarySelection()
{
  FILE* pipe = popen("xsel -o --primary", "r");
  int   c;
  m_equationInput.clear();
  while((c = getc(pipe)) != EOF)
    m_equationInput.push_back(char(c));
  m_equationInput.push_back('\0');
  m_wantSetEquation = true;
  fclose(pipe);
}
#endif

void Gui::updateT()
{
  double time = glfwGetTime();
  switch(m_tMode)
  {
    case tModeSawtooth:
      m_tSliderMin = 0.0f;
      m_tSliderMax = 1.0f;
      m_t          = float(fmod(time, 1.0));
      break;
    case tModeTriangle:
      m_tSliderMin = 0.0f;
      m_tSliderMax = 1.0f;
      m_t          = fabsf(1.0f - 2.0f * float(fmod(time, 1.0)));
      break;
    case tModeSin:
      m_tSliderMin = -1.0f;
      m_tSliderMax = 1.0f;
      m_t          = float(sin(time * 6.283185307179586));
      break;
    case tMode_0_to_2pi:
      m_tSliderMin = 0.0f;
      m_tSliderMax = 6.283185307179586f;
      m_t          = float(fmod(time, 1.0) * 6.283185307179586);
      break;
  }
}

void Gui::updateCamera()
{
  int x, y;
  glfwGetWindowSize(m_pWindow, &x, &y);
  m_cameraManipulator.setWindowSize(x, y);
  m_cameraManipulator.updateAnim();
}

void Gui::updateFpsSample()
{
  double now = glfwGetTime();
  if(m_lastUpdateTime == 0)
  {
    m_lastUpdateTime = now;
    return;
  }

  if(int64_t(now) != m_thisSecond)
  {
    m_displayedFPS       = m_frameCountThisSecond;
    m_displayedFrameTime = m_frameTimeThisSecond;

    m_thisSecond           = int64_t(now);
    m_frameCountThisSecond = 1;
    m_frameTimeThisSecond  = 0;
  }
  else
  {
    float frameTime = float(now - m_lastUpdateTime);
    m_frameCountThisSecond++;
    m_frameTimeThisSecond = std::max(m_frameTimeThisSecond, frameTime);
  }
  m_lastUpdateTime = now;
}

// 3d camera scroll wheel callback, moves you forwards and backwards.
void Gui::zoomCallback3d(double dy)
{
  m_cameraManipulator.wheel(int(copysign(1.0, dy)),
                            {m_lmb, m_mmb, m_rmb, bool(m_glfwMods & GLFW_MOD_SHIFT),
                             bool(m_glfwMods & GLFW_MOD_CONTROL), bool(m_glfwMods & GLFW_MOD_ALT)});
}

// 3d mouse move callback.
void Gui::mouseMoveCallback3d(float x, float y)
{
  m_cameraManipulator.mouseMove(int(x), int(y),
                                {m_lmb, m_mmb, m_rmb, bool(m_glfwMods & GLFW_MOD_SHIFT),
                                 bool(m_glfwMods & GLFW_MOD_CONTROL), bool(m_glfwMods & GLFW_MOD_ALT)});
}

Gui& Gui::getData(GLFWwindow* pWindow)
{
  void* userPointer = glfwGetWindowUserPointer(pWindow);
  assert(userPointer != nullptr);
  Gui& data = *static_cast<Gui*>(userPointer);
  assert(data.m_magicNumber == data.s_magicNumber);
  return data;
}

void Gui::scrollCallback(GLFWwindow* pWindow, double x, double y)
{
  Gui& g = getData(pWindow);
  ImGui_ImplGlfw_ScrollCallback(pWindow, x, y);
  if(ImGui::GetIO().WantCaptureMouse)
  {
  }
  else
  {
    g.zoomCallback3d(y * -0.25);
  }
}

void Gui::mouseCallback(GLFWwindow* pWindow, int button, int action, int mods)
{
  Gui& g       = getData(pWindow);
  g.m_glfwMods = mods;
  ImGui_ImplGlfw_MouseButtonCallback(pWindow, button, action, mods);
  bool mouseFlag = (action != GLFW_RELEASE) && !ImGui::GetIO().WantCaptureMouse;

  if(action == GLFW_PRESS)
  {
    g.m_cameraManipulator.setMousePosition(int(g.m_mouseX), int(g.m_mouseY));
  }

  switch(button)
  {
    case GLFW_MOUSE_BUTTON_RIGHT:
      g.m_rmb = mouseFlag;
      break;
    case GLFW_MOUSE_BUTTON_MIDDLE:
      g.m_mmb = mouseFlag;
      break;
    case GLFW_MOUSE_BUTTON_LEFT:
      g.m_lmb = mouseFlag;
      break;
    default:
      break;  // Get rid of warning.
  }

#ifdef __linux__  // Linux-ism: paste primary selection as test equation. This is very convenient for me.
  if(ImGui::GetIO().WantCaptureMouse && button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS)
  {
    g.equationPastePrimarySelection();
  }
#endif
  return;
}

void Gui::cursorPositionCallback(GLFWwindow* pWindow, double x, double y)
{
  Gui& g = getData(pWindow);
  g.mouseMoveCallback3d(float(x), float(y));
  g.m_mouseX = float(x);
  g.m_mouseY = float(y);
}

void Gui::charCallbackImpl(unsigned chr)
{
  switch(chr)
  {
    case 'b':
      m_wantOpenEquationHeader = true;
      m_wantFocusBoundingBox   = true;
      break;
    case 'c':
      m_wantComputeQueue ^= 1;
      break;
    case 'D':
      m_chunkDebugViewMode--;
      if(m_chunkDebugViewMode < 0)
        m_chunkDebugViewMode = chunkDebugViewModeCount - 1;
      break;
    case 'd':
      m_chunkDebugViewMode++;
      if(m_chunkDebugViewMode >= chunkDebugViewModeCount)
        m_chunkDebugViewMode = 0;
      break;
    case 'e':
      m_wantOpenEquationHeader = true;
      m_wantFocusEquation      = true;
      break;
    case 'M':
      m_wantOpenEquationHeader = true;
      if(m_tMode <= 0)
        m_tMode = tModeCount - 1;
      else
        m_tMode = m_tMode - 1;
      break;
    case 'm':
      m_wantOpenEquationHeader = true;
      m_tMode++;
      if(m_tMode >= tModeCount)
        m_tMode = 0;
      break;
    case 'n':
      if(m_colorByNormalAmount == 1.0f)
        m_colorByNormalAmount = 0.0f;
      else if(m_colorByNormalAmount == 0.0f)
        m_colorByNormalAmount = 0.5f;
      else
        m_colorByNormalAmount = 1.0f;
      break;
    case 'p':
      setEquation(glfwGetClipboardString(m_pWindow));
      break;
    case 'r':
      resetCamera();
      break;
    case 't':
      m_wantOpenEquationHeader = true;
      m_tMode                  = tModeManual;
      m_wantFocusT             = true;
      break;
    case 'u':
      m_guiVisible ^= 1;
      break;
    case 'v':
      m_vsync ^= 1;
      break;
    case 'x':
      m_wantOpenEquationHeader = true;
      m_targetCellCounts.x += MCUBES_CHUNK_EDGE_LENGTH_CELLS;
      break;
    case 'y':
      m_wantOpenEquationHeader = true;
      m_targetCellCounts.y += MCUBES_CHUNK_EDGE_LENGTH_CELLS;
      break;
    case 'z':
      m_wantOpenEquationHeader = true;
      m_targetCellCounts.z += MCUBES_CHUNK_EDGE_LENGTH_CELLS;
      break;
    case 'X':
      m_wantOpenEquationHeader = true;
      if(m_targetCellCounts.x > MCUBES_CHUNK_EDGE_LENGTH_CELLS)
        m_targetCellCounts.x -= MCUBES_CHUNK_EDGE_LENGTH_CELLS;
      break;
    case 'Y':
      m_wantOpenEquationHeader = true;
      if(m_targetCellCounts.y > MCUBES_CHUNK_EDGE_LENGTH_CELLS)
        m_targetCellCounts.y -= MCUBES_CHUNK_EDGE_LENGTH_CELLS;
      break;
    case 'Z':
      m_wantOpenEquationHeader = true;
      if(m_targetCellCounts.z > MCUBES_CHUNK_EDGE_LENGTH_CELLS)
        m_targetCellCounts.z -= MCUBES_CHUNK_EDGE_LENGTH_CELLS;
      break;
    case '+':
    case '=':
      m_batchSize++;
      if(m_batchSize > MCUBES_MAX_CHUNKS_PER_BATCH)
        m_batchSize = 1;
      break;
    case '-':
      m_batchSize--;
      if(m_batchSize < 1)
        m_batchSize = MCUBES_MAX_CHUNKS_PER_BATCH;
      break;
  }
}

void Gui::charCallback(GLFWwindow* pWindow, unsigned chr)
{
  ImGui_ImplGlfw_CharCallback(pWindow, chr);
  if(!ImGui::GetIO().WantTextInput)
  {
    getData(pWindow).charCallbackImpl(chr);
  }
}

void Gui::keyCallback(GLFWwindow* pWindow, int key, int scancode, int action, int mods)
{
  ImGui_ImplGlfw_KeyCallback(pWindow, key, scancode, action, mods);
}

void Gui::addCallbacks(GLFWwindow* pWindow)
{
  glfwSetScrollCallback(pWindow, scrollCallback);
  glfwSetMouseButtonCallback(pWindow, mouseCallback);
  glfwSetCursorPosCallback(pWindow, cursorPositionCallback);
  glfwSetCharCallback(pWindow, charCallback);
  glfwSetKeyCallback(pWindow, keyCallback);
}
