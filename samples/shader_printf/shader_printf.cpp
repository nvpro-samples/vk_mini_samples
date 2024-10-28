/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


//////////////////////////////////////////////////////////////////////////
/*

  This sample add DEBUG_PRINTF to the validation layer. This allows to 
  add debugPrintfEXT() in any shader and getting back results. 
  See #debug_printf

  The log is also rerouted in a Log window. (See nvvk::ElementLogger)

 */
//////////////////////////////////////////////////////////////////////////

#include <glm/glm.hpp>

// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
// clang-format on

#include <array>
#include <vulkan/vulkan_core.h>


#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>

#define VMA_IMPLEMENTATION
#include "common/vk_context.hpp"
#include "nvh/nvprint.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/error_vk.hpp"
#include "nvvk/extensions_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/element_logger.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvk/images_vk.hpp"


namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH


#if USE_HLSL
#include "_autogen/raster_vertexMain.spirv.h"
#include "_autogen/raster_fragmentMain.spirv.h"
const auto& vert_shd = std::vector<uint8_t>{std::begin(raster_vertexMain), std::end(raster_vertexMain)};
const auto& frag_shd = std::vector<uint8_t>{std::begin(raster_fragmentMain), std::end(raster_fragmentMain)};
#elif USE_SLANG
#include "_autogen/raster_slang.h"
#else
#include "_autogen/raster.frag.glsl.h"
#include "_autogen/raster.vert.glsl.h"
const auto& vert_shd = std::vector<uint32_t>{std::begin(raster_vert_glsl), std::end(raster_vert_glsl)};
const auto& frag_shd = std::vector<uint32_t>{std::begin(raster_frag_glsl), std::end(raster_frag_glsl)};
#endif  // USE_HLSL

static nvvkhl::SampleAppLog g_logger;

class ShaderPrintf : public nvvkhl::IAppElement
{
public:
  ShaderPrintf()           = default;
  ~ShaderPrintf() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app         = app;
    m_device      = m_app->getDevice();
    m_dutil       = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility
    m_alloc       = std::make_unique<nvvkhl::AllocVma>(VmaAllocatorCreateInfo{
              .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
              .physicalDevice = app->getPhysicalDevice(),
              .device         = app->getDevice(),
              .instance       = app->getInstance(),
    });                                                           // Allocator
    m_depthFormat = nvvk::findDepthFormat(m_app->getPhysicalDevice());  // Not all depth are supported

    createPipeline();
    createGeometryBuffers();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onUIMenu() override
  {
    static bool closeApp{false};

    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Exit", "Ctrl+Q"))
      {
        closeApp = true;
      }
      ImGui::EndMenu();
    }

    if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
    {
      closeApp = true;
    }

    if(closeApp)
    {
      m_app->close();
    }
  }

  void onResize(uint32_t width, uint32_t height) override { createGbuffers({width, height}); }

  void onUIRender() override
  {
    {  // Setting panel
      ImGui::Begin("Settings");
      ImGui::TextWrapped("Click on rectangle to print color under the mouse cursor.\n");
      ImGui::TextWrapped("Information is displayed in Log window.");
      ImGui::End();
    }

    {  // Viewport UI rendering
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Pick the mouse coordinate if the mouse is down
      if(ImGui::GetIO().MouseDown[0])
      {
        const glm::vec2 mousePos  = ImGui::GetMousePos();         // Current mouse pos in window
        const glm::vec2 corner    = ImGui::GetCursorScreenPos();  // Corner of the viewport
        m_pushConstant.mouseCoord = mousePos - corner;
      }
      else
      {
        m_pushConstant.mouseCoord = {-1, -1};
      }

      // Display the G-Buffer image
      if(m_gBuffers)
      {
        ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());
      }
      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    if(!m_gBuffers)
      return;

    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);
    nvvk::createRenderingInfo             renderingInfo({{0, 0}, m_viewSize}, {m_gBuffers->getColorImageView()},
                                                        m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                                        VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    renderingInfo.pStencilAttachment = nullptr;

    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    vkCmdBeginRendering(cmd, &renderingInfo);
    {
      m_app->setViewport(cmd);
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(DH::PushConstant), &m_pushConstant);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
      const VkDeviceSize offsets{0};
      vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT16);
      vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
  }

private:
  struct Vertex
  {
    glm::vec2 pos;
    glm::vec3 color;
  };

  void createPipeline()
  {

    const VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                                    sizeof(DH::PushConstant)};

    VkPipelineLayoutCreateInfo createInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRanges,
    };
    vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_pipelineLayout);

    VkPipelineRenderingCreateInfo prendInfo{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &m_colorFormat,
        .depthAttachmentFormat   = m_depthFormat,
    };

    nvvk::GraphicsPipelineState pstate;
    pstate.addBindingDescriptions({{0, sizeof(Vertex)}});
    pstate.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, pos))},       // Position
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, color))},  // Color
    });

    // Shader sources, pre-compiled to Spir-V (see Makefile)
    nvvk::GraphicsPipelineGenerator pgen(m_device, m_pipelineLayout, prendInfo, pstate);
#if(USE_SLANG)
    VkShaderModule shaderModule = nvvk::createShaderModule(m_device, &rasterSlang[0], sizeof(rasterSlang));
    pgen.addShader(shaderModule, VK_SHADER_STAGE_VERTEX_BIT, "vertexMain");
    pgen.addShader(shaderModule, VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain");
#else
    pgen.addShader(vert_shd, VK_SHADER_STAGE_VERTEX_BIT, USE_HLSL ? "vertexMain" : "main");
    pgen.addShader(frag_shd, VK_SHADER_STAGE_FRAGMENT_BIT, USE_HLSL ? "fragmentMain" : "main");
#endif

    m_graphicsPipeline = pgen.createPipeline();
    m_dutil->setObjectName(m_graphicsPipeline, "Graphics");
    pgen.clearShaders();
#if(USE_SLANG)
    vkDestroyShaderModule(m_device, shaderModule, nullptr);
#endif
  }

  void createGbuffers(VkExtent2D size)
  {
    m_viewSize = size;
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), m_viewSize, m_colorFormat, m_depthFormat);
  }

  void createGeometryBuffers()
  {
    const std::vector<Vertex>   vertices = {{{-0.5F, -0.5F}, {1.0F, 0.0F, 0.0F}},
                                            {{0.5F, -0.5F}, {0.0F, 1.0F, 0.0F}},
                                            {{0.5F, 0.5F}, {0.0F, 0.0F, 1.0F}},
                                            {{-0.5F, 0.5F}, {1.0F, 1.0F, 1.0F}}};
    const std::vector<uint16_t> indices  = {0, 2, 1, 2, 0, 3};

    {
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      m_vertices          = m_alloc->createBuffer(cmd, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      m_indices           = m_alloc->createBuffer(cmd, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      m_dutil->DBG_NAME(m_vertices.buffer);
      m_dutil->DBG_NAME(m_indices.buffer);
    }
  }

  void destroyResources()
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    m_alloc->destroy(m_vertices);
    m_alloc->destroy(m_indices);
    m_vertices = {};
    m_indices  = {};
    m_gBuffers.reset();
  }

  nvvkhl::Application* m_app{nullptr};

  std::unique_ptr<nvvkhl::GBuffer>  m_gBuffers;
  std::unique_ptr<nvvk::DebugUtil>  m_dutil;
  std::shared_ptr<nvvkhl::AllocVma> m_alloc;

  DH::PushConstant m_pushConstant{};

  VkExtent2D        m_viewSize{0, 0};
  VkFormat          m_colorFormat      = VK_FORMAT_R8G8B8A8_UNORM;  // Color format of the image
  VkFormat          m_depthFormat      = VK_FORMAT_UNDEFINED;       // Depth format of the depth buffer
  VkPipelineLayout  m_pipelineLayout   = VK_NULL_HANDLE;            // The description of the pipeline
  VkPipeline        m_graphicsPipeline = VK_NULL_HANDLE;            // The graphic pipeline to render
  nvvk::Buffer      m_vertices;                                     // Buffer of the vertices
  nvvk::Buffer      m_indices;                                      // Buffer of the indices
  VkClearColorValue m_clearColor{{0.0F, 0.0F, 0.0F, 1.0F}};         // Clear color
  VkDevice          m_device = VK_NULL_HANDLE;                      // Convenient
};


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)
{
  // #debug_printf : reroute the log to our nvvkhl::SampleAppLog class. The ElememtLogger will display it.
  nvprintSetCallback([](int level, const char* fmt) { g_logger.addLog(level, "%s", fmt); });
  g_logger.setLogLevel(LOGBITS_ALL);


  // #debug_printf
  // Adding the GPU debug information to the KHRONOS validation layer
  ValidationSettings validationLayer{
      .validate_gpu_based = {"GPU_BASED_DEBUG_PRINTF"},
      .printf_to_stdout   = VK_FALSE,
      .printf_buffer_size = 1024,
  };

  VkContextSettings vkSetup{
      .instanceExtensions    = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions      = {{VK_KHR_SWAPCHAIN_EXTENSION_NAME}, {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME}},
      .instanceCreateInfoExt = validationLayer.buildPNextChain(),
  };
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);

  // Create the Vulkan context with the above settings
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  if(!vkContext->isValid())
    std::exit(0);
  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);

  // Setting how we want the application
  nvvkhl::ApplicationCreateInfo appInfo;
  appInfo.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext->getInstance();
  appInfo.device         = vkContext->getDevice();
  appInfo.physicalDevice = vkContext->getPhysicalDevice();
  appInfo.queues         = vkContext->getQueueInfos();

  // Setting up the layout of the application
  appInfo.dockSetup = [](ImGuiID viewportID) {
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.5F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);
    ImGuiID logID = ImGui::DockBuilderSplitNode(settingID, ImGuiDir_Down, 0.85F, nullptr, &settingID);
    ImGui::DockBuilderDockWindow("Log", logID);
  };

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appInfo);

  //------
  // #debug_printf
  // Vulkan message callback - for receiving the printf in the shader
  auto dbgMessengerCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                 const VkDebugUtilsMessengerCallbackDataEXT* callbackData, void* userData) -> VkBool32 {
    // Get rid of all the extra message we don't need
    std::string cleanMsg = callbackData->pMessage;
    std::string delimiter = " | ";
    size_t      pos       = cleanMsg.rfind(delimiter);  // Remove everything before the last " | "
    if(pos != std::string::npos)
      cleanMsg = cleanMsg.substr(pos + delimiter.length());
    nvprintfLevel(LOGLEVEL_DEBUG, "%s", cleanMsg.c_str());  // <- This will end up in the Logger (only if DEBUG is on)
    return VK_FALSE;                                        // to continue
  };

  // Creating the callback
  VkDebugUtilsMessengerEXT           dbg_messenger{};
  VkDebugUtilsMessengerCreateInfoEXT dbg_messenger_create_info{
      .sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT,
      .messageType     = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
      .pfnUserCallback = dbgMessengerCallback,
  };
  NVVK_CHECK(vkCreateDebugUtilsMessengerEXT(app->getInstance(), &dbg_messenger_create_info, nullptr, &dbg_messenger));

  // Create a view/render
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);
  app->addElement(test);
  app->addElement(std::make_unique<nvvkhl::ElementLogger>(&g_logger, true));  // Add logger window
  app->addElement(std::make_shared<ShaderPrintf>());

  app->run();

  // #debug_printf : Removing the callback
  vkDestroyDebugUtilsMessengerEXT(app->getInstance(), dbg_messenger, nullptr);

  app.reset();
  vkContext.reset();

  return test->errorCode();
}
