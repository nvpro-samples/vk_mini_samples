/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
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

#define USE_SLANG true
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")


#include <array>
#include <vulkan/vulkan_core.h>


#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui.h>
#include <glm/glm.hpp>

#define VMA_IMPLEMENTATION

#include "shaders/shaderio.h"  // Shared between host and device


#include "_autogen/shader_printf.slang.h"
#include "_autogen/shader_printf.frag.glsl.h"
#include "_autogen/shader_printf.vert.glsl.h"

#include <nvapp/application.hpp>
#include <nvapp/elem_dbgprintf.hpp>
#include <nvapp/elem_logger.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/validation_settings.hpp>

#include <nvapp/imgui_texture.hpp>

class ShaderPrintf : public nvapp::IAppElement
{
public:
  ShaderPrintf()           = default;
  ~ShaderPrintf() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();
    m_alloc.init(VmaAllocatorCreateInfo{
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    });                                                                 // Allocator
    m_depthFormat = nvvk::findDepthFormat(m_app->getPhysicalDevice());  // Not all depth are supported

    // Initialization of the render target we want to use
    NVVK_CHECK(m_renderTarget.init(
        {.alloc = &m_alloc, .colorFormats = {m_colorFormat}, .depthFormat = m_depthFormat, .debugName = "ShaderPrintf"}));

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

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    NVVK_CHECK(m_renderTarget.update(cmd, size));
    m_viewportImage.update(m_renderTarget.getUiImageView());
  }

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

      // Display the rendered image
      {
        ImGui::Image(m_viewportImage, ImGui::GetContentRegionAvail());
      }
      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

    // Build the dynamic-rendering state from the render target. The render target
    // keeps its images in VK_IMAGE_LAYOUT_GENERAL, so no layout transitions are
    // needed to render into it or to sample it afterwards.
    nvvk::RenderTargetState rtState;
    m_renderTarget.fillState(rtState);
    rtState.colorAttachments[0].clearValue = {m_clearColor};
    rtState.depthAttachment.clearValue     = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

    nvvk::RenderTargetState::AttachmentOps ops{};  // default: clear+store on color & depth, don't care on stencil
    rtState.cmdBeginRendering(cmd, ops);
    {
      nvvk::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, m_renderTarget.getSize());
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(shaderio::PushConstant), &m_pushConstant);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
      const VkDeviceSize offsets{0};
      vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT16);
      vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
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
                                                    sizeof(shaderio::PushConstant)};

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

    pstate.vertexBindings = {
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, .stride = sizeof(Vertex), .divisor = 1}};
    pstate.vertexAttributes = {{.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                .location = 0,
                                .format   = VK_FORMAT_R32G32_SFLOAT,
                                .offset   = offsetof(Vertex, pos)},
                               {.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                .location = 1,
                                .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                .offset   = offsetof(Vertex, color)}};

    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = m_pipelineLayout;
    creator.colorFormats                         = {m_colorFormat};
    creator.renderingState.depthAttachmentFormat = m_depthFormat;

    // Shader sources, pre-compiled to Spir-V (see Makefile)
#if (USE_SLANG)
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", shader_printf_slang);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", shader_printf_slang);
#else
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", shader_printf_vert_glsl);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", shader_printf_frag_glsl);
#endif
    creator.createGraphicsPipeline(m_device, nullptr, pstate, &m_graphicsPipeline);
    NVVK_DBG_NAME(m_graphicsPipeline);
  }

  void createGeometryBuffers()
  {
    const std::vector<Vertex>   vertices = {{{-0.5F, -0.5F}, {1.0F, 0.0F, 0.0F}},
                                            {{0.5F, -0.5F}, {0.0F, 1.0F, 0.0F}},
                                            {{0.5F, 0.5F}, {0.0F, 0.0F, 1.0F}},
                                            {{-0.5F, 0.5F}, {1.0F, 1.0F, 1.0F}}};
    const std::vector<uint16_t> indices  = {0, 2, 1, 2, 0, 3};

    {
      VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
      nvvk::StagingUploader uploader;
      uploader.init(&m_alloc);

      NVVK_CHECK(m_alloc.createBuffer(m_vertices, std::span(vertices).size_bytes(), VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT));
      NVVK_CHECK(m_alloc.createBuffer(m_indices, std::span(indices).size_bytes(), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT));
      NVVK_CHECK(uploader.appendBuffer(m_vertices, 0, std::span(vertices)));
      NVVK_CHECK(uploader.appendBuffer(m_indices, 0, std::span(indices)));
      NVVK_DBG_NAME(m_vertices.buffer);
      NVVK_DBG_NAME(m_indices.buffer);

      uploader.cmdUploadAppended(cmd);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      uploader.deinit();
    }
  }

  void destroyResources()
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    m_alloc.destroyBuffer(m_vertices);
    m_alloc.destroyBuffer(m_indices);
    m_vertices = {};
    m_indices  = {};

    m_viewportImage.deinit();
    m_renderTarget.deinit();
    m_alloc.deinit();
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_renderTarget.getColorImage(), m_renderTarget.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  //----------------------------------------------------------------------------------
  nvapp::Application*     m_app{};
  nvvk::RenderTarget      m_renderTarget;   // Offscreen render target: color + depth
  nvapp::ImTexture        m_viewportImage;  // ImGui texture for the render target color image
  nvvk::ResourceAllocator m_alloc;

  shaderio::PushConstant m_pushConstant{};

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
  nvapp::ApplicationCreateInfo appInfo;

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.addVector({"size", "Size of the window to be created", "s"}, &appInfo.windowSize);
  reg.add({"headless"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);

  // This is the Element Logger, which will be used to display the log in the UI
  static auto elemLogger = std::make_shared<nvapp::ElementLogger>(true);
  elemLogger->setLevelFilter(nvapp::ElementLogger::eBitERROR | nvapp::ElementLogger::eBitWARNING | nvapp::ElementLogger::eBitINFO);

  // The logger will redirect the log to the Element Logger, to be displayed in the UI
  nvutils::Logger::getInstance().setLogCallback([](nvutils::Logger::LogLevel logLevel, const std::string& str) {
    elemLogger->addLog(logLevel, "%s", str.c_str());
  });


  // Create the Vulkan context
  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME}},
  };
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
    vkSetup.deviceExtensions.push_back({VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME});
  }

  // This element helps parsing the printf output, and clean it
  // It is also creating the VkDebugUtilsMessengerEXT with info level, which isn't set be default
  auto elemDbgPrintf = std::make_shared<nvapp::ElementDbgPrintf>();


  // #debug_printf
  // This is adding the settings to enable the validation layer for the debug printf
  nvvk::ValidationSettings validation{};
  validation.setPreset(nvvk::ValidationSettings::LayerPresets::eDebugPrintf);
  validation.printf_to_stdout   = VK_FALSE;  // Different from default to capture output
  vkSetup.instanceCreateInfoExt = validation.buildPNextChain();


  // Create the Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  // Setting how we want the application
  appInfo.name           = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Setting up the layout of the application
  appInfo.dockSetup = [](ImGuiID viewportID) {
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.5F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);
    ImGuiID logID = ImGui::DockBuilderSplitNode(settingID, ImGuiDir_Down, 0.85F, nullptr, &settingID);
    ImGui::DockBuilderDockWindow("Log", logID);
  };

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // Create a view/render
  app.addElement(elemLogger);     // Add logger window
  app.addElement(elemDbgPrintf);  // Add the debug printf element
  app.addElement(std::make_shared<ShaderPrintf>());

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
