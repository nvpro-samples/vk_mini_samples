/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION
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

// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const nvmath::vec2f& f) {x = f.x; y = f.y;} operator nvmath::vec2f() const { return nvmath::vec2f(x, y); }
// clang-format on

#include <array>
#include <vulkan/vulkan_core.h>
#include "nvmath/nvmath.h"
#include <imgui.h>

#define VMA_IMPLEMENTATION
#include "nvh/nvprint.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_logger.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"

#include "_autogen/raster.frag.h"
#include "_autogen/raster.vert.h"
#include "shaders/device_host.h"


nvvkhl::SampleAppLog g_logger;

namespace nvvkhl {


class ShaderPrintf : public nvvkhl::IAppElement
{
public:
  ShaderPrintf()           = default;
  ~ShaderPrintf() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app         = app;
    m_device      = m_app->getDevice();
    m_dutil       = std::make_unique<nvvk::DebugUtil>(m_device);            // Debug utility
    m_alloc       = std::make_unique<AllocVma>(m_app->getContext().get());  // Allocator
    m_depthFormat = nvvk::findDepthFormat(m_app->getPhysicalDevice());      // Not all depth are supported

    createPipeline();
    createGeometryBuffers();
  }

  void onDetach() override { detroyResources(); }

  void onUIMenu() override
  {
    static bool close_app{false};

    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Exit", "Ctrl+Q"))
      {
        close_app = true;
      }
      ImGui::EndMenu();
    }

    if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
    {
      close_app = true;
    }

    if(close_app)
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
        nvmath::vec2f mouse_pos   = ImGui::GetMousePos();         // Current mouse pos in window
        nvmath::vec2f corner      = ImGui::GetCursorScreenPos();  // Corner of the viewport
        m_pushConstant.mouseCoord = mouse_pos - corner;
      }
      else
      {
        m_pushConstant.mouseCoord = {-1, -1};
      }

      // Display the G-Buffer image
      ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());
      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    auto _sdbg = m_dutil->DBG_SCOPE(cmd);
    nvvk::createRenderingInfo r_info({{0, 0}, m_viewSize}, {m_gBuffers->getColorImageView()}, m_gBuffers->getDepthImageView(),
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    vkCmdBeginRendering(cmd, &r_info);
    m_app->setViewport(cmd);

    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstant), &m_pushConstant);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    VkDeviceSize offsets{0};
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
    vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);

    vkCmdEndRendering(cmd);
  }

private:
  struct Vertex
  {
    nvmath::vec2f pos;
    nvmath::vec3f color;
  };

  void createPipeline()
  {

    VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstant)};

    VkPipelineLayoutCreateInfo create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    create_info.pushConstantRangeCount = 1;
    create_info.pPushConstantRanges    = &push_constant_ranges;
    vkCreatePipelineLayout(m_device, &create_info, nullptr, &m_pipelineLayout);

    VkPipelineRenderingCreateInfo prend_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    prend_info.colorAttachmentCount    = 1;
    prend_info.pColorAttachmentFormats = &m_colorFormat;
    prend_info.depthAttachmentFormat   = m_depthFormat;

    nvvk::GraphicsPipelineState pstate;
    pstate.addBindingDescriptions({{0, sizeof(Vertex)}});
    pstate.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, pos))},       // Position
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, color))},  // Color
    });

    // Shader sources, pre-compiled to Spir-V (see Makefile)
    nvvk::GraphicsPipelineGenerator pgen(m_device, m_pipelineLayout, prend_info, pstate);
    pgen.addShader(std::vector<uint32_t>{std::begin(raster_vert), std::end(raster_vert)}, VK_SHADER_STAGE_VERTEX_BIT);
    pgen.addShader(std::vector<uint32_t>{std::begin(raster_frag), std::end(raster_frag)}, VK_SHADER_STAGE_FRAGMENT_BIT);

    m_graphicsPipeline = pgen.createPipeline();
    m_dutil->setObjectName(m_graphicsPipeline, "Graphics");
    pgen.clearShaders();
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
      auto* cmd  = m_app->createTempCmdBuffer();
      m_vertices = m_alloc->createBuffer(cmd, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      m_indices  = m_alloc->createBuffer(cmd, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      m_dutil->DBG_NAME(m_vertices.buffer);
      m_dutil->DBG_NAME(m_indices.buffer);
    }
  }

  void detroyResources()
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

  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;
  std::unique_ptr<nvvk::DebugUtil> m_dutil;
  std::shared_ptr<AllocVma>        m_alloc;

  PushConstant m_pushConstant{};

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

}  // namespace nvvkhl


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

auto main(int argc, char** argv) -> int
{
  // #ElementLogger For logging
  nvprintSetCallback([](int /*level*/, const char* fmt) { g_logger.addLog("%s", fmt); });

  nvvkhl::ApplicationCreateInfo spec;
  spec.name             = PROJECT_NAME " Example";
  spec.vSync            = true;
  spec.vkSetup.apiMajor = 1;
  spec.vkSetup.apiMinor = 3;

  // #debug_printf
  VkValidationFeaturesEXT                    features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
  std::vector<VkValidationFeatureEnableEXT>  enables{VK_VALIDATION_FEATURE_ENABLE_DEBUG_PRINTF_EXT};
  std::vector<VkValidationFeatureDisableEXT> disables{};
  features.enabledValidationFeatureCount  = static_cast<uint32_t>(enables.size());
  features.pEnabledValidationFeatures     = enables.data();
  features.disabledValidationFeatureCount = static_cast<uint32_t>(disables.size());
  features.pDisabledValidationFeatures    = disables.data();
  spec.vkSetup.instanceCreateInfoExt      = &features;


  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // #debug_printf : allow all messages to go through
  app->getContext()->setDebugSeverityFilterMask(VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT);

  // Create a view/render
  auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ShaderPrintf>());
  app->addElement(std::make_unique<nvvkhl::ElementLogger>(&g_logger, true));  // Add logger window

  app->run();
  app.reset();

  return test->errorCode();
}