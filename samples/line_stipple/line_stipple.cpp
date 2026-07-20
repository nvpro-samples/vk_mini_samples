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


/*---------------------------------------------------------------------------------------------------------------------
 This sample shows how to use the VK_EXT_line_rasterization extension to enable line stippling.
 ---------------------------------------------------------------------------------------------------------------------*/

#define USE_SLANG true
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION

#include <glm/gtc/type_ptr.hpp>  // value_ptr

// Pre-compiled shaders
#include "_autogen/line_stipple.slang.h"
#include "_autogen/line_stipple.vert.glsl.h"
#include "_autogen/line_stipple.frag.glsl.h"

#include "shaders/shaderio.h"

#include <nvapp/application.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvgui/property_editor.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/formats.hpp>
#include <nvapp/imgui_texture.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>

constexpr int32_t numStripVertices = 300;


class LineStippleElement : public nvapp::IAppElement
{
  struct Settings
  {
    // Define different stipple patterns
    const std::vector<std::pair<uint32_t, uint16_t>> stipplePatterns = {
        {1, 0xFFFF},  // Solid line
        {2, 0xAAAA},  // Dashed line
        {3, 0xEEEE},  // Dotted line
        {4, 0xFFF0},  // Dash-dot line
        {5, 0xE4E4},  // Complex pattern
    };
    float                      lineWidth{1.0f};
    VkLineRasterizationModeKHR lineRasterizationMode{VK_LINE_RASTERIZATION_MODE_BRESENHAM};
    shaderio::Transform        transform{.color = {1, 1, 1}};
    VkSampleCountFlagBits      msaaSamples{VK_SAMPLE_COUNT_1_BIT};
    bool                       crossLine{false};  // Make lines cross
    bool                       lineStrip{false};
  } m_settings;


public:
  LineStippleElement()           = default;
  ~LineStippleElement() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();
    m_alloc.init({
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });

    // Offscreen render target with built-in MSAA: the scene is rendered into a
    // multisampled color+depth image and the color is resolved to a single-sample
    // image for display. RenderTarget creates and resolves those images itself,
    // so the sample no longer manages any MSAA image/view by hand.
    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
    NVVK_CHECK(m_renderTarget.init({.alloc        = &m_alloc,
                                    .colorFormats = {m_colorFormat},
                                    .depthFormat  = m_depthFormat,
                                    .msaa = {.samples = m_settings.msaaSamples, .colorResolve = VK_RESOLVE_MODE_AVERAGE_BIT},
                                    .debugName = "LineStipple"}));


    // Check which features are supported
    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    m_lineFeature.pNext = &m_dynamicFeature;
    features2.pNext     = &m_lineFeature;
    vkGetPhysicalDeviceFeatures2(m_app->getPhysicalDevice(), &features2);

    // Find default (available) stipple pattern
    if(m_lineFeature.stippledBresenhamLines)
      m_settings.lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_BRESENHAM;
    else if(m_lineFeature.stippledRectangularLines)
      m_settings.lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR;
    else
      m_settings.lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_DEFAULT;

    createGraphicsPipeline();
    createGeometry();
  }

  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_device));
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
    m_alloc.destroyBuffer(m_vertexBuffer);
    m_viewportImage.deinit();
    m_renderTarget.deinit();
    m_alloc.deinit();
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);

    // #MSAA - RenderTarget owns the multisampled color+depth images and the
    // single-sample resolve target. fillState() wires the resolve automatically
    // when the sample count is > 1, so there is no manual MSAA bookkeeping here.
    nvvk::RenderTargetState rtState;
    m_renderTarget.fillState(rtState);
    rtState.colorAttachments[0].clearValue = {{0.2f, 0.2f, 0.3f, 1.0f}};
    rtState.depthAttachment.clearValue     = {.depthStencil = {1.0f, 0}};
    nvvk::RenderTargetState::AttachmentOps ops{};  // default: clear+store on color & depth, don't care on stencil
    rtState.cmdBeginRendering(cmd, ops);

    nvvk::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, m_renderTarget.getSize());
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    drawStippledLines(cmd);

    vkCmdEndRendering(cmd);
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    NVVK_CHECK(m_renderTarget.update(cmd, size));
    // Views are recreated on resize: refresh the ImGui texture that displays the
    // resolved image in the viewport.
    m_viewportImage.update(m_renderTarget.getUiImageView());
  }

  void onUIMenu() override
  {
    static bool close_app{false};

    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Exit", "Ctrl+Q"))
        close_app = true;
      ImGui::EndMenu();
    }

    if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
      close_app = true;

    if(close_app)
      m_app->close();
  }


  void onUIRender() override
  {
    settingUI();
    displayGbuffer();
  }


private:
  void createGraphicsPipeline()
  {
    // Push constant accessible to the vertex shaders
    VkPushConstantRange pushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(shaderio::Transform)};

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRange,
    };

    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_pipelineLayout));

    nvvk::GraphicsPipelineState graphicState;                                     // State of the graphic pipeline
    graphicState.multisampleState.rasterizationSamples = m_settings.msaaSamples;  // #MSAA

    graphicState.vertexBindings = {
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, .stride = sizeof(glm::vec3), .divisor = 1}};
    graphicState.vertexAttributes = {
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT, .format = VK_FORMAT_R32G32B32_SFLOAT}};


    graphicState.rasterizationState.lineWidth   = 1.0F;  // Width of the line : can be overridden
    graphicState.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
    graphicState.inputAssemblyState.topology    = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    // Depth test
    graphicState.depthStencilState.depthTestEnable  = VK_FALSE;  // VK_TRUE;
    graphicState.depthStencilState.depthWriteEnable = VK_TRUE;
    graphicState.depthStencilState.depthCompareOp   = VK_COMPARE_OP_ALWAYS;  // VK_COMPARE_OP_LESS_OR_EQUAL;

    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = m_pipelineLayout;
    creator.colorFormats                         = {m_colorFormat};
    creator.renderingState.depthAttachmentFormat = m_depthFormat;

    // Enable dynamic state
    creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_LINE_STIPPLE);
    creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_LINE_WIDTH);
    creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY);
    creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_LINE_STIPPLE_ENABLE_EXT);
    if(m_dynamicFeature.extendedDynamicState3LineRasterizationMode)
    {
      creator.dynamicStateValues.push_back(VK_DYNAMIC_STATE_LINE_RASTERIZATION_MODE_EXT);
    }

    // Shader sources, pre-compiled to Spir-V (see Makefile)
#if USE_SLANG
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", std::span(line_stipple_slang));
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", std::span(line_stipple_slang));
#else
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", std::span(line_stipple_vert_glsl));
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", std::span(line_stipple_frag_glsl));
#endif

    NVVK_CHECK(creator.createGraphicsPipeline(m_device, nullptr, graphicState, &m_graphicsPipeline));
    NVVK_DBG_NAME(m_graphicsPipeline);
  }

  // Creating all the lines and store in a buffer
  void createGeometry()
  {
    std::vector<glm::vec3> vertices;
    // 5 lines with different stipple patterns
    for(size_t i = 0; i < m_settings.stipplePatterns.size(); ++i)
    {
      float y = 0.8f - 0.4f * i;
      vertices.emplace_back(-0.9f, y, float(i) / 10.f);
      vertices.emplace_back(0.9f, y, 1.0 - float(i) / 10.f);
    }

    for(int32_t i = 0; i < numStripVertices; ++i)
    {
      float x = -0.9f + (1.8f * i / float(numStripVertices));
      float y = cos((16.f * i) / float(numStripVertices)) * 0.9f;
      vertices.emplace_back(x, y, 0.5f);
    }

    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    NVVK_CHECK(m_alloc.createBuffer(m_vertexBuffer, std::span(vertices).size_bytes(), VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT));
    NVVK_CHECK(uploader.appendBuffer(m_vertexBuffer, 0, std::span(vertices)));
    NVVK_DBG_NAME(m_vertexBuffer.buffer);
    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();
  }

  //-------------------------------------------------------------------------------------------------
  // Drawing the stippled lines
  //
  void drawStippledLines(VkCommandBuffer cmd)
  {
    float        rotation[] = {-10, 21, 135, 78, 12};
    VkDeviceSize offsets[]  = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer.buffer, offsets);

    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(shaderio::Transform), &m_settings.transform);
    vkCmdSetLineWidth(cmd, m_settings.lineWidth);  // Width of the line

    if(m_dynamicFeature.extendedDynamicState3LineRasterizationMode)
      vkCmdSetLineRasterizationModeEXT(cmd, m_settings.lineRasterizationMode);

    vkCmdSetLineStippleEnableEXT(cmd, VK_TRUE);  // Enable line stipple
    if(!m_settings.lineStrip)
    {
      // Show segment of lines with different stipple patterns
      vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
      for(size_t i = 0; i < m_settings.stipplePatterns.size(); ++i)
      {
        if(m_settings.crossLine)
        {
          shaderio::Transform transform = m_settings.transform;
          transform.rotation += glm::radians(rotation[i]);
          transform.translation.y += 0.5f - 0.25f * i;
          const glm::vec3 freq = glm::vec3(1.33333F, 2.33333F, 3.33333F) * static_cast<float>(i);
          transform.color      = static_cast<glm::vec3>(sin(freq) * 0.5F + 0.5F);
          vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(shaderio::Transform), &transform);
        }
        vkCmdSetLineStipple(cmd, m_settings.stipplePatterns[i].first, m_settings.stipplePatterns[i].second);
        vkCmdDraw(cmd, 2, 1, uint32_t(i * 2), 0);
      }
    }
    else
    {
      // Show a line strip with a stipple patterns
      vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(shaderio::Transform), &m_settings.transform);
      vkCmdSetLineStipple(cmd, m_settings.stipplePatterns[4].first, m_settings.stipplePatterns[4].second);
      uint32_t offset = uint32_t(m_settings.stipplePatterns.size() * 2);
      vkCmdDraw(cmd, numStripVertices, 1, offset, 0);
    }
  }


  //-------------------------------------------------------------------------------------------------
  // UI to change the settings
  void settingUI()
  {
    namespace PE = nvgui::PropertyEditor;

    if(ImGui::Begin("Settings"))
    {
      {
        // #MSAA
        ImGui::Separator();
        ImGui::Text("MSAA Settings");
        VkImageFormatProperties image_format_properties;
        NVVK_CHECK(vkGetPhysicalDeviceImageFormatProperties(m_app->getPhysicalDevice(), m_renderTarget.getColorFormat(),
                                                            VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 0, &image_format_properties));
        // sampleCounts is 3, 7 or 15, following line find n, in 2^n == sampleCounts+1
        const int max_sample_items = static_cast<int>(log2(static_cast<float>(image_format_properties.sampleCounts)) + 1.0F);
        // Same for the current VkSampleCountFlag, which is a power of two
        int                        item_combo = static_cast<int>(log2(static_cast<float>(m_settings.msaaSamples)));
        std::array<const char*, 7> items      = {"1", "2", "4", "8", "16", "32", "64"};
        ImGui::Text("Sample Count");
        ImGui::SameLine();
        if(ImGui::Combo("##Sample Count", &item_combo, items.data(), max_sample_items))
        {
          auto samples           = static_cast<int32_t>(powf(2, static_cast<float>(item_combo)));
          m_settings.msaaSamples = static_cast<VkSampleCountFlagBits>(samples);

          NVVK_CHECK(vkDeviceWaitIdle(m_device));  // Flushing the graphic pipeline

          // Reconfigure the render target for the new sample count and recreate its
          // GPU resources. MSAA requires a resolve mode; a 1x sample count uses none.
          const VkResolveModeFlagBits resolveMode =
              m_settings.msaaSamples == VK_SAMPLE_COUNT_1_BIT ? VK_RESOLVE_MODE_NONE : VK_RESOLVE_MODE_AVERAGE_BIT;
          NVVK_CHECK(m_renderTarget.setSampleCount(m_settings.msaaSamples, resolveMode));
          VkCommandBuffer cmd = m_app->createTempCmdBuffer();
          NVVK_CHECK(m_renderTarget.update(cmd, m_app->getViewportSize()));
          m_app->submitAndWaitTempCmdBuffer(cmd);
          m_viewportImage.update(m_renderTarget.getUiImageView());

          // The graphic pipeline contains MSAA information
          vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
          vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
          createGraphicsPipeline();
        }
      }

      PE::begin();  // PropertyEditor
      PE::SliderFloat("Line Width", &m_settings.lineWidth, 1.0f, 20.0f);
      PE::SliderFloat2("Translation", glm::value_ptr(m_settings.transform.translation), -0.5f, 0.5f);
      PE::SliderAngle("Rotation", &m_settings.transform.rotation, -180.0f, 180.0f);
      PE::Checkbox("Cross Line", &m_settings.crossLine);
      PE::Checkbox("Line Strip", &m_settings.lineStrip);

      ImGui::BeginDisabled(!m_dynamicFeature.extendedDynamicState3LineRasterizationMode);
      PE::entry("Rasterization Mode", [&] {
        return ImGui::RadioButton("Default", (int*)&m_settings.lineRasterizationMode, VK_LINE_RASTERIZATION_MODE_DEFAULT);
      });
      PE::entry("", [&] {
        return ImGui::RadioButton("Rectangular", (int*)&m_settings.lineRasterizationMode, VK_LINE_RASTERIZATION_MODE_RECTANGULAR);
      });
      PE::entry("", [&] {
        return ImGui::RadioButton("Bresenham", (int*)&m_settings.lineRasterizationMode, VK_LINE_RASTERIZATION_MODE_BRESENHAM);
      });
      PE::entry("", [&] {
        return ImGui::RadioButton("Smooth", (int*)&m_settings.lineRasterizationMode, VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH);
      });
      ImGui::EndDisabled();

      if(!m_settings.lineStrip)
      {
        for(size_t i = 0; i < m_settings.stipplePatterns.size(); ++i)
        {
          auto& [factor, pattern] = m_settings.stipplePatterns[i];
          if(PE::treeNode(fmt::format("Pattern {}", i)))
          {
            PE::SliderInt("factor", (int*)&factor, 1, 8, "%d", ImGuiSliderFlags_None);
            PE::InputScalar("pattern", ImGuiDataType_U16, (void*)&pattern, nullptr, nullptr, "%04X", ImGuiInputTextFlags_CharsHexadecimal);
            PE::treePop();
          }
        }
      }
      else
      {
        int i                   = 4;
        auto& [factor, pattern] = m_settings.stipplePatterns[i];
        if(PE::treeNode(fmt::format("Pattern {}", i)))
        {
          PE::SliderInt("factor", (int*)&factor, 1, 8, "%d", ImGuiSliderFlags_None);
          PE::InputScalar("pattern", ImGuiDataType_U16, (void*)&pattern, nullptr, nullptr, "%04X", ImGuiInputTextFlags_CharsHexadecimal);
          PE::treePop();
        }
      }
      PE::end();
      ImGui::End();
    }
  }

  // Display the G-Buffer image in Viewport
  void displayGbuffer()
  {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");

    // Display the resolved image
    ImGui::Image(m_viewportImage, ImGui::GetContentRegionAvail());

    ImGui::End();
    ImGui::PopStyleVar();
  }

  void onLastHeadlessFrame() override
  {
    // Save the displayed image (resolved single-sample when MSAA is active).
    m_app->saveImageToFile(m_renderTarget.getSampleImage(), m_renderTarget.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  //-------------------------------------------------------------------------------------------------
  nvapp::Application*     m_app{};          // Application
  nvvk::RenderTarget      m_renderTarget;   // Offscreen MSAA target (color+depth) with color resolve
  nvapp::ImTexture        m_viewportImage;  // ImGui texture displaying the resolved image
  nvvk::ResourceAllocator m_alloc;          // Allocator


  VkDevice         m_device           = VK_NULL_HANDLE;            // Convenient
  VkFormat         m_colorFormat      = VK_FORMAT_B8G8R8A8_UNORM;  // Color format of the image
  VkFormat         m_depthFormat      = VK_FORMAT_UNDEFINED;       // Depth format of the depth buffer
  VkPipelineLayout m_pipelineLayout   = VK_NULL_HANDLE;            // The description of the pipeline
  VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;            // The graphic pipeline to render

  nvvk::Buffer m_vertexBuffer;  // Buffer of the vertices
  VkPhysicalDeviceLineRasterizationFeaturesKHR m_lineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES};
  VkPhysicalDeviceExtendedDynamicState3FeaturesEXT m_dynamicFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
};

int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;  // Base application information

  // Command line parsing
  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);


  VkPhysicalDeviceExtendedDynamicState3FeaturesEXT extendedDynamicStateFeature{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
  nvvk::ContextInitInfo vkSetup = {
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions =
          {
              {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
              {VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, &extendedDynamicStateFeature, false},
          },
      .apiVersion = VK_API_VERSION_1_4,
  };
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  // Creation of the Vulkan context
  nvvk::Context vkContext;  // Vulkan context
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  // Check if line rasterization is supported
  if(extendedDynamicStateFeature.extendedDynamicState3LineStippleEnable == VK_FALSE)
  {
    LOGE("Missing dynamic feature to enable line stipple");
    return 1;
  }

  // Application setup
  appInfo.name           = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.windowSize     = {800, 600};
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  app.addElement(std::make_shared<LineStippleElement>());
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
