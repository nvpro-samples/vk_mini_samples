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


/*

 This sample shows how to use the VK_EXT_line_rasterization extension to enable line stippling.

*/


#include <glm/gtc/type_ptr.hpp>  // value_ptr

#include "common/alloc_dma.hpp"                     // For the memory allocation
#include "common/vk_context.hpp"                    // For the Vulkan context
#include "imgui/imgui_helper.h"                     // Helper for UI, PropertyEditor
#include "nvvk/debug_util_vk.hpp"                   // Utility to name objects
#include "nvvk/extensions_vk.hpp"                   // For the Vulkan extensions
#include "nvvk/images_vk.hpp"                       // For the image creation
#include "nvvk/pipeline_vk.hpp"                     // For the pipeline creation
#include "nvvkhl/element_benchmark_parameters.hpp"  // For the test/benchmark parameters
#include "nvvkhl/element_gui.hpp"                   // For menu UI
#include "nvvkhl/gbuffer.hpp"                       // For the G-Buffer
#include "nvvk/shaders_vk.hpp"                      // For create shader module

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


namespace DH {
using namespace glm;
#include "shaders/device_host.h"
}  // namespace DH

constexpr int32_t numStripVertices = 300;


class LineStippleElement : public nvvkhl::IAppElement
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
    VkLineRasterizationModeKHR lineRasterizationMode{VK_LINE_RASTERIZATION_MODE_DEFAULT_KHR};
    DH::Transform              transform{.color = {1, 1, 1}};
    VkSampleCountFlagBits      msaaSamples{VK_SAMPLE_COUNT_1_BIT};
    bool                       crossLine{false};  // Make lines cross
    bool                       lineStrip{false};
  } m_settings;


public:
  LineStippleElement()           = default;
  ~LineStippleElement() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();
    m_dutil  = std::make_unique<nvvk::DebugUtil>(m_device);                            // Debug utility
    m_alloc = std::make_unique<AllocDma>(app->getDevice(), app->getPhysicalDevice());  // Allocator for buffer, images, acceleration structures

    // Check which features are supported
    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    m_lineFeature.pNext = &m_dynamicFeature;
    features2.pNext     = &m_lineFeature;
    vkGetPhysicalDeviceFeatures2(m_app->getPhysicalDevice(), &features2);

    // Find default (available) stipple pattern
    if(m_lineFeature.stippledRectangularLines)
      m_settings.lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_KHR;
    else if(m_lineFeature.stippledBresenhamLines)
      m_settings.lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR;
    else
      m_settings.lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_DEFAULT_KHR;

    createGraphicsPipeline();
    createGeometry();
  }

  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_device));
    destroyResources();
  }

  void onRender(VkCommandBuffer cmd) override
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    VkRenderingAttachmentInfoKHR colorAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
        .imageView   = m_gBuffers->getColorImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = {{0.2f, 0.2f, 0.3f, 1.0f}},
    };

    VkRenderingAttachmentInfoKHR depthAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
        .imageView   = m_gBuffers->getDepthImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = {1.0f, 0},
    };

    VkRenderingInfoKHR renderingInfo{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
        .renderArea           = {{0, 0}, m_app->getViewportSize()},
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
        .pDepthAttachment     = &depthAttachment,
    };

    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Render to MSAA image and resolve to G-Buffer
    if(m_settings.msaaSamples != VK_SAMPLE_COUNT_1_BIT)
    {
      colorAttachment.imageView          = m_msaaGBuffers->getColorImageView();
      colorAttachment.resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;
      colorAttachment.resolveImageView   = m_gBuffers->getColorImageView();
      colorAttachment.resolveMode        = VK_RESOLVE_MODE_AVERAGE_BIT;
      depthAttachment.imageView          = m_msaaGBuffers->getDepthImageView();
    }

    vkCmdBeginRendering(cmd, &renderingInfo);
    {
      const VkDeviceSize offsets{0};
      m_app->setViewport(cmd);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
      drawStippledLines(cmd);
    }
    vkCmdEndRendering(cmd);
    nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), VkExtent2D{width, height}, m_colorFormat, m_depthFormat);
    createMsaaImage();
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
    VkPushConstantRange pushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .offset = 0, .size = sizeof(DH::Transform)};

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstantRange,
    };

    NVVK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_pipelineLayout));

    VkPipelineRenderingCreateInfo prend_info{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &m_colorFormat,
        .depthAttachmentFormat   = m_depthFormat,
    };

    nvvk::GraphicsPipelineState pstate;
    pstate.multisampleState.rasterizationSamples = m_settings.msaaSamples;  // #MSAA
    pstate.addBindingDescriptions({{0, sizeof(glm::vec3)}});
    pstate.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},  // Position
    });
    pstate.rasterizationState.lineWidth   = 1.0F;  // Width of the line : can be overridden
    pstate.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
    pstate.inputAssemblyState.topology    = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    // Depth test
    pstate.depthStencilState.depthTestEnable  = VK_FALSE;  // VK_TRUE;
    pstate.depthStencilState.depthWriteEnable = VK_TRUE;
    pstate.depthStencilState.depthCompareOp   = VK_COMPARE_OP_ALWAYS;  // VK_COMPARE_OP_LESS_OR_EQUAL;

    // Enable dynamic state
    pstate.addDynamicStateEnable(VK_DYNAMIC_STATE_LINE_STIPPLE_KHR);
    pstate.addDynamicStateEnable(VK_DYNAMIC_STATE_LINE_WIDTH);
    pstate.addDynamicStateEnable(VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY);
    if(m_dynamicFeature.extendedDynamicState3LineRasterizationMode)
      pstate.addDynamicStateEnable(VK_DYNAMIC_STATE_LINE_RASTERIZATION_MODE_EXT);

    // Enable line stipple
    VkPipelineRasterizationLineStateCreateInfoEXT lineRasterizationStateInfo{
        .sType                 = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_EXT,
        .lineRasterizationMode = m_settings.lineRasterizationMode,  // Maybe dynamically changeable
        .stippledLineEnable    = VK_TRUE,
        .lineStippleFactor     = 1,       // Dynamically changeable
        .lineStipplePattern    = 0xF0FF,  // Dynamically changeable
    };
    pstate.rasterizationState.pNext = &lineRasterizationStateInfo;


    // Shader sources, pre-compiled to Spir-V (see Makefile)
    nvvk::GraphicsPipelineGenerator pgen(m_device, m_pipelineLayout, prend_info, pstate);
#if USE_SLANG
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
#if USE_SLANG
    vkDestroyShaderModule(m_device, shaderModule, nullptr);
#endif
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

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_vertexBuffer      = m_alloc->createBuffer(cmd, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_dutil->setObjectName(m_vertexBuffer.buffer, "Vertex Buffer");
  }

  //-------------------------------------------------------------------------------------------------
  // Drawing the stippled lines
  //
  void drawStippledLines(VkCommandBuffer cmd)
  {
    float        rotation[] = {-10, 21, 135, 78, 12};
    VkDeviceSize offsets[]  = {0};
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer.buffer, offsets);

    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DH::Transform), &m_settings.transform);
    vkCmdSetLineWidth(cmd, m_settings.lineWidth);  // Width of the line

    if(m_dynamicFeature.extendedDynamicState3LineRasterizationMode)
      vkCmdSetLineRasterizationModeEXT(cmd, m_settings.lineRasterizationMode);

    if(!m_settings.lineStrip)
    {
      // Show segment of lines with different stipple patterns
      vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_LINE_LIST);
      for(size_t i = 0; i < m_settings.stipplePatterns.size(); ++i)
      {
        if(m_settings.crossLine)
        {
          DH::Transform transform = m_settings.transform;
          transform.rotation += glm::radians(rotation[i]);
          transform.translation.y += 0.5f - 0.25f * i;
          const glm::vec3 freq = glm::vec3(1.33333F, 2.33333F, 3.33333F) * static_cast<float>(i);
          transform.color      = static_cast<glm::vec3>(sin(freq) * 0.5F + 0.5F);
          vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DH::Transform), &transform);
        }
        vkCmdSetLineStippleEXT(cmd, m_settings.stipplePatterns[i].first, m_settings.stipplePatterns[i].second);
        vkCmdDraw(cmd, 2, 1, uint32_t(i * 2), 0);
      }
    }
    else
    {
      // Show a line strip with a stipple patterns
      vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP);
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DH::Transform), &m_settings.transform);
      vkCmdSetLineStippleEXT(cmd, m_settings.stipplePatterns[4].first, m_settings.stipplePatterns[4].second);
      uint32_t offset = uint32_t(m_settings.stipplePatterns.size() * 2);
      vkCmdDraw(cmd, numStripVertices, 1, offset, 0);
    }
  }


  //-------------------------------------------------------------------------------------------------
  // UI to change the settings
  void settingUI()
  {
    namespace PE = ImGuiH::PropertyEditor;

    if(ImGui::Begin("Settings"))
    {
      {
        // #MSAA
        ImGui::Separator();
        ImGui::Text("MSAA Settings");
        VkImageFormatProperties image_format_properties;
        NVVK_CHECK(vkGetPhysicalDeviceImageFormatProperties(m_app->getPhysicalDevice(), m_gBuffers->getColorFormat(),
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

          // The graphic pipeline contains MSAA information
          vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
          vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);

          createMsaaImage();
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
        return ImGui::RadioButton("Default", (int*)&m_settings.lineRasterizationMode, VK_LINE_RASTERIZATION_MODE_DEFAULT_KHR);
      });
      PE::entry("", [&] {
        return ImGui::RadioButton("Rectangular", (int*)&m_settings.lineRasterizationMode, VK_LINE_RASTERIZATION_MODE_RECTANGULAR_KHR);
      });
      PE::entry("", [&] {
        return ImGui::RadioButton("Bresenham", (int*)&m_settings.lineRasterizationMode, VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR);
      });
      PE::entry("", [&] {
        return ImGui::RadioButton("Smooth", (int*)&m_settings.lineRasterizationMode, VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_KHR);
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

  void createMsaaImage()
  {
    VkExtent2D viewSize = m_app->getViewportSize();
    m_msaaGBuffers      = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get());
    m_msaaGBuffers->create(viewSize, {m_colorFormat}, m_depthFormat, m_settings.msaaSamples);
  }


  // Display the G-Buffer image in Viewport
  void displayGbuffer()
  {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    ImGui::Begin("Viewport");
    ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());
    ImGui::End();
    ImGui::PopStyleVar();
  }

  void destroyResources()
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
    m_alloc->destroy(m_vertexBuffer);
    m_gBuffers.reset();
    m_msaaGBuffers.reset();
  }

  //-------------------------------------------------------------------------------------------------
  nvvkhl::Application* m_app{nullptr};
  VkDevice             m_device = VK_NULL_HANDLE;  // Convenient

  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;
  std::unique_ptr<nvvkhl::GBuffer> m_msaaGBuffers;
  std::unique_ptr<nvvk::DebugUtil> m_dutil;
  std::shared_ptr<AllocDma>        m_alloc;

  VkFormat         m_colorFormat      = VK_FORMAT_B8G8R8A8_UNORM;  // Color format of the image
  VkPipelineLayout m_pipelineLayout   = VK_NULL_HANDLE;            // The description of the pipeline
  VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;            // The graphic pipeline to render

  VkFormat m_depthFormat = VK_FORMAT_D32_SFLOAT;  // Depth format of the depth buffer

  nvvk::Buffer m_vertexBuffer;  // Buffer of the vertices
  VkPhysicalDeviceLineRasterizationFeaturesKHR m_lineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_KHR};
  VkPhysicalDeviceExtendedDynamicState3FeaturesEXT m_dynamicFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
};

int main(int argc, char** argv)
{
  VkPhysicalDeviceLineRasterizationFeaturesKHR lineRasterizationFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_LINE_RASTERIZATION_FEATURES_KHR};
  VkPhysicalDeviceExtendedDynamicState3FeaturesEXT extendedDynamicStateFeature{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};


  // Vulkan creation context information
  VkContextSettings vkSetup;
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_EXT_LINE_RASTERIZATION_EXTENSION_NAME, &lineRasterizationFeature});  // To enable line rasterization mode
  vkSetup.deviceExtensions.push_back({VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, &extendedDynamicStateFeature, false});  // [optional] To enable dynamic line rasterization mode

  // Create Vulkan context
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  if(!vkContext->isValid())
    std::exit(0);

  // Check if line rasterization is supported
  if(lineRasterizationFeature.bresenhamLines == VK_FALSE && lineRasterizationFeature.stippledRectangularLines == VK_FALSE
     && lineRasterizationFeature.stippledSmoothLines == VK_FALSE)
  {
    LOGE("Line rasterization is not supported by the device");
    return 1;
  }

  // Loading the Vulkan extensions
  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);  // Loading the Vulkan extension pointers

  // Application setup
  nvvkhl::ApplicationCreateInfo appSetup;
  appSetup.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appSetup.vSync          = true;
  appSetup.width          = 800;
  appSetup.height         = 600;
  appSetup.instance       = vkContext->getInstance();
  appSetup.device         = vkContext->getDevice();
  appSetup.physicalDevice = vkContext->getPhysicalDevice();
  appSetup.queues         = vkContext->getQueueInfos();

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appSetup);

  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);
  app->addElement(test);
  app->addElement(std::make_shared<LineStippleElement>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>());

  app->run();
  app.reset();
  vkContext.reset();

  return test->errorCode();
}
