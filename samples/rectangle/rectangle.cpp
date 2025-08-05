/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


/*
 This sample creates a 2D rectangle and display it in the middle of the viewport. 
 The rendering is done in gBuffers and the it is the image that is displayed.
 Clear color can be change.
*/


#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#include <array>
#include <vulkan/vulkan_core.h>
#include <imgui.h>


#include <GLFW/glfw3.h>
#undef APIENTRY

#define VMA_IMPLEMENTATION


#include "_autogen/rectangle.slang.h"
#include "_autogen/rectangle.frag.glsl.h"
#include "_autogen/rectangle.vert.glsl.h"


#include <nvapp/application.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>


class RectangleSample : public nvapp::IAppElement
{
public:
  RectangleSample()           = default;
  ~RectangleSample() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();
    m_alloc.init({.flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
                  .physicalDevice   = app->getPhysicalDevice(),
                  .device           = app->getDevice(),
                  .instance         = app->getInstance(),
                  .vulkanApiVersion = VK_API_VERSION_1_4});  // Allocator

    // Acquiring the sampler which will be used for displaying the GBuffer
    m_samplerPool.init(app->getDevice());
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // GBuffer
    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
    m_gBuffers.init({
        .allocator      = &m_alloc,
        .colorFormats   = {m_colorFormat},  // Only one GBuffer color attachment
        .depthFormat    = m_depthFormat,
        .imageSampler   = linearSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    });

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

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override { NVVK_CHECK(m_gBuffers.update(cmd, size)); }

  void onUIRender() override
  {

    {  // Setting panel
      ImGui::Begin("Settings");
      ImGui::ColorPicker4("Clear Color", &m_clearColor.float32[0], ImGuiColorEditFlags_Float | ImGuiColorEditFlags_PickerHueWheel);
      ImGui::End();
    }

    {  // Window Title
      static float dirty_timer = 0.0F;
      dirty_timer += ImGui::GetIO().DeltaTime;
      if(dirty_timer > 1.0F)  // Refresh every seconds
      {
        std::array<char, 256> buf{};

        const int ret = snprintf(buf.data(), buf.size(), "%s | %d FPS / %.3fms", TARGET_NAME,
                                 static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);
        assert(ret > 0);
        glfwSetWindowTitle(m_app->getWindowHandle(), buf.data());
        dirty_timer = 0;
      }
    }

    {  // Display the G-Buffer image
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");
      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());
      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);

    VkRenderingAttachmentInfoKHR colorAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
        .imageView   = m_gBuffers.getColorImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = {.color = m_clearColor},
    };

    VkRenderingAttachmentInfoKHR depthStencilAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO_KHR,
        .imageView   = m_gBuffers.getDepthImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_ATTACHMENT_OPTIMAL_KHR,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = {.depthStencil = {1.f, 0U}},
    };

    VkRenderingInfoKHR rInfo{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO_KHR,
        .renderArea           = {{0, 0}, m_gBuffers.getSize()},
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
        .pDepthAttachment     = &depthStencilAttachment,
    };


    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    vkCmdBeginRendering(cmd, &rInfo);
    {
      const VkDeviceSize offsets{0};
      nvvk::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, m_gBuffers.getSize());
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
      vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT16);
      vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
  }

private:
  struct Vertex
  {
    glm::vec2 pos;
    glm::vec3 color;
  };

  void createPipeline()
  {
    const VkPipelineLayoutCreateInfo create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    vkCreatePipelineLayout(m_device, &create_info, nullptr, &m_pipelineLayout);

    VkPipelineRenderingCreateInfo prend_info{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &m_colorFormat,
        .depthAttachmentFormat   = m_depthFormat,
    };

    // Creating the Pipeline
    nvvk::GraphicsPipelineState m_graphicState;  // State of the graphic pipeline
    m_graphicState.vertexBindings = {
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, .stride = sizeof(Vertex), .divisor = 1}};
    m_graphicState.vertexAttributes = {{.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                        .location = 0,
                                        .format   = VK_FORMAT_R32G32_SFLOAT,
                                        .offset   = offsetof(Vertex, pos)},
                                       {.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                        .location = 1,
                                        .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                        .offset   = offsetof(Vertex, color)}};


    // Shader sources, pre-compiled to Spir-V (see Makefile)
    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = m_pipelineLayout;
    creator.colorFormats                         = {m_colorFormat};
    creator.renderingState.depthAttachmentFormat = m_depthFormat;

#if USE_SLANG
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", rectangle_slang);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", rectangle_slang);
#else
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", rectangle_vert_glsl);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", rectangle_frag_glsl);
#endif

    NVVK_CHECK(creator.createGraphicsPipeline(m_device, nullptr, m_graphicState, &m_pipeline));
    NVVK_DBG_NAME(m_pipeline);
  }

  void createGeometryBuffers()
  {
    const std::vector<Vertex>   vertices = {{{-0.5F, -0.5F}, {1.0F, 1.0F, 0.0F}},
                                            {{0.5F, -0.5F}, {0.0F, 1.0F, 1.0F}},
                                            {{0.5F, 0.5F}, {1.0F, 0.0F, 1.0F}},
                                            {{-0.5F, 0.5F}, {1.0F, 1.0F, 1.0F}}};
    const std::vector<uint16_t> indices  = {0, 2, 1, 2, 0, 3};

    {
      VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
      nvvk::StagingUploader uploader;
      uploader.init(&m_alloc);

      NVVK_CHECK(m_alloc.createBuffer(m_vertices, std::span(vertices).size_bytes(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
      NVVK_CHECK(m_alloc.createBuffer(m_indices, std::span(indices).size_bytes(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT));
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
    vkDestroyPipeline(m_device, m_pipeline, nullptr);

    m_alloc.destroyBuffer(m_vertices);
    m_alloc.destroyBuffer(m_indices);
    m_vertices = {};
    m_indices  = {};
    m_gBuffers.deinit();
    m_samplerPool.deinit();
    m_alloc.deinit();
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers.getColorImage(), m_gBuffers.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string(), 95);
  }

  nvapp::Application*     m_app{};
  nvvk::GBuffer           m_gBuffers;
  nvvk::ResourceAllocator m_alloc;
  nvvk::SamplerPool       m_samplerPool{};  // The sampler pool, used to create a sampler for the texture


  VkFormat          m_colorFormat    = VK_FORMAT_B8G8R8A8_UNORM;       // Color format of the image
  VkFormat          m_depthFormat    = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkPipelineLayout  m_pipelineLayout = VK_NULL_HANDLE;                 // The description of the pipeline
  VkPipeline        m_pipeline       = VK_NULL_HANDLE;                 // The graphic pipeline to render
  nvvk::Buffer      m_vertices;                                        // Buffer of the vertices
  nvvk::Buffer      m_indices;                                         // Buffer of the indices
  VkClearColorValue m_clearColor{{0.1F, 0.4F, 0.1F, 1.0F}};            // Clear color
  VkDevice          m_device = VK_NULL_HANDLE;                         // Convenient
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

  nvvk::ContextInitInfo vkSetup = {
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
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


  appInfo.name           = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  app.addElement(std::make_shared<RectangleSample>());

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
