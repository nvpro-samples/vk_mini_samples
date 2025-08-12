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
 This sample shows how to integrate NSight Aftermath.

 Note: if NVVK_SUPPORTS_AFTERMATH is not defined, this means the path to NSight Aftermath wasn't set.
       - Download the NSight Aftermath SDK
       - Open `Ungrouped Entries` and set the NSIGHT_AFTERMATH_SDK path
       - Delete CMake cache and re-configure CMake
*/

#define VMA_IMPLEMENTATION


#include <fmt/format.h>
#include <glm/glm.hpp>

#include "shaders/shaderio.h"

#include "_autogen/crash_aftermath.frag.glsl.h"
#include "_autogen/crash_aftermath.slang.h"
#include "_autogen/crash_aftermath.vert.glsl.h"

// Enables the Nsight Aftermath code instrumentation for GPU crash dump creation.
#if defined(AFTERMATH_AVAILABLE)
#define USE_NSIGHT_AFTERMATH 1
#include <nvaftermath/aftermath.hpp>
#endif


#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/primitives.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/specialization.hpp>
#include <nvvk/staging.hpp>


#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

// The camera for the scene
std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};

class AftermathSample : public nvapp::IAppElement
{
  enum TdrReason
  {
    eNone,
    eOutOfBoundsVertexBufferOffset,
  };


public:
  AftermathSample()           = default;
  ~AftermathSample() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();
    m_alloc.init(VmaAllocatorCreateInfo{
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });

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
    createVkResources();
    updateDescriptorSet();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    m_gBuffers.update(cmd, size);
    //createGbuffers({width, height});
  }

  void onUIRender() override
  {
    {  // Setting panel
      ImGui::Begin("Settings");

#if !defined(NVVK_SUPPORTS_AFTERMATH)
      ImGui::TextColored(ImVec4(1, 0, 0, 1), "Aftermath not enabled");
#endif

      if(ImGui::Button("1. Crash"))
      {
        m_currentPipe = 1;
      }
      ImGui::SameLine();
      ImGui::Text("Infinite loop in vertex shader");

      if(ImGui::Button("2. Crash"))
      {
        m_currentPipe = 2;
      }
      ImGui::SameLine();
      ImGui::Text("Infinite loop in fragment shader");

      if(ImGui::Button("3. Bug"))
      {
        m_tdrReason = TdrReason::eOutOfBoundsVertexBufferOffset;
      }
      ImGui::SameLine();
      ImGui::Text("Out of bound vertex buffer");

      if(ImGui::Button("4. Bug"))
      {
        m_alloc.destroyBuffer(m_vertices);
      }
      ImGui::SameLine();
      ImGui::Text("Delete vertex buffer");

      if(ImGui::Button("5. Bug"))
      {
        wrongDescriptorSet();
      }
      ImGui::SameLine();
      ImGui::Text("Wrong buffer address");

      if(ImGui::Button("6. Crash"))
      {
        m_currentPipe = 3;
      }
      ImGui::SameLine();
      ImGui::Text("Out of bound buffer");

      if(ImGui::Button("7. Crash"))
      {
        m_currentPipe = 4;
      }
      ImGui::SameLine();
      ImGui::Text("Bad texture access");

      ImGui::End();
    }

    {  // Display the G-Buffer image
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

#if !defined(AFTERMATH_AVAILABLE)
      aftermathPopup();
#endif

      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());
      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void aftermathPopup()
  {
    static bool onlyOnce = true;
    if(onlyOnce)
      ImGui::OpenPopup("NSight Aftermath");
    // Always center this window when appearing
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0F, 20.0F));
    if(ImGui::BeginPopupModal("NSight Aftermath", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      onlyOnce = false;
      ImGui::Text("NSight Aftermath is not installed.\n");
      ImGui::Text("This sample will work but will not dump debugging crashes.\n");
      ImGui::Separator();
      if(ImGui::Button("OK", ImVec2(120, 0)))
      {
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }
    ImGui::PopStyleVar();
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);
    m_frameNumber++;

    // Drawing the primitives in a G-Buffer
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers.getColorImageView();
    colorAttachment.clearValue                = {m_clearColor};
    VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
    depthAttachment.imageView                 = m_gBuffers.getDepthImageView();
    depthAttachment.clearValue                = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

    // Create the rendering info
    VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea           = DEFAULT_VkRect2D(m_gBuffers.getSize());
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;
    renderingInfo.pDepthAttachment     = &depthAttachment;

    const glm::mat4 matv = g_cameraManip->getViewMatrix();
    glm::mat4       matp = g_cameraManip->getPerspectiveMatrix();

    shaderio::FrameInfo finfo{};
    finfo.time[0] = static_cast<float>(ImGui::GetTime());
    finfo.time[1] = 0;
    finfo.mpv     = matp * matv;

    finfo.resolution = glm::vec2(m_gBuffers.getSize().width, m_gBuffers.getSize().height);
    finfo.badOffset  = std::rand();  // 0xDEADBEEF;
    finfo.errorTest  = m_currentPipe;
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &finfo);

    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

    vkCmdBeginRendering(cmd, &renderingInfo);
    nvvk::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, m_gBuffers.getSize());

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descriptorPack.sets[0], 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline[m_currentPipe]);
    VkDeviceSize offsets{0};
    if(m_tdrReason == TdrReason::eOutOfBoundsVertexBufferOffset)
    {
      offsets = std::rand();
    }
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
    vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT32);
    auto index_count = static_cast<uint32_t>(m_meshes[0].triangles.size() * 3);
    vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

    vkCmdEndRendering(cmd);
  }

private:
  void createPipeline()
  {
    m_descriptorPack.bindings.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_descriptorPack.bindings.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_descriptorPack.bindings.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);

    NVVK_CHECK(m_descriptorPack.initFromBindings(m_device, 1));
    NVVK_DBG_NAME(m_descriptorPack.layout);
    NVVK_DBG_NAME(m_descriptorPack.pool);
    NVVK_DBG_NAME(m_descriptorPack.sets[0]);

    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorPack.layout}));
    NVVK_DBG_NAME(m_pipelineLayout);

    m_graphicState.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    m_graphicState.vertexBindings              = {{.sType   = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
                                                   .stride  = sizeof(nvutils::PrimitiveVertex),
                                                   .divisor = 1}};
    m_graphicState.vertexAttributes            = {{.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                                   .location = 0,
                                                   .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                                   .offset   = offsetof(nvutils::PrimitiveVertex, pos)},
                                                  {.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                                   .location = 1,
                                                   .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                                   .offset   = offsetof(nvutils::PrimitiveVertex, nrm)},
                                                  {.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                                   .location = 2,
                                                   .format   = VK_FORMAT_R32G32_SFLOAT,
                                                   .offset   = offsetof(nvutils::PrimitiveVertex, tex)}};

    // Helper to create the graphic pipeline
    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = m_pipelineLayout;
    creator.colorFormats                         = {m_colorFormat};
    creator.renderingState.depthAttachmentFormat = m_depthFormat;


#if defined(AFTERMATH_AVAILABLE)
    {
      auto&     aftermath = AftermathCrashTracker::getInstance();
      std::span data(crash_aftermath_slang);
      aftermath.addShaderBinary(data);
    }
#endif

    // Create all pipelines
    m_pipeline.resize(10);
    for(int i = 0; i < 10; i++)
    {
      creator.clearShaders();
      nvvk::Specialization specialization;
      specialization.add(0, i);
      const VkSpecializationInfo* pSpecializationInfo = specialization.getSpecializationInfo();

#if USE_SLANG
      creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", std::span(crash_aftermath_slang), pSpecializationInfo);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", std::span(crash_aftermath_slang), pSpecializationInfo);
#else
      creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", std::span(bary_vert_glsl), pSpecializationInfo);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", std::span(bary_frag_glsl), pSpecializationInfo);
#endif

      NVVK_CHECK(creator.createGraphicsPipeline(m_device, nullptr, m_graphicState, &m_pipeline[i]));
      NVVK_DBG_NAME(m_pipeline[i]);
    }
  }

  void updateDescriptorSet()
  {
    // Writing to descriptors
    const VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo dbi_val{m_bValues.buffer, 0, VK_WHOLE_SIZE};

    nvvk::WriteSetContainer         writeContainer;
    const nvvk::DescriptorBindings& bindings = m_descriptorPack.bindings;
    const VkDescriptorSet           set      = m_descriptorPack.sets[0];
    writeContainer.append(bindings.getWriteSet(0, set), m_bFrameInfo);
    writeContainer.append(bindings.getWriteSet(1, set), m_bValues);
    writeContainer.append(bindings.getWriteSet(2, set), m_image);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }

  void wrongDescriptorSet()
  {
    // Writing to descriptors
    VkDescriptorBufferInfo dbi_unif{nullptr, 0, VK_WHOLE_SIZE};
    dbi_unif.buffer = VkBuffer(0xDEADBEEFDEADBEEF);
    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(m_descriptorPack.bindings.getWriteSet(1, m_descriptorPack.sets[0]), dbi_unif);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }

  void createVkResources()
  {
    m_meshes.emplace_back(nvutils::createSphereUv(0.5F, 20, 20));

    {
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();

      nvvk::StagingUploader uploader;
      uploader.init(&m_alloc, true);

      // Create buffer of the mesh
      NVVK_CHECK(m_alloc.createBuffer(m_vertices, std::span(m_meshes[0].vertices).size_bytes(), VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT));
      NVVK_CHECK(uploader.appendBuffer(m_vertices, 0, std::span(m_meshes[0].vertices)));
      NVVK_DBG_NAME(m_vertices.buffer);
      NVVK_CHECK(m_alloc.createBuffer(m_indices, std::span(m_meshes[0].triangles).size_bytes(), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT));
      NVVK_CHECK(uploader.appendBuffer(m_indices, 0, std::span(m_meshes[0].triangles)));
      NVVK_DBG_NAME(m_indices.buffer);

      // Frame information
      NVVK_CHECK(m_alloc.createBuffer(m_bFrameInfo, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_HOST));
      NVVK_DBG_NAME(m_bFrameInfo.buffer);
      // Dummy buffer of values
      const std::vector<float> values = {0.5F};
      NVVK_CHECK(m_alloc.createBuffer(m_bValues, std::span(values).size_bytes(), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_HOST));
      NVVK_CHECK(uploader.appendBuffer(m_bValues, 0, std::span(values)));
      NVVK_DBG_NAME(m_bValues.buffer);

      // Create dummy texture
      {
        VkImageCreateInfo create_info = DEFAULT_VkImageCreateInfo;  // nvvk::makeImage2DCreateInfo({1, 1}, VK_FORMAT_R8G8B8A8_UNORM);
        create_info.extent = {1, 1, 1};
        create_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        create_info.usage  = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;

        const VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        std::vector<uint8_t>      image_data = {255, 0, 255, 255};

        NVVK_CHECK(m_alloc.createImage(m_image, create_info, DEFAULT_VkImageViewCreateInfo));
        NVVK_CHECK(uploader.appendImage(m_image, std::span(image_data), VK_IMAGE_LAYOUT_GENERAL));
        NVVK_DBG_NAME(m_image.image);
        NVVK_DBG_NAME(m_image.descriptor.imageView);
        m_samplerPool.acquireSampler(m_image.descriptor.sampler);
        NVVK_DBG_NAME(m_image.descriptor.sampler);
      }
      uploader.cmdUploadAppended(cmd);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      uploader.deinit();
    }

    // Camera position
    g_cameraManip->setLookat({0, 0, 1}, {0, 0, 0}, {0, 1, 0});
  }

  void destroyResources()
  {
    m_alloc.destroyBuffer(m_bFrameInfo);
    m_alloc.destroyBuffer(m_bValues);
    m_alloc.destroyBuffer(m_vertices);
    m_alloc.destroyBuffer(m_indices);
    m_alloc.destroyImage(m_image);

    m_vertices = {};
    m_indices  = {};

    for(auto& pipe : m_pipeline)
    {
      vkDestroyPipeline(m_device, pipe, nullptr);
    }
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    m_descriptorPack.deinit();

    m_gBuffers.deinit();
    m_samplerPool.deinit();
    m_alloc.deinit();
  }

  //--------------------------------------------------------------------------------------------------
  nvapp::Application*     m_app{};       // Application
  nvvk::GBuffer           m_gBuffers;    // G-Buffers: color + depth
  nvvk::ResourceAllocator m_alloc;       // Allocator
  nvvk::Buffer            m_bFrameInfo;  // The buffer used with frame data
  nvvk::Buffer            m_bValues;     // The buffer used to pass bad data

  TdrReason m_tdrReason{eNone};

  VkFormat m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;  // Color format of the image
  VkFormat m_depthFormat = VK_FORMAT_UNDEFINED;       // Depth format of the depth buffer

  nvvk::Buffer      m_vertices;                                  // Buffer of the vertices
  nvvk::Buffer      m_indices;                                   // Buffer of the indices
  VkClearColorValue m_clearColor  = {{0.0F, 0.0F, 0.0F, 1.0F}};  // Clear color
  VkDevice          m_device      = VK_NULL_HANDLE;              // Convenient
  int               m_currentPipe = 0;
  int               m_frameNumber = 0;

  std::vector<nvutils::PrimitiveMesh> m_meshes;
  nvvk::Image                         m_image;

  std::vector<VkPipeline> m_pipeline{};        // Graphic pipeline to render
  VkPipelineLayout        m_pipelineLayout{};  // Pipeline layout
  nvvk::DescriptorPack    m_descriptorPack{};  // Descriptor bindings, layout, pool, and set

  nvvk::GraphicsPipelineState m_graphicState;
  nvvk::SamplerPool           m_samplerPool{};  // The sampler pool, used to create a sampler for the texture
};

int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME}},
  };
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

#if defined(USE_NSIGHT_AFTERMATH)
  auto& aftermath = AftermathCrashTracker::getInstance();
  aftermath.initialize();
  aftermath.addExtensions(vkSetup.deviceExtensions);
  nvvk::CheckError::getInstance().setCallbackFunction([&](VkResult result) { aftermath.errorCallback(result); });
#endif


  // Create the Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }


  appInfo.name           = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();


  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // Camera manipulator (global)
  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  elemCamera->setCameraManipulator(g_cameraManip);

  app.addElement(elemCamera);
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app.addElement(std::make_shared<AftermathSample>());

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
