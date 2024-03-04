/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#include "nvh/primitives.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/specialization.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvk/shaders_vk.hpp"

namespace DH {
using namespace glm;
#include "shaders/device_host.h"
}  // namespace DH


#if USE_HLSL
#include "_autogen/raster_vertexMain.spirv.h"
#include "_autogen/raster_fragmentMain.spirv.h"
const auto& vert_shd = std::vector<uint8_t>{std::begin(raster_vertexMain), std::end(raster_vertexMain)};
const auto& frag_shd = std::vector<uint8_t>{std::begin(raster_fragmentMain), std::end(raster_fragmentMain)};
#elif USE_SLANG
#include "_autogen/raster_slang.h"
#else
#include "_autogen/raster.frag.h"
#include "_autogen/raster.vert.h"
const auto& vert_shd = std::vector<uint32_t>{std::begin(raster_vert), std::end(raster_vert)};
const auto& frag_shd = std::vector<uint32_t>{std::begin(raster_frag), std::end(raster_frag)};
#endif  // USE_HLSL

class AftermathSample : public nvvkhl::IAppElement
{
  enum TdrReason
  {
    eNone,
    eOutOfBoundsVertexBufferOffset,
  };


public:
  AftermathSample()           = default;
  ~AftermathSample() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app         = app;
    m_device      = m_app->getDevice();
    m_dutil       = std::make_unique<nvvk::DebugUtil>(m_device);                    // Debug utility
    m_alloc       = std::make_unique<nvvkhl::AllocVma>(m_app->getContext().get());  // Allocator
    m_depthFormat = nvvk::findDepthFormat(m_app->getPhysicalDevice());              // Not all depth are supported
    m_dset        = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    createPipeline();
    createVkResources();
    updateDescriptorSet();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(uint32_t width, uint32_t height) override { createGbuffers({width, height}); }

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
        m_alloc->destroy(m_vertices);
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

#if !defined(NVVK_SUPPORTS_AFTERMATH)
      aftermathPopup();
#endif

      if(m_gBuffers)
      {
        ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());
      }
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
    if(!m_gBuffers)
      return;

    const nvvk::DebugUtil::ScopedCmdLabel s_dbg = m_dutil->DBG_SCOPE(cmd);
    m_frameNumber++;

    nvvk::createRenderingInfo r_info({{0, 0}, m_viewSize}, {m_gBuffers->getColorImageView()}, m_gBuffers->getDepthImageView(),
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    const float      view_aspect_ratio = static_cast<float>(m_viewSize.width) / static_cast<float>(m_viewSize.height);
    const glm::vec2& clip              = CameraManip.getClipPlanes();
    const glm::mat4  matv              = CameraManip.getMatrix();
    glm::mat4 matp = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), view_aspect_ratio, clip.x, clip.y);
    matp[1][1] *= -1;

    DH::FrameInfo finfo{};
    finfo.time[0] = static_cast<float>(ImGui::GetTime());
    finfo.time[1] = 0;
    finfo.mpv     = matp * matv;

    finfo.resolution = glm::vec2(m_viewSize.width, m_viewSize.height);
    finfo.badOffset  = std::rand();  // 0xDEADBEEF;
    finfo.errorTest  = m_currentPipe;
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);

    vkCmdBeginRendering(cmd, &r_info);
    m_app->setViewport(cmd);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipe.layout, 0, 1, m_dset->getSets(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipe.plines[m_currentPipe]);
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
    m_dset->addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dset->addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dset->addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    m_dset->initLayout();
    m_dset->initPool(2);  // two frames - allow to change on the fly

    VkPipelineLayoutCreateInfo create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    create_info.setLayoutCount = 1;
    create_info.pSetLayouts    = &m_dset->getLayout();
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &create_info, nullptr, &m_pipe.layout));

    VkPipelineRenderingCreateInfo prend_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    prend_info.colorAttachmentCount    = 1;
    prend_info.pColorAttachmentFormats = &m_colorFormat;
    prend_info.depthAttachmentFormat   = m_depthFormat;

    nvvk::GraphicsPipelineState pstate;
    pstate.addBindingDescriptions({{0, sizeof(nvh::PrimitiveVertex)}});
    pstate.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, p))},  // Position
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, n))},  // Normal
        {2, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, t))},     // Normal

    });

    // Shader sources, pre-compiled to Spir-V (see Makefile)
    nvvk::GraphicsPipelineGenerator pgen(m_device, m_pipe.layout, prend_info, pstate);
#if USE_SLANG
    VkShaderModule shaderModule = nvvk::createShaderModule(m_device, &rasterSlang[0], sizeof(rasterSlang));
    for(int i = 0; i <= 10; i++)
    {
      nvvk::Specialization specialization;
      specialization.add(0, i);
      pgen.addShader(shaderModule, VK_SHADER_STAGE_VERTEX_BIT, "vertexMain").pSpecializationInfo =
          specialization.getSpecialization();
      pgen.addShader(shaderModule, VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain").pSpecializationInfo =
          specialization.getSpecialization();
      m_pipe.plines.push_back(pgen.createPipeline());
      m_dutil->DBG_NAME(m_pipe.plines[i]);
      pgen.clearShaders();
    }
    vkDestroyShaderModule(m_device, shaderModule, nullptr);
#else
    pgen.addShader(vert_shd, VK_SHADER_STAGE_VERTEX_BIT, USE_HLSL ? "vertexMain" : "main");
    pgen.addShader(frag_shd, VK_SHADER_STAGE_FRAGMENT_BIT, USE_HLSL ? "fragmentMain" : "main");
    m_pipe.plines.push_back(pgen.createPipeline());
    m_dutil->DBG_NAME(m_pipe.plines[0]);
    pgen.clearShaders();

    // Create many specializations (shader with constant values)
    // 1- Loop in vertex, 2- Loop in Fragment, 3- Over buffer
    for(int i = 1; i <= 10; i++)
    {
      nvvk::Specialization specialization;
      specialization.add(0, i);
      pgen.addShader(vert_shd, VK_SHADER_STAGE_VERTEX_BIT, USE_HLSL ? "vertexMain" : "main").pSpecializationInfo =
          specialization.getSpecialization();
      pgen.addShader(frag_shd, VK_SHADER_STAGE_FRAGMENT_BIT, USE_HLSL ? "fragmentMain" : "main").pSpecializationInfo =
          specialization.getSpecialization();
      m_pipe.plines.push_back(pgen.createPipeline());
      m_dutil->setObjectName(m_pipe.plines.back(), "Crash " + std::to_string(i));
      pgen.clearShaders();
    }
#endif
  }

  void updateDescriptorSet()
  {
    // Writing to descriptors
    const VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo dbi_val{m_bValues.buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_dset->makeWrite(0, 0, &dbi_unif));
    writes.emplace_back(m_dset->makeWrite(0, 1, &dbi_val));
    writes.emplace_back(m_dset->makeWrite(0, 2, &m_texture.descriptor));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void wrongDescriptorSet()
  {
    // Writing to descriptors
    VkDescriptorBufferInfo dbi_unif{nullptr, 0, VK_WHOLE_SIZE};
    dbi_unif.buffer = VkBuffer(0xDEADBEEFDEADBEEF);
    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_dset->makeWrite(0, 1, &dbi_unif));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void createGbuffers(VkExtent2D size)
  {
    m_viewSize = size;
    VkImageFormatProperties prop;
    NVVK_CHECK(vkGetPhysicalDeviceImageFormatProperties(
        m_app->getPhysicalDevice(), m_colorFormat, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, 0, &prop));

    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), m_viewSize, m_colorFormat, m_depthFormat);
  }

  void createVkResources()
  {
    m_meshes.emplace_back(nvh::createSphereUv(0.5F, 20, 20));

    {
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();

      // Create buffer of the mesh
      m_vertices = m_alloc->createBuffer(cmd, m_meshes[0].vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      m_indices  = m_alloc->createBuffer(cmd, m_meshes[0].triangles, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
      m_dutil->DBG_NAME(m_vertices.buffer);
      m_dutil->DBG_NAME(m_indices.buffer);

      // Frame information
      m_bFrameInfo = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      m_dutil->DBG_NAME(m_bFrameInfo.buffer);

      // Dummy buffer of values
      const std::vector<float> values = {0.5F};
      m_bValues = m_alloc->createBuffer(cmd, values, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      m_dutil->DBG_NAME(m_bValues.buffer);

      // Create dummy texture
      {
        const VkImageCreateInfo   create_info = nvvk::makeImage2DCreateInfo({1, 1}, VK_FORMAT_R8G8B8A8_UNORM);
        const VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        std::vector<uint8_t>      image_data = {255, 0, 255, 255};
        m_texture = m_alloc->createTexture(cmd, image_data.size() * sizeof(uint8_t), image_data.data(), create_info, sampler_info);
        m_dutil->DBG_NAME(m_texture.image);
        m_dutil->DBG_NAME(m_texture.descriptor.sampler);
      }

      m_app->submitAndWaitTempCmdBuffer(cmd);
    }

    // Camera position
    CameraManip.setLookat({0, 0, 1}, {0, 0, 0}, {0, 1, 0});
  }

  void destroyResources()
  {
    m_alloc->destroy(m_bFrameInfo);
    m_alloc->destroy(m_bValues);
    m_alloc->destroy(m_vertices);
    m_alloc->destroy(m_indices);
    m_alloc->destroy(m_texture);
    m_pipe.destroy(m_device);
    m_dset->deinit();
    m_vertices = {};
    m_indices  = {};
    m_gBuffers.reset();
  }

  nvvkhl::Application* m_app{nullptr};

  std::unique_ptr<nvvkhl::GBuffer>              m_gBuffers;
  std::unique_ptr<nvvk::DebugUtil>              m_dutil;
  std::unique_ptr<nvvkhl::AllocVma>             m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;  // Descriptor set
  nvvk::Buffer                                  m_bFrameInfo;
  nvvk::Buffer                                  m_bValues;

  TdrReason m_tdrReason{eNone};


  VkExtent2D                m_viewSize    = {0, 0};
  VkFormat                  m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;    // Color format of the image
  VkFormat                  m_depthFormat = VK_FORMAT_UNDEFINED;         // Depth format of the depth buffer
  nvvkhl::PipelineContainer m_pipe;                                      // Multiple pipelines
  nvvk::Buffer              m_vertices;                                  // Buffer of the vertices
  nvvk::Buffer              m_indices;                                   // Buffer of the indices
  VkClearColorValue         m_clearColor  = {{0.0F, 0.0F, 0.0F, 1.0F}};  // Clear color
  VkDevice                  m_device      = VK_NULL_HANDLE;              // Convenient
  int                       m_currentPipe = 0;
  int                       m_frameNumber = 0;

  std::vector<nvh::PrimitiveMesh> m_meshes;
  nvvk::Texture                   m_texture;
};

int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name  = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  spec.vSync = true;
  spec.vkSetup.setVersion(1, 3);
  //spec.vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  spec.vkSetup.enableAftermath = true;  // We want Aftermath

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app->addElement(std::make_shared<AftermathSample>());

  app->run();
  app.reset();

  return test->errorCode();
}
