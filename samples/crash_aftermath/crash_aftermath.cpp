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


/*

This sample shows how to integrate NSight Aftermath and demonstrates various
ways to trigger VK_ERROR_DEVICE_LOST on the GPU.

Crash mechanisms used:
  - TDR (Timeout Detection and Recovery): infinite loops that stall the GPU
    beyond the Windows ~2 second timeout.
  - GPU page faults via BDA (Buffer Device Address): writing to unmapped GPU
    virtual memory through buffer_reference pointers.

Note: if AFTERMATH_AVAILABLE is not defined, this means the path to NSight Aftermath wasn't set.
      - Download the NSight Aftermath SDK
      - In CMakeGUI; open `Ungrouped Entries` and set the NSIGHT_AFTERMATH_SDK path
        (i.e C:\Program Files\NVIDIA Corporation\Nsight Graphics 2026.x.x\SDKs\NsightAftermathSDK\202x.x.x.xxxxxx)
      - Delete CMake cache and re-configure CMake
*/

#define VMA_IMPLEMENTATION


#include <fmt/format.h>
#include <glm/glm.hpp>

#include "shaders/shaderio.h"

#include "_autogen/crash_aftermath.comp.glsl.h"
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

std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};

class AftermathSample : public nvapp::IAppElement
{
  // Each crash test maps to a shader CRASH_TEST specialization constant value
  // and/or a specific CPU-side setup (buffer address, compute dispatch, etc.).
  enum CrashTest
  {
    eNone = 0,         // Normal rendering, no crash
    eFragLoopSin,      // TDR: fragment infinite loop (sin)         -> CRASH_TEST=1
    eFragLoopSSBO,     // TDR: fragment infinite loop (SSBO write)  -> CRASH_TEST=2
    eBdaOverrun,       // Page fault: BDA write past buffer end     -> CRASH_TEST=3
    eBdaWildSpray,     // Page fault: BDA writes to random addrs    -> CRASH_TEST=4
    eBdaUseAfterFree,  // Page fault: BDA write to freed buffer     -> CRASH_TEST=3
    eBdaIndirect,      // Page fault: BDA pointer chain, inner freed-> CRASH_TEST=5
    eComputeLoop,      // TDR: compute shader infinite loop         -> compute pipeline
  };

  // Map CrashTest enum to the graphics pipeline index (CRASH_TEST specialization constant).
  static int getPipelineIndex(CrashTest test)
  {
    switch(test)
    {
      case eFragLoopSin:
        return 1;
      case eFragLoopSSBO:
        return 2;
      case eBdaOverrun:
      case eBdaUseAfterFree:
        return 3;
      case eBdaWildSpray:
        return 4;
      case eBdaIndirect:
        return 5;
      default:
        return 0;
    }
  }

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

    m_samplerPool.init(app->getDevice());
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
    m_gBuffers.init({
        .allocator      = &m_alloc,
        .colorFormats   = {m_colorFormat},
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

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override { m_gBuffers.update(cmd, size); }

  void onUIRender() override
  {
    {
      ImGui::Begin("Settings");

#if !defined(USE_NSIGHT_AFTERMATH)
      ImGui::TextColored(ImVec4(1, 0, 0, 1), "Aftermath SDK not integrated");
#endif

      // ---- TDR crashes: infinite loops that exceed the Windows GPU timeout ----
      ImGui::SeparatorText("TDR Crashes (infinite loops)");

      if(ImGui::Button("1. Crash"))
        m_crashTest = eFragLoopSin;
      ImGui::SameLine();
      ImGui::Text("Fragment infinite loop (sin)");

      if(ImGui::Button("2. Crash"))
        m_crashTest = eFragLoopSSBO;
      ImGui::SameLine();
      ImGui::Text("Fragment infinite loop (SSBO writes)");

      if(ImGui::Button("3. Crash"))
        m_crashTest = eComputeLoop;
      ImGui::SameLine();
      ImGui::Text("Compute shader infinite loop");

      // ---- Page-fault crashes: BDA writes to unmapped GPU virtual memory ----
      ImGui::SeparatorText("Page Fault Crashes (BDA writes)");

      if(ImGui::Button("4. Crash"))
        m_crashTest = eBdaOverrun;
      ImGui::SameLine();
      ImGui::Text("BDA buffer overrun (+1 GB past end)");

      if(ImGui::Button("5. Crash"))
        m_crashTest = eBdaWildSpray;
      ImGui::SameLine();
      ImGui::Text("BDA wild pointer spray");

      if(ImGui::Button("6. Crash"))
      {
        // Wait for the GPU to finish all prior work, then destroy the victim
        // buffer. This ensures the dedicated VkDeviceMemory is freed and the
        // GPU virtual address range is unmapped before the next frame's shader
        // writes through the stale address -> page fault.
        if(m_victimBuffer.buffer != VK_NULL_HANDLE)
        {
          vkDeviceWaitIdle(m_device);
          m_alloc.destroyBuffer(m_victimBuffer);
        }
        m_crashTest = eBdaUseAfterFree;
      }
      ImGui::SameLine();
      ImGui::Text("BDA use-after-free");

      if(ImGui::Button("7. Crash"))
      {
        // Destroy the inner target buffer. The indirect buffer still holds its
        // stale address. The shader follows the pointer chain and writes through
        // the dangling inner pointer -> page fault.
        if(m_indirectTarget.buffer != VK_NULL_HANDLE)
        {
          vkDeviceWaitIdle(m_device);
          m_alloc.destroyBuffer(m_indirectTarget);
        }
        m_crashTest = eBdaIndirect;
      }
      ImGui::SameLine();
      ImGui::Text("BDA indirect use-after-free (pointer chain)");

      ImGui::End();
    }

    {
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
    ImVec2 center = ImGui::GetMainViewport()->GetCenter();
    ImGui::SetNextWindowPos(center, ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20.0F, 20.0F));
    if(ImGui::BeginPopupModal("NSight Aftermath", NULL, ImGuiWindowFlags_AlwaysAutoResize))
    {
      onlyOnce = false;
      ImGui::Text("The Nsight Aftermath SDK is not integrated. \nYou can still test crash dumps by running the Aftermath Monitor;\npoint the Crash Dump Inspector at this executable's directory to load the .spv shader files (copied here at build time).\n");
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

    // Prepare frame data
    const glm::mat4 matv = g_cameraManip->getViewMatrix();
    glm::mat4       matp = g_cameraManip->getPerspectiveMatrix();

    shaderio::FrameInfo finfo{};
    finfo.time.x     = static_cast<float>(ImGui::GetTime());
    finfo.mpv        = matp * matv;
    finfo.resolution = glm::vec2(m_gBuffers.getSize().width, m_gBuffers.getSize().height);

    // Set the BDA for page-fault crash tests
    switch(m_crashTest)
    {
      case eBdaOverrun:
        // Valid buffer base + 1 GB offset: lands in unmapped GPU virtual memory
        finfo.bufferAddr = m_bValues.address + (1024ULL * 1024ULL * 1024ULL);
        break;
      case eBdaUseAfterFree:
        finfo.bufferAddr = m_victimAddress;
        break;
      case eBdaIndirect:
        // Points to the indirect buffer, which contains the (now stale) target address
        finfo.bufferAddr = m_indirectBuffer.address;
        break;
      default:
        finfo.bufferAddr = 0;
        break;
    }

    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &finfo);

    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                               | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Compute-shader crash test: dispatch and return (no rasterization needed)
    if(m_crashTest == eComputeLoop)
    {
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, m_descriptorPack.getSetPtr(), 0, nullptr);
      vkCmdDispatch(cmd, 1, 1, 1);
    }

    // Transition GBuffer images to be used as attachments
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getDepthImage(),
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                      {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});

    // Rasterize the sphere
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers.getColorImageView();
    colorAttachment.clearValue                = {m_clearColor};
    VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
    depthAttachment.imageView                 = m_gBuffers.getDepthImageView();
    depthAttachment.clearValue                = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

    VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea           = DEFAULT_VkRect2D(m_gBuffers.getSize());
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;
    renderingInfo.pDepthAttachment     = &depthAttachment;

    vkCmdBeginRendering(cmd, &renderingInfo);
    nvvk::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, m_gBuffers.getSize());

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, m_descriptorPack.getSetPtr(), 0, nullptr);

    int pipeIdx = getPipelineIndex(m_crashTest);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipelines[pipeIdx]);
    VkDeviceSize offsets{0};
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
    vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT32);
    auto indexCount = static_cast<uint32_t>(m_meshes[0].triangles.size() * 3);
    vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);

    vkCmdEndRendering(cmd);

    // Transition GBuffer images back for display
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getDepthImage(),
                                      VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}});
  }

private:
  void createPipeline()
  {
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

    NVVK_CHECK(m_descriptorPack.init(bindings, m_device, 1));
    NVVK_DBG_NAME(m_descriptorPack.getLayout());
    NVVK_DBG_NAME(m_descriptorPack.getPool());
    NVVK_DBG_NAME(m_descriptorPack.getSet(0));

    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorPack.getLayout()}));
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

    // Create graphics pipelines: one per CRASH_TEST specialization constant value (0..5)
    static constexpr int kNumGraphicsPipelines = 6;
    m_graphicsPipelines.resize(kNumGraphicsPipelines);
    for(int i = 0; i < kNumGraphicsPipelines; i++)
    {
      creator.clearShaders();
      nvvk::Specialization specialization;
      specialization.add(0, i);
      const VkSpecializationInfo* pSpecializationInfo = specialization.getSpecializationInfo();

#if USE_SLANG
      creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", std::span(crash_aftermath_slang), pSpecializationInfo);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", std::span(crash_aftermath_slang), pSpecializationInfo);
#else
      creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", std::span(crash_aftermath_vert_glsl), pSpecializationInfo);
      creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", std::span(crash_aftermath_frag_glsl), pSpecializationInfo);
#endif

      NVVK_CHECK(creator.createGraphicsPipeline(m_device, nullptr, m_graphicState, &m_graphicsPipelines[i]));
      NVVK_DBG_NAME(m_graphicsPipelines[i]);
    }

    // Create compute pipeline for the compute-shader infinite-loop crash test
    {
#if USE_SLANG
      const VkShaderModuleCreateInfo moduleInfo{
          .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
          .codeSize = std::span(crash_aftermath_slang).size_bytes(),
          .pCode    = crash_aftermath_slang,
      };
      const char* entryPoint = "computeMain";
#else
      const VkShaderModuleCreateInfo moduleInfo{
          .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
          .codeSize = std::span(crash_aftermath_comp_glsl).size_bytes(),
          .pCode    = crash_aftermath_comp_glsl,
      };
      const char* entryPoint = "main";
#endif

      VkPipelineShaderStageCreateInfo stageInfo{
          .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .pNext = &moduleInfo,
          .stage = VK_SHADER_STAGE_COMPUTE_BIT,
          .pName = entryPoint,
      };

      VkComputePipelineCreateInfo compInfo{
          .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
          .stage  = stageInfo,
          .layout = m_pipelineLayout,
      };
      NVVK_CHECK(vkCreateComputePipelines(m_device, {}, 1, &compInfo, nullptr, &m_computePipeline));
      NVVK_DBG_NAME(m_computePipeline);
    }
  }

  void updateDescriptorSet()
  {
    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(m_descriptorPack.makeWrite(0), m_bFrameInfo);
    writeContainer.append(m_descriptorPack.makeWrite(1), m_bValues);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }

  void createVkResources()
  {
    m_meshes.emplace_back(nvutils::createSphereUv(0.5F, 20, 20));

    {
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();

      nvvk::StagingUploader uploader;
      uploader.init(&m_alloc, true);

      // Mesh vertex and index buffers
      NVVK_CHECK(m_alloc.createBuffer(m_vertices, std::span(m_meshes[0].vertices).size_bytes(), VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT));
      NVVK_CHECK(uploader.appendBuffer(m_vertices, 0, std::span(m_meshes[0].vertices)));
      NVVK_DBG_NAME(m_vertices.buffer);
      NVVK_CHECK(m_alloc.createBuffer(m_indices, std::span(m_meshes[0].triangles).size_bytes(), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT));
      NVVK_CHECK(uploader.appendBuffer(m_indices, 0, std::span(m_meshes[0].triangles)));
      NVVK_DBG_NAME(m_indices.buffer);

      // Frame information UBO
      NVVK_CHECK(m_alloc.createBuffer(m_bFrameInfo, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_HOST));
      NVVK_DBG_NAME(m_bFrameInfo.buffer);

      // SSBO used by infinite-loop crash tests (writes prevent dead code elimination)
      const std::vector<float> values(64, 0.5F);
      NVVK_CHECK(m_alloc.createBuffer(m_bValues, std::span(values).size_bytes(), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_HOST));
      NVVK_CHECK(uploader.appendBuffer(m_bValues, 0, std::span(values)));
      NVVK_DBG_NAME(m_bValues.buffer);

      // Victim buffer for the BDA use-after-free crash test (test 6).
      // VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT forces a separate VkDeviceMemory
      // so that destroying the buffer actually unmaps the GPU virtual address range
      // (without it, VMA suballocates from a shared block that stays mapped).
      NVVK_CHECK(m_alloc.createBuffer(m_victimBuffer, 256, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                      VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT));
      m_victimAddress = m_victimBuffer.address;
      NVVK_DBG_NAME(m_victimBuffer.buffer);

      // Indirect use-after-free (test 7): a pointer chain where the inner
      // target is destroyed. The shader reads the target address from the
      // indirect buffer, then writes through the (now stale) inner pointer.
      NVVK_CHECK(m_alloc.createBuffer(m_indirectTarget, 256, VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                      VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT));
      NVVK_DBG_NAME(m_indirectTarget.buffer);
      NVVK_CHECK(m_alloc.createBuffer(m_indirectBuffer, sizeof(uint64_t), VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT));
      uint64_t indirectTargetAddr = m_indirectTarget.address;
      NVVK_CHECK(uploader.appendBuffer(m_indirectBuffer, 0, std::span(&indirectTargetAddr, 1)));
      NVVK_DBG_NAME(m_indirectBuffer.buffer);

      uploader.cmdUploadAppended(cmd);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      uploader.deinit();
    }

    g_cameraManip->setLookat({0, 0, 1}, {0, 0, 0}, {0, 1, 0});
  }

  void destroyResources()
  {
    m_alloc.destroyBuffer(m_bFrameInfo);
    m_alloc.destroyBuffer(m_bValues);
    m_alloc.destroyBuffer(m_vertices);
    m_alloc.destroyBuffer(m_indices);
    if(m_victimBuffer.buffer != VK_NULL_HANDLE)
      m_alloc.destroyBuffer(m_victimBuffer);
    if(m_indirectTarget.buffer != VK_NULL_HANDLE)
      m_alloc.destroyBuffer(m_indirectTarget);
    m_alloc.destroyBuffer(m_indirectBuffer);

    for(auto& pipe : m_graphicsPipelines)
      vkDestroyPipeline(m_device, pipe, nullptr);
    vkDestroyPipeline(m_device, m_computePipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    m_descriptorPack.deinit();

    m_gBuffers.deinit();
    m_samplerPool.deinit();
    m_alloc.deinit();
  }

  //--------------------------------------------------------------------------------------------------
  nvapp::Application*     m_app{};
  nvvk::GBuffer           m_gBuffers;
  nvvk::ResourceAllocator m_alloc;
  nvvk::Buffer            m_bFrameInfo;
  nvvk::Buffer            m_bValues;

  VkFormat m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
  VkFormat m_depthFormat = VK_FORMAT_UNDEFINED;

  nvvk::Buffer      m_vertices;
  nvvk::Buffer      m_indices;
  VkClearColorValue m_clearColor = {{0.0F, 0.0F, 0.0F, 1.0F}};
  VkDevice          m_device     = VK_NULL_HANDLE;

  CrashTest m_crashTest = eNone;

  // BDA use-after-free (test 6): address saved before the buffer is destroyed
  nvvk::Buffer    m_victimBuffer{};
  VkDeviceAddress m_victimAddress{};

  // BDA indirect use-after-free (test 7): pointer chain with stale inner pointer
  nvvk::Buffer m_indirectBuffer{};  // holds uint64_t pointing to m_indirectTarget
  nvvk::Buffer m_indirectTarget{};  // destroyed when test is triggered

  std::vector<nvutils::PrimitiveMesh> m_meshes;

  std::vector<VkPipeline>     m_graphicsPipelines{};
  VkPipeline                  m_computePipeline{};
  VkPipelineLayout            m_pipelineLayout{};
  nvvk::DescriptorPack        m_descriptorPack{};
  nvvk::GraphicsPipelineState m_graphicState;
  nvvk::SamplerPool           m_samplerPool{};
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

  // Disable validation layers to avoid interference with the crashes we want to track with Aftermath.
  // Validation layers can cause additional GPU work and synchronization that may prevent the crash
  // from happening or alter its behavior, making it harder to analyze with Aftermath.
  vkSetup.enableValidationLayers = false;

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

  nvapp::Application app;
  app.init(appInfo);

  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  elemCamera->setCameraManipulator(g_cameraManip);

  app.addElement(elemCamera);
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));
  app.addElement(std::make_shared<AftermathSample>());

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
