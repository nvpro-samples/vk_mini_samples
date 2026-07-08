/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//////////////////////////////////////////////////////////////////////////
/*
  Descriptor Heap Sample

  Demonstrates VK_EXT_descriptor_heap with three heap-index modes:
  - PushIndex: set/binding declarations mapped with
    HEAP_WITH_PUSH_INDEX_EXT (naturally per-cube draws)
  - ConstantOffset: set/binding unsized arrays mapped with
    HEAP_WITH_CONSTANT_OFFSET_EXT (naturally a single instanced draw)
  - DirectAccess: layout(descriptor_heap) syntax (same rendering pattern
    as ConstantOffset, but no mapping info)

  The scene is an NxNxN RGB color cube. Each small cube's position maps to
  an RGB color. Each face has its own texture showing the hex color code
  and face name, rendered with an embedded bitmap font.

  Cubes animate falling into place via a shader-driven drop effect.

  Architecture (search for #DESC_HEAP to find all descriptor-heap code):
  - initHeaps(): Query heap properties, allocate sampler heap buffer.
  - resizeResourceHeap(): Allocate resource heap buffer for image descriptors.
  - writeImageDescriptor(): Write descriptors to host staging memory.
  - rebuildScene(): Upload staging to device-local heap via StagingUploader.
  - createShaders(): Build all three mode shader pairs. PushIndex and
    ConstantOffset use VkShaderDescriptorSetAndBindingMappingInfoEXT; DirectAccess
    uses layout(descriptor_heap).
  - cmdBindHeaps(): Bind sampler and resource heaps once per frame.
  - cmdPushData(): Push frame and per-draw data via vkCmdPushDataEXT.
  - onRender(): PushIndex pushes DrawData per cube; ConstantOffset/DirectAccess
    push once and use instanced drawing (position from gl_InstanceIndex).
*/
//////////////////////////////////////////////////////////////////////////

#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#include <array>
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <vulkan/vulkan_core.h>

#include "font_texture.h"
#include "shaders/shaderio.h"

#include <fmt/format.h>

#define VMA_IMPLEMENTATION

#if USE_SLANG
#include "_autogen/constant_offset.slang.h"
#include "_autogen/direct_access.slang.h"
#include "_autogen/push_index.slang.h"
#else
#include "_autogen/constant_offset.frag.glsl.h"
#include "_autogen/direct_access.frag.glsl.h"
#include "_autogen/instanced.vert.glsl.h"
#include "_autogen/push_index.frag.glsl.h"
#include "_autogen/push_index.vert.glsl.h"
#endif

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvgui/camera.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/hash_operations.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/primitives.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvapp/imgui_texture.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/validation_settings.hpp>

// The camera for the scene
std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};

//////////////////////////////////////////////////////////////////////////
/// Rendering mode
enum class RenderingMode
{
  PushIndex      = 0,
  ConstantOffset = 1,
  DirectAccess   = 2,
};

//////////////////////////////////////////////////////////////////////////
/// Shader indices
enum class ShaderIndex
{
  eVertex   = 0,
  eFragment = 1,
};

//////////////////////////////////////////////////////////////////////////
/// Descriptor Heap Sample
class DescriptorHeapSample : public nvapp::IAppElement
{
public:
  DescriptorHeapSample()           = default;
  ~DescriptorHeapSample() override = default;

  // Descriptor heap setup runs here: initHeaps → rebuildScene → createShaders
  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = app->getDevice();

    m_allocator = std::make_unique<nvvk::ResourceAllocator>();
    NVVK_CHECK(m_allocator->init(VmaAllocatorCreateInfo{
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    }));

    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());

    // Offscreen render target: one color attachment + depth
    NVVK_CHECK(m_renderTarget.init({
        .alloc        = m_allocator.get(),
        .colorFormats = {m_colorFormat},
        .depthFormat  = m_depthFormat,
        .debugName    = "DescriptorHeap",
    }));

    // Initialize descriptor heaps
    if(!initHeaps(app->getPhysicalDevice()))
    {
      LOGW(
          "Failed to initialize descriptor heap. VK_EXT_descriptor_heap may "
          "not be supported.\n");
      return;
    }

    // Write a nearest sampler to host staging, then upload to device-local heap
    VkSamplerCreateInfo samplerCI{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerCI.magFilter    = VK_FILTER_NEAREST;
    samplerCI.minFilter    = VK_FILTER_NEAREST;
    samplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerCI.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

    std::vector<uint8_t>  samplerStaging(m_samplerDescSize);
    VkHostAddressRangeEXT dst{};
    dst.address = samplerStaging.data();
    dst.size    = m_samplerDescSize;
    vkWriteSamplerDescriptorsEXT(m_device, 1, &samplerCI, &dst);

    // Create cube mesh and upload mesh + sampler heap together
    m_cubeMesh = nvutils::createCube(1.0f, 1.0f, 1.0f);
    {
      nvvk::StagingUploader uploader;
      uploader.init(m_allocator.get());
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();

      NVVK_CHECK(m_allocator->createBuffer(m_vertexBuffer, std::span(m_cubeMesh.vertices).size_bytes(),
                                           VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT));
      NVVK_CHECK(m_allocator->createBuffer(m_indexBuffer, std::span(m_cubeMesh.triangles).size_bytes(),
                                           VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT));
      NVVK_CHECK(uploader.appendBuffer(m_vertexBuffer, 0, std::span(m_cubeMesh.vertices)));
      NVVK_CHECK(uploader.appendBuffer(m_indexBuffer, 0, std::span(m_cubeMesh.triangles)));
      NVVK_CHECK(uploader.appendBuffer(m_samplerHeapBuffer, 0, std::span(samplerStaging)));
      uploader.cmdUploadAppended(cmd);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      uploader.deinit();
    }

    rebuildScene(m_gridSize);
    createShaders();

    // Camera
    float halfExtent = m_gridSize * 0.6f;
    g_cameraManip->setClipPlanes({0.1f, 200.0f});
    g_cameraManip->setLookat({halfExtent * 2.5f, halfExtent * 2.0f, halfExtent * 2.5f}, {0, 0, 0}, {0, 1, 0});

    m_heapAvailable = true;
    m_startTime     = std::chrono::high_resolution_clock::now();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);

    if(m_heapAvailable)
    {
      for(size_t i = 0; i < m_pushIndexShaders.size(); i++)
        vkDestroyShaderEXT(m_device, m_pushIndexShaders[i], nullptr);
      for(size_t i = 0; i < m_constantOffsetShaders.size(); i++)
        vkDestroyShaderEXT(m_device, m_constantOffsetShaders[i], nullptr);
      for(size_t i = 0; i < m_directAccessShaders.size(); i++)
        vkDestroyShaderEXT(m_device, m_directAccessShaders[i], nullptr);
      destroyFaceImages();
      m_allocator->destroyBuffer(m_vertexBuffer);
      m_allocator->destroyBuffer(m_indexBuffer);
      m_allocator->destroyBuffer(m_samplerHeapBuffer);
      m_allocator->destroyBuffer(m_resourceHeapBuffer);
    }

    m_viewportImage.deinit();
    m_renderTarget.deinit();
    m_allocator->deinit();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    NVVK_CHECK(m_renderTarget.update(cmd, size));
    m_viewportImage.update(m_renderTarget.getUiImageView());
  }

  void onUIRender() override
  {
    {
      ImGui::Begin("Settings");

      if(!m_heapAvailable)
      {
        ImGui::TextColored(ImVec4(1, 0.3f, 0.3f, 1), "VK_EXT_descriptor_heap not available.");
        ImGui::TextWrapped("This sample requires a driver that supports the extension.");
        ImGui::End();
        return;
      }

      nvgui::CameraWidget(g_cameraManip);
      ImGui::Separator();

      ImGui::Text("Rendering Mode:");
      int modeInt = static_cast<int>(m_mode);
      ImGui::RadioButton("Push Index (per-cube draws)", &modeInt, static_cast<int>(RenderingMode::PushIndex));
      ImGui::SetItemTooltip(
          "Shader:  layout(set=0, binding=0) texture2D faceTextures[6];\n"
          "Mapping: HEAP_WITH_PUSH_INDEX (per-draw index from push data).");
      ImGui::RadioButton("Constant Offset (single instanced draw)", &modeInt, static_cast<int>(RenderingMode::ConstantOffset));
      ImGui::SetItemTooltip(
          "Shader:  layout(set=0, binding=0) texture2D heap[];\n"
          "Mapping: HEAP_WITH_CONSTANT_OFFSET (heap exposed as a single unsized array).");
      ImGui::RadioButton("Direct Access (single instanced draw)", &modeInt, static_cast<int>(RenderingMode::DirectAccess));
      m_mode = static_cast<RenderingMode>(modeInt);
      ImGui::SetItemTooltip(
          "Shader:  layout(descriptor_heap) texture2D heap[];\n"
          "Mapping: none.");
      ImGui::Separator();

      int newGridSize = m_gridSize;
      ImGui::SliderInt("Grid Size", &newGridSize, 2, 16);
      ImGui::SetItemTooltip("NxNxN cube grid. Each cube has 6 unique face textures.");
      if(newGridSize != m_gridSize)
      {
        vkDeviceWaitIdle(m_device);
        destroyFaceImages();
        m_gridSize = newGridSize;
        rebuildScene(m_gridSize);
      }

      ImGui::SliderFloat("Animation Speed", &m_animSpeed, 0.0f, 3.0f);
      ImGui::SetItemTooltip("Set to 0 to freeze with all cubes placed.");
      ImGui::Separator();

      int   numCubes    = m_gridSize * m_gridSize * m_gridSize;
      int   numTextures = numCubes * 6;
      float texMB       = numTextures * (s_TEX_SIZE * s_TEX_SIZE * 4) / (1024.0f * 1024.0f);
      ImGui::Text("Cubes: %d", numCubes);
      ImGui::Text("Textures: %d (%.1f MB)", numTextures, texMB);
      float heapMB = (m_resourceHeapSize + m_samplerHeapSize) / (1024.0f * 1024.0f);
      ImGui::Text("Descriptor heap: %.2f MB", heapMB);
      ImGui::SetItemTooltip(
          "Device-local heap buffers with "
          "VK_BUFFER_USAGE_DESCRIPTOR_HEAP_BIT_EXT.\n"
          "Descriptors are staged on host, then uploaded for fast GPU reads.");
      ImGui::Separator();
      ImGui::Text("Draw calls: %d", m_statsDrawCalls);
      ImGui::SetItemTooltip(
          "Push Index: one vkCmdDrawIndexed per cube.\n"
          "Constant Offset / Direct Access: one instanced vkCmdDrawIndexed.\n"
          "Push Index colors borders per cube; instanced modes share one border color.");
      ImGui::Text("Push data bytes: %d", m_statsPushDataBytes);
      ImGui::SetItemTooltip(
          "Bytes sent via vkCmdPushDataEXT (replaces vkCmdPushConstants\n"
          "when using descriptor heaps).");
      ImGui::Text("Heap binds: %d", m_statsHeapBinds);
      ImGui::SetItemTooltip(
          "Calls to vkCmdBindSamplerHeapEXT + vkCmdBindResourceHeapEXT.\n"
          "Typically once per frame.");
      float avgMappingsPerDraw = m_statsDrawCalls > 0 ? static_cast<float>(m_statsMappingsPerFrame) / m_statsDrawCalls : 0.0f;
      ImGui::Text("Mappings/frame: %d (%.0f/draw)", m_statsMappingsPerFrame, avgMappingsPerDraw);
      ImGui::SetItemTooltip(
          "Number of VkDescriptorSetAndBindingMappingEXT entries honored\n"
          "across all draws this frame. Mappings are baked into shaders at\n"
          "creation time; this counter reflects how often shaders declaring\n"
          "them are dispatched.\n"
          "  Push Index:      6 (HEAP_WITH_PUSH_INDEX) + 1 (HEAP_WITH_CONSTANT_OFFSET) per draw\n"
          "  Constant Offset: 2 (HEAP_WITH_CONSTANT_OFFSET) per draw\n"
          "  Direct Access:   0 (no mappings; layout(descriptor_heap))");

      ImGui::End();
    }

    {
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
      ImGui::Begin("Viewport");
      ImGui::Image(m_viewportImage, ImGui::GetContentRegionAvail());
      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    if(!m_heapAvailable)
      return;

    NVVK_DBG_SCOPE(cmd);

    // Time. The animation cycle in the vertex shader is:
    //   [stagger fall-in]  →  [rest, all placed]  →  [stagger fall-out]
    // Offset by restStartTime so t=0 lands at the start of the rest phase:
    // all cubes fully placed. Animation speed = 0 then freezes there.
    uint32_t    numCubes      = static_cast<uint32_t>(m_gridSize * m_gridSize * m_gridSize);
    const float restStartTime = (numCubes - 1) * shaderio::animationCubeDelay + shaderio::animationFallDuration;

    auto  now     = std::chrono::high_resolution_clock::now();
    float elapsed = std::chrono::duration<float>(now - m_startTime).count();
    float time    = restStartTime + elapsed * m_animSpeed;

    if(m_app->isHeadless())
      time = restStartTime;  // In headless mode, snapshot the fully-placed state

    // #DESC_HEAP: Bind descriptor heaps (once per frame)
    cmdBindHeaps(cmd);

    // Build the dynamic-rendering state from the render target. The render target
    // keeps its images in VK_IMAGE_LAYOUT_GENERAL, so no layout transitions are
    // needed to render into it or to sample it afterwards.
    nvvk::RenderTargetState rtState;
    m_renderTarget.fillState(rtState);
    rtState.colorAttachments[0].clearValue = {m_clearColor};
    rtState.depthAttachment.clearValue     = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

    nvvk::RenderTargetState::AttachmentOps ops{};  // default: clear+store on color & depth, don't care on stencil
    rtState.cmdBeginRendering(cmd, ops);

    m_graphicState.cmdSetViewportAndScissor(cmd, m_renderTarget.getSize());
    m_graphicState.cmdApplyAllStates(cmd);

    auto& shaders = (m_mode == RenderingMode::PushIndex)      ? m_pushIndexShaders :
                    (m_mode == RenderingMode::ConstantOffset) ? m_constantOffsetShaders :
                                                                m_directAccessShaders;
    m_graphicState.cmdBindShaders(cmd, {.vertex   = shaders[static_cast<size_t>(ShaderIndex::eVertex)],
                                        .fragment = shaders[static_cast<size_t>(ShaderIndex::eFragment)]});

    VkDeviceSize vbOffset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer.buffer, &vbOffset);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);

    // #DESC_HEAP: Push frame info at offset 0
    uint32_t numIndices = static_cast<uint32_t>(m_cubeMesh.triangles.size() * 3);

    shaderio::FrameInfo frameInfo{};
    frameInfo.proj       = g_cameraManip->getPerspectiveMatrix();
    frameInfo.view       = g_cameraManip->getViewMatrix();
    frameInfo.lightDir   = {0.5f, 0.7f, 0.3f};
    frameInfo.time       = time;
    frameInfo.numCubes   = numCubes;
    frameInfo.dropHeight = m_gridSize * 1.5f + 5.0f;
    cmdPushData(cmd, 0, sizeof(shaderio::FrameInfo), &frameInfo);

    m_statsDrawCalls        = 0;
    m_statsPushDataBytes    = static_cast<int>(sizeof(shaderio::FrameInfo));
    m_statsHeapBinds        = 1;
    m_statsMappingsPerFrame = 0;

    if(m_mode == RenderingMode::PushIndex)
    {
      // #DESC_HEAP: PushIndex mode — one draw call per cube
      // Each draw pushes DrawData after FrameInfo. The mapping reads
      // baseFaceTexIdx from push data to resolve faceTextures[0..5] from the
      // heap. The shader never reads baseFaceTexIdx directly.
      float spacing = 1.1f;
      float offset  = (m_gridSize - 1) * spacing * 0.5f;

      uint32_t cubeIdx = 0;
      for(int iz = 0; iz < m_gridSize; iz++)
      {
        for(int iy = 0; iy < m_gridSize; iy++)
        {
          for(int ix = 0; ix < m_gridSize; ix++, cubeIdx++)
          {
            glm::vec3 pos = glm::vec3(ix, iy, iz) * spacing - offset;

            shaderio::DrawData drawData{};
            drawData.transform      = glm::translate(glm::mat4(1.0f), pos);
            drawData.baseFaceTexIdx = cubeIdx * 6;
            drawData.cubeIndex      = cubeIdx;
            drawData.borderColor    = static_cast<uint32_t>(nvutils::hashVal(m_statsDrawCalls)) | 0xFF000000u;

            cmdPushData(cmd, sizeof(shaderio::FrameInfo), sizeof(shaderio::DrawData), &drawData);
            vkCmdDrawIndexed(cmd, numIndices, 1, 0, 0, 0);
            m_statsDrawCalls++;
            m_statsPushDataBytes += static_cast<int>(sizeof(shaderio::DrawData));
            m_statsMappingsPerFrame += 7;  // 6 images (PUSH_INDEX) + 1 sampler (CONSTANT_OFFSET)
          }
        }
      }
    }
    else
    {
      // #DESC_HEAP: ConstantOffset/DirectAccess modes — single instanced draw.
      shaderio::InstancedPushData instancedData{};
      instancedData.borderColor = static_cast<uint32_t>(nvutils::hashVal(m_statsDrawCalls)) | 0xFF000000u;
      instancedData.gridSize    = static_cast<uint32_t>(m_gridSize);

      cmdPushData(cmd, sizeof(shaderio::FrameInfo), sizeof(shaderio::InstancedPushData), &instancedData);

      // Instanced rendering: draw numCubes instances of the same cube mesh
      // in a single draw call. The vertex shader derives each cube from the
      // instance ID and grid size.
      vkCmdDrawIndexed(cmd, numIndices, numCubes, 0, 0, 0);
      m_statsDrawCalls = 1;
      m_statsPushDataBytes += static_cast<int>(sizeof(shaderio::InstancedPushData));

      if(m_mode == RenderingMode::ConstantOffset)
      {
        m_statsMappingsPerFrame += 2;  // 1 image array + 1 sampler array (both HEAP_WITH_CONSTANT_OFFSET)
      }
    }

    vkCmdEndRendering(cmd);
  }

private:
  //--------------------------------------------------------------------------------------------------
  // #DESC_HEAP: Descriptor heap initialization
  // Query VkPhysicalDeviceDescriptorHeapPropertiesEXT for descriptor sizes,
  // alignment, and reserved range requirements. Allocate sampler heap buffer
  // with DESCRIPTOR_HEAP_BIT_EXT. Resource heap is allocated later in
  // resizeResourceHeap() based on scene size.
  //
  bool initHeaps(VkPhysicalDevice physDevice)
  {
    // Query descriptor heap properties (sizes, alignment, reserved range)
    VkPhysicalDeviceDescriptorHeapPropertiesEXT heapProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &heapProps;
    vkGetPhysicalDeviceProperties2(physDevice, &props2);

    m_samplerDescSize       = heapProps.samplerDescriptorSize;
    m_imageDescSize         = heapProps.imageDescriptorSize;
    m_samplerReservedSize   = heapProps.minSamplerHeapReservedRange;
    m_resourceReservedSize  = heapProps.minResourceHeapReservedRange;
    m_samplerHeapAlignment  = heapProps.samplerHeapAlignment;
    m_resourceHeapAlignment = heapProps.resourceHeapAlignment;

    if(m_samplerDescSize == 0 || m_imageDescSize == 0)
      return false;

    // The spec requires heapRange.address to be aligned to *HeapAlignment when
    // binding. Query whether the driver's memory requirements already satisfy
    // this; if not, we pass the heap alignment as minAlignment to createBuffer
    // below.
    {
      VkBufferUsageFlags2CreateInfo usageFlags2{VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR};
      usageFlags2.usage = VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR | VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT;
      VkBufferCreateInfo               bufInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &usageFlags2, 0, 256, 0};
      VkDeviceBufferMemoryRequirements devBufReqs{VK_STRUCTURE_TYPE_DEVICE_BUFFER_MEMORY_REQUIREMENTS};
      devBufReqs.pCreateInfo = &bufInfo;
      VkMemoryRequirements2 memReqs{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
      vkGetDeviceBufferMemoryRequirements(m_device, &devBufReqs, &memReqs);
      VkDeviceSize heapAlign = std::max(m_samplerHeapAlignment, m_resourceHeapAlignment);
      if(memReqs.memoryRequirements.alignment < heapAlign)
      {
        LOGI(
            "vkGetDeviceBufferMemoryRequirements alignment (%zu) for "
            "VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT"
            " < "
            "VkPhysicalDeviceDescriptorHeapPropertiesEXT::"
            "resourceHeapAlignment (%zu); passing the latter as minAlignment\n",
            size_t(memReqs.memoryRequirements.alignment), size_t(heapAlign));
      }
    }

    // Allocate sampler heap (device-local for fast GPU reads; staged from host)
    uint32_t     maxSamplers     = 4;
    VkDeviceSize samplerHeapSize = m_samplerDescSize * maxSamplers + m_samplerReservedSize;
    samplerHeapSize = ((samplerHeapSize + m_samplerHeapAlignment - 1) / m_samplerHeapAlignment) * m_samplerHeapAlignment;
    NVVK_CHECK(m_allocator->createBuffer(m_samplerHeapBuffer, samplerHeapSize,
                                         VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT_KHR
                                             | VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT,
                                         VMA_MEMORY_USAGE_AUTO, {}, m_samplerHeapAlignment));
    NVVK_DBG_NAME(m_samplerHeapBuffer.buffer);

    VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addrInfo.buffer      = m_samplerHeapBuffer.buffer;
    m_samplerHeapAddress = vkGetBufferDeviceAddress(m_device, &addrInfo);
    m_samplerHeapSize    = samplerHeapSize;
    m_maxSamplers        = maxSamplers;

    return true;
  }

  //--------------------------------------------------------------------------------------------------
  // #DESC_HEAP: Resize resource heap to fit the given number of images
  // (Re)allocates the resource heap buffer for maxImages descriptors plus
  // reserved range. Also resizes host staging buffer; staging is filled by
  // writeImageDescriptor() and uploaded to the device-local heap in
  // rebuildScene().
  //
  void resizeResourceHeap(uint32_t maxImages)
  {
    if(m_resourceHeapBuffer.buffer)
      m_allocator->destroyBuffer(m_resourceHeapBuffer);

    VkDeviceSize resourceHeapSize = m_imageDescSize * maxImages + m_resourceReservedSize;
    resourceHeapSize = ((resourceHeapSize + m_resourceHeapAlignment - 1) / m_resourceHeapAlignment) * m_resourceHeapAlignment;
    NVVK_CHECK(m_allocator->createBuffer(m_resourceHeapBuffer, resourceHeapSize,
                                         VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT_KHR | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT_KHR
                                             | VK_BUFFER_USAGE_2_DESCRIPTOR_HEAP_BIT_EXT,
                                         VMA_MEMORY_USAGE_AUTO, {}, m_resourceHeapAlignment));
    NVVK_DBG_NAME(m_resourceHeapBuffer.buffer);

    VkBufferDeviceAddressInfo addrInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addrInfo.buffer       = m_resourceHeapBuffer.buffer;
    m_resourceHeapAddress = vkGetBufferDeviceAddress(m_device, &addrInfo);
    m_resourceHeapSize    = resourceHeapSize;
    m_maxImages           = maxImages;

    // Host staging for descriptor writes, uploaded to device-local heap in
    // rebuildScene
    m_resourceHeapStaging.resize(resourceHeapSize);
  }

  //--------------------------------------------------------------------------------------------------
  // #DESC_HEAP: Write an image descriptor into host staging at the given index
  // Uses vkWriteResourceDescriptorsEXT to write to host memory. The staging
  // buffer is uploaded to the device-local resource heap in rebuildScene().
  //
  void writeImageDescriptor(uint32_t index, VkImage image, VkFormat format, VkImageLayout layout)
  {
    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image            = image;
    viewInfo.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format           = format;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    VkImageDescriptorInfoEXT imageDescInfo{VK_STRUCTURE_TYPE_IMAGE_DESCRIPTOR_INFO_EXT};
    imageDescInfo.pView  = &viewInfo;
    imageDescInfo.layout = layout;

    VkResourceDescriptorInfoEXT resInfo{VK_STRUCTURE_TYPE_RESOURCE_DESCRIPTOR_INFO_EXT};
    resInfo.type        = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    resInfo.data.pImage = &imageDescInfo;

    // Write to host staging; uploaded to device-local heap in rebuildScene
    VkHostAddressRangeEXT dst{};
    dst.address = m_resourceHeapStaging.data() + index * m_imageDescSize;
    dst.size    = m_imageDescSize;
    vkWriteResourceDescriptorsEXT(m_device, 1, &resInfo, &dst);
  }

  //--------------------------------------------------------------------------------------------------
  // #DESC_HEAP: Bind sampler and resource heaps for the command buffer
  // Called once per frame. Subsequent draws use these heaps. The reserved range
  // (reservedRangeOffset/Size) is a spec requirement for driver-managed data.
  //
  void cmdBindHeaps(VkCommandBuffer cmd)
  {
    VkBindHeapInfoEXT samplerBind{VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT};
    samplerBind.heapRange           = {m_samplerHeapAddress, m_samplerHeapSize};
    samplerBind.reservedRangeOffset = m_samplerDescSize * m_maxSamplers;
    samplerBind.reservedRangeSize   = m_samplerReservedSize;
    vkCmdBindSamplerHeapEXT(cmd, &samplerBind);

    VkBindHeapInfoEXT resourceBind{VK_STRUCTURE_TYPE_BIND_HEAP_INFO_EXT};
    resourceBind.heapRange           = {m_resourceHeapAddress, m_resourceHeapSize};
    resourceBind.reservedRangeOffset = m_imageDescSize * m_maxImages;
    resourceBind.reservedRangeSize   = m_resourceReservedSize;
    vkCmdBindResourceHeapEXT(cmd, &resourceBind);
  }

  // #DESC_HEAP: Push data for shaders and descriptor mapping
  // Replaces vkCmdPushConstants. Layout: FrameInfo at offset 0; DrawData or
  // InstancedPushData at offset sizeof(FrameInfo). In push-index mode, the
  // mapping reads baseFaceTexIdx from push data at pushOffset to resolve heap
  // indices.
  void cmdPushData(VkCommandBuffer cmd, uint32_t offset, uint32_t size, const void* data)
  {
    VkPushDataInfoEXT pushInfo{VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT};
    pushInfo.offset       = offset;
    pushInfo.data.address = data;
    pushInfo.data.size    = size;
    vkCmdPushDataEXT(cmd, &pushInfo);
  }

  //--------------------------------------------------------------------------------------------------
  // #DESC_HEAP: Create shader objects for all three rendering modes.
  //
  // The descriptor-heap mapping struct
  // (VkShaderDescriptorSetAndBindingMappingInfoEXT, chained into
  // VkShaderCreateInfoEXT::pNext) is only used by shaders that declare
  // resources with traditional set/binding declarations. Direct Access
  // shaders declare layout(descriptor_heap) and bypass the mapping
  // entirely — mappingInfo is nullptr there.
  //
  void createShaders()
  {
    // Shared vertex state (used at draw time via cmdApplyAllStates)
    m_graphicState.rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
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
                                                   .offset   = offsetof(nvutils::PrimitiveVertex, nrm)}};

    // --- #DESC_HEAP: PushIndex shaders with set/binding mapping ---
    // This block demonstrates per-draw heap indexing. Before each draw,
    // vkCmdPushDataEXT writes a per-cube baseFaceTexIdx into push data;
    // HEAP_WITH_PUSH_INDEX reads it back from the mapping's pushOffset
    // field and uses it as the base heap index for the binding. The
    // shader's faceTextures[6] covers 6 consecutive descriptors from
    // that base, indexed by faceIdx. Binding 1 (sampler) uses
    // HEAP_WITH_CONSTANT_OFFSET: fixed at sampler-heap offset 0.
    //
    // On bindingCount vs. array bindings — from the VK_EXT_descriptor_heap
    // spec:
    //   "If an array of bindings are specified, each subsequent binding is
    //    offset by heapArrayStride. If a binding is itself an array, each
    //    subsequent shader index is offset by heapArrayStride."
    // Both knobs share heapArrayStride. In practice you pick one or the
    // other: either bindingCount=N over N adjacent scalar bindings, or
    // bindingCount=1 over a single array binding (this sample's
    // faceTextures[6]) and let the shader's array index select a heap
    // slot.
    {
      // Mapping for binding 0: texture2D faceTextures[6] ->
      // HEAP_WITH_PUSH_INDEX
      VkDescriptorSetAndBindingMappingEXT texMapping{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT};
      texMapping.descriptorSet                   = 0;
      texMapping.firstBinding                    = 0;
      texMapping.bindingCount                    = 1;
      texMapping.resourceMask                    = VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT;
      texMapping.source                          = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_PUSH_INDEX_EXT;
      texMapping.sourceData.pushIndex            = {};
      texMapping.sourceData.pushIndex.heapOffset = 0;
      texMapping.sourceData.pushIndex.pushOffset =
          static_cast<uint32_t>(sizeof(shaderio::FrameInfo) + offsetof(shaderio::DrawData, baseFaceTexIdx));
      texMapping.sourceData.pushIndex.heapIndexStride = static_cast<uint32_t>(m_imageDescSize);
      texMapping.sourceData.pushIndex.heapArrayStride = static_cast<uint32_t>(m_imageDescSize);

      // Mapping for binding 1: sampler -> HEAP_WITH_CONSTANT_OFFSET.
      // We only have one sampler, so a constant offset is enough and no
      // array stride is needed.
      VkDescriptorSetAndBindingMappingEXT samplerMapping{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT};
      samplerMapping.descriptorSet                        = 0;
      samplerMapping.firstBinding                         = 1;
      samplerMapping.bindingCount                         = 1;
      samplerMapping.resourceMask                         = VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
      samplerMapping.source                               = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
      samplerMapping.sourceData.constantOffset            = {};
      samplerMapping.sourceData.constantOffset.heapOffset = 0;

      std::array<VkDescriptorSetAndBindingMappingEXT, 2> mappings = {texMapping, samplerMapping};

      VkShaderDescriptorSetAndBindingMappingInfoEXT mappingInfo{VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT};
      mappingInfo.mappingCount = static_cast<uint32_t>(mappings.size());
      mappingInfo.pMappings    = mappings.data();

#if USE_SLANG
      createHeapShaders(m_pushIndexShaders, push_index_slang, push_index_slang, &mappingInfo, "vertexMain", "fragmentMain");
#else
      createHeapShaders(m_pushIndexShaders, push_index_vert_glsl, push_index_frag_glsl, &mappingInfo);
#endif
    }

    // --- #DESC_HEAP: ConstantOffset shaders with set/binding mapping ---
    // This block makes a contiguous region of the heap visible to the
    // shader as an array binding. HEAP_WITH_CONSTANT_OFFSET pins the
    // binding to a fixed heap offset; the shader supplies its own index
    // into the region. Here the shader computes instanceID*6 + faceIdx
    // and indexes the binding's unsized array.
    //
    // Aside: VkDescriptorMappingSourceConstantOffsetEXT also exposes
    // samplerHeapOffset / samplerHeapArrayStride. Those are the active
    // fields only when a binding is a combined image+sampler
    // (OpTypeSampledImage, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) — a
    // single descriptor that fuses an image and a sampler. This sample
    // uses separate texture and sampler bindings, so we don't need to set
    // samplerHeapOffset / samplerHeapArrayStride.
    {
      VkDescriptorSetAndBindingMappingEXT texMapping{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT};
      texMapping.descriptorSet                             = 0;
      texMapping.firstBinding                              = 0;
      texMapping.bindingCount                              = 1;
      texMapping.resourceMask                              = VK_SPIRV_RESOURCE_TYPE_SAMPLED_IMAGE_BIT_EXT;
      texMapping.source                                    = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
      texMapping.sourceData.constantOffset                 = {};
      texMapping.sourceData.constantOffset.heapOffset      = 0;
      texMapping.sourceData.constantOffset.heapArrayStride = static_cast<uint32_t>(m_imageDescSize);

      VkDescriptorSetAndBindingMappingEXT samplerMapping{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_AND_BINDING_MAPPING_EXT};
      samplerMapping.descriptorSet                        = 0;
      samplerMapping.firstBinding                         = 1;
      samplerMapping.bindingCount                         = 1;
      samplerMapping.resourceMask                         = VK_SPIRV_RESOURCE_TYPE_SAMPLER_BIT_EXT;
      samplerMapping.source                               = VK_DESCRIPTOR_MAPPING_SOURCE_HEAP_WITH_CONSTANT_OFFSET_EXT;
      samplerMapping.sourceData.constantOffset            = {};
      samplerMapping.sourceData.constantOffset.heapOffset = 0;
      samplerMapping.sourceData.constantOffset.heapArrayStride = static_cast<uint32_t>(m_samplerDescSize);

      std::array<VkDescriptorSetAndBindingMappingEXT, 2> mappings = {texMapping, samplerMapping};

      VkShaderDescriptorSetAndBindingMappingInfoEXT mappingInfo{VK_STRUCTURE_TYPE_SHADER_DESCRIPTOR_SET_AND_BINDING_MAPPING_INFO_EXT};
      mappingInfo.mappingCount = static_cast<uint32_t>(mappings.size());
      mappingInfo.pMappings    = mappings.data();

#if USE_SLANG
      createHeapShaders(m_constantOffsetShaders, constant_offset_slang, constant_offset_slang, &mappingInfo,
                        "vertexMain", "fragmentMain");
#else
      createHeapShaders(m_constantOffsetShaders, instanced_vert_glsl, constant_offset_frag_glsl, &mappingInfo);
#endif
    }

    // --- #DESC_HEAP: DirectAccess shaders ---
    // No mapping struct: layout(descriptor_heap) handles binding entirely
    // in SPIR-V via the DescriptorHeapEXT capability. mappingInfo = nullptr.
    {
#if USE_SLANG
      createHeapShaders(m_directAccessShaders, direct_access_slang, direct_access_slang, nullptr, "vertexMain", "fragmentMain");
#else
      createHeapShaders(m_directAccessShaders, instanced_vert_glsl, direct_access_frag_glsl, nullptr);
#endif
    }
  }

  // #DESC_HEAP: Create linked shader objects with
  // VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT. If mappingInfo is provided, it's
  // chained into pNext for set/binding -> heap mapping. No descriptor set
  // layout is needed (layout = VK_NULL_HANDLE).
  template <size_t NV, size_t NF>
  void createHeapShaders(std::array<VkShaderEXT, 2>& shaders,
                         const uint32_t (&vertSpirv)[NV],
                         const uint32_t (&fragSpirv)[NF],
                         VkShaderDescriptorSetAndBindingMappingInfoEXT* mappingInfo,
                         const char*                                    vertEntry = "main",
                         const char*                                    fragEntry = "main")
  {
    VkShaderCreateFlagsEXT flags = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT | VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT;

    std::array<VkShaderCreateInfoEXT, 2> createInfos{};
    createInfos[0].sType     = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
    createInfos[0].pNext     = mappingInfo;
    createInfos[0].flags     = flags;
    createInfos[0].stage     = VK_SHADER_STAGE_VERTEX_BIT;
    createInfos[0].nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
    createInfos[0].codeType  = VK_SHADER_CODE_TYPE_SPIRV_EXT;
    createInfos[0].codeSize  = sizeof(vertSpirv);
    createInfos[0].pCode     = vertSpirv;
    createInfos[0].pName     = vertEntry;

    createInfos[1].sType     = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT;
    createInfos[1].pNext     = mappingInfo;
    createInfos[1].flags     = flags;
    createInfos[1].stage     = VK_SHADER_STAGE_FRAGMENT_BIT;
    createInfos[1].nextStage = 0;
    createInfos[1].codeType  = VK_SHADER_CODE_TYPE_SPIRV_EXT;
    createInfos[1].codeSize  = sizeof(fragSpirv);
    createInfos[1].pCode     = fragSpirv;
    createInfos[1].pName     = fragEntry;

    NVVK_CHECK(vkCreateShadersEXT(m_device, 2, createInfos.data(), nullptr, shaders.data()));
  }

  //--------------------------------------------------------------------------------------------------
  // #DESC_HEAP: Build scene and populate resource heap
  // Generates face textures, writes descriptors to host staging via
  // writeImageDescriptor(), then uploads both textures and staging buffer to
  // device-local memory.
  //
  void rebuildScene(int gridSize)
  {
    static const char* s_faceNames[] = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"};
    int                N             = gridSize;
    int                numCubes      = N * N * N;

    resizeResourceHeap(static_cast<uint32_t>(numCubes * 6));

    nvvk::StagingUploader uploader;
    uploader.init(m_allocator.get(), true);
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    uint32_t texIdx = 0;
    for(int iz = 0; iz < N; iz++)
    {
      for(int iy = 0; iy < N; iy++)
      {
        for(int ix = 0; ix < N; ix++)
        {
          uint8_t ir = (N > 1) ? static_cast<uint8_t>(ix * 255 / (N - 1)) : 128;
          uint8_t ig = (N > 1) ? static_cast<uint8_t>(iy * 255 / (N - 1)) : 128;
          uint8_t ib = (N > 1) ? static_cast<uint8_t>(iz * 255 / (N - 1)) : 128;

          for(int face = 0; face < 6; face++)
          {
            std::vector<uint8_t> pixels;
            font_texture::generateFaceTexture(pixels, ir, ig, ib, s_faceNames[face]);

            VkImageCreateInfo imgCI{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            imgCI.imageType   = VK_IMAGE_TYPE_2D;
            imgCI.format      = VK_FORMAT_R8G8B8A8_UNORM;
            imgCI.extent      = {s_TEX_SIZE, s_TEX_SIZE, 1};
            imgCI.mipLevels   = 1;
            imgCI.arrayLayers = 1;
            imgCI.samples     = VK_SAMPLE_COUNT_1_BIT;
            imgCI.tiling      = VK_IMAGE_TILING_OPTIMAL;
            imgCI.usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

            nvvk::Image img;
            NVVK_CHECK(m_allocator->createImage(img, imgCI));
            NVVK_CHECK(uploader.appendImage(img, std::span(pixels), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
            m_faceImages.push_back(img);

            writeImageDescriptor(texIdx, img.image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            texIdx++;
          }
        }
      }
    }

    // Upload host-staged descriptors to device-local resource heap
    NVVK_CHECK(uploader.appendBuffer(m_resourceHeapBuffer, 0, std::span(m_resourceHeapStaging)));

    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();
  }

  void destroyFaceImages()
  {
    for(size_t i = 0; i < m_faceImages.size(); i++)
      m_allocator->destroyImage(m_faceImages[i]);
    m_faceImages.clear();
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_renderTarget.getColorImage(), m_renderTarget.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvapp::Application*                      m_app{};
  VkDevice                                 m_device{};
  std::unique_ptr<nvvk::ResourceAllocator> m_allocator;

  VkFormat           m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;
  VkFormat           m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;
  VkClearColorValue  m_clearColor  = {{0.2f, 0.2f, 0.2f, 1.0f}};
  nvvk::RenderTarget m_renderTarget;   // Offscreen render target: color + depth
  nvapp::ImTexture   m_viewportImage;  // ImGui texture for the render target color image

  // #DESC_HEAP: Descriptor heap state
  nvvk::Buffer         m_samplerHeapBuffer;      // Device-local sampler heap
  nvvk::Buffer         m_resourceHeapBuffer;     // Device-local resource (image) heap
  VkDeviceAddress      m_samplerHeapAddress{};   // GPU address for vkCmdBindSamplerHeapEXT
  VkDeviceAddress      m_resourceHeapAddress{};  // GPU address for vkCmdBindResourceHeapEXT
  VkDeviceSize         m_samplerHeapSize{};
  VkDeviceSize         m_resourceHeapSize{};
  VkDeviceSize         m_samplerDescSize{};      // Bytes per sampler descriptor
  VkDeviceSize         m_imageDescSize{};        // Bytes per image descriptor
  VkDeviceSize         m_samplerReservedSize{};  // Spec-required reserved range at end of heap
  VkDeviceSize         m_resourceReservedSize{};
  VkDeviceSize         m_samplerHeapAlignment{};
  VkDeviceSize         m_resourceHeapAlignment{};
  uint32_t             m_maxSamplers{};
  uint32_t             m_maxImages{};
  std::vector<uint8_t> m_resourceHeapStaging;  // Host staging for vkWriteResourceDescriptorsEXT

  // Shader objects
  std::array<VkShaderEXT, 2>  m_pushIndexShaders{};
  std::array<VkShaderEXT, 2>  m_constantOffsetShaders{};
  std::array<VkShaderEXT, 2>  m_directAccessShaders{};
  nvvk::GraphicsPipelineState m_graphicState;

  // Scene
  nvutils::PrimitiveMesh   m_cubeMesh;
  nvvk::Buffer             m_vertexBuffer;
  nvvk::Buffer             m_indexBuffer;
  std::vector<nvvk::Image> m_faceImages;

  // Settings
  bool                                           m_heapAvailable         = false;
  RenderingMode                                  m_mode                  = RenderingMode::ConstantOffset;
  int                                            m_gridSize              = 6;
  float                                          m_animSpeed             = 1.0f;
  int                                            m_statsDrawCalls        = 0;
  int                                            m_statsPushDataBytes    = 0;
  int                                            m_statsHeapBinds        = 0;
  int                                            m_statsMappingsPerFrame = 0;
  std::chrono::high_resolution_clock::time_point m_startTime;
};

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);

  // Vulkan context setup
  VkPhysicalDeviceDescriptorHeapFeaturesEXT heapFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT};
  heapFeatures.descriptorHeap = VK_TRUE;

  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};

  VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untypedPtrFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR};
  untypedPtrFeatures.shaderUntypedPointers = VK_TRUE;

  nvvk::ContextInitInfo vkSetup = {.instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
                                   .deviceExtensions   = {
                                       {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
                                       {VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, &heapFeatures, false},  // Request VK_EXT_descriptor_heap (optional — sample shows
                                       // error if not available)
                                       {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjFeatures, false},  // Request VK_EXT_shader_object
                                       {VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME, &untypedPtrFeatures, false},  // Required by SPIR-V emitted for layout(descriptor_heap)
                                   }};

  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }

  nvvk::ValidationSettings vvlInfo{};
  vvlInfo.setPreset(nvvk::ValidationSettings::LayerPresets::eStandard);
  vkSetup.instanceCreateInfoExt = vvlInfo.buildPNextChain();

  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE(
        "Failed to create Vulkan context. VK_EXT_descriptor_heap may not be "
        "supported by your driver.\n");
    return 1;
  }

  appInfo.name           = fmt::format("{} ({})", nvutils::getExecutablePath().stem().string(), SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
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
  app.addElement(std::make_shared<DescriptorHeapSample>());

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
