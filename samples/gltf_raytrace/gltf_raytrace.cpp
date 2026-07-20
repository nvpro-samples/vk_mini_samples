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

    This shows the rendering of a glTF scene with a ray-query path tracer.

    Resource binding is fully bindless via nvvk::DescriptorHeap (VK_EXT_descriptor_heap): a sampler
    heap (one shared linear sampler) and a host-visible resource heap holding the output storage
    image, the HDR environment, the glTF textures, and the environment importance-sampling table.
    The compute shader is a VK_EXT_shader_object created with VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT
    and reads every resource via <Resource>.Handle(...); per-frame data is delivered with
    vkCmdPushDataEXT. The TLAS is passed by device address in the push constant and turned into an
    acceleration structure in-shader (RaytracingAccelerationStructure(address))

    It also uses many helper classes to create the scene, the ray tracing structures and the rendering.

*/
//////////////////////////////////////////////////////////////////////////

// The defines must be done here, to avoid having multiple definitions

#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }
#define IMGUI_DEFINE_MATH_OPERATORS


#include <string>
#include <vector>

#include <fmt/format.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>

#include "GLFW/glfw3.h"
#undef APIENTRY

// Shader Input/Output
#include "shaders/shaderio.h"  // Shared between host and device

// Pre-compiled shaders
#include "_autogen/gltf_pathtrace.slang.h"
#include "_autogen/tonemapper.slang.h"

#define USE_NSIGHT_AFTERMATH 1
#include <nvaftermath/aftermath.hpp>

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_dbgprintf.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/property_editor.hpp>
#include <nvapp/imgui_texture.hpp>
#include <nvgui/sky.hpp>
#include <nvgui/tonemapper.hpp>
#include <nvshaders_host/tonemapper.hpp>
#include <nvslang/slang.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/descriptor_heap.hpp>
#include <nvvk/hdr_ibl.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/ray_picker.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/validation_settings.hpp>
#include <nvvkgltf/camera_utils.hpp>
#include <nvvkgltf/scene.hpp>
#include <nvvkgltf/scene_rtx.hpp>
#include <nvvkgltf/scene_vk.hpp>
#include <nvvk/compute_pipeline.hpp>

#include "common/utils.hpp"

std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};


std::filesystem::path g_sceneFilename = "shader_ball.gltf";  // Default scene
std::filesystem::path g_hdrFilename   = "env3.hdr";          // Default HDR

/// </summary> Ray trace multiple primitives using Ray Query
class GltfRaytrace : public nvapp::IAppElement
{
  enum
  {
    eImgTonemapped,
    eImgRendered
  };

  // The per-region heap slot layout (kHeapImg* / kHeapBuf*) is shared with the shader in shaderio.h.
  // Only the region bases (imageShaderIndexBase / bufferShaderIndexBase) are pushed; the shader adds
  // the slot constants to form the absolute heap index. By default the TLAS is NOT in the heap - it is
  // passed by device address and converted to an acceleration structure in the shader.
  static constexpr uint32_t kMaxTextures    = 500;  // Reserved texture slots (matches legacy reservation)
  static constexpr uint32_t kImageHeapSize  = shaderio::kHeapImgTexturesStart + kMaxTextures;
  static constexpr uint32_t kBufferHeapSize = 1;  // env alias-table

public:
  GltfRaytrace()           = default;
  ~GltfRaytrace() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = app->getDevice();

    // Create the Vulkan allocator (VMA)
    m_allocator.init({
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });  // Allocator

    // The texture sampler to use (needed to sample the rendered image in the tonemapper)
    m_samplerPool.init(m_device);
    NVVK_CHECK(m_samplerPool.acquireSampler(m_linearSampler));
    NVVK_DBG_NAME(m_linearSampler);

    // IBL environment map
    m_hdrIbl.init(&m_allocator, &m_samplerPool);

    // Offscreen render target: rendered (HDR) + tonemapped color images, no depth
    NVVK_CHECK(m_renderTarget.init(
        {.alloc = &m_allocator, .colorFormats = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32G32B32A32_SFLOAT}, .debugName = "GltfRaytrace"}));
    {
      VkCommandBuffer cmd = app->createTempCmdBuffer();
      NVVK_CHECK(m_renderTarget.update(cmd, {100, 100}));
      app->submitAndWaitTempCmdBuffer(cmd);
    }

    // Ray picker
    m_rayPicker.init(&m_allocator);

    // Tonemapper
    {
      auto code = std::span<const uint32_t>(tonemapper_slang);  // Pre-compiled
      m_tonemapper.init(&m_allocator, code);
    }

    m_sceneVk.init(&m_allocator, &m_samplerPool);
    m_sceneRtx.init(&m_allocator);

    // Slang compiler
    {
      using namespace slang;
      m_slangCompiler.addSearchPaths(nvsamples::getShaderDirs());
      m_slangCompiler.defaultTarget();
      m_slangCompiler.defaultOptions();
      m_slangCompiler.addOption(
          {CompilerOptionName::DebugInformation, {CompilerOptionValueKind::Int, SLANG_DEBUG_INFO_LEVEL_STANDARD}});
      m_slangCompiler.addOption({CompilerOptionName::Optimization, {CompilerOptionValueKind::Int, SLANG_OPTIMIZATION_LEVEL_DEFAULT}});
      m_slangCompiler.addCapability("spvRayQueryKHR");
      m_slangCompiler.addCapability("spvDescriptorHeapEXT");  // Bindless via VK_EXT_descriptor_heap

#if defined(AFTERMATH_AVAILABLE)
      // This aftermath callback is used to report the shader hash (Spirv) to the Aftermath library.
      m_slangCompiler.setCompileCallback([&](const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize) {
        std::span<const uint32_t> data(spirvCode, spirvSize / sizeof(uint32_t));
        AftermathCrashTracker::getInstance().addShaderBinary(data);
      });
#endif
    }

    // Bindless descriptor heap: created before any descriptor writes (HDR / scene / textures).
    createDescriptorHeap();

    // Create resources
    createHDR();
    createScene();
    createVkBuffers();
    compileShader();
    updateTextures();
    updateSamplers();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    NVVK_CHECK(m_renderTarget.update(cmd, size));
    // The image views are recreated on resize: refresh the ImGui texture that
    // displays the tonemapped image, and the output storage-image heap slot.
    m_viewportImage.update(m_renderTarget.getUiImageView(eImgTonemapped));
    writeOutputImageDescriptor();  // The render target image view changed; refresh its heap slot
    resetFrame();                  // Reset frame to restart the rendering
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");

      nvgui::CameraWidget(g_cameraManip);

      namespace PE = nvgui::PropertyEditor;
      bool changed{false};
      if(ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
      {
        PE::begin();
        changed |= PE::Combo("Environment Type", &m_pushConst.environmentType, "Sky\0HDR\0\0");  // 0: Sky, 1: HDR
        PE::end();

        if(ImGui::TreeNode("Ray Tracer"))
        {
          PE::begin();
          changed |= PE::SliderInt("Depth", &m_pushConst.maxDepth, 0, 20);
          changed |= PE::SliderInt("Samples", &m_pushConst.maxSamples, 1, 10);

          // Combo box to select the environment type

          PE::end();
          ImGui::TreePop();
        }
        if(ImGui::TreeNode("Sky"))
        {
          changed |= nvgui::skyPhysicalParameterUI(m_skyParams);
          ImGui::TreePop();
        }
      }

      if(ImGui::CollapsingHeader("Tonemapper"))
      {
        nvgui::tonemapperWidget(m_tonemapperData);
      }

      ImGui::End();
      if(changed)
        resetFrame();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // If double-clicking in the "Viewport", shoot a ray to the scene under the mouse.
      // If the ray hit something, set the camera center to the hit position.
      if(ImGui::IsWindowHovered(ImGuiFocusedFlags_RootWindow) && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
      {
        nvutils::ScopedTimer st("RayPicker");
        VkCommandBuffer      cmd = m_app->createTempCmdBuffer();
        // Convert screen coordinates to normalized viewport coordinates [0,1]
        ImVec2 localMousePos = (ImGui::GetMousePos() - ImGui::GetCursorScreenPos()) / ImGui::GetContentRegionAvail();

        m_rayPicker.run(cmd, {.modelViewInv   = glm::inverse(g_cameraManip->getViewMatrix()),
                              .perspectiveInv = glm::inverse(g_cameraManip->getPerspectiveMatrix()),
                              .pickPos        = {localMousePos.x, localMousePos.y},
                              .tlas           = m_sceneRtx.tlas()});
        m_app->submitAndWaitTempCmdBuffer(cmd);
        nvvk::RayPicker ::PickResult pickResult = m_rayPicker.getResult();
        if(pickResult.instanceID > -1)  // Hit something
        {
          // Set the camera CENTER to the hit position
          glm::vec3  worldPos = pickResult.worldRayOrigin + pickResult.worldRayDirection * pickResult.hitT;
          glm::dvec3 eye, center, up;
          g_cameraManip->getLookat(eye, center, up);
          g_cameraManip->setLookat(eye, worldPos, up, false);  // Nice with CameraManip.updateAnim();
        }
      }

      // Display the tonemapped image
      ImGui::Image(m_viewportImage, ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

    if(!updateFrame())
    {
      return;
    }

    if(!m_scene.valid())
    {
      return;
    }

    raytrace(cmd);
    tonemap(cmd);
  }
  void raytrace(VkCommandBuffer cmd)
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

    // Update Camera uniform buffer
    shaderio::CameraInfo finfo{
        .projInv = glm::inverse(g_cameraManip->getPerspectiveMatrix()),
        .viewInv = glm::inverse(g_cameraManip->getViewMatrix()),
    };
    vkCmdUpdateBuffer(cmd, m_bCameraInfo.buffer, 0, sizeof(shaderio::CameraInfo), &finfo);
    vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(shaderio::SkyPhysicalParameters), &m_skyParams);  // Update the sky

    // Update the push constant: the camera information, sky parameters and the scene to render.
    // (The heap region bases / sampler index were set once in createDescriptorHeap.)
    m_pushConst.frame      = m_frame;
    m_pushConst.cameraInfo = (shaderio::CameraInfo*)m_bCameraInfo.address;
    m_pushConst.skyParams  = (shaderio::SkyPhysicalParameters*)m_bSkyParams.address;
    m_pushConst.gltfScene  = (shaderio::GltfScene*)m_sceneVk.sceneDesc().address;
    m_pushConst.mouseCoord = nvapp::ElementDbgPrintf::getMouseCoord();  // Use for debugging: printf in shader

    // Make sure buffer is ready to be used
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Bind the shader to use
    VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
    vkCmdBindShadersEXT(cmd, 1, &stage, &m_shader);

    // Bind the bindless heaps (sampler + resource); the shader resolves everything via .Handle()
    m_heap.cmdBindHeaps(cmd, m_samplerHeapBuffer.address, m_resourceHeapBuffer.address);

    // Push the constant data straight from SPIR-V layout (no pipeline layout with descriptor heap)
    VkPushDataInfoEXT pushInfo{.sType  = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
                               .offset = 0,
                               .data   = {.address = &m_pushConst, .size = sizeof(shaderio::PushConstant)}};
    vkCmdPushDataEXT(cmd, &pushInfo);

    // Dispatch the raytracing shader
    const VkExtent2D& size      = m_app->getViewportSize();
    VkExtent2D        numGroups = nvvk::getGroupCounts(size, WORKGROUP_SIZE);
    vkCmdDispatch(cmd, numGroups.width, numGroups.height, 1);

    // Making sure the rendered image is ready to be used by tonemapper
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  }

  void tonemap(VkCommandBuffer cmd)
  {
    // Input is sampled (COMBINED_IMAGE_SAMPLER) so it needs a sampler; output is a storage image.
    m_tonemapper.runCompute(cmd, m_renderTarget.getSize(), m_tonemapperData,
                            m_renderTarget.getColorSampleDescriptorImageInfo(eImgRendered, m_linearSampler),
                            m_renderTarget.getColorSampleDescriptorImageInfo(eImgTonemapped));
  }

  void onUIMenu() override
  {
    bool reloadShader = false;
    if(ImGui::BeginMenu("Tools"))
    {
      reloadShader = ImGui::MenuItem("Reload Shaders");
      ImGui::EndMenu();
    }
    if(ImGui::IsKeyPressed(ImGuiKey_F5) || reloadShader)
    {
      vkQueueWaitIdle(m_app->getQueue(0).queue);
      compileShader();
      resetFrame();
    }
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_renderTarget.getColorImage(eImgTonemapped), m_renderTarget.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  void onFileDrop(const std::filesystem::path& filename) override
  {
    vkQueueWaitIdle(m_app->getQueue(0).queue);
    if(nvutils::extensionMatches(filename, ".gltf") || nvutils::extensionMatches(filename, ".glb"))
    {
      g_sceneFilename = filename;
      createScene();
      updateTextures();
      updateSamplers();
    }
    else if(nvutils::extensionMatches(filename, ".hdr"))
    {
      g_hdrFilename = filename;
      createHDR();
    }

    resetFrame();
  }

private:
  void createScene()
  {
    nvutils::ScopedTimer st(std::string(__FUNCTION__) + "\n");

    std::filesystem::path filename = nvutils::findFile(g_sceneFilename, nvsamples::getResourcesDirs());
    LOGI("%sLoading scene: %s\n", st.indent().c_str(), nvutils::utf8FromPath(filename).c_str());
    if(!m_scene.load(filename))  // Loading the scene
    {
      LOGE("%sError loading scene: %s\n", st.indent().c_str(), nvutils::utf8FromPath(filename).c_str());
      // Clear the displayed image
      VkCommandBuffer         cmd        = m_app->createTempCmdBuffer();
      const VkClearColorValue clearValue = {{0.F, 0.F, 0.F, 0.F}};
      VkImageSubresourceRange range      = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
      vkCmdClearColorImage(cmd, m_renderTarget.getColorImage(eImgTonemapped), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &range);
      m_app->submitAndWaitTempCmdBuffer(cmd);  // Submit and wait for the command buffer
      return;
    }
    {
      nvvk::StagingUploader staging;
      staging.init(&m_allocator, true);

      // Create the scene in Vulkan buffers
      {
        VkCommandBuffer cmd = m_app->createTempCmdBuffer();

        m_sceneVk.create(cmd, staging, m_scene, false);  // Creating the scene in Vulkan buffers
        staging.cmdUploadAppended(cmd);
        m_app->submitAndWaitTempCmdBuffer(cmd);  // Submit and wait for the command buffer
      }

      // Create the bottom-level acceleration structures
      m_sceneRtx.createBottomLevelAccelerationStructure(m_scene, m_sceneVk,
                                                        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                            | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR);

      // Build the bottom-level acceleration structures and compact them
      // It is done in a loop, using the BLAS size hint to create the scratch buffer
      // First build the BLAS then the compaction is done which allow to use less memory
      // The loop is done until all the BLAS are built (within budget)
      bool finished = false;
      do
      {
        // First step is to build the BLAS
        VkCommandBuffer cmd = m_app->createTempCmdBuffer();
        finished            = m_sceneRtx.cmdBuildBottomLevelAccelerationStructure(cmd, 512'000'000);
        m_app->submitAndWaitTempCmdBuffer(cmd);  // Submit and wait for the command buffer

        // Second step is to compact the BLAS
        // Note that BLAS build must be finished before compacting, which explains the two steps
        cmd = m_app->createTempCmdBuffer();
        m_sceneRtx.cmdCompactBlas(cmd);          // Compact the BLAS
        m_app->submitAndWaitTempCmdBuffer(cmd);  // Submit and wait for the command buffer
      } while(!finished);

      // Create the top-level acceleration structure
      {
        VkCommandBuffer cmd = m_app->createTempCmdBuffer();
        m_sceneRtx.cmdCreateBuildTopLevelAccelerationStructure(cmd, staging, m_scene);
        staging.cmdUploadAppended(cmd);
        m_app->submitAndWaitTempCmdBuffer(cmd);  // Submit and wait for the command buffer
      }


      staging.deinit();
    }

    // The TLAS was (re)built: refresh its device address used by the shader.
    m_pushConst.tlasAddress = m_sceneRtx.tlasAccel().address;

    nvvkgltf::addSceneCamerasToWidget(g_cameraManip, filename, m_scene.getRenderCameras(), m_scene.getSceneBounds());  // Set camera from scene

    // Default parameters for overall material
    m_pushConst.maxDepth              = 5;
    m_pushConst.frame                 = 0;
    m_pushConst.fireflyClampThreshold = 1;
    m_pushConst.maxSamples            = 2;

    // Default sky parameters
    m_skyParams = {};
  }

  // Create all Vulkan buffer data
  void createVkBuffers()
  {
    // Create the buffer of the current camera transformation, changing at each frame
    NVVK_CHECK(m_allocator.createBuffer(m_bCameraInfo, sizeof(shaderio::CameraInfo),
                                        VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                        VMA_MEMORY_USAGE_CPU_TO_GPU));
    NVVK_DBG_NAME(m_bCameraInfo.buffer);
    // Create the buffer of sky parameters, updated at each frame
    NVVK_CHECK(m_allocator.createBuffer(m_bSkyParams, sizeof(shaderio::SkyPhysicalParameters),
                                        VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
                                        VMA_MEMORY_USAGE_CPU_TO_GPU));
    NVVK_DBG_NAME(m_bSkyParams.buffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Creating the bindless descriptor heap (VK_EXT_descriptor_heap). Both heaps are host-visible and
  // persistently mapped (Method B), so descriptors are written directly into the mapping:
  // - The sampler heap holds a default sampler (slot 0) plus one slot per glTF scene sampler.
  // - The resource heap is sized for (outImage + HDR + kMaxTextures) images and 1 buffer (the env
  //   importance-sampling table).
  // Only the region bases + sampler index are stored in the push constant here.
  //
  void createDescriptorHeap()
  {
    nvutils::ScopedTimer st(__FUNCTION__);

    NVVK_CHECK(m_heap.init(m_app->getPhysicalDevice(), m_app->getDevice()));

    constexpr VkBufferUsageFlags2 heapUsage = nvvk::DescriptorHeap::getRequiredBufferUsage();
    const VkDeviceSize samplerBufSize  = m_heap.setupSamplerHeap(shaderio::kHeapSmpSceneStart + shaderio::kMaxSamplers);
    const VkDeviceSize resourceBufSize = m_heap.setupResourceHeap(kImageHeapSize, kBufferHeapSize);

    // Both heaps are host-visible and persistently mapped (Method B): descriptors are written
    // directly into the mapping, so no staging upload or command submission is needed.
    constexpr VmaAllocationCreateFlags kMappedFlags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    NVVK_CHECK(m_allocator.createBuffer(m_samplerHeapBuffer, samplerBufSize, heapUsage, VMA_MEMORY_USAGE_AUTO,
                                        kMappedFlags, m_heap.getSamplerHeapAlignment()));
    NVVK_DBG_NAME(m_samplerHeapBuffer.buffer);
    NVVK_CHECK(m_allocator.createBuffer(m_resourceHeapBuffer, resourceBufSize, heapUsage, VMA_MEMORY_USAGE_AUTO,
                                        kMappedFlags, m_heap.getResourceHeapAlignment()));
    NVVK_DBG_NAME(m_resourceHeapBuffer.buffer);

    // Default sampler (slot kHeapSmpDefault): linear, repeat wrapping. Used by the HDR/env map, the
    // output image, and any glTF texture without an explicit sampler. Per-scene samplers are written
    // at slots kHeapSmpSceneStart + i by updateSamplers() when a scene is loaded.
    VkSamplerCreateInfo samplerCI{
        .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter    = VK_FILTER_LINEAR,
        .minFilter    = VK_FILTER_LINEAR,
        .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .maxLod       = VK_LOD_CLAMP_NONE,
    };
    NVVK_CHECK(m_heap.writeSamplerDescriptor(shaderio::kHeapSmpDefault, samplerCI, m_samplerHeapBuffer.mapping));

    // Push only the region bases; the shader adds the shared slot constants (shaderio::kHeap*).
    // The sampler heap is 0-based, so a sampler slot index is already its shader handle.
    m_pushConst.imageHeapBase   = m_heap.imageShaderIndexBase();
    m_pushConst.bufferHeapBase  = m_heap.bufferShaderIndexBase();
    m_pushConst.samplerHeapBase = 0;

    // The render-target image already exists (G-Buffer created in onAttach); write its slot.
    writeOutputImageDescriptor();
  }

  // Write/refresh the path-trace output (storage image) descriptor in the resource heap.
  void writeOutputImageDescriptor()
  {
    if(m_resourceHeapBuffer.mapping == nullptr)
      return;
    // Render target color image is created in onResize; skip until it exists.
    if(m_renderTarget.getSize().width == 0 || m_renderTarget.getSize().height == 0)
      return;
    NVVK_CHECK(m_heap.writeStorageImageDescriptor(shaderio::kHeapImgOutput, m_renderTarget.getColorImage(eImgRendered),
                                                  m_renderTarget.getColorFormat(eImgRendered), VK_IMAGE_LAYOUT_GENERAL,
                                                  m_resourceHeapBuffer.mapping));
  }

  void compileShader()
  {
    nvutils::ScopedTimer st(__FUNCTION__);

    // Bindless: the shader sources its descriptors from the bound heap, so there are no descriptor
    // set layouts and no push-constant range here (VUID-VkShaderCreateInfoEXT-flags-11290/11293).
    // The push-constant layout is read directly from the SPIR-V.
    VkShaderCreateInfoEXT shaderInfo = nvsamples::makeShaderCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, 0, gltf_pathtrace_slang,
                                                                       "computeMain", VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT);
    if(m_slangCompiler.compileFile("gltf_pathtrace.slang"))
    {
      shaderInfo.codeSize = m_slangCompiler.getSpirvSize();
      shaderInfo.pCode    = m_slangCompiler.getSpirv();
    }
    else
    {
      LOGE("Error compiling gltf_pathtrace.slang\n");
    }
    vkDestroyShaderEXT(m_device, m_shader, nullptr);
    NVVK_CHECK(vkCreateShadersEXT(m_app->getDevice(), 1U, &shaderInfo, nullptr, &m_shader));
    NVVK_DBG_NAME(m_shader);
  }

  // Write all scene textures into the resource heap (sampled images), starting at kHeapImgTexturesStart.
  void updateTextures()
  {
    const uint32_t numTextures = m_sceneVk.nbTextures();
    assert(numTextures <= kMaxTextures && "Increase kMaxTextures to fit this scene");
    for(uint32_t i = 0; i < numTextures; i++)
    {
      NVVK_CHECK(m_heap.writeSampledImageDescriptor(shaderio::kHeapImgTexturesStart + i, m_sceneVk.textures()[i],
                                             m_resourceHeapBuffer.mapping));
    }
  }

  // Write the scene's glTF samplers into the sampler heap, starting at kHeapSmpSceneStart. glTF
  // sampler `i` lands at slot kHeapSmpSceneStart + i, so the shader maps a texture's samplerIndex
  // straight to a heap slot (see getSceneSampler in gltf_pathtrace.slang). Textures with no explicit
  // sampler (samplerIndex < 0) fall back to the default sampler at kHeapSmpDefault.
  void updateSamplers()
  {
    const tinygltf::Model& model       = m_scene.getModel();
    const uint32_t         numSamplers = static_cast<uint32_t>(model.samplers.size());
    assert(numSamplers <= shaderio::kMaxSamplers && "Increase kMaxSamplers to fit this scene");
    const uint32_t count = std::min(numSamplers, shaderio::kMaxSamplers);
    for(uint32_t i = 0; i < count; i++)
    {
      VkSamplerCreateInfo samplerCI = nvvkgltf::getVkSamplerCreateInfo(model, static_cast<int>(i));
      NVVK_CHECK(m_heap.writeSamplerDescriptor(shaderio::kHeapSmpSceneStart + i, samplerCI, m_samplerHeapBuffer.mapping));
    }
  }


  //--------------------------------------------------------------------------------------------------
  // To be call when renderer need to re-start
  //
  void resetFrame() { m_frame = -1; }

  //--------------------------------------------------------------------------------------------------
  // If the camera matrix has changed, resets the frame.
  // otherwise, increments frame.
  //
  bool updateFrame()
  {
    static double     ref_fov{0};
    static glm::dmat4 ref_cam_matrix;

    const auto& m   = g_cameraManip->getViewMatrix();
    const auto  fov = g_cameraManip->getFov();

    if(ref_cam_matrix != m || ref_fov != fov)
    {
      resetFrame();
      ref_cam_matrix = m;
      ref_fov        = fov;
    }

    if(m_frame >= m_maxFrames)
    {
      return false;
    }
    m_frame++;
    return true;
  }

  // Loading the HDR
  void createHDR()
  {
    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_allocator, true);

    std::filesystem::path filename = nvutils::findFile(g_hdrFilename, nvsamples::getResourcesDirs());
    m_hdrIbl.destroyEnvironment();
    m_hdrIbl.loadEnvironment(cmd, uploader, filename);

    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();

    // Expose the HDR environment (sampled image) and its importance-sampling alias table (SSBO)
    // through the descriptor heap.
    NVVK_CHECK(m_heap.writeSampledImageDescriptor(shaderio::kHeapImgHdr, m_hdrIbl.getHdrImage(), m_resourceHeapBuffer.mapping));
    NVVK_CHECK(m_heap.writeBufferDescriptor(shaderio::kHeapBufEnvSampling, m_hdrIbl.getEnvAccel(),
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, m_resourceHeapBuffer.mapping));
  }


  void destroyResources()
  {
    m_allocator.destroyBuffer(m_bCameraInfo);
    m_allocator.destroyBuffer(m_bSkyParams);

    vkDestroyShaderEXT(m_device, m_shader, nullptr);

    // Descriptor heap teardown (m_heap.deinit() destroys all samplers written into the heap)
    m_allocator.destroyBuffer(m_samplerHeapBuffer);
    m_allocator.destroyBuffer(m_resourceHeapBuffer);
    m_heap.deinit();

    m_tonemapper.deinit();
    m_viewportImage.deinit();
    m_renderTarget.deinit();
    m_sceneVk.deinit();
    m_sceneRtx.deinit();
    m_hdrIbl.deinit();
    m_rayPicker.deinit();
    m_allocator.deinit();
    m_samplerPool.deinit();
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvapp::Application*     m_app{};
  nvvk::ResourceAllocator m_allocator{};
  nvvk::SamplerPool       m_samplerPool{};
  nvvk::RayPicker         m_rayPicker{};


  nvslang::SlangCompiler m_slangCompiler{};
  VkShaderEXT            m_shader{};

  VkDevice m_device{};  // Convenient


  nvvkgltf::Scene    m_scene;     // GLTF Scene
  nvvkgltf::SceneVk  m_sceneVk;   // GLTF Scene buffers
  nvvkgltf::SceneRtx m_sceneRtx;  // GLTF Scene BLAS/TLAS

  nvvk::HdrIbl m_hdrIbl;

  // Resources
  nvvk::RenderTarget m_renderTarget;     // Offscreen target: rendered (HDR) + tonemapped color images
  nvapp::ImTexture   m_viewportImage;    // ImGui texture displaying the tonemapped image
  VkSampler          m_linearSampler{};  // Sampler used to feed the rendered image to the tonemapper
  nvvk::Buffer       m_bCameraInfo;      // Camera information
  nvvk::Buffer       m_bSkyParams;       // Sky parameters

  // Data and setting
  shaderio::SkyPhysicalParameters m_skyParams = {};
  nvshaders::Tonemapper           m_tonemapper{};
  shaderio::TonemapperData        m_tonemapperData;

  // Bindless descriptor heap (replaces descriptor set layouts + push-constant ranges)
  nvvk::DescriptorHeap m_heap{};
  nvvk::Buffer         m_samplerHeapBuffer{};   // host-visible sampler heap (written directly)
  nvvk::Buffer         m_resourceHeapBuffer{};  // host-visible resource heap (written directly)

  shaderio::PushConstant m_pushConst{};  // Information sent to the shader
  int                    m_frame{0};
  int                    m_maxFrames{10000};
};

//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char** argv) -> int
{
  // Flush every log line to the file so a crash doesn't lose the most recent (and most relevant) output.
  nvutils::Logger::getInstance().setFileFlush(true);

  nvapp::ApplicationCreateInfo appInfo;

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;

  reg.add({"modelfile", "Input filename"}, {".gltf"}, &g_sceneFilename);
  reg.addVector({"size", "Size of the window to be created", "s"}, &appInfo.windowSize);
  reg.add({"headless"}, &appInfo.headless, true);
  reg.add({"frames", "Number of frames to run in headless mode"}, &appInfo.headlessFrameCount);
  reg.add({"vsync"}, &appInfo.vSync);
  cli.add(reg);
  cli.parse(argc, argv);

  // Extension feature needed.
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR computeDerivativesFeature{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR};
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  // Bindless descriptor heap (resources/samplers accessed via .Handle in the shader) and the
  // untyped pointers it relies on. Push descriptors are no longer needed.
  VkPhysicalDeviceDescriptorHeapFeaturesEXT heapFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT};
  VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untypedPtrFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR};

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},  // still used by the tonemapper compute pass
                             {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
                             {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature},
                             {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature},
                             {VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature},
                             {VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, &computeDerivativesFeature},
                             {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures},
                             {VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, &heapFeatures},
                             {VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME, &untypedPtrFeatures}},

      .queues = {VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_COMPUTE_BIT},
  };
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

#if defined(USE_NSIGHT_AFTERMATH)
  // Adding the Aftermath extension to the device and initialize the Aftermath
  auto& aftermath = AftermathCrashTracker::getInstance();
  aftermath.initialize();
  aftermath.addExtensions(vkSetup.deviceExtensions);
  // The callback function is called when a validation error is triggered. This will wait to give time to dump the GPU crash.
  nvvk::CheckError::getInstance().setCallbackFunction([&](VkResult result) { aftermath.errorCallback(result); });
#endif

  nvvk::ValidationSettings validation{};
  // validation.setDebugPrintf();
  vkSetup.instanceCreateInfoExt = validation.buildPNextChain();

  // Create the Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  // Application information
  appInfo.name           = fmt::format("{} ({})", nvutils::getExecutablePath().stem().string(), "Slang");
  appInfo.vSync          = false;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();


  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // Add all application elements
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  elemCamera->setCameraManipulator(g_cameraManip);
  app.addElement(elemCamera);
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());                         // Menu / Quit
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", appInfo.name));  // Window title info
  app.addElement(std::make_shared<GltfRaytrace>());

  app.run();
  app.deinit();
  vkContext.deinit();
}
