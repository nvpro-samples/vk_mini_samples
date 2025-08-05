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

//////////////////////////////////////////////////////////////////////////
/*

    This shows the rendering of a GLTF scene.
    It uses the ray tracing extension to render the scene.
    It also used many of the helper classes to create the scene, the ray tracing structures and the rendering.

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


#include <array>
#include <string>
#include <vector>

#include <fmt/format.h>
#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>

#include "GLFW/glfw3.h"
#undef APIENTRY

// Shader Input/Output
namespace shaderio {
using namespace glm;
#include "shaders/shaderio.h"  // Shared between host and device
}  // namespace shaderio

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
#include <nvvk/descriptors.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/hdr_ibl.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/ray_picker.hpp>
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

public:
  GltfRaytrace()           = default;
  ~GltfRaytrace() override = default;

  void onAttach(nvapp::Application* app) override
  {
    VkDevice         device         = app->getDevice();
    VkPhysicalDevice physicalDevice = app->getPhysicalDevice();

    m_app                        = app;
    m_device                     = device;
    const uint32_t c_queue_index = app->getQueue(1).familyIndex;

    // Create the Vulkan allocator (VMA)
    m_allocator.init({
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });  // Allocator

    // The texture sampler to use
    m_samplerPool.init(m_device);
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // IBL environment map
    m_hdrIbl.init(&m_allocator, &m_samplerPool);

    // G-Buffer
    m_gBuffers.init({.allocator      = &m_allocator,
                     .colorFormats   = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32G32B32A32_SFLOAT},
                     .imageSampler   = linearSampler,
                     .descriptorPool = m_app->getTextureDescriptorPool()});
    {
      VkCommandBuffer cmd = app->createTempCmdBuffer();
      m_gBuffers.update(cmd, {100, 100});
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

#if defined(AFTERMATH_AVAILABLE)
      // This aftermath callback is used to report the shader hash (Spirv) to the Aftermath library.
      m_slangCompiler.setCompileCallback([&](const std::filesystem::path& sourceFile, const uint32_t* spirvCode, size_t spirvSize) {
        std::span<const uint32_t> data(spirvCode, spirvSize / sizeof(uint32_t));
        AftermathCrashTracker::getInstance().addShaderBinary(data);
      });
#endif
    }

    // Create resources
    createHDR();
    createCompPipelines();
    createScene();
    createVkBuffers();
    compileShader();
    updateTextures();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    m_gBuffers.update(cmd, size);
    resetFrame();  // Reset frame to restart the rendering
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
          glm::vec3 worldPos = pickResult.worldRayOrigin + pickResult.worldRayDirection * pickResult.hitT;
          glm::vec3 eye, center, up;
          g_cameraManip->getLookat(eye, center, up);
          g_cameraManip->setLookat(eye, worldPos, up, false);  // Nice with CameraManip.updateAnim();
        }
      }

      // Display the G-Buffer tonemapped image
      ImGui::Image(ImTextureID(m_gBuffers.getDescriptorSet(eImgTonemapped)), ImGui::GetContentRegionAvail());

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

    // Update the push constant: the camera information, sky parameters and the scene to render
    m_pushConst.frame      = m_frame;
    m_pushConst.cameraInfo = (shaderio::CameraInfo*)m_bCameraInfo.address;
    m_pushConst.skyParams  = (shaderio::SkyPhysicalParameters*)m_bSkyParams.address;
    m_pushConst.gltfScene  = (shaderio::GltfScene*)m_sceneVk.sceneDesc().address;
    m_pushConst.mouseCoord = nvapp::ElementDbgPrintf::getMouseCoord();  // Use for debugging: printf in shader
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);

    // Make sure buffer is ready to be used
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Bind the shader to use
    VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
    vkCmdBindShadersEXT(cmd, 1, &stage, &m_shader);

    // Bind the descriptor set: TLAS, output image, textures, etc. (Set: 0)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descriptorPack[0].sets[0], 0, nullptr);

    // Set the Descriptor for HDR (Set: 2)
    VkDescriptorSet hdrDescSet = m_hdrIbl.getDescriptorSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 2, 1, &hdrDescSet, 0, nullptr);

    pushDescriptorSet(cmd);

    // Dispatch the raytracing shader
    const VkExtent2D& size      = m_app->getViewportSize();
    VkExtent2D        numGroups = nvvk::getGroupCounts(size, WORKGROUP_SIZE);
    vkCmdDispatch(cmd, numGroups.width, numGroups.height, 1);

    // Making sure the rendered image is ready to be used by tonemapper
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
  }

  void tonemap(VkCommandBuffer cmd)
  {
    m_tonemapper.runCompute(cmd, m_gBuffers.getSize(), m_tonemapperData, m_gBuffers.getDescriptorImageInfo(eImgRendered),
                            m_gBuffers.getDescriptorImageInfo(eImgTonemapped));
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
    m_app->saveImageToFile(m_gBuffers.getColorImage(eImgTonemapped), m_gBuffers.getSize(),
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
      // Clear the GBuffer
      VkCommandBuffer         cmd        = m_app->createTempCmdBuffer();
      const VkClearColorValue clearValue = {{0.F, 0.F, 0.F, 0.F}};
      VkImageSubresourceRange range      = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
      vkCmdClearColorImage(cmd, m_gBuffers.getColorImage(eImgTonemapped), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &range);
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
        m_app->submitAndWaitTempCmdBuffer(cmd);  // Submit and wait for the command buffer/nvpro_core/nvvk/shaders/compile.bat
      }


      staging.deinit();
    }

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
  // Creating the pipeline: shader ...
  //
  void createCompPipelines()
  {
    // Reserve 500 textures
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(m_app->getPhysicalDevice(), &deviceProperties);
    uint32_t maxTextures = std::min(500U, deviceProperties.limits.maxDescriptorSetSampledImages - 1);

    // 0: Descriptor SET: all textures
    m_descriptorPack[0].bindings.addBinding(B_textures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, maxTextures,
                                            VK_SHADER_STAGE_ALL, nullptr,
                                            VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                                                | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
    NVVK_CHECK(m_descriptorPack[0].initFromBindings(m_device, 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                                    VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT |  // allows descriptor sets to be updated after they have been bound to a command buffer
                                                        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT));  // individual descriptor sets can be freed from the descriptor pool
    NVVK_DBG_NAME(m_descriptorPack[0].layout);
    NVVK_DBG_NAME(m_descriptorPack[0].pool);
    NVVK_DBG_NAME(m_descriptorPack[0].sets[0]);

    // 1: Descriptor PUSH: top level acceleration structure and the output image
    m_descriptorPack[1].bindings.addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_descriptorPack[1].bindings.addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    NVVK_CHECK(m_descriptorPack[1].initFromBindings(m_device, 0,  // 0 == Don't allocate pool or sets
                                                    VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR));
    NVVK_DBG_NAME(m_descriptorPack[1].layout);

    // Creating the pipeline layout
    const VkPushConstantRange pushConstant{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(shaderio::PushConstant)};
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout,
                                          {m_descriptorPack[0].layout, m_descriptorPack[1].layout, m_hdrIbl.getDescriptorSetLayout()},  //
                                          {pushConstant}));
    NVVK_DBG_NAME(m_pipelineLayout);
  }

  void compileShader()
  {
    nvutils::ScopedTimer st(__FUNCTION__);

    VkPushConstantRange pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant)};

    std::array<VkDescriptorSetLayout, 3> descriptorSetLayouts{m_descriptorPack[0].layout, m_descriptorPack[1].layout,
                                                              m_hdrIbl.getDescriptorSetLayout()};

    VkShaderCreateInfoEXT shaderInfo{
        .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .stage                  = VK_SHADER_STAGE_COMPUTE_BIT,
        .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
        .codeSize               = gltf_pathtrace_slang_sizeInBytes,
        .pCode                  = gltf_pathtrace_slang,
        .pName                  = "computeMain",
        .setLayoutCount         = uint32_t(descriptorSetLayouts.size()),
        .pSetLayouts            = descriptorSetLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstant,
    };
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


  // Pushing the information to the shader: acceleration structure and output image
  void pushDescriptorSet(VkCommandBuffer cmd)
  {
    nvvk::WriteSetContainer write{};
    write.append(m_descriptorPack[1].bindings.getWriteSet(B_tlas), m_sceneRtx.tlas());
    write.append(m_descriptorPack[1].bindings.getWriteSet(B_outImage), m_gBuffers.getColorImageView(eImgRendered),
                 VK_IMAGE_LAYOUT_GENERAL);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 1, write.size(), write.data());
  }

  // Updating all textures in the descriptor set
  void updateTextures()
  {
    // Now do the textures
    nvvk::WriteSetContainer write{};
    VkWriteDescriptorSet    allTextures = m_descriptorPack[0].bindings.getWriteSet(B_textures);
    allTextures.dstSet                  = m_descriptorPack[0].sets[0];
    allTextures.descriptorCount         = m_sceneVk.nbTextures();
    if(allTextures.descriptorCount == 0)
      return;
    write.append(allTextures, m_sceneVk.textures().data());
    vkUpdateDescriptorSets(m_device, write.size(), write.data(), 0, nullptr);
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
    static float     ref_fov{0};
    static glm::mat4 ref_cam_matrix;

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
  }


  void destroyResources()
  {
    m_allocator.destroyBuffer(m_bCameraInfo);
    m_allocator.destroyBuffer(m_bSkyParams);

    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    m_descriptorPack[0].deinit();
    m_descriptorPack[1].deinit();
    vkDestroyShaderEXT(m_device, m_shader, nullptr);

    m_tonemapper.deinit();
    m_gBuffers.deinit();
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
  nvvk::GBuffer m_gBuffers;     // G-Buffers: color + depth
  nvvk::Buffer  m_bCameraInfo;  // Camera information
  nvvk::Buffer  m_bSkyParams;   // Sky parameters

  // Data and setting
  shaderio::SkyPhysicalParameters m_skyParams = {};
  nvshaders::Tonemapper           m_tonemapper{};
  shaderio::TonemapperData        m_tonemapperData;

  // Pipeline
  nvvk::DescriptorPack m_descriptorPack[2]{};  // 0 = textures, 1 = push-only
  VkPipeline           m_pipeline{};
  VkPipelineLayout     m_pipelineLayout{};

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

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
                             {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
                             {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
                             {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature},
                             {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature},
                             {VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature},
                             {VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, &computeDerivativesFeature},
                             {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures}},

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
