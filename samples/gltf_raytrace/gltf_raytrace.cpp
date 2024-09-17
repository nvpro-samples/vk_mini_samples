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

//////////////////////////////////////////////////////////////////////////
/*

    This shows the rendering of a GLTF scene.
    It uses the ray tracing extension to render the scene.
    It also used many of the helper classes to create the scene, the ray tracing structures and the rendering.

*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <vulkan/vulkan_core.h>


#ifdef _MSC_VER
#pragma warning(disable : 4018)  // signed/unsigned mismatch (tinygltf)
#pragma warning(disable : 4267)  // conversion from 'size_t' to 'uint32_t', possible loss of data (tinygltf)
#endif


#include "imgui/imgui_helper.h"
#include "nvh/gltfscene.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/gltf_scene_rtx.hpp"
#include "nvvkhl/scene_camera.hpp"
#include "nvvkhl/sky.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"
#include "nvvk/extensions_vk.hpp"

#include "shaders/dh_bindings.h"
#include "common/utils.hpp"
#include "common/vk_context.hpp"
#include "common/alloc_dma.hpp"

// The defines must be done here, to avoid having multiple definitions
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>


namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

#if USE_HLSL
#include "_autogen/gltf_pathtrace_computeMain.spirv.h"
const auto& comp_shd = std::vector<char>{std::begin(gltf_pathtrace_computeMain), std::end(gltf_pathtrace_computeMain)};
#elif USE_SLANG
#include "_autogen/gltf_pathtrace_slang.h"
const auto& comp_shd = std::vector<uint32_t>{std::begin(gltf_pathtraceSlang), std::end(gltf_pathtraceSlang)};
#else
#include "_autogen/gltf_pathtrace.comp.glsl.h"
const auto& comp_shd = std::vector<uint32_t>{std::begin(gltf_pathtrace_comp_glsl), std::end(gltf_pathtrace_comp_glsl)};
#endif


#define GROUP_SIZE 16  // Same group size as in compute shader


std::string g_sceneFilename = "shader_ball.gltf";  // Default scene

/// </summary> Ray trace multiple primitives using Ray Query
class RayQuery : public nvvkhl::IAppElement
{
  enum
  {
    eImgTonemapped,
    eImgRendered
  };

public:
  RayQuery()           = default;
  ~RayQuery() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    VkDevice         device         = app->getDevice();
    VkPhysicalDevice physicalDevice = app->getPhysicalDevice();

    m_app                        = app;
    m_device                     = device;
    const uint32_t c_queue_index = app->getQueue(1).familyIndex;

    m_dutil      = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility
    m_alloc      = std::make_unique<AllocDma>(device, physicalDevice);
    m_rtSet      = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_tonemapper = std::make_unique<nvvkhl::TonemapperPostProcess>(device, m_alloc.get());
    m_scene      = std::make_unique<nvh::gltf::Scene>();                                      // GLTF scene
    m_sceneVk    = std::make_unique<nvvkhl::SceneVk>(device, physicalDevice, m_alloc.get());  // GLTF Scene buffers
    m_sceneRtx = std::make_unique<nvvkhl::SceneRtx>(device, physicalDevice, m_alloc.get(), c_queue_index);  // GLTF Scene BLAS/TLAS

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    int32_t gctQueueIndex = m_app->getQueue(0).familyIndex;

    // Create resources
    createScene();
    createVkBuffers();
    createCompPipelines();

    m_tonemapper->createComputePipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    // Create two G-Buffers: the tonemapped image and the original rendered image
    std::vector<VkFormat> color_buffers = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32G32B32A32_SFLOAT};  // tonemapped, original
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), VkExtent2D{width, height}, color_buffers);

    // Update the tonemapper with the information of the new G-Buffers
    m_tonemapper->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(eImgRendered),
                                              m_gBuffers->getDescriptorImageInfo(eImgTonemapped));
    resetFrame();  // Reset frame to restart the rendering
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");

      ImGuiH::CameraWidget();

      using namespace ImGuiH;
      namespace PE = PropertyEditor;
      bool changed{false};
      if(ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
      {
        PropertyEditor::begin();
        if(PropertyEditor::treeNode("Ray Tracer"))
        {
          changed |= PropertyEditor::entry("Depth", [&] { return ImGui::SliderInt("#1", &m_pushConst.maxDepth, 0, 20); });
          changed |=
              PropertyEditor::entry("Samples", [&] { return ImGui::SliderInt("#1", &m_pushConst.maxSamples, 1, 10); });
          PropertyEditor::treePop();
        }
        if(PropertyEditor::treeNode("Sky"))
        {
          changed |= nvvkhl::physicalSkyUI(m_skyParams);
          PropertyEditor::treePop();
        }
        PropertyEditor::end();
      }

      if(ImGui::CollapsingHeader("Tonemapper"))
      {
        changed |= m_tonemapper->onUI();
      }

      ImGui::End();
      if(changed)
        resetFrame();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer tonemapped image
      ImGui::Image(m_gBuffers->getDescriptorSet(eImgTonemapped), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    auto sdbg = m_dutil->DBG_SCOPE(cmd);

    if(!updateFrame())
    {
      return;
    }

    // Update Camera uniform buffer
    const auto& clip = CameraManip.getClipPlanes();
    glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), CameraManip.getAspectRatio(), clip.x, clip.y);
    proj[1][1] *= -1;
    DH::CameraInfo finfo{.projInv = glm::inverse(proj), .viewInv = glm::inverse(CameraManip.getMatrix())};
    vkCmdUpdateBuffer(cmd, m_bCameraInfo.buffer, 0, sizeof(DH::CameraInfo), &finfo);
    vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(nvvkhl_shaders::PhysicalSkyParameters), &m_skyParams);  // Update the sky

    m_pushConst.frame = m_frame;

    // Make sure buffer is ready to be used
    memoryBarrier(cmd);

    // Ray trace
    const VkExtent2D& size = m_app->getViewportSize();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipe.plines[0]);
    pushDescriptorSet(cmd);
    vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant), &m_pushConst);
    vkCmdDispatch(cmd, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);

    // Making sure the rendered image is ready to be used by tonemapper
    memoryBarrier(cmd);

    m_tonemapper->runCompute(cmd, size);
  }

private:
  void createScene()
  {
    std::string filename = getFilePath(g_sceneFilename, getMediaDirs());
    if(!m_scene->load(filename))  // Loading the scene
    {
      LOGE("Error loading scene: %s\n", filename.c_str());
      exit(1);
    }
    {
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      m_sceneVk->create(cmd, *m_scene);  // Creating the scene in Vulkan buffers
      m_sceneRtx->create(cmd, *m_scene, *m_sceneVk, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);  // Creating the acceleration structures
      m_app->submitAndWaitTempCmdBuffer(cmd);  // Submit and wait for the command buffer
      m_alloc->finalizeAndReleaseStaging();    // Make sure there are no pending staging buffers and clear them up
    }

    nvvkhl::setCamera(filename, m_scene->getRenderCameras(), m_scene->getSceneBounds());  // Set camera from scene

    // Default parameters for overall material
    m_pushConst.maxDepth              = 5;
    m_pushConst.frame                 = 0;
    m_pushConst.fireflyClampThreshold = 1;
    m_pushConst.maxSamples            = 2;

    // Default sky parameters
    m_skyParams = nvvkhl_shaders::initPhysicalSkyParameters();
  }

  // Create all Vulkan buffer data
  void createVkBuffers()
  {
    // Create the buffer of the current camera transformation, changing at each frame
    m_bCameraInfo = m_alloc->createBuffer(sizeof(DH::CameraInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bCameraInfo.buffer);
    // Create the buffer of sky parameters, updated at each frame
    m_bSkyParams = m_alloc->createBuffer(sizeof(nvvkhl_shaders::SimpleSkyParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bSkyParams.buffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Creating the pipeline: shader ...
  //
  void createCompPipelines()
  {
    m_rtPipe.destroy(m_device);
    m_rtSet->deinit();
    m_rtSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_rtPipe.plines.resize(1);

    // This descriptor set, holds the top level acceleration structure and the output image
    m_rtSet->addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_cameraInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_sceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_textures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, m_sceneVk->nbTextures(), VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_skyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);

    m_rtSet->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

    // pushing time
    VkPushConstantRange        pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant)};
    VkPipelineLayoutCreateInfo plCreateInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1U,
        .pSetLayouts            = &m_rtSet->getLayout(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstant,
    };
    vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_rtPipe.layout);

    VkComputePipelineCreateInfo cpCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = nvvk::createShaderStageInfo(m_device, comp_shd, VK_SHADER_STAGE_COMPUTE_BIT, USE_GLSL ? "main" : "computeMain"),
        .layout = m_rtPipe.layout,
    };

    vkCreateComputePipelines(m_device, {}, 1, &cpCreateInfo, nullptr, &m_rtPipe.plines[0]);

    vkDestroyShaderModule(m_device, cpCreateInfo.stage.module, nullptr);
  }


  void pushDescriptorSet(VkCommandBuffer cmd)
  {
    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_sceneRtx->tlas();

    VkWriteDescriptorSetAccelerationStructureKHR descASInfo{
        .sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures    = &tlas,
    };
    VkDescriptorImageInfo        imageInfo{{}, m_gBuffers->getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo       dbi_unif{m_bCameraInfo.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo       sceneDesc{m_sceneVk->sceneDesc().buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo dbi_sky{m_bSkyParams.buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtSet->makeWrite(0, B_tlas, &descASInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, B_outImage, &imageInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, B_cameraInfo, &dbi_unif));
    writes.emplace_back(m_rtSet->makeWrite(0, B_sceneDesc, &sceneDesc));
    writes.emplace_back(m_rtSet->makeWrite(0, B_skyParam, &dbi_sky));

    std::vector<VkDescriptorImageInfo> diit;
    for(const auto& texture : m_sceneVk->textures())  // All texture samplers
    {
      diit.emplace_back(texture.descriptor);
    }
    writes.emplace_back(m_rtSet->makeWriteArray(0, B_textures, diit.data()));

    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipe.layout, 0,
                              static_cast<uint32_t>(writes.size()), writes.data());
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

    const auto& m   = CameraManip.getMatrix();
    const auto  fov = CameraManip.getFov();

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


  void destroyResources()
  {
    m_alloc->destroy(m_bCameraInfo);
    m_alloc->destroy(m_bSkyParams);

    m_tonemapper.reset();
    m_gBuffers.reset();
    m_rtSet->deinit();
    m_rtPipe.destroy(m_device);
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*                           m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>               m_dutil{};
  std::unique_ptr<AllocDma>                      m_alloc{};
  std::unique_ptr<nvvk::DescriptorSetContainer>  m_rtSet{};  // Descriptor set
  std::unique_ptr<nvvkhl::TonemapperPostProcess> m_tonemapper{};
  std::unique_ptr<nvvkhl::GBuffer>               m_gBuffers{};  // G-Buffers: color + depth
  std::unique_ptr<nvh::gltf::Scene>              m_scene{};     // GLTF Scene
  std::unique_ptr<nvvkhl::SceneVk>               m_sceneVk{};   // GLTF Scene buffers
  std::unique_ptr<nvvkhl::SceneRtx>              m_sceneRtx{};  // GLTF Scene BLAS/TLAS

  VkDevice m_device = VK_NULL_HANDLE;  // Convenient

  // Resources
  nvvk::Buffer m_bCameraInfo;  // Camera information
  nvvk::Buffer m_bSkyParams;   // Sky parameters

  // Data and setting
  nvvkhl_shaders::PhysicalSkyParameters m_skyParams = {};

  // Pipeline
  DH::PushConstant m_pushConst{};  // Information sent to the shader
  int              m_frame{0};
  int              m_maxFrames{10000};

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};

  nvvkhl::PipelineContainer m_rtPipe;
};

//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char** argv) -> int
{
  // Extension feature needed.
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};

  // Setting up how Vulkan context must be created
  VkContextSettings vkSetup;
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);  // WIN32, XLIB, ...
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});  // required with VK_KHR_acceleration_structure
  vkSetup.deviceExtensions.push_back({VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature});
#if USE_SLANG
  VkPhysicalDeviceComputeShaderDerivativesFeaturesNV computeDerivativesFeature{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_NV};
  vkSetup.deviceExtensions.push_back({VK_NV_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, &computeDerivativesFeature});
#endif
  vkSetup.queues.push_back(VK_QUEUE_COMPUTE_BIT);  // Extra queue for building BLAS and TLAS

  // Create the Vulkan context
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);  // Loading the Vulkan extension pointers
  if(!vkContext->isValid())
    std::exit(0);

  nvvkhl::ApplicationCreateInfo appInfo;
  appInfo.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = false;
  appInfo.instance       = vkContext->getInstance();
  appInfo.device         = vkContext->getDevice();
  appInfo.physicalDevice = vkContext->getPhysicalDevice();
  appInfo.queues         = vkContext->getQueueInfos();

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appInfo);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);
  test->parameterLists().addFilename(".gltf|Scene to render", &g_sceneFilename);


  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app->addElement(std::make_shared<RayQuery>());

  app->run();
  app.reset();
  vkContext.reset();
  return test->errorCode();
}
