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

#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION
#include <glm/glm.hpp>
#include <vector>


namespace shaderio {
using namespace glm;
#include "shaders/shaderio.h"  // Shared between host and device
}  // namespace shaderio
#include "_autogen/compute_only.comp.glsl.h"  // Generated compiled shader
#include "_autogen/compute_only.slang.h"

#define SHOW_MENU 1      // Enabling the standard Window menu.
#define SHOW_SETTINGS 1  // Show the setting panel

#include <nvapp/application.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>


class ComputeOnlyElement : public nvapp::IAppElement
{
public:
  ComputeOnlyElement()           = default;
  ~ComputeOnlyElement() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app = app;
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
    m_gBuffers.init({
        .allocator      = &m_alloc,
        .colorFormats   = {VK_FORMAT_R8G8B8A8_UNORM},  // Only one GBuffer color attachment
        .imageSampler   = linearSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    });

    createShaderObjectAndLayout();
  }

  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
    vkDestroyShaderEXT(m_app->getDevice(), m_shader, NULL);

    vkDestroyPipelineLayout(m_app->getDevice(), m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_app->getDevice(), m_descriptorSetLayout, nullptr);

    m_samplerPool.deinit();
    m_gBuffers.deinit();
    m_alloc.deinit();
  }

  void onUIRender() override
  {
#if SHOW_SETTINGS
    // [optional] convenient setting panel
    ImGui::Begin("Settings");
    ImGui::TextDisabled("%d FPS / %.3fms", static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);
    ImGui::SliderFloat("Zoom", &m_pushConst.zoom, 0.1f, 3.f);
    ImGui::SliderInt("Iteration", &m_pushConst.iter, 1, 8);
    ImGui::End();
#endif

    // Rendered image displayed fully in 'Viewport' window
    ImGui::Begin("Viewport");
    ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());
    ImGui::End();
  }

  void onRender(VkCommandBuffer cmd)
  {
    // Push descriptor set
    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(m_bindings.getWriteSet(0), m_gBuffers.getDescriptorImageInfo());
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,
                              static_cast<uint32_t>(writeContainer.size()), writeContainer.data());

    // Bind compute shader
    const VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};
    vkCmdBindShadersEXT(cmd, 1, stages, &m_shader);

    // Pushing constants
    m_pushConst.time = static_cast<float>(ImGui::GetTime());
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);

    // Dispatch compute shader
    VkExtent2D group_counts = nvvk::getGroupCounts(m_gBuffers.getSize(), WORKGROUP_SIZE);
    vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);
  }

  // Called if showMenu is true
  void onUIMenu() override
  {
    if(ImGui::BeginMenu("File"))
    {
      if(ImGui::MenuItem("Exit", "Ctrl+Q"))
        m_app->close();
      ImGui::EndMenu();
    }
    if(ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Q))
      m_app->close();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override { NVVK_CHECK(m_gBuffers.update(cmd, size)); }

  //-------------------------------------------------------------------------------------------------
  // Creating the pipeline layout and shader object
  void createShaderObjectAndLayout()
  {
    VkPushConstantRange pushConstant = {.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(shaderio::PushConstant)};

    // Create the layout used by the shader
    m_bindings.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    NVVK_CHECK(m_bindings.createDescriptorSetLayout(m_app->getDevice(), VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                                    &m_descriptorSetLayout));
    NVVK_DBG_NAME(m_descriptorSetLayout);

    NVVK_CHECK(nvvk::createPipelineLayout(m_app->getDevice(), &m_pipelineLayout, {m_descriptorSetLayout}, {pushConstant}));
    NVVK_DBG_NAME(m_pipelineLayout);

    // Compute shader description
    VkShaderCreateInfoEXT shaderCreateInfos = {
        .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .pNext                  = NULL,
        .flags                  = VK_SHADER_CREATE_DISPATCH_BASE_BIT_EXT,
        .stage                  = VK_SHADER_STAGE_COMPUTE_BIT,
        .nextStage              = 0,
        .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_descriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstant,
        .pSpecializationInfo    = NULL,
    };
#if USE_SLANG
    shaderCreateInfos.codeSize = compute_only_slang_sizeInBytes;
    shaderCreateInfos.pCode    = compute_only_slang;
    shaderCreateInfos.pName    = "computeMain";
#else
    shaderCreateInfos.codeSize = std::span(compute_only_comp_glsl).size_bytes();
    shaderCreateInfos.pCode    = std::span(compute_only_comp_glsl).data();
    shaderCreateInfos.pName    = "main";
#endif

    // Create the shader
    NVVK_CHECK(vkCreateShadersEXT(m_app->getDevice(), 1, &shaderCreateInfos, NULL, &m_shader));
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers.getColorImage(), m_gBuffers.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

private:
  nvapp::Application*      m_app{};        // Application instance
  nvvk::ResourceAllocator  m_alloc;        // Allocator
  nvvk::GBuffer            m_gBuffers;     // G-Buffers: color + depth
  nvvk::SamplerPool        m_samplerPool;  // The sampler pool, used to create a sampler for the texture
  nvvk::DescriptorBindings m_bindings;     // Descriptor bindings helper

  VkShaderEXT           m_shader{};
  VkPipelineLayout      m_pipelineLayout{};       // Pipeline layout
  VkDescriptorSetLayout m_descriptorSetLayout{};  // Descriptor set layout

  shaderio::PushConstant m_pushConst = {.zoom = 1.5f, .iter = 2};
};

int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;

  // Command parser
  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);

  // Extension feature needed.
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions = {{VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjFeature}, {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME}},
  };
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  // Create the Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  // Setting up how the the application must be created
  appInfo.name           = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);
  appInfo.useMenu        = SHOW_MENU ? true : false;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  app.addElement(std::make_shared<ComputeOnlyElement>());  // Add our sample to the application

  app.run();  // Loop infinitely, and call IAppElement virtual functions at each frame

  app.deinit();
  vkContext.deinit();

  return 0;
}
