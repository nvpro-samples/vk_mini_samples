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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#define VMA_IMPLEMENTATION
#include "nvvk/descriptorsets_vk.hpp"               // Descriptor set helper
#include "nvvkhl/alloc_vma.hpp"                     // Our allocator
#include "nvvkhl/application.hpp"                   // For Application and IAppElememt
#include "nvvkhl/gbuffer.hpp"                       // G-Buffer helper
#include "nvvkhl/shaders/dh_comp.h"                 // Workgroup size and count
#include "nvvkhl/element_benchmark_parameters.hpp"  // For testing

namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

#define SHOW_MENU 1      // Enabling the standard Window menu.
#define SHOW_SETTINGS 1  // Show the setting panel


// Shader spir-v source code, compiled from CMake
#if USE_HLSL
#include "_autogen/shader_computeMain.spirv.h"
const auto& comp_shd = std::vector<uint8_t>{std::begin(shader_computeMain), std::end(shader_computeMain)};
#elif USE_SLANG
#include "_autogen/shader_slang.h"
const auto& comp_shd = std::vector<uint32_t>{std::begin(shaderSlang), std::end(shaderSlang)};
#else
#include "_autogen/shader.comp.glsl.h"  // Generated compiled shader
const auto& comp_shd = std::vector<uint32_t>{std::begin(shader_comp_glsl), std::end(shader_comp_glsl)};
#endif

template <typename T>  // Return memory usage size
size_t getShaderSize(const std::vector<T>& vec)
{
  using baseType = typename std::remove_reference<T>::type;
  return sizeof(baseType) * vec.size();
}

DH::PushConstant g_pushC = {.zoom = 1.5f, .iter = 2};

class ComputeOnlyElement : public nvvkhl::IAppElement
{
public:
  ComputeOnlyElement()           = default;
  ~ComputeOnlyElement() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app   = app;
    m_alloc = std::make_unique<nvvkhl::AllocVma>(VmaAllocatorCreateInfo{
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    });  // Allocator
    m_dset  = std::make_unique<nvvk::DescriptorSetContainer>(m_app->getDevice());
    createShaderObjectAndLayout();
  }

  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
    for(auto shader : m_shaders)
      vkDestroyShaderEXT(m_app->getDevice(), shader, NULL);
  }

  void onUIRender() override
  {
#if SHOW_SETTINGS
    // [optional] convenient setting panel
    ImGui::Begin("Settings");
    ImGui::TextDisabled("%d FPS / %.3fms", static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);
    ImGui::SliderFloat("Zoom", &g_pushC.zoom, 0.1f, 3.f);
    ImGui::SliderInt("Iteration", &g_pushC.iter, 1, 8);
    ImGui::End();
#endif

    // Rendered image displayed fully in 'Viewport' window
    ImGui::Begin("Viewport");
    ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());
    ImGui::End();
  }

  void onRender(VkCommandBuffer cmd)
  {
    // Push descriptor set
    const VkDescriptorImageInfo       in_desc = m_gBuffers->getDescriptorImageInfo();
    std::vector<VkWriteDescriptorSet> writes;
    writes.push_back(m_dset->makeWrite(0, 0, &in_desc));
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dset->getPipeLayout(), 0,
                              static_cast<uint32_t>(writes.size()), writes.data());

    // Bind compute shader
    const VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};
    vkCmdBindShadersEXT(cmd, 1, stages, m_shaders.data());

    // Pushing constants
    g_pushC.time = static_cast<float>(ImGui::GetTime());
    vkCmdPushConstants(cmd, m_dset->getPipeLayout(), VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant), &g_pushC);

    // Dispatch compute shader
    VkExtent2D group_counts = getGroupCounts(m_gBuffers->getSize());
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
    if(ImGui::IsKeyPressed(ImGuiKey_Q) && ImGui::IsKeyDown(ImGuiKey_LeftCtrl))
      m_app->close();
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    // Re-creating the G-Buffer (RGBA8) when the viewport size change
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_app->getDevice(), m_alloc.get(), VkExtent2D{width, height}, VK_FORMAT_R8G8B8A8_UNORM);
  }

  //-------------------------------------------------------------------------------------------------
  // Creating the pipeline layout and shader object
  void createShaderObjectAndLayout()
  {
    VkPushConstantRange push_constant_ranges = {.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(DH::PushConstant)};

    // Create the layout used by the shader
    m_dset->addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_dset->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
    m_dset->initPipeLayout(1, &push_constant_ranges);

    // Compute shader description
    std::vector<VkShaderCreateInfoEXT> shaderCreateInfos;
    shaderCreateInfos.push_back(VkShaderCreateInfoEXT{
        .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .pNext                  = NULL,
        .flags                  = VK_SHADER_CREATE_DISPATCH_BASE_BIT_EXT,
        .stage                  = VK_SHADER_STAGE_COMPUTE_BIT,
        .nextStage              = 0,
        .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
        .codeSize               = getShaderSize(comp_shd),
        .pCode                  = comp_shd.data(),
        .pName                  = USE_GLSL ? "main" : "computeMain",
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_dset->getLayout(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &push_constant_ranges,
        .pSpecializationInfo    = NULL,
    });
    // Create the shader
    NVVK_CHECK(vkCreateShadersEXT(m_app->getDevice(), 1, shaderCreateInfos.data(), NULL, m_shaders.data()));
  }

private:
  nvvkhl::Application*                          m_app = {nullptr};
  std::unique_ptr<nvvkhl::AllocVma>             m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;
  std::unique_ptr<nvvkhl::GBuffer>              m_gBuffers;
  std::array<VkShaderEXT, 1>                    m_shaders = {};
};

int main(int argc, char** argv)
{
  nvvk::ContextCreateInfo vkSetup{false};  // Vulkan creation context information (see nvvk::Context)
  vkSetup.setVersion(1, 3);
  vkSetup.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
  static VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR baryFeat{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  vkSetup.addDeviceExtension(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, false, &baryFeat);

  // Required extra extensions
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  vkSetup.addDeviceExtension(VK_EXT_SHADER_OBJECT_EXTENSION_NAME, false, &shaderObjFeature);
  vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

  nvvk::Context vkContext;
  vkContext.init(vkSetup);

  nvvkhl::ApplicationCreateInfo appInfo;
  appInfo.name = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.useMenu = SHOW_MENU ? true : false;
  appInfo.instance       = vkContext.m_instance;
  appInfo.device         = vkContext.m_device;
  appInfo.physicalDevice = vkContext.m_physicalDevice;
  appInfo.queues         = {vkContext.m_queueGCT, vkContext.m_queueC};


  auto app  = std::make_unique<nvvkhl::Application>(appInfo);                       // Create the application
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);  // Create the test framework
  app->addElement(test);                                                         // Add the test element (--test ...)
  app->addElement(std::make_shared<ComputeOnlyElement>());                       // Add our sample to the application
  app->run();  // Loop infinitely, and call IAppElement virtual functions at each frame

  app.reset();  // Clean up
  vkContext.deinit();

  return test->errorCode();
}
