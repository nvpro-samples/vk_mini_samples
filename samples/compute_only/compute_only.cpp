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

#define USE_SLANG true
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION

#include "shaders/shaderio.h"  // Shared between host and device
#include "common/utils.hpp"
#include "_autogen/compute_only.comp.glsl.h"  // Generated compiled shader
#include "_autogen/compute_only.slang.h"

#define SHOW_MENU true      // Enabling the standard Window menu.
#define SHOW_SETTINGS true  // Show the setting panel

#include <nvapp/application.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/descriptor_heap.hpp>
#include <nvapp/imgui_texture.hpp>
#include <nvvk/render_target.hpp>
#include <nvvk/resource_allocator.hpp>


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

    // Offscreen render target: single color image
    NVVK_CHECK(m_renderTarget.init({
        .alloc        = &m_alloc,
        .colorFormats = {VK_FORMAT_R8G8B8A8_UNORM},  // Only one GBuffer color attachment
        .debugName    = "ComputeOnly",
    }));

    createDescriptorHeap();
    createShaderObject();
  }

  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
    vkDestroyShaderEXT(m_app->getDevice(), m_shader, nullptr);

    m_alloc.destroyBuffer(m_resourceHeapBuffer);
    m_resourceHeapBuffer = {};
    m_heap.deinit();

    m_viewportImage.deinit();
    m_renderTarget.deinit();
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
    ImGui::Image(m_viewportImage, ImGui::GetContentRegionAvail());
    ImGui::End();
  }

  void onRender(VkCommandBuffer cmd) override
  {
    // Bind compute shader
    const std::array<VkShaderStageFlagBits, 1> stages = {VK_SHADER_STAGE_COMPUTE_BIT};
    vkCmdBindShadersEXT(cmd, 1, stages.data(), &m_shader);

    m_heap.cmdBindHeaps(cmd, 0, m_resourceHeapBuffer.address);

    m_pushConst.time = static_cast<float>(ImGui::GetTime());
    VkPushDataInfoEXT pushInfo{.sType  = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
                               .offset = 0,
                               .data   = {.address = &m_pushConst, .size = sizeof(shaderio::PushConstant)}};
    vkCmdPushDataEXT(cmd, &pushInfo);

    // Dispatch compute shader
    VkExtent2D group_counts = nvvk::getGroupCounts(m_renderTarget.getSize(), WORKGROUP_SIZE);
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

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    NVVK_CHECK(m_renderTarget.update(cmd, size));
    writeOutputImageDescriptor();
    m_viewportImage.update(m_renderTarget.getUiImageView());
  }

  void createShaderObject()
  {
    const VkShaderCreateInfoEXT shaderCreateInfos =
#if USE_SLANG
        nvsamples::makeShaderCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, 0, compute_only_slang, "computeMain",
                                        VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT);
#else
        nvsamples::makeShaderCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, 0, compute_only_comp_glsl, "main",
                                        VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT);
#endif

    NVVK_CHECK(vkCreateShadersEXT(m_app->getDevice(), 1, &shaderCreateInfos, nullptr, &m_shader));
    NVVK_DBG_NAME(m_shader);
  }

  void createDescriptorHeap()
  {
    NVVK_CHECK(m_heap.init(m_app->getPhysicalDevice(), m_app->getDevice()));

    constexpr VkBufferUsageFlags2 heapUsage       = nvvk::DescriptorHeap::getRequiredBufferUsage();
    const VkDeviceSize            resourceBufSize = m_heap.setupResourceHeap(1, 0);  // 1 storage image, 0 buffers

    constexpr VmaAllocationCreateFlags kMappedFlags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    NVVK_CHECK(m_alloc.createBuffer(m_resourceHeapBuffer, resourceBufSize, heapUsage, VMA_MEMORY_USAGE_AUTO,
                                    kMappedFlags, m_heap.getResourceHeapAlignment()));
    NVVK_DBG_NAME(m_resourceHeapBuffer.buffer);
  }

  void writeOutputImageDescriptor()
  {
    assert(m_resourceHeapBuffer.buffer != VK_NULL_HANDLE);
    NVVK_CHECK(m_heap.writeStorageImageDescriptor(shaderio::kHeapImgOutput, m_renderTarget.getColorImage(),
                                                  m_renderTarget.getColorFormat(), VK_IMAGE_LAYOUT_GENERAL,
                                                  m_resourceHeapBuffer.mapping));
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_renderTarget.getSampleImage(), m_renderTarget.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

private:
  nvapp::Application*     m_app{};  // Application instance
  nvvk::ResourceAllocator m_alloc;  // Allocator
  nvvk::DescriptorHeap    m_heap;
  nvvk::RenderTarget      m_renderTarget;   // Offscreen color image
  nvapp::ImTexture        m_viewportImage;  // ImGui texture displaying the rendered image
  nvvk::Buffer            m_resourceHeapBuffer;

  VkShaderEXT m_shader{};

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

  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  VkPhysicalDeviceDescriptorHeapFeaturesEXT heapFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT};
  VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untypedPtrFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR};

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions =
          {
              {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjFeature},
              {VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, &heapFeatures},
              {VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME, &untypedPtrFeatures},
          },
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
