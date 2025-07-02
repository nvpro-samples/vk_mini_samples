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


/** @DOC_START

This sample demonstrate how to use compute shader in a separate thread to render an image. 
The image is then copied to the main thread for display. The compute shader is rendering 
as fast as possible, while the main thread is displaying the image at the screen refresh rate.

Note: The amount of compute frame it can do per display iteration depends on the GPU. On laptops, 
      and integrated GPUs, the difference is minimal. On high-end GPUs, the difference can be significant.

@DOC_END */

#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")
#define VMA_IMPLEMENTATION

#include <condition_variable>
#include <iostream>
#include <thread>
#include <vector>

#ifdef NVP_SUPPORTS_NVAPI
#include "common/nvapi_manager.hpp"
#endif

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#undef APIENTRY

namespace shaderio {
using namespace glm;
#include "shaders/shaderio.h"  // Shared between host and device
}  // namespace shaderio
#include "_autogen/shader.comp.glsl.h"
#include "_autogen/shader.slang.h"


#include <nvapp/application.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>


#define SHOW_MENU 1  // Enabling the standard Window menu.
constexpr auto g_defaultWindowSize = VkExtent2D{1024, 1024};


class MultiThreadedSample : public nvapp::IAppElement
{
public:
  MultiThreadedSample()           = default;
  ~MultiThreadedSample() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app = app;
    // Allocator
    m_alloc.init(VmaAllocatorCreateInfo{
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });

    createShaderObjectAndLayout();

    // Acquiring the sampler which will be used for displaying the GBuffer
    m_samplerPool.init(app->getDevice());
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // GBuffer
    // The Display version
    m_gBuffers.init({
        .allocator      = &m_alloc,
        .colorFormats   = {VK_FORMAT_R8G8B8A8_UNORM},  // Only one GBuffer color attachment
        .imageSampler   = linearSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    });
    {
      // The rendering thread
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      m_gCompBuffers.init({
          .allocator      = &m_alloc,
          .colorFormats   = {VK_FORMAT_R8G8B8A8_UNORM},  // Only one GBuffer color attachment
          .imageSampler   = linearSampler,
          .descriptorPool = m_app->getTextureDescriptorPool(),
      });
      m_gCompBuffers.update(cmd, g_defaultWindowSize);
      m_app->submitAndWaitTempCmdBuffer(cmd);
    }


    // Create a thread to render the compute shader
    startRenderThread();
  }

  void onDetach() override
  {
    stopRenderThread();

    NVVK_CHECK(vkDeviceWaitIdle(m_app->getDevice()));
    vkDestroyShaderEXT(m_app->getDevice(), m_shader, NULL);

    vkDestroyPipelineLayout(m_app->getDevice(), m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_app->getDevice(), m_descriptorSetLayout, nullptr);

    m_samplerPool.deinit();
    m_gBuffers.deinit();
    m_gCompBuffers.deinit();

    m_alloc.deinit();
  }

  // Displaying the rendered image and some text
  void onUIRender() override
  {
    static float stats = 0;

    ImGui::Begin("Viewport");
    ImVec2 pos = ImGui::GetCursorPos();  // Remember position to put back text
    ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());
    ImGui::SetCursorPos(pos);
    m_frameCounter++;
    if(m_frameCounter > 30)
    {
      stats           = m_threadCounter / float(m_frameCounter);
      m_threadCounter = 0;
      m_frameCounter  = 0;
    }
    ImGui::Text("Number of compute per frame: %f", stats);
    ImGui::Text("Framerate: %f FPS / %f IPS", ImGui::GetIO().Framerate, stats * ImGui::GetIO().Framerate);
    ImGui::End();
  }

  // Copying the rendered image to the main thread
  void onRender(VkCommandBuffer cmd) override
  {
    VkImageSubresourceLayers srcSubresource{.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1};

    VkImageCopy imgCopy    = {};
    imgCopy.srcSubresource = srcSubresource;
    imgCopy.dstSubresource = srcSubresource;
    imgCopy.extent         = {std::min(m_gBuffers.getSize().width, m_gCompBuffers.getSize().width),
                              std::min(m_gBuffers.getSize().height, m_gCompBuffers.getSize().height), 1};
    vkCmdCopyImage(cmd, m_gCompBuffers.getColorImage(), VK_IMAGE_LAYOUT_GENERAL, m_gBuffers.getColorImage(),
                   VK_IMAGE_LAYOUT_GENERAL, 1, &imgCopy);
  }

  // Utility to start and stop the render thread
  void startRenderThread()
  {
    if(m_isRunning.load())
    {
      std::cerr << "Thread already running.\n";
      return;
    }

    m_shouldRun.store(true);
    m_workerThread = std::thread(&MultiThreadedSample::renderThread, this);
  }

  void stopRenderThread()
  {
    if(!m_isRunning.load())
    {
      std::cerr << "Thread is not running.\n";
      return;
    }

    m_shouldRun.store(false);
    {
      std::unique_lock lock(m_mutex);
      m_cv.wait(lock, [this]() { return !m_isRunning.load(); });
    }

    if(m_workerThread.joinable())
    {
      m_workerThread.join();
    }
  }

  // The render thread: rendering the compute shader in a infinite loop
  void renderThread()
  {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_isRunning.store(true);
    }

    constexpr int32_t computeQueueIndex = 1;

    VkCommandPool                 transientCmdPool{};
    const VkCommandPoolCreateInfo commandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,  // Hint that commands will be short-lived
        .queueFamilyIndex = m_app->getQueue(computeQueueIndex).familyIndex,
    };
    NVVK_CHECK(vkCreateCommandPool(m_app->getDevice(), &commandPoolCreateInfo, nullptr, &transientCmdPool));
    NVVK_DBG_NAME(transientCmdPool);

    while(m_shouldRun.load())
    {
      VkCommandBuffer cmd;
      NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_app->getDevice(), transientCmdPool));
      VkExtent2D group_counts = nvvk::getGroupCounts(m_gCompBuffers.getSize(), WORKGROUP_SIZE);

      // Wait for the frame to be consumed
      nvvk::WriteSetContainer writeContainer;
      writeContainer.append(m_bindings.getWriteSet(0), m_gCompBuffers.getDescriptorImageInfo());
      vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,
                                static_cast<uint32_t>(writeContainer.size()), writeContainer.data());

      // Bind compute shader
      const VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};
      vkCmdBindShadersEXT(cmd, 1, stages, &m_shader);

      // Pushing constants
      m_pushConst.time += 0.001f;  //static_cast<float>(ImGui::GetTime());
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);

      // Dispatch compute shader
      vkCmdDispatch(cmd, group_counts.width, group_counts.height, 1);

      NVVK_CHECK(nvvk::endSingleTimeCommands(cmd, m_app->getDevice(), transientCmdPool,
                                             m_app->getQueue(computeQueueIndex).queue));
      NVVK_CHECK(vkResetCommandPool(m_app->getDevice(), transientCmdPool, 0));
      m_threadCounter++;
    };

    vkDestroyCommandPool(m_app->getDevice(), transientCmdPool, nullptr);

    {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_isRunning.store(false);
      m_cv.notify_all();
    }
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
    bool v_sync = m_app->isVsync();
    if(ImGui::BeginMenu("View"))
    {
      ImGui::MenuItem("V-Sync", "Ctrl+Shift+V", &v_sync);
      ImGui::EndMenu();
    }

    if(!m_isRunning.load())
    {
      // Stopped by v-sync change, must be started again, but a frame later
      // because the v-sync change will invoke vkDeviceWaitIdle and cannot be done
      // while the render thread is running.
      startRenderThread();
    }

    if(m_app->isVsync() != v_sync)
    {
      stopRenderThread();
      m_app->setVsync(v_sync);
    }
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    // Re-creating the G-Buffer (RGBA8) when the viewport size change
    m_gBuffers.update(cmd, size);
  }

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

    VkShaderCreateInfoEXT shaderInfo = {
        .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .flags                  = VK_SHADER_CREATE_DISPATCH_BASE_BIT_EXT,
        .stage                  = VK_SHADER_STAGE_COMPUTE_BIT,
        .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_descriptorSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstant,
        .pSpecializationInfo    = NULL,
    };
#if USE_SLANG
    shaderInfo.codeSize = shader_slang_sizeInBytes;
    shaderInfo.pCode    = shader_slang;
    shaderInfo.pName    = "computeMain";
#else
    shaderInfo.codeSize = std::span(shader_comp_glsl).size_bytes();
    shaderInfo.pCode    = std::span(shader_comp_glsl).data();
    shaderInfo.pName    = "main";
#endif

    // Create the shader
    NVVK_CHECK(vkCreateShadersEXT(m_app->getDevice(), 1, &shaderInfo, NULL, &m_shader));
    NVVK_DBG_NAME(m_shader);
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers.getColorImage(), m_gBuffers.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

private:
  nvapp::Application*     m_app{};           // Application instance
  nvvk::ResourceAllocator m_alloc{};         // Allocator
  nvvk::GBuffer           m_gBuffers{};      // G-Buffers: color + depth
  nvvk::GBuffer           m_gCompBuffers{};  // G-Buffers: color + depth
  VkShaderEXT             m_shader{};
  int                     m_threadCounter{};
  int                     m_frameCounter{};
  shaderio::PushConstant  m_pushConst = {.zoom = 1.5f, .iter = 2};

  nvvk::SamplerPool        m_samplerPool{};          // The sampler pool, used to create a sampler for the texture
  VkPipelineLayout         m_pipelineLayout{};       // Pipeline layout
  VkDescriptorSetLayout    m_descriptorSetLayout{};  // Descriptor set layout
  nvvk::DescriptorBindings m_bindings;

  std::thread             m_workerThread{};
  std::atomic<bool>       m_shouldRun{false};
  std::atomic<bool>       m_isRunning{false};
  std::mutex              m_mutex{};
  std::condition_variable m_cv{};
};

int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;

  // Command parser
  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  reg.add({"frames", "Number of frames to render in headless mode"}, &appInfo.headlessFrameCount, true);
  cli.add(reg);
  cli.parse(argc, argv);


#ifdef NVP_SUPPORTS_NVAPI
  NVAPIManager nvapiManager;
  nvapiManager.init();
  nvapiManager.pushSetting({
      .version         = NVDRS_SETTING_VER,
      .settingId       = OGL_CPL_PREFER_DXPRESENT_ID,
      .settingType     = NVDRS_DWORD_TYPE,
      .u32CurrentValue = OGL_CPL_PREFER_DXPRESENT_PREFER_DISABLED,
  });
#endif

  // Required extra extensions

  // Vulkan context and extension feature needed.
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions =
          {
              {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjFeature},
              {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
          },
      .queues = {VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_COMPUTE_BIT},
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

  // Application setup
  appInfo.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.useMenu        = SHOW_MENU ? true : false;
  appInfo.windowSize     = {g_defaultWindowSize.width, g_defaultWindowSize.height};
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  app.addElement(std::make_shared<MultiThreadedSample>());  // Add our sample to the application

  if(!appInfo.headless)
    glfwSetWindowAttrib(app.getWindowHandle(), GLFW_RESIZABLE, GLFW_FALSE);


  app.run();  // Loop infinitely, and call IAppElement virtual functions at each frame

#ifdef NVP_SUPPORTS_NVAPI
  nvapiManager.popSettings();
  nvapiManager.deinit();
#endif

  app.deinit();
  vkContext.deinit();

  return 0;
}
