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
  This sample shows how a texture3d can be made on the CPU or the GPU.
*/

#define USE_SLANG true
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION
#define IMGUI_DEFINE_MATH_OPERATORS
#include <future>
#include <span>

#include "glm/gtc/noise.hpp"  // For perlin noise
#include <glm/glm.hpp>        // Math library

#include <GLFW/glfw3.h>  // Windowing
#undef APIENTRY

#include "shaders/shaderio.h"  // Shared between host and device

#include "common/utils.hpp"
#include "_autogen/perlin.comp.glsl.h"
#include "_autogen/perlin.slang.h"
#include "_autogen/texture_3d.frag.glsl.h"
#include "_autogen/texture_3d.slang.h"
#include "_autogen/texture_3d.vert.glsl.h"

#include <volk.h>


#include <fmt/format.h>                    // String formating
#include <nvapp/application.hpp>           // The Application base
#include <nvapp/elem_camera.hpp>           // To handle the camera movement
#include <nvapp/elem_default_menu.hpp>     // Display a menu
#include <nvapp/elem_default_title.hpp>    // Change the window title
#include <nvgui/camera.hpp>                // Camera widget
#include <nvgui/property_editor.hpp>       // Formatting UI
#include <nvutils/camera_manipulator.hpp>  // To manipulate the camera
#include <nvutils/file_operations.hpp>     // Various
#include <nvutils/logger.hpp>              // LOGE, LOGI, etc.
#include <nvutils/parameter_parser.hpp>    // To parse the command line
#include <nvutils/primitives.hpp>          // Create a cube
#include <nvutils/timers.hpp>              // Timing
#include <nvvk/check_error.hpp>            // Vulkan error checking
#include <nvvk/compute_pipeline.hpp>       // Get group counts
#include <nvvk/context.hpp>                // Vulkan context creation
#include <nvvk/debug_util.hpp>             // Debug names and more
#include <nvvk/default_structs.hpp>        // Default Vulkan structure
#include <nvvk/descriptor_heap.hpp>        // Bindless descriptor heap (Method B)
#include <nvvk/formats.hpp>                // Find format, etc.
#include <nvvk/render_target.hpp>          // Rendering in a render target
#include <nvvk/graphics_pipeline.hpp>      // Dynamic graphics state
#include <nvvk/helpers.hpp>                // Find format
#include <nvvk/resource_allocator.hpp>     // The GPU resource allocator
#include <nvvk/staging.hpp>                // Staging manager

#include <nvapp/imgui_texture.hpp>

std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};


class Texture3dSample : public nvapp::IAppElement
{
  struct Settings
  {
    uint32_t                 powerOfTwoSize = 6;
    bool                     useGpu         = true;
    VkFilter                 magFilter      = VK_FILTER_LINEAR;
    VkSamplerAddressMode     addressMode    = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    shaderio::PerlinSettings perlin         = {};
    int                      headlight      = 1;
    glm::vec3                toLight        = {1.F, 1.F, 1.F};
    int                      steps          = 100;
    float                    threshold      = 0.05f;
    glm::vec4                surfaceColor   = {0.8F, 0.8F, 0.8F, 1.0F};
    uint32_t                 getSize() { return 1 << powerOfTwoSize; }
    uint32_t                 getTotalSize() { return getSize() * getSize() * getSize(); }
  };

public:
  Texture3dSample()           = default;
  ~Texture3dSample() override = default;

  // Implementation of nvvk::IApplication interface
  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    NVVK_CHECK(nvvk::createTimelineSemaphore(m_device, 0, m_timelineSemaphore));

    // Create the Vulkan allocator (VMA)
    m_alloc.init({
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });  // Allocator

    m_stagingUploader.init(&m_alloc, true);

    // Initialization of the render target we want to use
    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
    NVVK_CHECK(m_renderTarget.init(
        {.alloc = &m_alloc, .colorFormats = {VK_FORMAT_R8G8B8A8_UNORM}, .depthFormat = m_depthFormat, .debugName = "Texture3dSample"}));

    createDescriptorHeap();
    createShaders();
    createVkBuffers();
    createTexture();

    // Setting the default camera
    g_cameraManip->setClipPlanes({0.01F, 100.0F});
    g_cameraManip->setLookat({-0.5F, 0.5F, 2.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  };

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);

    vkDestroyShaderEXT(m_device, m_computeShader, nullptr);
    vkDestroyShaderEXT(m_device, m_vertexShader, nullptr);
    vkDestroyShaderEXT(m_device, m_fragmentShader, nullptr);
    m_computeShader  = VK_NULL_HANDLE;
    m_vertexShader   = VK_NULL_HANDLE;
    m_fragmentShader = VK_NULL_HANDLE;

    if(m_hasVolumeSampler)
    {
      m_heap.releaseSamplerDescriptor(m_volumeSamplerIdx);
      m_hasVolumeSampler = false;
    }

    m_alloc.destroyBuffer(m_samplerHeapBuffer);
    m_alloc.destroyBuffer(m_resourceHeapBuffer);
    m_samplerHeapBuffer  = {};
    m_resourceHeapBuffer = {};
    m_heap.deinit();

    m_viewportImage.deinit();
    m_renderTarget.deinit();

    m_alloc.destroyBuffer(m_vertices);
    m_alloc.destroyBuffer(m_indices);
    m_alloc.destroyBuffer(m_frameInfo);
    m_alloc.destroyImage(m_image);

    m_stagingUploader.deinit();
    m_alloc.deinit();

    vkDestroySemaphore(m_device, m_timelineSemaphore, nullptr);
  };


  void onUIRender() override
  {
    namespace PE      = nvgui::PropertyEditor;
    auto& s           = m_settings;
    bool  redoTexture = false;

    // Settings
    if(ImGui::Begin("Settings"))
    {


      nvgui::CameraWidget(g_cameraManip);

      ImGui::Text("Shading");
      PE::begin();
      PE::ColorEdit3("Color", &m_settings.surfaceColor.x);
      redoTexture |= PE::Combo("Filter Mode", (int*)&s.magFilter, "Nearest\0Linear\0");
      redoTexture |= PE::Combo("Address Mode", (int*)&s.addressMode,
                               "Repeat\0Mirror Repeat\0Clamp to Edge\0Clamp to Border\0Mirror Clamp to Edge\0");
      PE::Checkbox("Head light", (bool*)&m_settings.headlight);
      ImGui::BeginDisabled(m_settings.headlight);
      PE::SliderFloat3("Light Dir", &m_settings.toLight.x, -1.0F, 1.0F);
      ImGui::EndDisabled();
      PE::end();
      /// ----
      std::string s_size = "Texture Size: " + std::to_string(1 << s.powerOfTwoSize) + std::string("^3");
      ImGui::Text("Perlin");
      PE::begin();
      redoTexture |= PE::SliderInt(s_size.c_str(), (int*)&s.powerOfTwoSize, 4, 7);
      m_needsTextureUpdate |= PE::SliderInt("Octave", (int*)&s.perlin.octave, 1, 8, "%.3f", {}, "Looping the noise n-times");
      m_needsTextureUpdate |= PE::SliderFloat("Power", &s.perlin.power, 0.001F, 3, "%.3f", ImGuiSliderFlags_Logarithmic,
                                              "Increase the values. Low power equal to sharp edges, higher equal to "
                                              "smooth transition.");
      m_needsTextureUpdate |= PE::SliderFloat("Frequency", &s.perlin.frequency, 0.1F, 5.F, "%.3f", ImGuiSliderFlags_Logarithmic,
                                              "Number of time the noise is sampled in the domain.");
      m_needsTextureUpdate |= PE::Checkbox("Gpu Creation", &s.useGpu, "Use compute shader to generate the texture data");
      PE::end();
      /// ----
      ImGui::Text("Ray Marching");
      PE::begin();
      PE::SliderFloat("Threshold", &m_settings.threshold, -1.0F, 1.0, "%.3f", {},
                      "Values below the threshold are ignored. High Power value is needed, for the threshold to be "
                      "effective.");
      PE::SliderInt("Steps", (int*)&m_settings.steps, 1, 500, "%d", {}, "Number of maximum steps.");
      PE::end();
      /// ----
      ImGui::Text("Presets");
      PE::begin();
      {
        static int preset = 0;
        if(PE::SliderInt("Presets", &preset, 0, 9))
        {
          m_needsTextureUpdate = true;
          redoTexture          = true;
          switch(preset)
          {
            case 0:
              m_settings.perlin         = {};
              m_settings.powerOfTwoSize = 6;
              m_settings.threshold      = 0.05F;
              break;
            case 1:
              m_settings.perlin         = {8, 3, 5};
              m_settings.powerOfTwoSize = 7;
              m_settings.threshold      = 0.1F;
              break;
            case 2:
              m_settings.perlin         = {8, .3F, .2F};
              m_settings.powerOfTwoSize = 7;
              m_settings.threshold      = 0.7F;
              break;
            case 3:
              m_settings.perlin         = {8, 1.7F, 3.0F};
              m_settings.powerOfTwoSize = 7;
              m_settings.threshold      = 0.14F;
              break;
            case 4:
              m_settings.perlin         = {8, 2.3F, 1.4F};
              m_settings.powerOfTwoSize = 7;
              m_settings.threshold      = 0.009F;
              break;
            case 5:
              m_settings.perlin         = {2, 0.86F, 1.42F};
              m_settings.powerOfTwoSize = 7;
              m_settings.threshold      = 0.28F;
              break;
            case 6:
              m_settings.perlin         = {3, 0.005F, 0.92F};
              m_settings.powerOfTwoSize = 6;
              m_settings.threshold      = 0.1F;
              break;
            case 7:
              m_settings.perlin         = {8, 3.0F, 5.F};
              m_settings.powerOfTwoSize = 7;
              m_settings.threshold      = 0.009F;
              break;
            case 8:
              m_settings.perlin         = {2, 2.0F, 4.5F};
              m_settings.powerOfTwoSize = 5;
              m_settings.threshold      = 0.226F;
              break;
            case 9:
              m_settings.perlin         = {1, 2.0F, 25.F};
              m_settings.powerOfTwoSize = 1;
              m_settings.threshold      = 0.045F;
              break;
            default:
              m_settings = Settings();
              break;
          }
        }
      }
      PE::end();

      if(redoTexture)
      {
        vkDeviceWaitIdle(m_device);
        m_alloc.destroyImage(m_image);
        createTexture();
      }

      ImGui::TextDisabled("%d FPS / %.3fms", static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);


      // Show computation status in red if work is in progress
      if(m_perlinPercent > 0.f)
      {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.0f, 0.0f, 1.0f));
        ImGui::Text("Computing Perlin noise...");
        ImGui::PopStyleColor();
        ImGui::ProgressBar(m_perlinPercent);
      }

      ImGui::End();
    }

    // Using viewport Window
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
    if(ImGui::Begin("Viewport"))
    {
      if(m_image.image != nullptr)
      {
        ImGui::Image(m_viewportImage, ImGui::GetContentRegionAvail());
      }

      ImGui::End();
    }
    ImGui::PopStyleVar();
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

    if(!m_settings.useGpu)
    {
      m_stagingUploader.releaseStaging();
    }

    if(m_needsTextureUpdate)
    {
      updateTextureData(cmd, true);
    }


    // Update Frame buffer uniform buffer
    shaderio::FrameInfo finfo{};
    finfo.view      = g_cameraManip->getViewMatrix();
    finfo.proj      = g_cameraManip->getPerspectiveMatrix();
    finfo.camPos    = g_cameraManip->getEye();
    finfo.headlight = m_settings.headlight;
    finfo.toLight   = m_settings.toLight;
    vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &finfo);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

    // Build the dynamic-rendering state from the render target. The render target
    // keeps its images in VK_IMAGE_LAYOUT_GENERAL, so no layout transitions are
    // needed to render into it or to sample it afterwards.
    nvvk::RenderTargetState rtState;
    m_renderTarget.fillState(rtState);
    rtState.colorAttachments[0].clearValue = {m_clearColor};
    rtState.depthAttachment.clearValue     = {.depthStencil = DEFAULT_VkClearDepthStencilValue};
    nvvk::RenderTargetState::AttachmentOps ops{};  // default: clear+store on color & depth, don't care on stencil
    rtState.cmdBeginRendering(cmd, ops);

    {
      const VkDeviceSize offsets{0};

      writeVolumeSampledDescriptor();

      m_dynamicPipeline.cmdApplyAllStates(cmd);
      m_dynamicPipeline.cmdSetViewportAndScissor(cmd, m_app->getViewportSize());
      m_dynamicPipeline.cmdBindShaders(cmd, {.vertex = m_vertexShader, .fragment = m_fragmentShader});
      m_heap.cmdBindHeaps(cmd, m_samplerHeapBuffer.address, m_resourceHeapBuffer.address);

      shaderio::PushConstant pushConstant{};
      pushConstant.threshold      = m_settings.threshold;
      pushConstant.steps          = m_settings.steps;
      pushConstant.color          = m_settings.surfaceColor;
      pushConstant.transfo        = glm::mat4(1);  // Identity
      pushConstant.bufferHeapBase = m_heap.bufferShaderIndexBase();

      VkPushDataInfoEXT pushInfo{.sType  = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
                                 .offset = 0,
                                 .data   = {.address = &pushConstant, .size = sizeof(shaderio::PushConstant)}};
      vkCmdPushDataEXT(cmd, &pushInfo);

      vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT32);
      int32_t numIndices = 36;
      vkCmdDrawIndexed(cmd, numIndices, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);

    VkSemaphoreSubmitInfo signalInfo = {
        .sType       = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore   = m_timelineSemaphore,
        .value       = m_timelineSemaphoreNextValue,
        .stageMask   = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
        .deviceIndex = 0,
    };

    m_app->addSignalSemaphore(signalInfo);

    m_timelineSemaphoreNextValue++;
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    NVVK_CHECK(m_renderTarget.update(cmd, size));
    m_viewportImage.update(m_renderTarget.getUiImageView());
  }

private:
  void createDescriptorHeap()
  {
    NVVK_CHECK(m_heap.init(m_app->getPhysicalDevice(), m_device));

    constexpr VkBufferUsageFlags2 heapUsage       = nvvk::DescriptorHeap::getRequiredBufferUsage();
    const VkDeviceSize            samplerBufSize  = m_heap.setupSamplerHeap(1);
    const VkDeviceSize            resourceBufSize = m_heap.setupResourceHeap(1, 1);
    NVVK_CHECK(m_alloc.createBuffer(m_samplerHeapBuffer, samplerBufSize, heapUsage, VMA_MEMORY_USAGE_AUTO, {},
                                    m_heap.getSamplerHeapAlignment()));
    NVVK_CHECK(m_alloc.createBuffer(m_resourceHeapBuffer, resourceBufSize, heapUsage, VMA_MEMORY_USAGE_AUTO,
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                    m_heap.getResourceHeapAlignment()));
    NVVK_DBG_NAME(m_samplerHeapBuffer.buffer);
    NVVK_DBG_NAME(m_resourceHeapBuffer.buffer);
  }

  void createShaders()
  {
    m_dynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    m_dynamicPipeline.vertexBindings              = {{.sType   = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
                                                      .stride  = sizeof(nvutils::PrimitiveVertex),
                                                      .divisor = 1}};
    m_dynamicPipeline.vertexAttributes = {{.sType  = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                           .format = VK_FORMAT_R32G32B32_SFLOAT,
                                           .offset = offsetof(nvutils::PrimitiveVertex, pos)}};

    const VkShaderCreateFlagsEXT rasterFlags = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT | VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT;

#if USE_SLANG
    const VkShaderCreateInfoEXT computeInfo =
        nvsamples::makeShaderCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, 0, perlin_slang, "computeMain",
                                        VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT);
#else
    const VkShaderCreateInfoEXT computeInfo = nvsamples::makeShaderCreateInfo(VK_SHADER_STAGE_COMPUTE_BIT, 0, perlin_comp_glsl,
                                                                              "main", VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT);
#endif
    NVVK_CHECK(vkCreateShadersEXT(m_device, 1, &computeInfo, nullptr, &m_computeShader));
    NVVK_DBG_NAME(m_computeShader);

#if USE_SLANG
    const std::array<VkShaderCreateInfoEXT, 2> rasterInfos{
        nvsamples::makeShaderCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, texture_3d_slang,
                                        "vertexMain", rasterFlags),
        nvsamples::makeShaderCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, 0, texture_3d_slang, "fragmentMain", rasterFlags),
    };
#else
    const std::array<VkShaderCreateInfoEXT, 2> rasterInfos{
        nvsamples::makeShaderCreateInfo(VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT, texture_3d_vert_glsl, "main", rasterFlags),
        nvsamples::makeShaderCreateInfo(VK_SHADER_STAGE_FRAGMENT_BIT, 0, texture_3d_frag_glsl, "main", rasterFlags),
    };
#endif

    std::array<VkShaderEXT, 2> rasterShaders{};
    NVVK_CHECK(vkCreateShadersEXT(m_device, static_cast<uint32_t>(rasterInfos.size()), rasterInfos.data(), nullptr,
                                  rasterShaders.data()));
    m_vertexShader   = rasterShaders[0];
    m_fragmentShader = rasterShaders[1];
    NVVK_DBG_NAME(m_vertexShader);
    NVVK_DBG_NAME(m_fragmentShader);
  }

  void writeVolumeStorageDescriptor()
  {
    NVVK_CHECK(m_heap.writeStorageImageDescriptor(shaderio::kHeapImgVolume, m_image, m_resourceHeapBuffer.mapping, VK_IMAGE_VIEW_TYPE_3D));
  }

  void writeVolumeSampledDescriptor()
  {
    NVVK_CHECK(m_heap.writeSampledImageDescriptor(shaderio::kHeapImgVolume, m_image, m_resourceHeapBuffer.mapping, VK_IMAGE_VIEW_TYPE_3D));
  }

  void writeVolumeHeapDescriptors()
  {
    writeVolumeStorageDescriptor();
    writeVolumeSampledDescriptor();
  }

  void createTexture()
  {
    nvutils::ScopedTimer st(__FUNCTION__);

    assert(!m_image.image);

    uint32_t realSize  = m_settings.getSize();
    VkFormat imgFormat = VK_FORMAT_R32_SFLOAT;

    std::array<uint32_t, 2> queueFamilies = {
        m_app->getQueue(0).familyIndex,
        m_app->getQueue(1).familyIndex,
    };

    VkImageCreateInfo create_info{
        .sType       = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType   = VK_IMAGE_TYPE_3D,
        .format      = imgFormat,
        .extent      = {realSize, realSize, realSize},
        .mipLevels   = 1,
        .arrayLayers = 1,
        .samples     = VK_SAMPLE_COUNT_1_BIT,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
        .sharingMode           = VK_SHARING_MODE_CONCURRENT,
        .queueFamilyIndexCount = 2,
        .pQueueFamilyIndices   = queueFamilies.data(),
    };

    VkImageViewCreateInfo view_info{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext    = nullptr,
        .image    = m_image.image,
        .viewType = VK_IMAGE_VIEW_TYPE_3D,
        .format   = imgFormat,
        .subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = VK_REMAINING_MIP_LEVELS, .layerCount = VK_REMAINING_ARRAY_LAYERS},
    };

    NVVK_CHECK(m_alloc.createImage(m_image, create_info, view_info));
    m_image.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    NVVK_DBG_NAME(m_image.image);
    NVVK_DBG_NAME(m_image.descriptor.imageView);

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    nvvk::cmdImageMemoryBarrier(cmd, {m_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkClearColorValue       clearColor = {{1.0f, 1.0f, 1.0f, 1.0f}};
    vkCmdClearColorImage(cmd, m_image.image, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);

    if(m_hasVolumeSampler)
    {
      m_heap.releaseSamplerDescriptor(m_volumeSamplerIdx);
      m_hasVolumeSampler = false;
    }

    void* smpMapping = nullptr;
    NVVK_CHECK(m_stagingUploader.appendBufferMapping(m_samplerHeapBuffer, 0, m_heap.getSamplerHeapSize(), smpMapping));
    VkSamplerCreateInfo samplerInfo{
        .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter    = m_settings.magFilter,
        .addressModeU = m_settings.addressMode,
        .addressModeV = m_settings.addressMode,
        .addressModeW = m_settings.addressMode,
    };
    m_volumeSamplerIdx = m_heap.acquireSamplerDescriptor(samplerInfo, smpMapping);
    m_hasVolumeSampler = true;

    writeVolumeHeapDescriptors();

    updateTextureData(cmd, false);
    m_stagingUploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    NVVK_DBG_NAME(m_image.image);
    NVVK_DBG_NAME(m_image.descriptor.imageView);
  }

  void fillPerlinImage(std::vector<float>& imageData)
  {
    nvutils::ScopedTimer st(__FUNCTION__);

    // Make local copies of all settings we need to avoid any changes during computation
    const uint32_t realSize  = m_settings.getSize();
    const float    power     = m_settings.perlin.power;
    const float    frequency = m_settings.perlin.frequency;
    const int      octaves   = m_settings.perlin.octave;
    m_perlinPercent          = 0.0f;
    const float increment    = 1.0f / (realSize * realSize * realSize);

    // Simple perlin noise
    for(uint32_t x = 0; x < realSize; x++)
    {
      for(uint32_t y = 0; y < realSize; y++)
      {
        for(uint32_t z = 0; z < realSize; z++)
        {
          float v     = 0.0F;
          float scale = power;
          float freq  = frequency / realSize;

          for(int oct = 0; oct < octaves; oct++)
          {
            v += glm::perlin(glm::vec3(x, y, z) * freq) / scale;
            freq *= 2.0F;    // Double the frequency
            scale *= power;  // Next power of b
          }
          imageData[static_cast<size_t>(z) * realSize * realSize + static_cast<uint64_t>(y) * realSize + x] = v;
          m_perlinPercent += increment;
        }
      }
    }
    m_perlinPercent = 0.0f;
  }

  void updateTextureData(VkCommandBuffer cmd, bool isPerFrame)
  {
    NVVK_DBG_SCOPE(cmd);
    assert(m_image.image);

    uint32_t realSize = m_settings.getSize();
    if(m_settings.useGpu)
    {
      runCompute(cmd, {realSize, realSize, realSize});
    }
    else
    {
      // No computation in progress, start a new one
      if(!m_perlinFuture.valid())
      {
        m_needsTextureUpdate = true;
        // Launch the computation asynchronously
        m_perlinFuture = std::async(std::launch::async, [this, realSize]() {
          std::vector<float> imageData;
          imageData.resize(m_settings.getTotalSize());
          fillPerlinImage(imageData);
          return imageData;
        });
      }
      // Check if we have a pending computation and it's ready
      else if(m_perlinFuture.valid() && m_perlinFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready)
      {
        m_needsTextureUpdate         = false;
        std::vector<float> imageData = m_perlinFuture.get();

        nvvk::SemaphoreState cmdSemaphoreState{};
        if(isPerFrame)
        {
          cmdSemaphoreState = nvvk::SemaphoreState::makeFixed(m_timelineSemaphore, m_timelineSemaphoreNextValue);
        }

        assert(m_stagingUploader.isAppendedEmpty());
        m_stagingUploader.appendImage(m_image, std::span(imageData), m_image.descriptor.imageLayout, cmdSemaphoreState);
        m_stagingUploader.cmdUploadAppended(cmd);
      }
    }
  }


  void runCompute(VkCommandBuffer cmd, const VkExtent3D& size)
  {
    NVVK_DBG_SCOPE(cmd);
    uint32_t realSize = m_settings.getSize();

    const VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};
    vkCmdBindShadersEXT(cmd, 1, stages, &m_computeShader);
    m_heap.cmdBindHeaps(cmd, 0, m_resourceHeapBuffer.address);

    writeVolumeStorageDescriptor();

    shaderio::PerlinSettings perlin = m_settings.perlin;
    perlin.frequency /= float(realSize);
    VkPushDataInfoEXT pushInfo{.sType  = VK_STRUCTURE_TYPE_PUSH_DATA_INFO_EXT,
                               .offset = 0,
                               .data   = {.address = &perlin, .size = sizeof(shaderio::PerlinSettings)}};
    vkCmdPushDataEXT(cmd, &pushInfo);

    VkExtent2D group_counts = nvvk::getGroupCounts({size.width, size.height}, WORKGROUP_SIZE);
    vkCmdDispatch(cmd, group_counts.width, group_counts.height, size.depth);
  }


  void createVkBuffers()
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    // Creating the Cube on the GPU
    nvutils::PrimitiveMesh mesh = nvutils::createCube();
    NVVK_CHECK(m_alloc.createBuffer(m_vertices, std::span(mesh.vertices).size_bytes(), VK_BUFFER_USAGE_2_VERTEX_BUFFER_BIT));
    NVVK_CHECK(m_alloc.createBuffer(m_indices, std::span(mesh.triangles).size_bytes(), VK_BUFFER_USAGE_2_INDEX_BUFFER_BIT));
    NVVK_DBG_NAME(m_vertices.buffer);
    NVVK_DBG_NAME(m_indices.buffer);
    NVVK_CHECK(m_stagingUploader.appendBuffer(m_vertices, 0, std::span(mesh.vertices)));
    NVVK_CHECK(m_stagingUploader.appendBuffer(m_indices, 0, std::span(mesh.triangles)));

    m_stagingUploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    // Frame information: camera matrix
    constexpr VkBufferUsageFlags2 kHeapBufUsage = VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT
                                                  | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT;
    NVVK_CHECK(m_alloc.createBuffer(m_frameInfo, sizeof(shaderio::FrameInfo), kHeapBufUsage, VMA_MEMORY_USAGE_CPU_TO_GPU));
    NVVK_DBG_NAME(m_frameInfo.buffer);

    NVVK_CHECK(m_heap.writeBufferDescriptor(shaderio::kHeapBufFrameInfo, m_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                            m_resourceHeapBuffer.mapping));
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_renderTarget.getColorImage(), m_renderTarget.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

private:
  nvapp::Application* m_app = nullptr;

  VkDevice m_device             = VK_NULL_HANDLE;
  bool     m_needsTextureUpdate = false;

  nvvk::ResourceAllocator m_alloc;
  nvvk::StagingUploader   m_stagingUploader;

  nvvk::DescriptorHeap m_heap{};
  nvvk::Buffer         m_samplerHeapBuffer{};
  nvvk::Buffer         m_resourceHeapBuffer{};
  uint32_t             m_volumeSamplerIdx = 0;
  bool                 m_hasVolumeSampler = false;

  VkShaderEXT                 m_computeShader  = VK_NULL_HANDLE;
  VkShaderEXT                 m_vertexShader   = VK_NULL_HANDLE;
  VkShaderEXT                 m_fragmentShader = VK_NULL_HANDLE;
  nvvk::GraphicsPipelineState m_dynamicPipeline;

  VkSemaphore m_timelineSemaphore{};
  uint64_t    m_timelineSemaphoreNextValue = 1;

  nvvk::Image        m_image;          // The 3D texture holding the perlin noise
  nvvk::RenderTarget m_renderTarget;   // Offscreen render target: color + depth
  nvapp::ImTexture   m_viewportImage;  // ImGui texture for the render target color image

  nvvk::Buffer m_vertices;   // Buffer of the vertices
  nvvk::Buffer m_indices;    // Buffer of the indices
  nvvk::Buffer m_frameInfo;  // Frame information passed to the GPU


  Settings m_settings = {};

  VkFormat          m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;    // Color format of the image
  VkFormat          m_depthFormat = VK_FORMAT_UNDEFINED;         // Depth format of the depth buffer
  VkClearColorValue m_clearColor  = {{0.3F, 0.3F, 0.3F, 1.0F}};  // Clear color

  float m_perlinPercent = 0.f;
  // Thread for async operations
  std::thread                     m_perlinThread;  // Thread for Perlin noise computation
  std::future<std::vector<float>> m_perlinFuture;  // Future to store the result
};

//--------
int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;
  nvvk::Context                vkContext;  // The Vulkan context

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);

  VkPhysicalDeviceExtendedDynamicState3FeaturesEXT dStateFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  VkPhysicalDeviceDescriptorHeapFeaturesEXT heapFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT};
  VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untypedPtrFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR};

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions =
          {
              {VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME, &dStateFeatures},
              {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures},
              {VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME, &heapFeatures},
              {VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME, &untypedPtrFeatures},
          },
      .queues = {VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_TRANSFER_BIT},
  };
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  // Vulkan context creation
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  appInfo.name           = fmt::format("{} ({})", nvutils::getExecutablePath().stem().string(), SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // Create this example
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  elemCamera->setCameraManipulator(g_cameraManip);

  app.addElement(std::make_shared<Texture3dSample>());
  app.addElement(elemCamera);
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info

  app.run();

  app.deinit();
  vkContext.deinit();

  return 0;
}
