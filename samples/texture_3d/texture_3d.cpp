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

/*
  This sample shows how a texture3d can be made on the CPU or the GPU.
*/

#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION
#define IMGUI_DEFINE_MATH_OPERATORS
#include <future>
#include <span>

#include "glm/gtc/noise.hpp"  // For perlin noise
#include <glm/glm.hpp>        // Math library

#include <GLFW/glfw3.h>  // Windowing
#undef APIENTRY

namespace shaderio {
using namespace glm;
#include "shaders/shaderio.h"  // Shared between host and device
}  // namespace shaderio

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
#include <nvvk/descriptors.hpp>            // Help creation descriptor sets
#include <nvvk/formats.hpp>                // Find format, etc.
#include <nvvk/gbuffers.hpp>               // Rendering in GBuffers
#include <nvvk/graphics_pipeline.hpp>      // Helper to create a graphic pipeline
#include <nvvk/helpers.hpp>                // Find format
#include <nvvk/resource_allocator.hpp>     // The GPU resource allocator
#include <nvvk/sampler_pool.hpp>           // Texture sampler
#include <nvvk/staging.hpp>                // Staging manager

#include "common/utils.hpp"

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

    // The texture sampler to use
    m_samplerPool.init(m_device);
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // Initialization of the G-Buffers we want use
    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
    m_gBuffers.init({.allocator      = &m_alloc,
                     .colorFormats   = {VK_FORMAT_R8G8B8A8_UNORM},
                     .depthFormat    = m_depthFormat,
                     .imageSampler   = linearSampler,
                     .descriptorPool = m_app->getTextureDescriptorPool()});


    createComputePipeline();
    createTexture();
    createVkBuffers();
    createGraphicPipeline();

    // Setting the default camera
    g_cameraManip->setClipPlanes({0.01F, 100.0F});
    g_cameraManip->setLookat({-0.5F, 0.5F, 2.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  };

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    vkDestroyPipeline(m_device, m_computePipeline, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    vkDestroyPipelineLayout(m_device, m_computePipelineLayout, nullptr);
    vkDestroyPipelineLayout(m_device, m_rasterPipelineLayout, nullptr);

    vkDestroyDescriptorSetLayout(m_device, m_computeDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_rasterDescriptorSetLayout, nullptr);

    m_gBuffers.deinit();
    m_samplerPool.deinit();

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
      PE::SliderInt("Steps", (int*)&m_settings.steps, 1, 500, "%.3f", {}, "Number of maximum steps.");
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
        ImGui::Image(ImTextureID(m_gBuffers.getDescriptorSet()), ImGui::GetContentRegionAvail());
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

    // Drawing the quad in a G-Buffer
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers.getColorImageView();
    colorAttachment.clearValue                = {.color = m_clearColor};
    VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
    depthAttachment.imageView                 = m_gBuffers.getDepthImageView();
    depthAttachment.clearValue                = {{{1.0F, 0}}};

    // Create the rendering info
    VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea           = DEFAULT_VkRect2D(m_gBuffers.getSize());
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;
    renderingInfo.pDepthAttachment     = &depthAttachment;


    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    vkCmdBeginRendering(cmd, &renderingInfo);
    {
      const VkDeviceSize offsets{0};

      nvvk::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, m_app->getViewportSize());
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

      nvvk::WriteSetContainer writes;
      writes.append(m_rasterBind.getWriteSet(0), m_frameInfo);
      writes.append(m_rasterBind.getWriteSet(1), m_image.descriptor);
      vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_rasterPipelineLayout, 0,
                                static_cast<uint32_t>(writes.size()), writes.data());

      // Push constant information
      shaderio::PushConstant pushConstant{};
      pushConstant.threshold = m_settings.threshold;
      pushConstant.steps     = m_settings.steps;
      pushConstant.color     = m_settings.surfaceColor;
      pushConstant.transfo   = glm::mat4(1);  // Identity
      vkCmdPushConstants(cmd, m_rasterPipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(shaderio::PushConstant), &pushConstant);

      vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT32);
      int32_t num_indices = 36;
      vkCmdDrawIndexed(cmd, num_indices, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});

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

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override { NVVK_CHECK(m_gBuffers.update(cmd, size)); }

private:
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
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
    NVVK_DBG_NAME(m_image.image);
    NVVK_DBG_NAME(m_image.descriptor.imageView);

    VkSamplerCreateInfo samplerInfo{
        .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter    = m_settings.magFilter,
        .addressModeU = m_settings.addressMode,
        .addressModeV = m_settings.addressMode,
        .addressModeW = m_settings.addressMode,
    };

    // Creating the sampler
    NVVK_CHECK(m_samplerPool.acquireSampler(m_image.descriptor.sampler, samplerInfo));

    m_image.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    nvvk::cmdImageMemoryBarrier(cmd, {m_image.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL});
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkClearColorValue       clearColor = {{1.0f, 1.0f, 1.0f, 1.0f}};
    vkCmdClearColorImage(cmd, m_image.image, VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &range);

    updateTextureData(cmd, false);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    // Debugging information
    NVVK_DBG_NAME(m_image.image);
    NVVK_DBG_NAME(m_image.descriptor.sampler);
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


  void createComputePipeline()
  {
    m_computeBind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    NVVK_CHECK(m_computeBind.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
                                                       &m_computeDescriptorSetLayout));
    NVVK_DBG_NAME(m_computeDescriptorSetLayout);

    const VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = sizeof(shaderio::PerlinSettings)};
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_computePipelineLayout, {m_computeDescriptorSetLayout}, {pushConstantRange}));
    NVVK_DBG_NAME(m_computePipelineLayout);

#if USE_SLANG
    const VkShaderModuleCreateInfo  moduleInfo = nvsamples::getShaderModuleCreateInfo(perlin_slang);
    VkPipelineShaderStageCreateInfo stageInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = &moduleInfo,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .pName = "computeMain",
    };
#else
    const VkShaderModuleCreateInfo  moduleInfo = nvsamples::getShaderModuleCreateInfo(perlin_comp_glsl);
    VkPipelineShaderStageCreateInfo stageInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = &moduleInfo,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .pName = "main",
    };
#endif

    VkComputePipelineCreateInfo compInfo{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = stageInfo,
        .layout = m_computePipelineLayout,
    };
    vkCreateComputePipelines(m_device, {}, 1, &compInfo, nullptr, &m_computePipeline);
    NVVK_DBG_NAME(m_computePipeline);
  }

  void runCompute(VkCommandBuffer cmd, const VkExtent3D& size)
  {
    NVVK_DBG_SCOPE(cmd);
    uint32_t realSize = m_settings.getSize();

    nvvk::WriteSetContainer writeContainer;
    writeContainer.append(m_computeBind.getWriteSet(0), m_image.descriptor);

    shaderio::PerlinSettings perlin = m_settings.perlin;
    perlin.frequency /= float(realSize);
    vkCmdPushConstants(cmd, m_computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shaderio::PerlinSettings), &perlin);
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipelineLayout, 0,
                              static_cast<uint32_t>(writeContainer.size()), writeContainer.data());
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
    VkExtent2D group_counts = nvvk::getGroupCounts({size.width, size.height}, WORKGROUP_SIZE);
    vkCmdDispatch(cmd, group_counts.width, group_counts.height, size.depth);
  }


  void createVkBuffers()
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    // Creating the Cube on the GPU
    nvutils::PrimitiveMesh mesh = nvutils::createCube();
    NVVK_CHECK(m_alloc.createBuffer(m_vertices, std::span(mesh.vertices).size_bytes(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
    NVVK_CHECK(m_alloc.createBuffer(m_indices, std::span(mesh.triangles).size_bytes(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT));
    NVVK_DBG_NAME(m_vertices.buffer);
    NVVK_DBG_NAME(m_indices.buffer);
    NVVK_CHECK(m_stagingUploader.appendBuffer(m_vertices, 0, std::span(mesh.vertices)));
    NVVK_CHECK(m_stagingUploader.appendBuffer(m_indices, 0, std::span(mesh.triangles)));

    m_stagingUploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_stagingUploader.releaseStaging();

    // Frame information: camera matrix
    NVVK_CHECK(m_alloc.createBuffer(m_frameInfo, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VMA_MEMORY_USAGE_CPU_TO_GPU));
    NVVK_DBG_NAME(m_frameInfo.buffer);
  }

  void createGraphicPipeline()
  {
    m_rasterBind.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rasterBind.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL);
    m_rasterBind.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           &m_rasterDescriptorSetLayout);
    NVVK_DBG_NAME(m_rasterDescriptorSetLayout);

    const VkPushConstantRange pushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                                .offset     = 0,
                                                .size       = sizeof(shaderio::PushConstant)};
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_rasterPipelineLayout, {m_rasterDescriptorSetLayout}, {pushConstantRange}));
    NVVK_DBG_NAME(m_rasterPipelineLayout);


    nvvk::GraphicsPipelineState graphicState;

    // Creating the Pipeline
    graphicState.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    graphicState.vertexBindings              = {{.sType   = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
                                                 .stride  = sizeof(nvutils::PrimitiveVertex),
                                                 .divisor = 1}};
    graphicState.vertexAttributes            = {{.sType  = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                                 .format = VK_FORMAT_R32G32B32_SFLOAT,
                                                 .offset = offsetof(nvutils::PrimitiveVertex, pos)}};
    // Helper to create the graphic pipeline
    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = m_rasterPipelineLayout;
    creator.colorFormats                         = {m_colorFormat};
    creator.renderingState.depthAttachmentFormat = m_depthFormat;

    std::array<VkShaderModule, 2> shaderModules{};
#if USE_SLANG
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", texture_3d_slang);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", texture_3d_slang);
#else
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", texture_3d_vert_glsl);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", texture_3d_frag_glsl);
#endif

    NVVK_CHECK(creator.createGraphicsPipeline(m_device, nullptr, graphicState, &m_graphicsPipeline));
    NVVK_DBG_NAME(m_graphicsPipeline);
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers.getColorImage(), m_gBuffers.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

private:
  nvapp::Application* m_app = nullptr;

  VkDevice m_device             = VK_NULL_HANDLE;
  bool     m_needsTextureUpdate = false;

  nvvk::ResourceAllocator m_alloc;
  nvvk::StagingUploader   m_stagingUploader;
  nvvk::SamplerPool       m_samplerPool;

  // Pipelines: compute to generate the 3d, graphic to draw, ray-marching
  nvvk::DescriptorBindings m_computeBind;
  VkDescriptorSetLayout    m_computeDescriptorSetLayout{};
  VkPipelineLayout         m_computePipelineLayout{};
  VkPipeline               m_computePipeline = VK_NULL_HANDLE;
  nvvk::DescriptorBindings m_rasterBind;
  VkDescriptorSetLayout    m_rasterDescriptorSetLayout{};
  VkPipelineLayout         m_rasterPipelineLayout{};
  VkPipeline               m_graphicsPipeline = VK_NULL_HANDLE;

  VkSemaphore m_timelineSemaphore{};
  uint64_t    m_timelineSemaphoreNextValue = 1;

  nvvk::Image   m_image;     // The 3D texture holding the perlin noise
  nvvk::GBuffer m_gBuffers;  // G-Buffers: color + depth

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

  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME}},
      .queues             = {VK_QUEUE_GRAPHICS_BIT, VK_QUEUE_TRANSFER_BIT},
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
