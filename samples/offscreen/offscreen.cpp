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
 This sample renders in an image without creating any window context, and save the 
 rendered image to disk.
*/

#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define STBIW_WINDOWS_UTF8
#define STB_IMAGE_WRITE_IMPLEMENTATION
#pragma warning(disable : 4996)  // sprintf warning
#define VMA_IMPLEMENTATION
#include <glm/glm.hpp>
#include <stb/stb_image_write.h>
#include <vulkan/vulkan_core.h>

// Shaders
#include "shaders/shaderio.h"  // Shared between host and device

#include "_autogen/offscreen.frag.glsl.h"
#include "_autogen/offscreen.slang.h"
#include "_autogen/offscreen.vert.glsl.h"

#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/commands.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/resources.hpp>


namespace nvvkhl {
class OfflineRender
{
public:
  OfflineRender()  = default;
  ~OfflineRender() = default;

  void init(VkInstance instance, VkDevice device, VkPhysicalDevice physicalDevice, const nvvk::QueueInfo& queue)
  {
    m_device = device;
    m_queue  = queue;

    // Initialize the resource allocator
    m_alloc.init({.physicalDevice = physicalDevice, .device = device, .instance = instance});

    // Initialize the GBuffer
    m_gBuffers.init({.allocator = &m_alloc, .colorFormats = {VK_FORMAT_R8G8B8A8_UNORM}});

    // Create a command pool, for creating the command buffer
    const VkCommandPoolCreateInfo commandPoolCreateInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,  // Hint that commands will be short-lived
        .queueFamilyIndex = m_queue.familyIndex,
    };
    NVVK_CHECK(vkCreateCommandPool(m_device, &commandPoolCreateInfo, nullptr, &m_commandPool));
    NVVK_DBG_NAME(m_commandPool);
  }

  //--------------------------------------------------------------------------------------------------
  // Rendering the scene to a frame buffer
  void offlineRender(float animTime)
  {
    const nvutils::ScopedTimer s_timer("Offline rendering");

    std::array<VkClearValue, 2> clear_values{};
    clear_values[0].color        = {{0.1F, 0.1F, 0.4F, 0.F}};
    clear_values[1].depthStencil = {1.0F, 0};

    // Preparing the rendering
    VkCommandBuffer cmd;
    NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, m_commandPool));

    // Render the scene to the frame buffer (GBuffer)
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    nvvk::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, m_gBuffers.getSize());

    // Dynamic rendering
    VkRenderingAttachmentInfoKHR colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                    = m_gBuffers.getColorImageView();

    VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea           = DEFAULT_VkRect2D(m_gBuffers.getSize());
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;

    vkCmdBeginRendering(cmd, &renderingInfo);

    // Pushing the time and aspect ratio to the shader
    shaderio::PushConstant pushConstant{
        .iTime       = animTime,
        .aspectRatio = m_gBuffers.getAspectRatio(),
    };
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(shaderio::PushConstant), &pushConstant);

    // Rendering the scene
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdDraw(cmd, 3, 1, 0, 0);  // No vertices, it is implicitly done in the vertex shader

    // Done and submit execution
    vkCmdEndRendering(cmd);
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
    NVVK_CHECK(nvvk::endSingleTimeCommands(cmd, m_device, m_commandPool, m_queue.queue));
  }

  //--------------------------------------------------------------------------------------------------
  // Save the image to disk
  //
  void saveImage(const std::filesystem::path& outFilename)
  {
    const nvutils::ScopedTimer s_timer("Save Image\n");

    // Create a temporary buffer to hold the pixels of the image
    const VkBufferUsageFlags usage{VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT};
    const VkDeviceSize bufferSize = 4 * sizeof(uint8_t) * m_gBuffers.getSize().width * m_gBuffers.getSize().height;
    nvvk::Buffer       pixelBuffer;
    NVVK_CHECK(m_alloc.createBuffer(pixelBuffer, bufferSize, usage, VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));
    NVVK_DBG_NAME(pixelBuffer.buffer);

    imageToBuffer(m_gBuffers.getColorImage(), pixelBuffer.buffer);

    // Write the buffer to disk
    const std::string outFilenameUtf8 = nvutils::utf8FromPath(outFilename);
    LOGI(" - Size: %d, %d\n", m_gBuffers.getSize().width, m_gBuffers.getSize().height);
    LOGI(" - Bytes: %d\n", m_gBuffers.getSize().width * m_gBuffers.getSize().height * 4);
    LOGI(" - Out name: %s\n", outFilenameUtf8.c_str());
    const void* data = pixelBuffer.mapping;
    stbi_write_jpg(outFilenameUtf8.c_str(), m_gBuffers.getSize().width, m_gBuffers.getSize().height, 4, data, 100);

    // Destroy temporary buffer
    m_alloc.destroyBuffer(pixelBuffer);
  }


  //--------------------------------------------------------------------------------------------------
  // Copy the image to a buffer - this linearize the image memory
  //
  void imageToBuffer(const VkImage& imageIn, const VkBuffer& pixelBufferOut) const
  {
    const nvutils::ScopedTimer s_timer(" - Image To Buffer");

    VkCommandBuffer cmd;
    NVVK_CHECK(nvvk::beginSingleTimeCommands(cmd, m_device, m_commandPool));

    // Set the image to the right layout
    nvvk::cmdImageMemoryBarrier(cmd, {imageIn, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL});

    // Copy the image to the buffer
    VkBufferImageCopy copyRegion{};
    copyRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copyRegion.imageExtent      = VkExtent3D{m_gBuffers.getSize().width, m_gBuffers.getSize().height, 1};
    vkCmdCopyImageToBuffer(cmd, imageIn, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pixelBufferOut, 1, &copyRegion);

    // Put back the image as it was
    nvvk::cmdImageMemoryBarrier(cmd, {imageIn, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});

    // Submit the command buffer and wait for it to finish
    NVVK_CHECK(nvvk::endSingleTimeCommands(cmd, m_device, m_commandPool, m_queue.queue));
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline of this example
  //
  void createPipeline()
  {
    const nvutils::ScopedTimer s_timer("Create Pipeline");

    // Pipeline Layout: The layout of the shader needs only Push Constants: we are using parameters, time and aspect ratio
    const VkPushConstantRange  pushConstants = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(shaderio::PushConstant)};
    VkPipelineLayoutCreateInfo layoutInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstants,
    };
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipelineLayout));

    // Holds the state of the graphic pipeline
    nvvk::GraphicsPipelineState graphicState;
    graphicState.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    // Helper to create the graphic pipeline
    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout = m_pipelineLayout;


#if USE_SLANG
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", offscreen_slang);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", offscreen_slang);
#else
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", offscreen_vert_glsl);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", offscreen_frag_glsl);
#endif

    NVVK_CHECK(creator.createGraphicsPipeline(m_device, nullptr, graphicState, &m_pipeline));
    NVVK_DBG_NAME(m_pipeline);
  }


  //--------------------------------------------------------------------------------------------------
  // Creating an off screen frame buffer and the associated render pass
  //
  void createFramebuffer(const VkExtent2D& size)
  {
    const nvutils::ScopedTimer s_timer("Create Framebuffer");
    VkCommandBuffer            cmd;
    nvvk::beginSingleTimeCommands(cmd, m_device, m_commandPool);
    NVVK_CHECK(m_gBuffers.update(cmd, size));
    nvvk::endSingleTimeCommands(cmd, m_device, m_commandPool, m_queue.queue);
  }

  void deinit()
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyCommandPool(m_device, m_commandPool, nullptr);

    m_gBuffers.deinit();
    m_alloc.deinit();
  }

private:
  nvvk::ResourceAllocator m_alloc;     // Resource allocator for the buffers
  nvvk::GBuffer           m_gBuffers;  // GBuffer for the offscreen rendering

  nvvk::QueueInfo  m_queue;             // Queue information
  VkCommandPool    m_commandPool{};     // Command pool for the command buffer
  VkDevice         m_device{};          // Vulkan device
  VkPipelineLayout m_pipelineLayout{};  // The description of the pipeline
  VkPipeline       m_pipeline{};        // The graphic pipeline to render
};
}  // namespace nvvkhl

int main(int argc, char** argv)
{
  float                 animTime{0.0F};
  glm::uvec2            renderSize{800, 600};
  std::filesystem::path outputFile = nvutils::getExecutablePath().replace_extension("jpg");

  std::string                projectName = nvutils::getExecutablePath().stem().string();
  nvutils::ParameterParser   cli(projectName);
  nvutils::ParameterRegistry reg;
  reg.add({"time", "Run in headless mode", "t"}, &animTime, true);
  reg.addVector({"size", "Render size width", "s"}, &renderSize);
  reg.add({"output", "Output filename (must end with .jpg)", "o"}, &outputFile);
  cli.add(reg);
  cli.parse(argc, argv);


  // Creating the Vulkan instance and device, with only defaults, no extension
  nvvk::ContextInitInfo vkSetup{.instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME}};
  nvvk::Context         vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  // Create the application
  nvvkhl::OfflineRender app;
  app.init(vkContext.getInstance(), vkContext.getDevice(), vkContext.getPhysicalDevice(), vkContext.getQueueInfo(0));
  app.createFramebuffer({renderSize.x, renderSize.y});  // Framebuffer where it will render
  app.createPipeline();                                 // How the quad will be rendered: shaders and more
  app.offlineRender(animTime);                          // Rendering
  app.saveImage(outputFile);                            // Saving rendered image
  app.deinit();                                         // Destroying the application

  vkContext.deinit();

  return 0;
}
