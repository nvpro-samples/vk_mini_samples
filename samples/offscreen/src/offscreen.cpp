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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 This sample renders in an image without creating any window context, and save the 
 rendered image to disk.
*/

#include <array>
#include <memory>

#define VMA_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION


#include "nvh/commandlineparser.hpp"
#include "nvh/timesampler.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/error_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/gbuffer.hpp"

#include "stb_image_write.h"

// Shaders
#include "shaders/device_host.h"

#if USE_HLSL
#include "_autogen/raster_vertexMain.spirv.h"
#include "_autogen/raster_fragmentMain.spirv.h"
const auto& vert_shd = std::vector<uint8_t>{std::begin(raster_vertexMain), std::end(raster_vertexMain)};
const auto& frag_shd = std::vector<uint8_t>{std::begin(raster_fragmentMain), std::end(raster_fragmentMain)};
#elif USE_SLANG
#include "_autogen/raster_vertexMain.spirv.h"
#include "_autogen/raster_fragmentMain.spirv.h"
const auto& vert_shd = std::vector<uint32_t>{std::begin(raster_vertexMain), std::end(raster_vertexMain)};
const auto& frag_shd = std::vector<uint32_t>{std::begin(raster_fragmentMain), std::end(raster_fragmentMain)};
#else
#include "_autogen/raster.frag.h"
#include "_autogen/raster.vert.h"
const auto& vert_shd = std::vector<uint32_t>{std::begin(raster_vert), std::end(raster_vert)};
const auto& frag_shd = std::vector<uint32_t>{std::begin(raster_frag), std::end(raster_frag)};
#endif  // USE_HLSL


namespace nvvkhl {
class OfflineRender
{
public:
  explicit OfflineRender(nvvk::Context* ctx)
      : m_ctx(ctx)
  {
    m_alloc   = std::make_unique<nvvkhl::AllocVma>(ctx);
    m_cmdPool = std::make_unique<nvvk::CommandPool>(m_ctx->m_device, m_ctx->m_queueGCT.familyIndex);
  }

  ~OfflineRender() { destroy(); };


  //--------------------------------------------------------------------------------------------------
  // Rendering the scene to a frame buffer
  void offlineRender(float anim_time)
  {
    const nvh::ScopedTimer s_timer("Offline rendering");

    std::array<VkClearValue, 2> clear_values{};
    clear_values[0].color        = {{0.1F, 0.1F, 0.4F, 0.F}};
    clear_values[1].depthStencil = {1.0F, 0};

    // Preparing the rendering
    VkCommandBuffer cmd = m_cmdPool->createCommandBuffer();

    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView()},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    vkCmdBeginRendering(cmd, &r_info);

    const glm::vec2 size_f = {static_cast<float>(m_gBuffers->getSize().width), static_cast<float>(m_gBuffers->getSize().height)};

    // Viewport and scissor
    const VkViewport viewport{0.0F, 0.0F, size_f.x, size_f.y, 0.0F, 1.0F};
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    const VkRect2D scissor{{0, 0}, m_gBuffers->getSize()};
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Rendering the full-screen pixel shader
    PushConstant push_c{};
    push_c.aspectRatio = size_f.x / size_f.y;
    push_c.iTime       = anim_time;

    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstant), &push_c);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdDraw(cmd, 3, 1, 0, 0);  // No vertices, it is implicitly done in the vertex shader

    // Done and submit execution
    vkCmdEndRendering(cmd);

    m_cmdPool->submitAndWait(cmd);
  }

  //--------------------------------------------------------------------------------------------------
  // Save the image to disk
  //
  void saveImage(const std::string& outFilename)
  {
    const nvh::ScopedTimer s_timer("Save Image\n");

    // Create a temporary buffer to hold the pixels of the image
    const VkBufferUsageFlags usage{VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    const VkDeviceSize buffer_size  = 4 * sizeof(uint8_t) * m_gBuffers->getSize().width * m_gBuffers->getSize().height;
    nvvk::Buffer       pixel_buffer = m_alloc->createBuffer(buffer_size, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    imageToBuffer(m_gBuffers->getColorImage(), pixel_buffer.buffer);

    // Write the buffer to disk
    LOGI(" - Size: %d, %d\n", m_gBuffers->getSize().width, m_gBuffers->getSize().height);
    LOGI(" - Bytes: %d\n", m_gBuffers->getSize().width * m_gBuffers->getSize().height * 4);
    LOGI(" - Out name: %s\n", outFilename.c_str());
    const void* data = m_alloc->map(pixel_buffer);
    stbi_write_jpg(outFilename.c_str(), m_gBuffers->getSize().width, m_gBuffers->getSize().height, 4, data, 0);
    m_alloc->unmap(pixel_buffer);

    // Destroy temporary buffer
    m_alloc->destroy(pixel_buffer);
  }


  //--------------------------------------------------------------------------------------------------
  // Copy the image to a buffer - this linearize the image memory
  //
  void imageToBuffer(const VkImage& imgIn, const VkBuffer& pixelBufferOut)
  {
    const nvh::ScopedTimer s_timer(" - Image To Buffer");

    VkCommandBuffer cmd = m_cmdPool->createCommandBuffer();

    // Make the image layout eTransferSrcOptimal to copy to buffer
    const VkImageSubresourceRange subresource_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    nvvk::cmdBarrierImageLayout(cmd, imgIn, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, subresource_range);

    // Copy the image to the buffer
    VkBufferImageCopy copy_region{};
    copy_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy_region.imageExtent      = VkExtent3D{m_gBuffers->getSize().width, m_gBuffers->getSize().height, 1};
    vkCmdCopyImageToBuffer(cmd, imgIn, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, pixelBufferOut, 1, &copy_region);

    // Put back the image as it was
    nvvk::cmdBarrierImageLayout(cmd, imgIn, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresource_range);
    m_cmdPool->submitAndWait(cmd);
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline of this example
  //
  void createPipeline()
  {
    const nvh::ScopedTimer s_timer("Create Pipeline");

    // Pipeline Layout: The layout of the shader needs only Push Constants: we are using parameters, time and aspect ratio
    const VkPushConstantRange  push_constants = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstant)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges    = &push_constants;
    NVVK_CHECK(vkCreatePipelineLayout(m_ctx->m_device, &layout_info, nullptr, &m_pipelineLayout));

    VkPipelineRenderingCreateInfo prend_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    prend_info.colorAttachmentCount    = 1;
    prend_info.pColorAttachmentFormats = &m_colorFormat;
    prend_info.depthAttachmentFormat   = m_depthFormat;

    nvvk::GraphicsPipelineState pstate;
    pstate.rasterizationState.cullMode = VK_CULL_MODE_NONE;

    nvvk::GraphicsPipelineGenerator pgen(m_ctx->m_device, m_pipelineLayout, prend_info, pstate);
    pgen.addShader(vert_shd, VK_SHADER_STAGE_VERTEX_BIT, USE_HLSL ? "vertexMain" : "main");
    pgen.addShader(frag_shd, VK_SHADER_STAGE_FRAGMENT_BIT, USE_HLSL ? "fragmentMain" : "main");

    m_pipeline = pgen.createPipeline();
  }


  //--------------------------------------------------------------------------------------------------
  // Creating an offscreen frame buffer and the associated render pass
  //
  void createFramebuffer(const VkExtent2D& size)
  {
    const nvh::ScopedTimer s_timer("Create Framebuffer");
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_ctx->m_device, m_alloc.get(), size, m_colorFormat, m_depthFormat);
  }

  void destroy()
  {
    vkDestroyPipelineLayout(m_ctx->m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_ctx->m_device, m_pipeline, nullptr);
    m_gBuffers.reset();
    m_cmdPool.reset();
    m_alloc.reset();
  }

private:
  nvvk::Context*                     m_ctx{nullptr};
  std::unique_ptr<nvvkhl::AllocVma>  m_alloc;
  std::unique_ptr<nvvk::CommandPool> m_cmdPool;
  std::unique_ptr<nvvkhl::GBuffer>   m_gBuffers;

  VkClearColorValue m_clearColor{{0.1F, 0.4F, 0.1F, 1.0F}};  // Clear color
  VkPipelineLayout  m_pipelineLayout{VK_NULL_HANDLE};        // The description of the pipeline
  VkPipeline        m_pipeline{VK_NULL_HANDLE};              // The graphic pipeline to render
  VkFormat          m_colorFormat{VK_FORMAT_R8G8B8A8_UNORM};
  VkFormat          m_depthFormat{VK_FORMAT_D32_SFLOAT};
};
}  // namespace nvvkhl

int main(int argc, char** argv)
{
  // Logging to file
  const std::string logfile = std::string("log_") + std::string(PROJECT_NAME) + std::string(".txt");
  nvprintSetLogFileName(logfile.c_str());

  float       anim_time{0.0F};
  VkExtent2D  render_size{800, 600};
  std::string output_file{"result.jpg"};

  nvh::CommandLineParser parser("Offline Render");
  parser.addArgument({"-t", "--time"}, &anim_time, "Animation time");
  parser.addArgument({"-w", "--width"}, &render_size.width, "Render size width");
  parser.addArgument({"-h", "--height"}, &render_size.height, "Render size height");
  parser.addArgument({"-o", "--output"}, &output_file, "Output filename (must end with .jpg)");
  if(!parser.parse(argc, argv))
  {
    parser.printHelp();
    return 1;
  }

  // Creating the Vulkan instance and device, with only defaults, no extension
  nvvk::Context           vkctx;
  nvvk::ContextCreateInfo vkctx_info{};
  vkctx_info.apiMajor = 1;
  vkctx_info.apiMinor = 3;
  vkctx.init(vkctx_info);


  // Create the application
  auto app = std::make_unique<nvvkhl::OfflineRender>(&vkctx);

  app->createFramebuffer(render_size);  // Framebuffer where it will render
  app->createPipeline();                // How the quad will be rendered: shaders and more
  app->offlineRender(anim_time);        // Rendering

  app->saveImage(output_file);  // Saving rendered image

  app.reset();
  vkctx.deinit();

  return 0;
}
