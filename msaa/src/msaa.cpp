/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#include "msaa.hpp"
#include "nvvk/pipeline_vk.hpp"


#include "_autogen/raster.vert.h"
#include "_autogen/raster.frag.h"
#include "_autogen/raster_overlay.frag.h"
#include "nvvk/images_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"


//--------------------------------------------------------------------------------------------------
// Override: forcing to be in raster
//
void MsaaSample::create(const nvvk::AppBaseVkCreateInfo& info)
{
  VulkanSample::create(info);
  m_renderMode = RenderMode::eRaster;
}


//--------------------------------------------------------------------------------------------------
// Override: Create `createMsaaRender()` and no ray tracer
//
void MsaaSample::createScene(const std::string& filename)
{
  nvh::Stopwatch sw;
  LOGI("\nCreate Sample\n");

  loadScene(filename);
  fitCamera(m_gltfScene.m_dimensions.min, m_gltfScene.m_dimensions.max);

  LOGI("- Pipeline creation\n");
  {  // Graphic
    nvh::Stopwatch sw_;

    createUniformBuffer();
    createOffscreenRender();
    createMsaaImages(m_msaaSamples);  // #MSAA
    createGraphicPipeline();
    LOGI(" - %6.2fms: Graphic\n", sw_.elapsed());
  }

  {  // Post
    nvh::Stopwatch sw_;
    createPostPipeline();
    updatePostDescriptorSet(m_offscreenColor.descriptor);
    LOGI(" - %6.2fms: Post\n", sw_.elapsed());
  }

  LOGI("TOTAL: %7.2fms\n\n", sw.elapsed());
}

//--------------------------------------------------------------------------------------------------
// Override: using MSAA renderpass and MSAA nb sample information
// See #MSAA
//
void MsaaSample::createGraphicPipeline()
{
  auto& p = m_pContainer[eGraphic];

  // Descriptors
  nvvk::DescriptorSetBindings bind;
  auto                        nbTextures = static_cast<uint32_t>(m_textures.size());
  bind.addBinding(SceneBindings::eFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
  bind.addBinding(SceneBindings::eSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
  bind.addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nbTextures, VK_SHADER_STAGE_ALL);

  p.dstLayout = bind.createLayout(m_device);
  p.dstPool   = bind.createPool(m_device, 1);
  p.dstSet    = nvvk::allocateDescriptorSet(m_device, p.dstPool, p.dstLayout);

  // Writing to descriptors
  VkDescriptorBufferInfo             dbiUnif{m_frameInfoBuf.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo             sceneDesc{m_sceneDesc.buffer, 0, VK_WHOLE_SIZE};
  std::vector<VkDescriptorImageInfo> diit;
  std::vector<VkWriteDescriptorSet>  writes;
  writes.emplace_back(bind.makeWrite(p.dstSet, SceneBindings::eFrameInfo, &dbiUnif));
  writes.emplace_back(bind.makeWrite(p.dstSet, SceneBindings::eSceneDesc, &sceneDesc));
  for(auto& texture : m_textures)  // All texture samplers
    diit.emplace_back(texture.descriptor);
  writes.emplace_back(bind.makeWriteArray(p.dstSet, SceneBindings::eTextures, diit.data()));
  // Writing the information
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  // Creating the Pipeline Layout
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(RasterPushConstant)};
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &p.dstLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &p.pipelineLayout);

  // Shader source (Spir-V)
  std::vector<uint32_t> vertexShader(std::begin(raster_vert), std::end(raster_vert));
  std::vector<uint32_t> fragShader(std::begin(raster_frag), std::end(raster_frag));

  VkPipelineRenderingCreateInfoKHR rfInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
  rfInfo.colorAttachmentCount    = 1;
  rfInfo.pColorAttachmentFormats = &m_offscreenColorFormat;
  rfInfo.depthAttachmentFormat   = m_offscreenDepthFormat;
  rfInfo.stencilAttachmentFormat = m_offscreenDepthFormat;

  // Creating the Pipeline
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, p.pipelineLayout, nullptr /*pass*/);
  gpb.rasterizationState.depthBiasEnable         = VK_TRUE;
  gpb.rasterizationState.depthBiasConstantFactor = -1;
  gpb.rasterizationState.depthBiasSlopeFactor    = 1;
  gpb.multisampleState.rasterizationSamples      = m_msaaSamples;  // #MSAA
  gpb.addShader(vertexShader, VK_SHADER_STAGE_VERTEX_BIT);
  gpb.addShader(fragShader, VK_SHADER_STAGE_FRAGMENT_BIT);
  gpb.addBindingDescriptions({{0, sizeof(Vertex)}});
  gpb.addAttributeDescriptions({
      {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, position)},  // Position + texcoord U
      {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, normal)},    // Normal + texcoord V
      {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent)},   // Tangents
  });
  gpb.createInfo.pNext = &rfInfo;
  p.pipeline           = gpb.createPipeline();
  NAME2_VK(p.pipeline, "Graphics");

  // Wireframe
  {
    gpb.clearShaders();
    std::vector<uint32_t> fragShader(std::begin(raster_overlay_frag), std::end(raster_overlay_frag));
    gpb.addShader(vertexShader, VK_SHADER_STAGE_VERTEX_BIT);
    gpb.addShader(fragShader, VK_SHADER_STAGE_FRAGMENT_BIT);
    gpb.rasterizationState.depthBiasEnable = VK_FALSE;
    gpb.rasterizationState.polygonMode     = VK_POLYGON_MODE_LINE;
    gpb.rasterizationState.lineWidth       = 2.0f;
    gpb.depthStencilState.depthWriteEnable = VK_FALSE;
    m_wireframe                            = gpb.createPipeline();
    NAME2_VK(m_wireframe, "Wireframe");
  }
}


//--------------------------------------------------------------------------------------------------
// Override: freeing up also the MSAA information
//
void MsaaSample::freeResources()
{
  VulkanSample::freeResources();

  //#MSAA
  m_alloc.destroy(m_msaaColor);
  m_alloc.destroy(m_msaaDepth);
  vkDestroyImageView(m_device, m_msaaColorIView, nullptr);
  vkDestroyImageView(m_device, m_msaaDepthIView, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Override: using m_msaaRenderPass when samples are > 1
// See #MSAA
//
void MsaaSample::rasterize(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);

  // Recording the commands to draw the scene if not done yet
  if(m_recordedCmdBuffer[0] == VK_NULL_HANDLE)
    recordRendering();

  if(m_msaaSamples == VK_SAMPLE_COUNT_1_BIT)  //  #MSAA
  {
    nvvk::createRenderingInfo rInfo({{0, 0}, getSize()}, {m_offscreenColor.descriptor.imageView},
                                    m_offscreenDepth.descriptor.imageView, VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_LOAD_OP_CLEAR,
                                    m_clearColor, {1.0f, 0}, VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT_KHR);
    vkCmdBeginRendering(cmdBuf, &rInfo);
  }
  else
  {
    nvvk::createRenderingInfo rInfo({{0, 0}, getSize()}, {m_msaaColorIView}, m_msaaDepthIView,
                                    VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor, {1.0f, 0},
                                    VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT_KHR);
    rInfo.colorAttachments[0].resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;
    rInfo.colorAttachments[0].resolveImageView = m_offscreenColor.descriptor.imageView;  // Resolving MSAA to offscreen
    rInfo.colorAttachments[0].resolveMode      = VK_RESOLVE_MODE_AVERAGE_BIT;
    vkCmdBeginRendering(cmdBuf, &rInfo);
  }


  // Executing the drawing of the recorded commands
  vkCmdExecuteCommands(cmdBuf, m_showWireframe ? 2 : 1, m_recordedCmdBuffer.data());
  vkCmdEndRendering(cmdBuf);

  // <ake sure it is finished
  auto imageMemoryBarrier = nvvk::makeImageMemoryBarrier(m_offscreenColor.image, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                       nullptr, 0, nullptr, 1, &imageMemoryBarrier);
}


//--------------------------------------------------------------------------------------------------
// Record commands for rendering fill and wireframe
//
void MsaaSample::recordRendering()
{
  // Create the command buffer to record the drawing commands
  VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocInfo.commandPool        = m_cmdPool;
  allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
  allocInfo.commandBufferCount = 2;
  vkAllocateCommandBuffers(m_device, &allocInfo, m_recordedCmdBuffer.data());

  VkCommandBufferInheritanceRenderingInfoKHR inheritanceRenderingInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO_KHR};
  inheritanceRenderingInfo.colorAttachmentCount    = 1;
  inheritanceRenderingInfo.pColorAttachmentFormats = &m_offscreenColorFormat;
  inheritanceRenderingInfo.depthAttachmentFormat   = m_offscreenDepthFormat;
  inheritanceRenderingInfo.stencilAttachmentFormat = m_offscreenDepthFormat;
  inheritanceRenderingInfo.rasterizationSamples    = m_msaaSamples;

  VkCommandBufferInheritanceInfo inheritInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
  inheritInfo.pNext = &inheritanceRenderingInfo;

  for(int i = 0; i < 2; i++)
  {
    nvh::Stopwatch sw;
    auto&          p = m_pContainer[eGraphic];

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    beginInfo.pInheritanceInfo = &inheritInfo;
    vkBeginCommandBuffer(m_recordedCmdBuffer[i], &beginInfo);

    // Dynamic Viewport
    setViewport(m_recordedCmdBuffer[i]);

    // Drawing all instances
    vkCmdBindPipeline(m_recordedCmdBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, i == 0 ? p.pipeline : m_wireframe);
    vkCmdBindDescriptorSets(m_recordedCmdBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipelineLayout, 0, 1, &p.dstSet, 0, nullptr);

    uint32_t     nodeId{0};
    VkDeviceSize offsets{0};
    for(auto& node : m_gltfScene.m_nodes)
    {
      auto& primitive = m_gltfScene.m_primMeshes[node.primMesh];
      // Push constant information
      m_pcRaster.materialId = primitive.materialIndex;
      m_pcRaster.instanceId = nodeId++;
      vkCmdPushConstants(m_recordedCmdBuffer[i], p.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(RasterPushConstant), &m_pcRaster);

      vkCmdBindVertexBuffers(m_recordedCmdBuffer[i], 0, 1, &m_vertices[node.primMesh].buffer, &offsets);
      vkCmdBindIndexBuffer(m_recordedCmdBuffer[i], m_indices[node.primMesh].buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(m_recordedCmdBuffer[i], primitive.indexCount, 1, 0, 0, 0);
    }
    vkEndCommandBuffer(m_recordedCmdBuffer[i]);
    LOGI("Recoreded Command Buffer: %7.2fms\n", sw.elapsed());
  }
}

//--------------------------------------------------------------------------------------------------
// Override: add MSAA and remove ray tracing
//
void MsaaSample::onResize(int /*w*/, int /*h*/)
{
  vkFreeCommandBuffers(m_device, m_cmdPool, 2, m_recordedCmdBuffer.data());
  m_recordedCmdBuffer = {VK_NULL_HANDLE};

  createOffscreenRender();
  updatePostDescriptorSet(m_offscreenColor.descriptor);
  resetFrame();
  createMsaaImages(m_msaaSamples);  // #MSAA
}


//--------------------------------------------------------------------------------------------------
// Creating MSAA image/depth and renderpass resolving in the offscreen image
//
// #MSAA
void MsaaSample::createMsaaImages(VkSampleCountFlagBits samples)
{
  m_alloc.destroy(m_msaaColor);
  m_alloc.destroy(m_msaaDepth);
  vkDestroyImageView(m_device, m_msaaColorIView, nullptr);
  vkDestroyImageView(m_device, m_msaaDepthIView, nullptr);

  // Default create image info
  VkImageCreateInfo createInfo = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  createInfo.imageType         = VK_IMAGE_TYPE_2D;
  createInfo.samples           = samples;  // #MSAA
  createInfo.mipLevels         = 1;
  createInfo.arrayLayers       = 1;
  createInfo.extent.width      = m_size.width;
  createInfo.extent.height     = m_size.height;
  createInfo.extent.depth      = 1;

  // Creating color
  {
    createInfo.format = m_offscreenColorFormat;
    createInfo.usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;  // #MSAA - Optimization
    m_msaaColor = m_alloc.createImage(createInfo);
    NAME_VK(m_msaaColor.image);
    auto ivInfo = nvvk::makeImageViewCreateInfo(m_msaaColor.image, createInfo);
    vkCreateImageView(m_device, &ivInfo, nullptr, &m_msaaColorIView);
  }

  // Creating the depth buffer
  {
    createInfo.format = m_offscreenDepthFormat;
    createInfo.usage  = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

    m_msaaDepth = m_alloc.createImage(createInfo);
    NAME_VK(m_msaaDepth.image);
    auto ivInfo                        = nvvk::makeImageViewCreateInfo(m_msaaDepth.image, createInfo);
    ivInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vkCreateImageView(m_device, &ivInfo, nullptr, &m_msaaDepthIView);
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_msaaColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_msaaDepth.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    genCmdBuf.submitAndWait(cmdBuf);
  }
}


inline uint32_t floor_log2i(uint32_t v)
{
  uint32_t l = 0;
  while(v > 1U)
  {
    v >>= 1;
    l++;
  }
  return l;
}
//--------------------------------------------------------------------------------------------------
// Rendering UI
//
void MsaaSample::renderUI()
{
  if(showGui() == false)
    return;

  bool changed{false};

  ImGuiH::Panel::Begin();
  ImGui::Checkbox("Show wireframe", &m_showWireframe);
  if(ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGuiH::CameraWidget();
  }

  if(ImGui::CollapsingHeader("Environment", ImGuiTreeNodeFlags_DefaultOpen))
  {
    changed |= ImGui::ColorEdit3("Color", &m_clearColor.float32[0], ImGuiColorEditFlags_Float);
    if(ImGui::TreeNode("Tonemapper"))
    {
      ImGui::SliderFloat("Exposure", &m_tonemapper.exposure, 0.001f, 5.0f);
      ImGui::SliderFloat("Brightness", &m_tonemapper.brightness, 0.0f, 2.0f);
      ImGui::SliderFloat("Contrast", &m_tonemapper.contrast, 0.0f, 2.0f);
      ImGui::SliderFloat("Saturation", &m_tonemapper.saturation, 0.0f, 2.0f);
      ImGui::SliderFloat("Vignette", &m_tonemapper.vignette, 0.0f, 1.0f);
      ImGui::TreePop();
    }
  }


  // #MSAA
  VkImageFormatProperties imageFormatProperties;
  vkGetPhysicalDeviceImageFormatProperties(m_physicalDevice, m_offscreenColorFormat, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                           VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, 0, &imageFormatProperties);
  auto        maxSamples = static_cast<uint32_t>(log2(imageFormatProperties.sampleCounts) + 1);
  int         item_combo = static_cast<uint32_t>(log2((double)m_msaaSamples));
  const char* items[]    = {"1", "2", "4", "8", "16", "32", "64"};
  ImGui::Text("Sample Count");
  ImGui::SameLine();
  if(ImGui::Combo("##Sample Count", &item_combo, items, maxSamples))
  {
    int32_t samples = static_cast<int32_t>(powf(2, (float)item_combo));
    m_msaaSamples   = static_cast<VkSampleCountFlagBits>(samples);

    vkDeviceWaitIdle(m_device);  // Flushing the graphic pipeline

    // The graphic pipeline will use offscreen or msaa render pass
    vkDestroyPipeline(m_device, m_pContainer[eGraphic].pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pContainer[eGraphic].pipelineLayout, nullptr);
    vkDestroyDescriptorPool(m_device, m_pContainer[eGraphic].dstPool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_pContainer[eGraphic].dstLayout, nullptr);
    vkDestroyPipeline(m_device, m_wireframe, nullptr);
    createMsaaImages(m_msaaSamples);  // #MSAA
    createGraphicPipeline();

    // Need to re-record the scene, dependency on renderpass
    vkFreeCommandBuffers(m_device, m_cmdPool, 2, m_recordedCmdBuffer.data());
    m_recordedCmdBuffer = {VK_NULL_HANDLE};
  }


  ImGuiH::Panel::End();

  if(changed)
    resetFrame();
}
