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

#include "inherited.hpp"

//--------------------------------------------------------------------------------------------------
// Overload: requesting inherited feature
//
void InheritedSample::create(const nvvk::AppBaseVkCreateInfo& info)
{
  VulkanSample::create(info);

  // Raster mode only
  m_renderMode = RenderMode::eRaster;

  // Requesting if inherited viewport is supported
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &m_inheritedViewport;
  vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features);
}

//--------------------------------------------------------------------------------------------------
// Oveload: no raytracing
//
void InheritedSample::createScene(const std::string& filename)
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
    createGraphicPipeline();
    LOGI(" - %6.2fms: Graphic\n", sw_.elapsed());
  }
  //{  // Ray tracing
  //  nvh::Stopwatch sw_;
  //  initRayTracing();
  //  createBottomLevelAS();
  //  createTopLevelAS();
  //  createRtPipeline();
  //  LOGI(" - %6.2fms: Ray tracing\n", sw_.elapsed());
  //}
  {  // Post
    nvh::Stopwatch sw_;
    createPostPipeline();
    updatePostDescriptorSet(m_offscreenColor.descriptor);
    LOGI(" - %6.2fms: Post\n", sw_.elapsed());
  }

  LOGI("TOTAL: %7.2fms\n\n", sw.elapsed());
}


//--------------------------------------------------------------------------------------------------
// Overload: using inherited feature
//
void InheritedSample::rasterize(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);

  // Recording the commands to draw the scene if not done yet
  if(m_recordedCmdBuffer[0] == VK_NULL_HANDLE)
  {
    nvh::Stopwatch sw;
    // Create the command buffer to record the drawing commands
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool        = m_cmdPool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    allocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(m_device, &allocInfo, m_recordedCmdBuffer.data());


    // #inherit
    // The extension struct needed to enable inheriting 2D viewport+scisors.
    VkCommandBufferInheritanceViewportScissorInfoNV inheritViewportInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_VIEWPORT_SCISSOR_INFO_NV};
    inheritViewportInfo.viewportScissor2D  = m_inheritedViewport.inheritedViewportScissor2D;
    inheritViewportInfo.viewportDepthCount = 1;
    inheritViewportInfo.pViewportDepths    = &m_viewportDepth;

    VkCommandBufferInheritanceInfo inheritInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
    inheritInfo.renderPass = m_offscreenRenderPass;
    inheritInfo.pNext      = &inheritViewportInfo;

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    beginInfo.pInheritanceInfo = &inheritInfo;
    vkBeginCommandBuffer(m_recordedCmdBuffer[0], &beginInfo);


    // Dynamic Viewport
    if(m_inheritedViewport.inheritedViewportScissor2D == VK_FALSE)
      setViewport(m_recordedCmdBuffer[0]);


    // Drawing all instances
    auto& p = m_pContainer[eGraphic];
    vkCmdBindPipeline(m_recordedCmdBuffer[0], VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipeline);
    vkCmdBindDescriptorSets(m_recordedCmdBuffer[0], VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipelineLayout, 0, 1, &p.dstSet, 0, nullptr);

    uint32_t     nodeId{0};
    VkDeviceSize offsets{0};
    for(auto& node : m_gltfScene.m_nodes)
    {
      auto& primitive = m_gltfScene.m_primMeshes[node.primMesh];
      // Push constant information
      m_pcRaster.materialId = primitive.materialIndex;
      m_pcRaster.instanceId = nodeId++;
      vkCmdPushConstants(m_recordedCmdBuffer[0], p.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(RasterPushConstant), &m_pcRaster);

      vkCmdBindVertexBuffers(m_recordedCmdBuffer[0], 0, 1, &m_vertices[node.primMesh].buffer, &offsets);
      vkCmdBindIndexBuffer(m_recordedCmdBuffer[0], m_indices[node.primMesh].buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(m_recordedCmdBuffer[0], primitive.indexCount, 1, 0, 0, 0);
    }
    vkEndCommandBuffer(m_recordedCmdBuffer[0]);
    LOGI("Recoreded Command Buffer: %7.2fms\n", sw.elapsed());
  }

  // Executing the drawing of the recorded commands
  vkCmdExecuteCommands(cmdBuf, 1, m_recordedCmdBuffer.data());
}

//--------------------------------------------------------------------------------------------------
// Overload: resize no longer need to delete the recorded command buffer
//
void InheritedSample::onResize(int /*w*/, int /*h*/)
{
  if(m_inheritedViewport.inheritedViewportScissor2D == VK_FALSE)
  {
    vkFreeCommandBuffers(m_device, m_cmdPool, (uint32_t)m_recordedCmdBuffer.size(), m_recordedCmdBuffer.data());
    m_recordedCmdBuffer = {VK_NULL_HANDLE};
  }

  createOffscreenRender();
  updatePostDescriptorSet(m_offscreenColor.descriptor);
  updateRtDescriptorSet();
  resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Rendering UI
//
void InheritedSample::renderUI()
{
  if(showGui() == false)
    return;

  ImGuiH::Panel::Begin();

  if(ImGui::Checkbox("Inherited Viewport", (bool*)&m_inheritedViewport.inheritedViewportScissor2D))
  {
    vkDeviceWaitIdle(m_device);
    vkFreeCommandBuffers(m_device, m_cmdPool, (uint32_t)m_recordedCmdBuffer.size(), m_recordedCmdBuffer.data());
    m_recordedCmdBuffer = {VK_NULL_HANDLE};
  }
  ImGui::SliderFloat2("View Center", &m_viewCenter.x, 0.2f, 0.8f);

  ImGuiH::Panel::End();
}


//--------------------------------------------------------------------------------------------------
// Modified: using size for aspect ratio
//
void InheritedSample::updateUniformBuffer(VkCommandBuffer cmdBuf, const VkExtent2D& size)
{
  // Prepare new UBO contents on host.
  LABEL_SCOPE_VK(cmdBuf);
  CameraManip.updateAnim();

  // Prepare new UBO contents on host.
  const float aspectRatio = size.width / static_cast<float>(size.height);
  auto&       clip        = CameraManip.getClipPlanes();

  FrameInfo hostUBO{};
  hostUBO.view       = CameraManip.getMatrix();
  hostUBO.proj       = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, clip.x, clip.y);
  hostUBO.viewInv    = nvmath::invert(hostUBO.view);
  hostUBO.projInv    = nvmath::invert(hostUBO.proj);
  hostUBO.light[0]   = m_lights[0];
  hostUBO.light[1]   = m_lights[1];
  hostUBO.clearColor = m_clearColor.float32;

  // Schedule the host-to-device upload. (hostUBO is copied into the cmd buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_frameInfo.buffer, 0, sizeof(FrameInfo), &hostUBO);
}


//--------------------------------------------------------------------------------------------------
// Rendering the scene in 4 different corners using always the same
// camera perspective (see: updateUniformBuffer())
//
void InheritedSample::fourViews(VkCommandBuffer cmdBuf)
{
  // Clearing values
  std::array<VkClearValue, 2> clearValues{};
  clearValues[0].color        = m_clearColor;
  clearValues[1].depthStencil = {1.0f, 0};

  auto*             c = &m_clearColor.float32[0];
  VkClearColorValue clearDarker{c[0] * 0.8f, c[1] * 0.8f, c[2] * 0.8f, c[3]};

  // Viewport information, same min/max depth for all
  VkViewport viewport{};
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;
  VkRect2D scissor{};

  // Center of the divided window
  nvmath::vec2f midPoint{m_size.width * m_viewCenter.x, m_size.height * m_viewCenter.y};

  // Same rendering pass default values, except for `renderArea`
  VkRenderPassBeginInfo offscreenRenderPassBeginInfo{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
  offscreenRenderPassBeginInfo.clearValueCount = 2;
  offscreenRenderPassBeginInfo.pClearValues    = clearValues.data();
  offscreenRenderPassBeginInfo.renderPass      = m_offscreenRenderPass;
  offscreenRenderPassBeginInfo.framebuffer     = m_offscreenFramebuffer;

  // Upper left
  viewport.x      = 0.0f;
  viewport.y      = 0.0f;
  viewport.width  = midPoint.x;
  viewport.height = midPoint.y;
  vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
  scissor.offset.x      = (int32_t)viewport.x;
  scissor.offset.y      = (int32_t)viewport.y;
  scissor.extent.width  = (uint32_t)viewport.width;
  scissor.extent.height = (uint32_t)viewport.height;
  vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
  updateUniformBuffer(cmdBuf, scissor.extent);
  offscreenRenderPassBeginInfo.renderArea = {scissor.offset, scissor.extent};
  clearValues[0].color                    = m_clearColor;
  vkCmdBeginRenderPass(cmdBuf, &offscreenRenderPassBeginInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
  rasterize(cmdBuf);
  vkCmdEndRenderPass(cmdBuf);

  // upper right
  viewport.x      = midPoint.x;
  viewport.y      = 0.0f;
  viewport.width  = m_size.width - midPoint.x;
  viewport.height = midPoint.y;
  vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
  scissor.offset.x      = (int32_t)viewport.x;
  scissor.offset.y      = (int32_t)viewport.y;
  scissor.extent.width  = (uint32_t)viewport.width;
  scissor.extent.height = (uint32_t)viewport.height;
  vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
  updateUniformBuffer(cmdBuf, scissor.extent);
  offscreenRenderPassBeginInfo.renderArea = {scissor.offset, scissor.extent};
  clearValues[0].color                    = clearDarker;
  vkCmdBeginRenderPass(cmdBuf, &offscreenRenderPassBeginInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
  rasterize(cmdBuf);
  vkCmdEndRenderPass(cmdBuf);

  // Lower left
  viewport.x      = 0.0;
  viewport.y      = midPoint.y;
  viewport.width  = midPoint.x;
  viewport.height = m_size.height - midPoint.y;
  vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
  scissor.offset.x      = (int32_t)viewport.x;
  scissor.offset.y      = (int32_t)viewport.y;
  scissor.extent.width  = (uint32_t)viewport.width;
  scissor.extent.height = (uint32_t)viewport.height;
  vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
  updateUniformBuffer(cmdBuf, scissor.extent);
  offscreenRenderPassBeginInfo.renderArea = {scissor.offset, scissor.extent};
  clearValues[0].color                    = clearDarker;
  vkCmdBeginRenderPass(cmdBuf, &offscreenRenderPassBeginInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
  rasterize(cmdBuf);
  vkCmdEndRenderPass(cmdBuf);

  // Lower right
  viewport.x      = midPoint.x;
  viewport.y      = midPoint.y;
  viewport.width  = m_size.width - midPoint.x;
  viewport.height = m_size.height - midPoint.y;
  vkCmdSetViewport(cmdBuf, 0, 1, &viewport);
  scissor.offset.x      = (int32_t)viewport.x;
  scissor.offset.y      = (int32_t)viewport.y;
  scissor.extent.width  = (uint32_t)viewport.width;
  scissor.extent.height = (uint32_t)viewport.height;
  vkCmdSetScissor(cmdBuf, 0, 1, &scissor);
  updateUniformBuffer(cmdBuf, scissor.extent);
  offscreenRenderPassBeginInfo.renderArea = {scissor.offset, scissor.extent};
  clearValues[0].color                    = m_clearColor;
  vkCmdBeginRenderPass(cmdBuf, &offscreenRenderPassBeginInfo, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
  rasterize(cmdBuf);
  vkCmdEndRenderPass(cmdBuf);
}
