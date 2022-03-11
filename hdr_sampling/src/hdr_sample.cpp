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

#include <filesystem>

#include "hdr_sample.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/pipeline_vk.hpp"


// Glsl Shaders compiled to Spir-V (See Makefile)
#include "_autogen/raster.vert.h"
#include "_autogen/raster.frag.h"
#include "_autogen/raster_overlay.frag.h"
#include "_autogen/pathtrace.rahit.h"
#include "_autogen/pathtrace.rchit.h"
#include "_autogen/pathtrace.rgen.h"
#include "_autogen/pathtrace.rmiss.h"
#include "nvvk/dynamicrendering_vk.hpp"

// #VMA
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

//--------------------------------------------------------------------------------------------------
// Overload: adding HDR
//--------------------------------------------------------------------------------------------------
void HdrSample::create(const nvvk::AppBaseVkCreateInfo& info)
{
  AppBaseVk::create(info);

  // #VMA
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.physicalDevice         = info.physicalDevice;
  allocatorInfo.device                 = info.device;
  allocatorInfo.instance               = info.instance;
  allocatorInfo.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
  vmaCreateAllocator(&allocatorInfo, &m_vmaAlloc);

  m_vma = std::make_unique<nvvk::VMAMemoryAllocator>(info.device, info.physicalDevice, m_vmaAlloc);
  m_alloc.init(info.device, info.physicalDevice, m_vma.get());  // #VMA
  m_debug.setup(m_device);
  m_picker.setup(m_device, m_physicalDevice, info.queueIndices[0], &m_alloc);

  m_hdrEnv.setup(m_device, m_physicalDevice, info.queueIndices[0], &m_alloc);
  m_hdrDome.setup(m_device, m_physicalDevice, info.queueIndices[0], &m_alloc);

  m_clearColor = {0.5f, 0.5f, 0.5f, 1.f};
}

//--------------------------------------------------------------------------------------------------
// Overload: adding HDR  dome image
//
void HdrSample::createScene(const std::string& filename)
{
  VulkanSample::createScene(filename);

  m_hdrDome.setOutImage(m_offscreenColor.descriptor);
}

//--------------------------------------------------------------------------------------------------
// Loading and creating HDR support
//
void HdrSample::createHdr(const std::string& hdrFilename)
{
  LOGI("- HDR section \n");
  {  // HDR
    nvh::Stopwatch sw_;
    m_hdrEnv.loadEnvironment(hdrFilename);
    m_hdrDome.create(m_hdrEnv.getDescSet(), m_hdrEnv.getDescLayout());
    m_pcRay.maxLuminance = m_hdrEnv.getIntegral();
    LOGI(" = Total HDR: %6.2fms\n", sw_.elapsed());

    // Forced to regenerate the raster recorded command buffer
    vkFreeCommandBuffers(m_device, m_cmdPool, 2, m_recordedCmdBuffer.data());
    m_recordedCmdBuffer = {VK_NULL_HANDLE};

    if(m_offscreenColor.image != VK_NULL_HANDLE)
      m_hdrDome.setOutImage(m_offscreenColor.descriptor);
  }
}


//--------------------------------------------------------------------------------------------------
// Overload: adding HDR
//
void HdrSample::destroy()
{
  freeResources();
  m_hdrEnv.destroy();
  m_hdrDome.destroy();
  m_alloc.deinit();

  // #VMA
  vmaDestroyAllocator(m_vmaAlloc);
  m_vma->deinit();
  AppBaseVk::destroy();
}

//--------------------------------------------------------------------------------------------------
// Drawing in the background a dome
//
void HdrSample::drawDome(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);
  auto&       view        = CameraManip.getMatrix();
  auto        proj        = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);

  m_hdrDome.draw(cmdBuf, view, proj, m_size, &m_clearColor.float32[0], m_frameInfo.envRotation);
}


//--------------------------------------------------------------------------------------------------
// Overload: see #HDR
//
void HdrSample::createGraphicPipeline()
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
  std::vector<VkDescriptorSetLayout> layouts{p.dstLayout, m_hdrDome.getDescLayout()};  // #HDR
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(RasterPushConstant)};
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = static_cast<uint32_t>(layouts.size());
  createInfo.pSetLayouts            = layouts.data();
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
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, p.pipelineLayout, {} /*m_offscreenRenderPass*/);
  gpb.rasterizationState.depthBiasEnable         = VK_TRUE;
  gpb.rasterizationState.depthBiasConstantFactor = -1;
  gpb.rasterizationState.depthBiasSlopeFactor    = 1;
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
// Overload: see #HDR
//
//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode: record all command and "play back"
//
void HdrSample::rasterize(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);

  // Recording the commands to draw the scene if not done yet
  if(m_recordedCmdBuffer[0] == VK_NULL_HANDLE)
    recordRendering();

  nvvk::createRenderingInfo rInfo({{0, 0}, getSize()}, {m_offscreenColor.descriptor.imageView},
                                  m_offscreenDepth.descriptor.imageView, VK_ATTACHMENT_LOAD_OP_LOAD,
                                  VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor,  //#HDR
                                  {1.0f, 0}, VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT_KHR);

  vkCmdBeginRendering(cmdBuf, &rInfo);

  // Executing the drawing of the recorded commands
  vkCmdExecuteCommands(cmdBuf, m_showWireframe ? 2 : 1, m_recordedCmdBuffer.data());

  vkCmdEndRendering(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Record commands for rendering fill and wireframe
//
void HdrSample::recordRendering()
{
  nvh::Stopwatch sw;
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
  inheritanceRenderingInfo.rasterizationSamples    = VK_SAMPLE_COUNT_1_BIT;

  VkCommandBufferInheritanceInfo inheritInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
  inheritInfo.pNext = &inheritanceRenderingInfo;

  for(int i = 0; i < 2; i++)
  {
    auto& p = m_pContainer[eGraphic];

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    beginInfo.pInheritanceInfo = &inheritInfo;
    vkBeginCommandBuffer(m_recordedCmdBuffer[i], &beginInfo);

    // Dynamic Viewport
    setViewport(m_recordedCmdBuffer[i]);

    // Drawing all instances
    std::vector<VkDescriptorSet> dstSets{p.dstSet, m_hdrDome.getDescSet()};  // #HDR
    vkCmdBindPipeline(m_recordedCmdBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, i == 0 ? p.pipeline : m_wireframe);
    vkCmdBindDescriptorSets(m_recordedCmdBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipelineLayout, 0,
                            static_cast<uint32_t>(dstSets.size()), dstSets.data(), 0, nullptr);

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
// Overload: adding HDR
//
void HdrSample::onResize(int /*w*/, int /*h*/)
{
  VulkanSample::onResize();
  m_hdrDome.setOutImage(m_offscreenColor.descriptor);
}

//--------------------------------------------------------------------------------------------------
// Overload: adding HDR to the descriptor set layout. See #HDR
//
void HdrSample::createRtPipeline()
{
  auto& p = m_pContainer[eRaytrace];

  // This descriptor set, holds the top level acceleration structure and the output image
  nvvk::DescriptorSetBindings bind;

  // Create Binding Set
  bind.addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
  bind.addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  p.dstPool   = bind.createPool(m_device);
  p.dstLayout = bind.createLayout(m_device);
  p.dstSet    = nvvk::allocateDescriptorSet(m_device, p.dstPool, p.dstLayout);

  // Write to descriptors
  VkAccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure();
  VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  descASInfo.accelerationStructureCount = 1;
  descASInfo.pAccelerationStructures    = &tlas;
  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(p.dstSet, RtxBindings::eTlas, &descASInfo));
  writes.emplace_back(bind.makeWrite(p.dstSet, RtxBindings::eOutImage, &imageInfo));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  // Creating all shaders
  enum StageIndices
  {
    eRaygen,
    eMiss,
    eClosestHit,
    eAnyHit,
    eShaderGroupCount
  };
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.pName = "main";  // All the same entry point
  // Raygen
  stage.module    = nvvk::createShaderModule(m_device, pathtrace_rgen, sizeof(pathtrace_rgen));
  stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eRaygen] = stage;
  NAME2_VK(stage.module, "Raygen");
  // Miss
  stage.module  = nvvk::createShaderModule(m_device, pathtrace_rmiss, sizeof(pathtrace_rmiss));
  stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss] = stage;
  NAME2_VK(stage.module, "Miss");
  // Hit Group - Closest Hit
  stage.module        = nvvk::createShaderModule(m_device, pathtrace_rchit, sizeof(pathtrace_rchit));
  stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[eClosestHit] = stage;
  NAME2_VK(stage.module, "Closest Hit");
  // AnyHit
  stage.module    = nvvk::createShaderModule(m_device, pathtrace_rahit, sizeof(pathtrace_rahit));
  stage.stage     = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  stages[eAnyHit] = stage;
  NAME2_VK(stage.module, "AnyHit");

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups;
  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  shaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  shaderGroups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  group.anyHitShader     = eAnyHit;
  shaderGroups.push_back(group);

  // any hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = VK_SHADER_UNUSED_KHR;
  group.anyHitShader     = eAnyHit;
  shaderGroups.push_back(group);


  // Push constant: we want to be able to update constants used by the shaders
  VkPushConstantRange pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(RtxPushConstant)};

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  // #HDR
  std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {p.dstLayout, m_pContainer[eGraphic].dstLayout,
                                                         m_hdrEnv.getDescLayout()};  // #HDR
  pipelineLayoutCreateInfo.setLayoutCount             = static_cast<uint32_t>(rtDescSetLayouts.size());
  pipelineLayoutCreateInfo.pSetLayouts                = rtDescSetLayouts.data();
  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &p.pipelineLayout);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  rayPipelineInfo.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
  rayPipelineInfo.pStages                      = stages.data();
  rayPipelineInfo.groupCount                   = static_cast<uint32_t>(shaderGroups.size());
  rayPipelineInfo.pGroups                      = shaderGroups.data();
  rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
  rayPipelineInfo.layout                       = p.pipelineLayout;
  vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &p.pipeline);

  // Creating the SBT
  m_sbt.create(p.pipeline, rayPipelineInfo);

  // Removing temp modules
  for(auto& s : stages)
    vkDestroyShaderModule(m_device, s.module, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Overload: adding HDR descriptor set (see #HDR)
//
void HdrSample::raytrace(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);

  if(!updateFrame())
    return;

  std::vector<VkDescriptorSet> descSets{m_pContainer[eRaytrace].dstSet, m_pContainer[eGraphic].dstSet, m_hdrEnv.getDescSet()};  // #HDR
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pContainer[eRaytrace].pipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pContainer[eRaytrace].pipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_pContainer[eRaytrace].pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(RtxPushConstant), &m_pcRay);

  const auto& regions = m_sbt.getRegions();
  vkCmdTraceRaysKHR(cmdBuf, &regions[0], &regions[1], &regions[2], &regions[3], m_size.width, m_size.height, 1);
}

//--------------------------------------------------------------------------------------------------
// Overload: Allow to drop .hdr files
//
void HdrSample::onFileDrop(const char* filename)
{
  namespace fs = std::filesystem;
  vkDeviceWaitIdle(m_device);
  std::string extension = fs::path(filename).extension().string();
  if(extension == ".gltf" || extension == ".glb")
  {
    freeResources();
    createScene(filename);
  }
  else if(extension == ".hdr")  // #HDR
  {
    createHdr(filename);
  }

  resetFrame();
}


//--------------------------------------------------------------------------------------------------
// Override: adding HDR button
//
void HdrSample::renderUI()
{
  if(showGui() == false)
    return;
  bool changed{false};

  ImGuiH::Panel::Begin();
  if(ImGui::Button("Load glTF"))
  {  // Loading file dialog
    auto filename = NVPSystem::windowOpenFileDialog(m_window, "Load glTF", "glTF(.gltf, .glb)|*.gltf;*.glb");
    onFileDrop(filename.c_str());
  }
  ImGui::SameLine();
  if(ImGui::Button("Load HDR"))
  {  // #HDR Loading file dialog
    auto filename = NVPSystem::windowOpenFileDialog(m_window, "Load HDR", ".hdr");
    onFileDrop(filename.c_str());
  }

  if(ImGui::CollapsingHeader("Render Mode"))
  {
    changed |= ImGui::RadioButton("Raster", (int*)&m_renderMode, (int)RenderMode::eRaster);
    ImGui::SameLine();
    changed |= ImGui::RadioButton("Ray Tracing", (int*)&m_renderMode, (int)RenderMode::eRayTracer);
    if(m_renderMode == RenderMode::eRayTracer && ImGui::TreeNode("Ray Tracing"))
    {
      changed = uiRaytrace(changed);
      ImGui::TreePop();
    }
    if(m_renderMode == RenderMode::eRaster && ImGui::TreeNode("Raster"))
    {
      ImGui::Checkbox("Show wireframe", &m_showWireframe);
      ImGui::TreePop();
    }
  }

  if(ImGui::CollapsingHeader("Camera"))
    ImGuiH::CameraWidget();

  if(ImGui::CollapsingHeader("Environment"))
  {
    changed |= ImGui::SliderAngle("Rotation", &m_frameInfo.envRotation);  //, -180, 180);
    uiEnvironment(changed);
  }

  uiInfo();

  ImGuiH::Panel::End();
  if(changed)
    resetFrame();
}
