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

#include "optix_denoiser_sample.hpp"

#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/pipeline_vk.hpp"


#include "_autogen/passthrough.vert.h"
#include "_autogen/pathtrace.rahit.h"
#include "_autogen/pathtrace.rchit.h"
#include "_autogen/pathtrace.rgen.h"
#include "_autogen/pathtrace.rmiss.h"
#include "_autogen/post.frag.h"
#include "_autogen/gbuffers.rchit.h"
#include "_autogen/gbuffers.rmiss.h"


//--------------------------------------------------------------------------------------------------
// Override: adding HDR and the denoiser
//--------------------------------------------------------------------------------------------------
void OptixDenoiserSample::create(const nvvk::AppBaseVkCreateInfo& info)
{
  VulkanSample::create(info);

  // #OPTIX_D
  m_clearColor = {0.5f, 0.5f, 0.5f, 1.f};
  m_hdrEnv.setup(m_device, m_physicalDevice, info.queueIndices[0], &m_alloc);

#ifdef NVP_SUPPORTS_OPTIX7
  m_denoiser.setup(m_device, m_physicalDevice, info.queueIndices[0]);

  OptixDenoiserOptions dOptions;
  dOptions.guideAlbedo = true;
  dOptions.guideNormal = true;
  m_denoiser.initOptiX(dOptions, OPTIX_PIXEL_FORMAT_FLOAT4, true);
#endif  // NVP_SUPPORTS_OPTIX7
}

//--------------------------------------------------------------------------------------------------
// Override: Adding the creation of Gbuffers and the creation of the denoiser
//
void OptixDenoiserSample::createScene(const std::string& filename)
{
  createGbuffers();  // #OPTIX_D
  // #OPTIX_D
#ifdef NVP_SUPPORTS_OPTIX7
  {  // Denoiser
    nvh::Stopwatch sw_;
    m_denoiser.allocateBuffers(m_size);
    m_denoiser.createSemaphore();
    m_denoiser.createCopyPipeline();
    LOGI(" - %6.2fms: Denoiser\n", sw_.elapsed());
  }
#endif  // NVP_SUPPORTS_OPTIX7

  VulkanSample::createScene(filename);
}

//--------------------------------------------------------------------------------------------------
// #OPTIX_D
// Load and create HDR image, same as in #HDR
//
void OptixDenoiserSample::createHdr(const std::string& hdrFilename)
{
  LOGI("- HDR section \n");
  {  // HDR
    nvh::Stopwatch sw_;
    m_hdrEnv.loadEnvironment(hdrFilename);
    m_pcRay.maxLuminance = m_hdrEnv.getIntegral();
    LOGI(" = Total HDR: %6.2fms\n", sw_.elapsed());

    // Forced to regenerate the raster recorded command buffer
    vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_recordedCmdBuffer);
    m_recordedCmdBuffer = VK_NULL_HANDLE;
  }
}

//--------------------------------------------------------------------------------------------------
// Override: destroy Gbuffers and denoiser
//
void OptixDenoiserSample::freeResources()
{
  VulkanSample::freeResources();


  // #OPTIX_D
  // Denoiser
  m_alloc.destroy(m_gAlbedo);
  m_alloc.destroy(m_gNormal);
  m_alloc.destroy(m_gDenoised);

  for(auto& p : m_postDstPool)  // All in flight pool
    vkDestroyDescriptorPool(m_device, p, nullptr);

#ifdef NVP_SUPPORTS_OPTIX7
  m_denoiser.destroy();
#endif  // NVP_SUPPORTS_OPTIX7
}

//--------------------------------------------------------------------------------------------------
// Override: Destroy HDR
//
void OptixDenoiserSample::destroy()
{
  freeResources();

  m_hdrEnv.destroy();  // #OPTIX_D #HDR
  m_alloc.deinit();
  AppBaseVk::destroy();
}

//--------------------------------------------------------------------------------------------------
// Override: recreate the Gbuffers and the denoiser buffers
//
void OptixDenoiserSample::onResize(int /*w*/, int /*h*/)
{
  vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_recordedCmdBuffer);
  m_recordedCmdBuffer = VK_NULL_HANDLE;

  createGbuffers();  // #OPTIX_D
  createOffscreenRender();
  updatePostDescriptorSet(m_offscreenColor.descriptor);
  updateRtDescriptorSet();
// #OPTIX_D
#ifdef NVP_SUPPORTS_OPTIX7
  m_denoiser.allocateBuffers(m_size);
#endif  // NVP_SUPPORTS_OPTIX7
  resetFrame();
}


//--------------------------------------------------------------------------------------------------
// Override: creating 3 post descriptor set, such that we can modify the image that will be
// displayed, while it is rendering. This is needed as we cannot write to the descriptor set
// while in use. (see #OPTIX_D and updatePostDescriptorSet())
//
void OptixDenoiserSample::createPostPipeline()
{
  auto& p = m_pContainer[ePost];

  // The descriptor layout is the description of the data that is passed to the vertex or the  fragment program.

  nvvk::DescriptorSetBindings bind;
  bind.addBinding(PostBindings::ePostImage, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  p.dstLayout = bind.createLayout(m_device);
  // #OPTIX_D
  for(int i = 0; i < FRAMES_IN_FLIGHT; i++)
  {
    m_postDstPool[i] = bind.createPool(m_device);
    m_postDstSet[i]  = nvvk::allocateDescriptorSet(m_device, m_postDstPool[i], p.dstLayout);
  }

  // Push constants in the fragment shader
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Tonemapper)};

  // Creating the pipeline layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &p.dstLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &p.pipelineLayout);

  // Pipeline: completely generic, no vertices
  std::vector<uint32_t> vertexShader(std::begin(passthrough_vert), std::end(passthrough_vert));
  std::vector<uint32_t> fragShader(std::begin(post_frag), std::end(post_frag));

  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, p.pipelineLayout, m_renderPass);
  pipelineGenerator.addShader(vertexShader, VK_SHADER_STAGE_VERTEX_BIT);
  pipelineGenerator.addShader(fragShader, VK_SHADER_STAGE_FRAGMENT_BIT);
  pipelineGenerator.rasterizationState.cullMode = VK_CULL_MODE_NONE;

  p.pipeline = pipelineGenerator.createPipeline();
  m_debug.setObjectName(p.pipeline, "post");
}


//--------------------------------------------------------------------------------------------------
// Override: Updating the NEXT descriptor set, as the current one might be in use.
//
void OptixDenoiserSample::updatePostDescriptorSet(const VkDescriptorImageInfo& descriptor)
{
  // #OPTIX_D
  m_postFrame            = (m_postFrame + 1) % FRAMES_IN_FLIGHT;
  VkDescriptorSet dstSet = m_postDstSet[m_postFrame];

  VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wds.dstSet          = dstSet;
  wds.dstBinding      = PostBindings::ePostImage;
  wds.descriptorCount = 1;
  wds.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  wds.pImageInfo      = &descriptor;
  vkUpdateDescriptorSets(m_device, 1, &wds, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Override: Using the right Post descriptor set
//
void OptixDenoiserSample::drawPost(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);

  // #OPTIX_D
  VkDescriptorSet dstSet = m_postDstSet[m_postFrame];

  auto& p = m_pContainer[ePost];
  setViewport(cmdBuf);
  vkCmdPushConstants(cmdBuf, p.pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Tonemapper), &m_tonemapper);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipelineLayout, 0, 1, &dstSet, 0, nullptr);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);
}

//--------------------------------------------------------------------------------------------------
// Override:
// - adding extra Gbuffers for autput.
// - adding Gbuffer rchit/miss to extract only the extra information.
//
void OptixDenoiserSample::createRtPipeline()
{
  auto& p = m_pContainer[eRaytrace];

  // This descriptor set, holds the top level acceleration structure and the output image
  nvvk::DescriptorSetBindings bind;

  // Create Binding Set
  bind.addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
  bind.addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  // #OPTIX_D
  bind.addBinding(RtxBindings::eOutAlbedo, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
  bind.addBinding(RtxBindings::eOutNormal, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);


  p.dstPool   = bind.createPool(m_device);
  p.dstLayout = bind.createLayout(m_device);
  p.dstSet    = nvvk::allocateDescriptorSet(m_device, p.dstPool, p.dstLayout);

  // Write to descriptors
  VkAccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure();
  VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  descASInfo.accelerationStructureCount = 1;
  descASInfo.pAccelerationStructures    = &tlas;
  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  // #OPTIX_D
  VkDescriptorImageInfo albedoInfo{{}, m_gAlbedo.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorImageInfo normalInfo{{}, m_gNormal.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(p.dstSet, RtxBindings::eTlas, &descASInfo));
  writes.emplace_back(bind.makeWrite(p.dstSet, RtxBindings::eOutImage, &imageInfo));
  // #OPTIX_D
  writes.emplace_back(bind.makeWrite(p.dstSet, RtxBindings::eOutAlbedo, &albedoInfo));
  writes.emplace_back(bind.makeWrite(p.dstSet, RtxBindings::eOutNormal, &normalInfo));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  // Creating all shaders
  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMissGbuf,  // #OPTIX_D
    eClosestHit,
    eAnyHit,
    eClosestHitGbuf,  // #OPTIX_D
    eShaderGroupCount
  };
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.pName = "main";  // All the same entry point
  // Raygen
  stage.module    = nvvk::createShaderModule(m_device, pathtrace_rgen, sizeof(pathtrace_rgen));
  stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eRaygen] = stage;
  // Miss
  stage.module  = nvvk::createShaderModule(m_device, pathtrace_rmiss, sizeof(pathtrace_rmiss));
  stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss] = stage;
  // Hit Group - Closest Hit
  stage.module        = nvvk::createShaderModule(m_device, pathtrace_rchit, sizeof(pathtrace_rchit));
  stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[eClosestHit] = stage;
  // AnyHit
  stage.module    = nvvk::createShaderModule(m_device, pathtrace_rahit, sizeof(pathtrace_rahit));
  stage.stage     = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
  stages[eAnyHit] = stage;
  // #OPTIX_D
  // Miss G-Buffers
  stage.module      = nvvk::createShaderModule(m_device, gbuffers_rmiss, sizeof(gbuffers_rmiss));
  stage.stage       = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMissGbuf] = stage;
  // Hit Group - Closest Hit
  stage.module            = nvvk::createShaderModule(m_device, gbuffers_rchit, sizeof(gbuffers_rchit));
  stage.stage             = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[eClosestHitGbuf] = stage;


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

  // #OPTIX_D
  // Miss - G-Buf
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMissGbuf;
  shaderGroups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  group.anyHitShader     = eAnyHit;
  shaderGroups.push_back(group);

  // #OPTIX_D
  // any hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHitGbuf;
  group.anyHitShader     = VK_SHADER_UNUSED_KHR;
  shaderGroups.push_back(group);

  // Push constant: we want to be able to update constants used by the shaders
  VkPushConstantRange pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(RtxPushConstant)};

  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {p.dstLayout, m_pContainer[eGraphic].dstLayout,
                                                         m_hdrEnv.getDescLayout()};  // #OPTIX_D
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
// Override: adding the extra GBuffers
//
void OptixDenoiserSample::updateRtDescriptorSet()
{
  // #OPTIX_D
  auto makeWrite = [](const auto& dstSet, uint32_t dstBinding, const auto& imageInfo) {
    VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wds.dstSet          = dstSet;
    wds.dstBinding      = dstBinding;
    wds.descriptorCount = 1;
    wds.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wds.pImageInfo      = &imageInfo;
    return wds;
  };

  // (1) Output buffer
  VkDescriptorImageInfo colorInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorImageInfo albedoInfo{{}, m_gAlbedo.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorImageInfo normalInfo{{}, m_gNormal.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};

  std::vector<VkWriteDescriptorSet> writes;
  writes.push_back(makeWrite(m_pContainer[eRaytrace].dstSet, RtxBindings::eOutImage, colorInfo));
  writes.push_back(makeWrite(m_pContainer[eRaytrace].dstSet, RtxBindings::eOutAlbedo, albedoInfo));
  writes.push_back(makeWrite(m_pContainer[eRaytrace].dstSet, RtxBindings::eOutNormal, normalInfo));

  vkUpdateDescriptorSets(m_device, (int)writes.size(), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Override: adding the HDR descriptor set
//
void OptixDenoiserSample::raytrace(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);

  if(!updateFrame())
    return;

  std::vector<VkDescriptorSet> descSets{m_pContainer[eRaytrace].dstSet, m_pContainer[eGraphic].dstSet, m_hdrEnv.getDescSet()};  // #OPTIX_D
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pContainer[eRaytrace].pipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pContainer[eRaytrace].pipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_pContainer[eRaytrace].pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(RtxPushConstant), &m_pcRay);

  const auto& regions = m_sbt.getRegions();
  vkCmdTraceRaysKHR(cmdBuf, &regions[0], &regions[1], &regions[2], &regions[3], m_size.width, m_size.height, 1);
}

//--------------------------------------------------------------------------------------------------
// Override: no raster, and adding denoiser settings
//
void OptixDenoiserSample::renderUI()
{
  if(showGui() == false)
    return;

  bool changed{false};

  ImGuiH::Panel::Begin();
  float widgetWidth = std::min(std::max(ImGui::GetWindowWidth() - 150.0f, 100.0f), 300.0f);
  ImGui::PushItemWidth(widgetWidth);

  if(ImGui::TreeNode("Ray Tracing"))
  {
    changed |= ImGui::SliderFloat("Max Luminance", &m_pcRay.maxLuminance, 0.01f, 20.f);
    changed |= ImGui::SliderInt("Depth", (int*)&m_pcRay.maxDepth, 1, 15);
    changed |= ImGui::SliderInt("Samples", (int*)&m_pcRay.maxSamples, 1, 20);
    ImGui::SliderInt("Max frames", &m_maxFrames, 1, 50000);
    if(m_maxFrames < m_pcRay.frame)
      changed = true;
    ImGui::TreePop();
  }

  if(ImGui::CollapsingHeader("Camera"))
  {
    ImGuiH::CameraWidget();
  }

  if(ImGui::CollapsingHeader("Environment"))
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

    uint32_t i = 0;
    for(auto& light : m_lights)
    {
      if(ImGui::TreeNode((void*)(intptr_t)i, "Light %d", i))
      {
        changed |= ImGui::RadioButton("Point", &light.type, 0);
        ImGui::SameLine();
        changed |= ImGui::RadioButton("Infinite", &light.type, 1);
        changed |= ImGui::DragFloat3("Position", &light.position.x, 0.1f);
        changed |= ImGui::DragFloat("Intensity", &light.intensity, 1.f, 0.0f, 1000.f, nullptr,
                                    ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
        changed |= ImGui::ColorEdit3("Color", reinterpret_cast<float*>(&light.color));
        ImGui::TreePop();
      }
      i++;
    }
  }

  // #OPTIX_D
  if(ImGui::CollapsingHeader("Denoiser", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGui::Checkbox("Denoise", (bool*)&m_denoiseApply);
    ImGui::Checkbox("First Frame", &m_denoiseFirstFrame);
    ImGui::SliderInt("N-frames", &m_denoiseEveryNFrames, 1, 500);
    ImGui::SliderInt("Max frames", &m_maxFrames, 1, 5000);
    ImGui::Text("Frame: %d", m_maxFrames);
    int denoisedFrame = -1;
    if(m_denoiseApply)
    {
      if(m_pcRay.frame >= m_maxFrames)
        denoisedFrame = m_maxFrames;
      else if(m_denoiseFirstFrame && (m_pcRay.frame < m_denoiseEveryNFrames))
        denoisedFrame = 0;
      else if(m_pcRay.frame > m_denoiseEveryNFrames)
        denoisedFrame = (m_pcRay.frame / m_denoiseEveryNFrames) * m_denoiseEveryNFrames;
    }
    ImGui::Text("Denoised Frame: %d", denoisedFrame);
  }


  ImGui::PopItemWidth();
  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
  ImGui::TextDisabled("(M) Toggle Render Mode");
  ImGui::TextDisabled("(F) Frame All");
  ImGui::TextDisabled("(R) Restart rendering");
  ImGui::TextDisabled("(SPACE) Set Eye position");
  ImGui::TextDisabled("(F10) Toggle Pane");
  ImGui::TextDisabled("(ESC) Quit");
  ImGuiH::Panel::End();

  if(changed)
    resetFrame();
}


//--------------------------------------------------------------------------------------------------
// Override: adding to drop .hdr files
//
void OptixDenoiserSample::onFileDrop(const char* filename)
{
  namespace fs = std::filesystem;
  vkDeviceWaitIdle(m_device);
  std::string extension = fs::path(filename).extension().string();
  if(extension == ".gltf")
  {
    freeResources();
    createScene(filename);
  }
  else if(extension == ".hdr")  // #OPTIX_D
  {
    createHdr(filename);
  }

  resetFrame();
}

//////////////////////////////////////////////////////////////////////////
// #OPTIX_D
// Return true if the current frame need to be denoised, as we are not
// dnoising all frames.
bool OptixDenoiserSample::needToDenoise()
{
  if(m_denoiseApply)
  {
    if(m_pcRay.frame == m_maxFrames)
      return true;
    if(m_denoiseFirstFrame && m_pcRay.frame == 0)
      return true;
    if(m_pcRay.frame % m_denoiseEveryNFrames == 0)
      return true;
  }
  return false;
}

// Will copy the Vulkan images to Cuda buffers
void OptixDenoiserSample::copyImagesToCuda(const VkCommandBuffer& cmdBuf)
{
  if(needToDenoise())
  {
#ifdef NVP_SUPPORTS_OPTIX7
    m_denoiser.imageToBuffer(cmdBuf, {m_offscreenColor, m_gAlbedo, m_gNormal});
#endif  // NVP_SUPPORTS_OPTIX7
  }
}

// Copy the denoised buffer to Vulkan image
void OptixDenoiserSample::copyCudaImagesToVulkan(const VkCommandBuffer& cmdBuf)
{
  if(needToDenoise())
  {
#ifdef NVP_SUPPORTS_OPTIX7
    m_denoiser.bufferToImage(cmdBuf, &m_gDenoised);
#endif  // NVP_SUPPORTS_OPTIX7
  }
}

// Invoke the Optix denoiser
void OptixDenoiserSample::denoise()
{
  if(needToDenoise())
  {
#ifdef NVP_SUPPORTS_OPTIX7
    m_denoiser.denoiseImageBuffer(m_fenceValue);
#endif  // NVP_SUPPORTS_OPTIX7
  }
}

// Determine which image will be displayed, the original from ray tracer or the denoised one
void OptixDenoiserSample::setImageToDisplay()
{
  bool showDenoised =
      m_denoiseApply && ((m_pcRay.frame >= m_denoiseEveryNFrames) || m_denoiseFirstFrame || (m_pcRay.frame >= m_maxFrames));
  updatePostDescriptorSet(showDenoised ? m_gDenoised.descriptor : m_offscreenColor.descriptor);
}


//--------------------------------------------------------------------------------------------------
// Override: Creating more command buffers (x2) because the frame will be in 2 parts
// before and after denoising
void OptixDenoiserSample::createSwapchain(const VkSurfaceKHR& surface,
                                          uint32_t            width,
                                          uint32_t            height,
                                          VkFormat            colorFormat /*= vk::Format::eB8G8R8A8Unorm*/,
                                          VkFormat            depthFormat /*= vk::Format::eUndefined*/,
                                          bool                vsync /*= false*/)
{
  AppBaseVk::createSwapchain(surface, width, height, colorFormat, depthFormat, vsync);

  std::vector<VkCommandBuffer> commandBuffers(m_swapChain.getImageCount());
  VkCommandBufferAllocateInfo  allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  allocateInfo.commandPool        = m_cmdPool;
  allocateInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocateInfo.commandBufferCount = m_swapChain.getImageCount();

  VkResult result = vkAllocateCommandBuffers(m_device, &allocateInfo, commandBuffers.data());
  m_commandBuffers.insert(m_commandBuffers.end(), commandBuffers.begin(), commandBuffers.end());
}


//--------------------------------------------------------------------------------------------------
// We are submitting the command buffer using a timeline semaphore, semaphore used by Cuda to wait
// for the Vulkan execution before denoising
//
void OptixDenoiserSample::submitWithTLSemaphore(const VkCommandBuffer& cmdBuf)
{
  // Increment for signaling
  m_fenceValue++;

  VkCommandBufferSubmitInfoKHR cmdBufInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR};
  cmdBufInfo.commandBuffer = cmdBuf;

  VkSemaphoreSubmitInfoKHR waitSemaphore{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR};
  waitSemaphore.semaphore = m_swapChain.getActiveReadSemaphore();
  waitSemaphore.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT_KHR;

#ifdef NVP_SUPPORTS_OPTIX7
  VkSemaphoreSubmitInfoKHR signalSemaphore{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR};
  signalSemaphore.semaphore = m_denoiser.getTLSemaphore();
  signalSemaphore.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;
  signalSemaphore.value     = m_fenceValue;
#endif  // NVP_SUPPORTS_OPTIX7

  VkSubmitInfo2KHR submits{VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR};
  submits.commandBufferInfoCount = 1;
  submits.pCommandBufferInfos    = &cmdBufInfo;
  submits.waitSemaphoreInfoCount = 1;
  submits.pWaitSemaphoreInfos    = &waitSemaphore;
#ifdef NVP_SUPPORTS_OPTIX7
  submits.signalSemaphoreInfoCount = 1;
  submits.pSignalSemaphoreInfos    = &signalSemaphore;
#endif  // _DEBUG

  vkQueueSubmit2KHR(m_queue, 1, &submits, {});
}

//--------------------------------------------------------------------------------------------------
// Similar to AppBaseVk::submitFrame() but taking a command buffer
// It is also using the timeline semaphore from Cuda and waiting that the denoiser
// is done with the work before submitting the command buffer.
//
void OptixDenoiserSample::submitFrame(const VkCommandBuffer& cmdBuf)
{
  uint32_t imageIndex = m_swapChain.getActiveImageIndex();
  VkFence  fence      = m_waitFences[imageIndex];
  vkResetFences(m_device, 1, &fence);

  VkCommandBufferSubmitInfoKHR cmdBufInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO_KHR};
  cmdBufInfo.commandBuffer = cmdBuf;

#ifdef NVP_SUPPORTS_OPTIX7
  VkSemaphoreSubmitInfoKHR waitSemaphore{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR};
  waitSemaphore.semaphore = m_denoiser.getTLSemaphore();
  waitSemaphore.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;
  waitSemaphore.value     = m_fenceValue;
#endif  // NVP_SUPPORTS_OPTIX7

  VkSemaphoreSubmitInfoKHR signalSemaphore{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO_KHR};
  signalSemaphore.semaphore = m_swapChain.getActiveWrittenSemaphore();
  signalSemaphore.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT_KHR;

  VkSubmitInfo2KHR submits{VK_STRUCTURE_TYPE_SUBMIT_INFO_2_KHR};
  submits.commandBufferInfoCount = 1;
  submits.pCommandBufferInfos    = &cmdBufInfo;
#ifdef NVP_SUPPORTS_OPTIX7
  submits.waitSemaphoreInfoCount = 1;
  submits.pWaitSemaphoreInfos    = &waitSemaphore;
#endif  // NVP_SUPPORTS_OPTIX7
  submits.signalSemaphoreInfoCount = 1;
  submits.pSignalSemaphoreInfos    = &signalSemaphore;

  vkQueueSubmit2KHR(m_queue, 1, &submits, fence);

  // Presenting frame
  m_swapChain.present(m_queue);
}


//--------------------------------------------------------------------------------------------------
// Creating all extra buffers: albedo, normal and denoised image
//
void OptixDenoiserSample::createGbuffers()
{
  m_alloc.destroy(m_gAlbedo);
  m_alloc.destroy(m_gNormal);
  m_alloc.destroy(m_gDenoised);

  VkImageUsageFlags   usage{VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT};
  VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};

  {  // Albedo RGBA8
    auto                  colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, VK_FORMAT_R8G8B8A8_UNORM, usage);
    nvvk::Image           image           = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo          = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    m_gAlbedo                             = m_alloc.createTexture(image, ivInfo, sampler);
    m_gAlbedo.descriptor.imageLayout      = VK_IMAGE_LAYOUT_GENERAL;
    NAME_VK(m_gAlbedo.image);
  }

  {  // Normal RGBA8
    auto                  colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, VK_FORMAT_R8G8B8A8_UNORM, usage);
    nvvk::Image           image           = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo          = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    m_gNormal                             = m_alloc.createTexture(image, ivInfo, sampler);
    m_gNormal.descriptor.imageLayout      = VK_IMAGE_LAYOUT_GENERAL;
    NAME_VK(m_gNormal.image);
  }

  {  // Denoised RGBA32
    auto                  colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, VK_FORMAT_R32G32B32A32_SFLOAT, usage);
    nvvk::Image           image           = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo          = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    m_gDenoised                           = m_alloc.createTexture(image, ivInfo, sampler);
    m_gDenoised.descriptor.imageLayout    = VK_IMAGE_LAYOUT_GENERAL;
    NAME_VK(m_gDenoised.image);
  }

  // Setting the image layout  to general
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_gAlbedo.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_gNormal.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_gDenoised.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    genCmdBuf.submitAndWait(cmdBuf);
  }
}
