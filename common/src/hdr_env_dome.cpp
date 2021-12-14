/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#define _USE_MATH_DEFINES
#include <chrono>
#include <iostream>
#include <cmath>
#include <numeric>

#include "nvmath/nvmath.h"
#include "nvh/nvprint.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "hdr_env_dome.hpp"
#include "stb_image.h"
#include "shaders/host_device.h"


#include "_autogen/hdr_dome.comp.h"
#include "_autogen/hdr_integrate_brdf.comp.h"
#include "_autogen/hdr_prefilter_diffuse.comp.h"
#include "_autogen/hdr_prefilter_glossy.comp.h"


extern std::vector<std::string> defaultSearchPaths;


inline VkExtent2D getGridSize(const VkExtent2D& size)
{
  return VkExtent2D{(size.width + (GRID_SIZE - 1)) / GRID_SIZE, (size.height + (GRID_SIZE - 1)) / GRID_SIZE};
}

//--------------------------------------------------------------------------------------------------
//
//
void HdrEnvDome::setup(const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t familyIndex, nvvk::ResourceAllocator* allocator)
{
  m_device      = device;
  m_alloc       = allocator;
  m_familyIndex = familyIndex;
  m_debug.setup(device);
}


//--------------------------------------------------------------------------------------------------
// The descriptor set and layout are from HDR_ENV
// - it consists of the HDR image and the acceleration structure
// - those will be used to create the diffuse and glossy image
// - Also use to 'clear' the image with the background image
//
void HdrEnvDome::create(VkDescriptorSet dstSet, VkDescriptorSetLayout dstSetLayout)
{
  destroy();
  m_hdrEnvSet    = dstSet;
  m_hdrEnvLayout = dstSetLayout;

  VkShaderModule diffModule = nvvk::createShaderModule(m_device, hdr_prefilter_diffuse_comp, sizeof(hdr_prefilter_diffuse_comp));
  VkShaderModule glossModule = nvvk::createShaderModule(m_device, hdr_prefilter_glossy_comp, sizeof(hdr_prefilter_glossy_comp));

  createDrawPipeline();
  integrateBrdf(512, m_textures.lutBrdf);
  prefilterHdr(128, m_textures.diffuse, diffModule, false);
  prefilterHdr(512, m_textures.glossy, glossModule, true);
  createDescriptorSetLayout();

  m_debug.setObjectName(m_textures.lutBrdf.image, "HDR_BRDF");
  m_debug.setObjectName(m_textures.diffuse.image, "HDR_Diffuse");
  m_debug.setObjectName(m_textures.glossy.image, "HDR_Glossy");

  vkDestroyShaderModule(m_device, diffModule, nullptr);
  vkDestroyShaderModule(m_device, glossModule, nullptr);
}

//--------------------------------------------------------------------------------------------------
// This is the image the HDR will be write to, a framebuffer image or an offsceen image
//
void HdrEnvDome::setOutImage(const VkDescriptorImageInfo& outimage)
{
  VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wds.dstSet          = m_domeSet;
  wds.dstBinding      = 0;
  wds.descriptorCount = 1;
  wds.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  wds.pImageInfo      = &outimage;
  vkUpdateDescriptorSets(m_device, 1, &wds, 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Compute Pipeline to "Clear" the image with the HDR as background
//
void HdrEnvDome::createDrawPipeline()
{
  // Descriptor: the output image
  nvvk::DescriptorSetBindings bind;
  bind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  m_domeLayout = bind.createLayout(m_device);
  m_domePool   = bind.createPool(m_device);
  m_domeSet    = nvvk::allocateDescriptorSet(m_device, m_domePool, m_domeLayout);

  // Creating the pipeline layout
  VkPushConstantRange                pushConstantRanges = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HdrDomePushConstant)};
  std::vector<VkDescriptorSetLayout> layouts{m_domeLayout, m_hdrEnvLayout};
  VkPipelineLayoutCreateInfo         createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = static_cast<uint32_t>(layouts.size());
  createInfo.pSetLayouts            = layouts.data();
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_domePipelineLayout);

  // HDR Dome compute shader
  VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module = nvvk::createShaderModule(m_device, hdr_dome_comp, sizeof(hdr_dome_comp));
  stageInfo.pName  = "main";

  VkComputePipelineCreateInfo compInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  compInfo.layout = m_domePipelineLayout;
  compInfo.stage  = stageInfo;

  vkCreateComputePipelines(m_device, {}, 1, &compInfo, nullptr, &m_domePipeline);
  NAME_VK(m_domePipeline);

  // Clean up
  vkDestroyShaderModule(m_device, compInfo.stage.module, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Draw the HDR to the image (setOutImage)
// - view and projection matrix should come from the camera
// - size is the image output size (framebuffer size)
// - color is the color multiplier of the HDR (intensity)
//
void HdrEnvDome::draw(const VkCommandBuffer& cmdBuf, const nvmath::mat4f& view, const nvmath::mat4f& proj, const VkExtent2D& size, const float* color)
{
  LABEL_SCOPE_VK(cmdBuf);

  // This will be to have a world direction vector pointing to the pixel
  nvmath::mat4f m = nvmath::invert(proj);
  m.a30 = m.a31 = m.a32 = m.a33 = 0.0f;
  m                             = nvmath::invert(view) * m;

  // Information to the compute shader
  HdrDomePushConstant pc;
  pc.mvp       = m;
  pc.multColor = color;

  // Execution
  std::vector<VkDescriptorSet> dstSets{m_domeSet, m_hdrEnvSet};
  vkCmdPushConstants(cmdBuf, m_domePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HdrDomePushConstant), &pc);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_domePipelineLayout, 0,
                          static_cast<uint32_t>(dstSets.size()), dstSets.data(), 0, nullptr);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, m_domePipeline);
  VkExtent2D grid = getGridSize(size);
  vkCmdDispatch(cmdBuf, grid.width, grid.height, 1);
}


//--------------------------------------------------------------------------------------------------
//
//
void HdrEnvDome::destroy()
{
  m_alloc->destroy(m_textures.diffuse);
  m_alloc->destroy(m_textures.lutBrdf);
  m_alloc->destroy(m_textures.glossy);

  vkDestroyPipeline(m_device, m_domePipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_domePipelineLayout, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_domeLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_domePool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_hdrLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_hdrPool, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Descriptors of the HDR and the acceleration structure
//
void HdrEnvDome::createDescriptorSetLayout()
{
  nvvk::DescriptorSetBindings bind;
  VkShaderStageFlags          flags = VK_SHADER_STAGE_ALL;

  bind.addBinding(EnvDomeBindings::eHdrBrdf, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, flags);      // HDR image
  bind.addBinding(EnvDomeBindings::eHdrDiffuse, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, flags);   // HDR image
  bind.addBinding(EnvDomeBindings::eHdrSpecular, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, flags);  // HDR image

  m_hdrPool = bind.createPool(m_device, 1);
  CREATE_NAMED_VK(m_hdrLayout, bind.createLayout(m_device));
  CREATE_NAMED_VK(m_hdrSet, nvvk::allocateDescriptorSet(m_device, m_hdrPool, m_hdrLayout));

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(m_hdrSet, EnvDomeBindings::eHdrBrdf, &m_textures.lutBrdf.descriptor));
  writes.emplace_back(bind.makeWrite(m_hdrSet, EnvDomeBindings::eHdrDiffuse, &m_textures.diffuse.descriptor));
  writes.emplace_back(bind.makeWrite(m_hdrSet, EnvDomeBindings::eHdrSpecular, &m_textures.glossy.descriptor));

  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Pre-integrate glossy BRDF, see
// http://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
void HdrEnvDome::integrateBrdf(uint32_t dimension, nvvk::Texture& target)
{
  auto tStart = std::chrono::high_resolution_clock::now();

  // Create an image RG16 to store the BRDF
  VkImageCreateInfo     imageCI       = nvvk::makeImage2DCreateInfo({dimension, dimension}, VK_FORMAT_R16G16_SFLOAT,
                                                          VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
  nvvk::Image           image         = m_alloc->createImage(imageCI);
  VkImageViewCreateInfo imageViewInfo = nvvk::makeImageViewCreateInfo(image.image, imageCI);
  VkSamplerCreateInfo   samplerCI     = nvvk::makeSamplerCreateInfo();
  target                              = m_alloc->createTexture(image, imageViewInfo, samplerCI);
  target.descriptor.imageLayout       = VK_IMAGE_LAYOUT_GENERAL;

  // Compute shader
  VkDescriptorSet       dstSet{VK_NULL_HANDLE};
  VkDescriptorSetLayout dstLayout{VK_NULL_HANDLE};
  VkDescriptorPool      dstPool{VK_NULL_HANDLE};
  VkPipeline            pipeline{VK_NULL_HANDLE};
  VkPipelineLayout      pipelineLayout{VK_NULL_HANDLE};

  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_familyIndex);
    LABEL_SCOPE_VK(cmdBuf);

    // Change image layout to general
    nvvk::cmdBarrierImageLayout(cmdBuf, target.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    // The output image is the one we have just created
    nvvk::DescriptorSetBindings bind;
    bind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
    dstLayout = bind.createLayout(m_device);
    dstPool   = bind.createPool(m_device);
    dstSet    = nvvk::allocateDescriptorSet(m_device, dstPool, dstLayout);

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(bind.makeWrite(dstSet, 0, &target.descriptor));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // Creating the pipeline
    VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    createInfo.setLayoutCount = 1;
    createInfo.pSetLayouts    = &dstLayout;
    vkCreatePipelineLayout(m_device, &createInfo, nullptr, &pipelineLayout);
    NAME_VK(pipelineLayout);

    VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = nvvk::createShaderModule(m_device, hdr_integrate_brdf_comp, sizeof(hdr_integrate_brdf_comp));
    stageInfo.pName  = "main";

    VkComputePipelineCreateInfo compInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    compInfo.layout = pipelineLayout;
    compInfo.stage  = stageInfo;

    vkCreateComputePipelines(m_device, {}, 1, &compInfo, nullptr, &pipeline);
    vkDestroyShaderModule(m_device, compInfo.stage.module, nullptr);

    // Run shader
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &dstSet, 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    VkExtent2D grid = getGridSize({dimension, dimension});
    vkCmdDispatch(cmdBuf, grid.width, grid.height, 1);
  }

  // Clean up
  vkDestroyDescriptorSetLayout(m_device, dstLayout, nullptr);
  vkDestroyDescriptorPool(m_device, dstPool, nullptr);
  vkDestroyPipeline(m_device, pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, pipelineLayout, nullptr);

  auto tEnd  = std::chrono::high_resolution_clock::now();
  auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
  LOGI(" - Generating BRDF LUT: %f ms \n", tDiff);
}


//--------------------------------------------------------------------------------------------------
//
//
void HdrEnvDome::prefilterHdr(uint32_t dim, nvvk::Texture& target, const VkShaderModule& module, bool doMipmap)
{
  auto tStart = std::chrono::high_resolution_clock::now();

  const VkExtent2D size{dim, dim};
  VkFormat         format  = VK_FORMAT_R16G16B16A16_SFLOAT;
  const uint32_t   numMips = doMipmap ? static_cast<uint32_t>(floor(log2(dim))) + 1 : 1;

  VkSamplerCreateInfo samplerCreateInfo = nvvk::makeSamplerCreateInfo();
  samplerCreateInfo.maxLod              = static_cast<float>(numMips);


  {  // Target - cube
    VkImageCreateInfo imageCreateInfo = nvvk::makeImageCubeCreateInfo(
        size, format, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, doMipmap);
    nvvk::Image           image         = m_alloc->createImage(imageCreateInfo);
    VkImageViewCreateInfo imageViewInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo, true);
    target                              = m_alloc->createTexture(image, imageViewInfo, samplerCreateInfo);
    target.descriptor.imageLayout       = VK_IMAGE_LAYOUT_GENERAL;
  }

  nvvk::Texture scratchTexture;
  {  // Scratch texture
    VkImageCreateInfo     imageCI         = nvvk::makeImage2DCreateInfo(size, format, VK_IMAGE_USAGE_STORAGE_BIT);
    nvvk::Image           image           = m_alloc->createImage(imageCI);
    VkImageViewCreateInfo imageViewInfo   = nvvk::makeImageViewCreateInfo(image.image, imageCI);
    VkSamplerCreateInfo   samplerCI       = nvvk::makeSamplerCreateInfo();
    scratchTexture                        = m_alloc->createTexture(image, imageViewInfo, samplerCI);
    scratchTexture.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }


  // Compute shader
  VkDescriptorSet       dstSet{VK_NULL_HANDLE};
  VkDescriptorSetLayout dstLayout{VK_NULL_HANDLE};
  VkDescriptorPool      dstPool{VK_NULL_HANDLE};
  VkPipeline            pipeline{VK_NULL_HANDLE};
  VkPipelineLayout      pipelineLayout{VK_NULL_HANDLE};

  // Descriptors
  nvvk::DescriptorSetBindings bind;
  bind.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT);
  dstLayout = bind.createLayout(m_device);
  dstPool   = bind.createPool(m_device);
  dstSet    = nvvk::allocateDescriptorSet(m_device, dstPool, dstLayout);

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(bind.makeWrite(dstSet, 0, &scratchTexture.descriptor));  // Writing to scratch
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

  // Creating the pipeline
  VkPushConstantRange                pushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HdrPushBlock)};
  std::vector<VkDescriptorSetLayout> layouts{dstLayout, m_hdrEnvLayout};
  VkPipelineLayoutCreateInfo         createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = static_cast<uint32_t>(layouts.size());
  createInfo.pSetLayouts            = layouts.data();
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRange;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &pipelineLayout);

  VkPipelineShaderStageCreateInfo stageInfo{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
  stageInfo.module = module;
  stageInfo.pName  = "main";

  VkComputePipelineCreateInfo compInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
  compInfo.layout = pipelineLayout;
  compInfo.stage  = stageInfo;

  vkCreateComputePipelines(m_device, {}, 1, &compInfo, nullptr, &pipeline);

  {
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_familyIndex);
    // Change scratch to general
    nvvk::cmdBarrierImageLayout(cmdBuf, scratchTexture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    // Change target to destination
    VkImageSubresourceRange subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, numMips, 0, 6};
    nvvk::cmdBarrierImageLayout(cmdBuf, target.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);

    std::vector<VkDescriptorSet> dstSets{dstSet, m_hdrEnvSet};
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0,
                            static_cast<uint32_t>(dstSets.size()), dstSets.data(), 0, nullptr);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    renderToCube(cmdBuf, target, scratchTexture, pipelineLayout, dim, numMips);
  }

  // Clean up
  vkDestroyDescriptorSetLayout(m_device, dstLayout, nullptr);
  vkDestroyDescriptorPool(m_device, dstPool, nullptr);
  vkDestroyPipeline(m_device, pipeline, nullptr);
  vkDestroyPipelineLayout(m_device, pipelineLayout, nullptr);

  m_alloc->destroy(scratchTexture);

  auto tEnd  = std::chrono::high_resolution_clock::now();
  auto tDiff = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
  LOGI(" - Prefilter cube, %d mip levels: %f ms \n", numMips, tDiff);
}


//--------------------------------------------------------------------------------------------------
//
//
void HdrEnvDome::renderToCube(const VkCommandBuffer& cmdBuf,
                              nvvk::Texture&         target,
                              nvvk::Texture&         scratch,
                              VkPipelineLayout       pipelineLayout,
                              uint32_t               dim,
                              const uint32_t         numMips)
{
  LABEL_SCOPE_VK(cmdBuf);

  nvmath::mat4f matPers = nvmath::perspectiveVK(90.0f, 1.0f, 0.1f, 10.0f);
  matPers               = nvmath::invert(matPers);
  matPers.a30 = matPers.a31 = matPers.a32 = matPers.a33 = 0.0f;

  std::array<nvmath::mat4f, 6> mv;
  const nvmath::vec3f          pos(0.0f, 0.0f, 0.0f);
  mv[0] = nvmath::look_at(pos, nvmath::vec3f(1.0f, 0.0f, 0.0f), nvmath::vec3f(0.0f, -1.0f, 0.0f));   // Positive X
  mv[1] = nvmath::look_at(pos, nvmath::vec3f(-1.0f, 0.0f, 0.0f), nvmath::vec3f(0.0f, -1.0f, 0.0f));  // Negative X
  mv[2] = nvmath::look_at(pos, nvmath::vec3f(0.0f, -1.0f, 0.0f), nvmath::vec3f(0.0f, 0.0f, -1.0f));  // Positive Y
  mv[3] = nvmath::look_at(pos, nvmath::vec3f(0.0f, 1.0f, 0.0f), nvmath::vec3f(0.0f, 0.0f, 1.0f));    // Negative Y
  mv[4] = nvmath::look_at(pos, nvmath::vec3f(0.0f, 0.0f, 1.0f), nvmath::vec3f(0.0f, -1.0f, 0.0f));   // Positive Z
  mv[5] = nvmath::look_at(pos, nvmath::vec3f(0.0f, 0.0f, -1.0f), nvmath::vec3f(0.0f, -1.0f, 0.0f));  // Negative Z
  for(auto& m : mv)
    m = nvmath::invert(m);

  // Change image layout for all cubemap faces to transfer destination
  VkImageSubresourceRange subresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, numMips, 0, 6};
  nvvk::cmdBarrierImageLayout(cmdBuf, target.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, subresourceRange);


  // Image barrier for compute stage
  auto barrier = [&](VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageMemoryBarrier    imageMemoryBarrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    imageMemoryBarrier.oldLayout           = oldLayout;
    imageMemoryBarrier.newLayout           = newLayout;
    imageMemoryBarrier.image               = scratch.image;
    imageMemoryBarrier.subresourceRange    = range;
    imageMemoryBarrier.srcAccessMask       = VK_ACCESS_MEMORY_WRITE_BIT;
    imageMemoryBarrier.dstAccessMask       = VK_ACCESS_MEMORY_READ_BIT;
    imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    VkPipelineStageFlags srcStageMask      = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags destStageMask     = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    vkCmdPipelineBarrier(cmdBuf, srcStageMask, destStageMask, 0, 0, nullptr, 0, nullptr, 1, &imageMemoryBarrier);
  };


  VkExtent3D   extent{dim, dim, 1};
  HdrPushBlock pushBlock{};

  for(uint32_t mip = 0; mip < numMips; mip++)
  {
    for(uint32_t f = 0; f < 6; f++)
    {
      // Update shader push constant block
      float roughness      = (float)mip / (float)(numMips - 1);
      pushBlock.roughness  = roughness;
      pushBlock.mvp        = mv[f] * matPers;
      pushBlock.size       = {extent.width, extent.height};
      pushBlock.numSamples = 1024 / (mip + 1);
      vkCmdPushConstants(cmdBuf, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(HdrPushBlock), &pushBlock);

      // Execute compute shader
      VkExtent2D grid = getGridSize({extent.width, extent.height});
      vkCmdDispatch(cmdBuf, grid.width, grid.height, 1);

      // Wait for compute to finish before copying
      barrier(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

      // Copy region for transfer from framebuffer to cube face
      VkImageCopy copyRegion{};
      copyRegion.srcSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      copyRegion.srcSubresource.baseArrayLayer = 0;
      copyRegion.srcSubresource.mipLevel       = 0;
      copyRegion.srcSubresource.layerCount     = 1;
      copyRegion.dstSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
      copyRegion.dstSubresource.baseArrayLayer = f;
      copyRegion.dstSubresource.mipLevel       = mip;
      copyRegion.dstSubresource.layerCount     = 1;
      copyRegion.extent                        = extent;

      vkCmdCopyImage(cmdBuf, scratch.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target.image,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

      // Transform scratch texture back to general
      barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    }

    // Next mipmap level
    if(extent.width > 1)
      extent.width /= 2;
    if(extent.height > 1)
      extent.height /= 2;
  }


  nvvk::cmdBarrierImageLayout(cmdBuf, target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, subresourceRange);
}
