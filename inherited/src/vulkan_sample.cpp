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

#include <sstream>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "vulkan_sample.hpp"

#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvh/gltfscene.hpp"
#include "nvh/nvprint.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"

#include "nvh/alignment.hpp"
#include "nvvk/buffers_vk.hpp"

// Glsl Shaders compiled to Spir-V (See Makefile)
#include "_autogen/frag_shader.frag.h"
#include "_autogen/passthrough.vert.h"
#include "_autogen/pathtrace.rchit.h"
#include "_autogen/pathtrace.rgen.h"
#include "_autogen/pathtrace.rmiss.h"
#include "_autogen/post.frag.h"
#include "_autogen/vert_shader.vert.h"
#include <iomanip>
#include "nvh/timesampler.hpp"

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void VulkanSample::create(const nvvk::AppBaseVkCreateInfo& info)
{
  AppBaseVk::create(info);

  m_alloc.init(m_instance, m_device, m_physicalDevice);
  m_debug.setup(m_device);
  m_picker.setup(m_device, m_physicalDevice, info.queueIndices[0], &m_alloc);

  m_offscreenDepthFormat = nvvk::findDepthFormat(m_physicalDevice);

  // One global light
  m_light[0].type      = 0;
  m_light[0].position  = {0.f, -2.5f, 0.4f};
  m_light[0].intensity = 15.f;
  m_light[0].color     = {0.9f, 0.9f, 1.0f};

  m_rtxPushConstant.frame = -1;


  // Requesting if inherited viewport is supported
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  features.pNext = &m_inheritedViewport;
  vkGetPhysicalDeviceFeatures2(m_physicalDevice, &features);
}

//--------------------------------------------------------------------------------------------------
// Load the glTF and create the resources and pipeline for the sample
//
void VulkanSample::createScene(const std::string& filename)
{
  nvh::Stopwatch sw;

  LOGI("\nCreate Sample\n");
  loadScene(filename);

  LOGI("- Pipeline creation\n");
  {  // Graphic
    nvh::Stopwatch sw_;
    createOffscreenRender();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createUniformBuffer();
    updateDescriptorSet();
    LOGI(" - %6.2fms: Graphic\n", sw_.elapsed());
  }
  {  // Ray tracing
    nvh::Stopwatch sw_;
    initRayTracing();
    createBottomLevelAS();
    createTopLevelAS();
    createRtDescriptorSet();
    createRtPipeline();
    LOGI(" - %6.2fms: Ray tracing\n", sw_.elapsed());
  }
  {  // Post
    nvh::Stopwatch sw_;
    createPostDescriptor();
    createPostPipeline();
    updatePostDescriptorSet();
    LOGI(" - %6.2fms: Post\n", sw_.elapsed());
  }

  setupGlfwCallbacks(m_window);
  LOGI("TOTAL: %7.2fms\n\n", sw.elapsed());
}


//--------------------------------------------------------------------------------------------------
// Loading the glTF file and setting up all buffers
//
void VulkanSample::loadScene(const std::string& filename)
{
  nvh::Stopwatch sw;
  using vkBU = VkBufferUsageFlagBits;
  tinygltf::Model    tmodel;
  tinygltf::TinyGLTF tcontext;
  std::string        warn, error;

  LOGI("- Loading file: %s\n", filename.c_str());
  if(!tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename))
  {
    assert(!"Error while loading scene");
  }
  LOGW(warn.c_str());
  LOGE(error.c_str());

  m_gltfScene.importMaterials(tmodel);
  m_gltfScene.importDrawableNodes(tmodel, nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0);

  // Create the buffers, copy vertices, indices and materials
  nvvk::CommandPool cmdPool(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = cmdPool.createCommandBuffer();

  createMaterialBuffer(cmdBuf);
  createInstanceInfoBuffer(cmdBuf);
  createVertexBuffer(cmdBuf);
  createTextureImages(cmdBuf, tmodel);

  // Buffer references
  SceneDescription sceneDesc;
  sceneDesc.materialAddress = nvvk::getBufferDeviceAddress(m_device, m_materialBuffer.buffer);
  sceneDesc.primInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_primInfo.buffer);
  sceneDesc.instInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_instInfoBuffer.buffer);
  m_sceneDesc               = m_alloc.createBuffer(cmdBuf, sizeof(SceneDescription), &sceneDesc,
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  NAME_VK(m_sceneDesc.buffer);

  cmdPool.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
  LOGI("  --> %7.2fms\n", sw.elapsed());
}


//--------------------------------------------------------------------------------------------------
// Creating information per primitive
// - Create a buffer of Vertex and Index for each primitive
// - Each primInfo has a reference to the vertex and index buffer, and which material id it uses
//
void VulkanSample::createVertexBuffer(VkCommandBuffer cmdBuf)
{
  std::vector<PrimMeshInfo> primInfo;  // The array of all primitive information
  uint32_t                  primIdx{0};

  auto usageFlag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                   | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

  for(auto& primMesh : m_gltfScene.m_primMeshes)
  {
    // Buffer of Vertex per primitive
    std::vector<Vertex> vertex;
    for(size_t v_ctx = 0; v_ctx < primMesh.vertexCount; v_ctx++)
    {
      Vertex v;
      size_t idx = primMesh.vertexOffset + v_ctx;
      v.position = m_gltfScene.m_positions[idx];
      v.normal   = m_gltfScene.m_normals[idx];
      // Adding texcoord to the end of position and normal vector
      v.position.w = m_gltfScene.m_texcoords0[idx].x;
      v.normal.w   = m_gltfScene.m_texcoords0[idx].y;
      vertex.emplace_back(v);
    }
    auto v_buffer = m_alloc.createBuffer(cmdBuf, vertex, usageFlag | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    NAME_IDX_VK(v_buffer.buffer, primIdx);
    m_vertices.emplace_back(v_buffer);

    // Buffer of indices
    std::vector<uint32_t> indices(primMesh.indexCount);
    for(size_t idx = 0; idx < primMesh.indexCount; idx++)
    {
      indices[idx] = m_gltfScene.m_indices[idx + primMesh.firstIndex];
    }
    auto i_buffer = m_alloc.createBuffer(cmdBuf, indices, usageFlag | VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    NAME_IDX_VK(i_buffer.buffer, primIdx);
    m_indices.emplace_back(i_buffer);

    // Primitive information, material Id and addresses of buffers
    PrimMeshInfo info;
    info.materialIndex = primMesh.materialIndex;
    info.vertexAddress = nvvk::getBufferDeviceAddress(m_device, v_buffer.buffer);
    info.indexAddress  = nvvk::getBufferDeviceAddress(m_device, i_buffer.buffer);
    primInfo.emplace_back(info);

    primIdx++;
  }

  // Creating the buffer of all primitive information
  m_primInfo = m_alloc.createBuffer(cmdBuf, primInfo, usageFlag);
  NAME_VK(m_primInfo.buffer);
}

//--------------------------------------------------------------------------------------------------
// Copying all materials, only the elements we need
// - Base color + texture, emissive
//
void VulkanSample::createMaterialBuffer(VkCommandBuffer cmdBuf)
{
  std::vector<ShadingMaterial> shadeMaterials;
  for(auto& m : m_gltfScene.m_materials)
  {
    shadeMaterials.emplace_back(ShadingMaterial{m.baseColorFactor, m.emissiveFactor, m.baseColorTexture});
  }
  m_materialBuffer = m_alloc.createBuffer(cmdBuf, shadeMaterials,
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  NAME_VK(m_materialBuffer.buffer);
}

//--------------------------------------------------------------------------------------------------
// Array of instance information
// - Use by the vertex shader to retrieve the position of the instance
void VulkanSample::createInstanceInfoBuffer(VkCommandBuffer cmdBuf)
{
  std::vector<InstanceInfo> instInfo;
  for(auto& node : m_gltfScene.m_nodes)
  {
    auto&        primitive = m_gltfScene.m_primMeshes[node.primMesh];
    InstanceInfo info;
    info.objMatrix   = node.worldMatrix;
    info.objMatrixIT = nvmath::transpose(nvmath::invert(node.worldMatrix));
    instInfo.emplace_back(info);
  }
  m_instInfoBuffer =
      m_alloc.createBuffer(cmdBuf, instInfo, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  NAME_VK(m_instInfoBuffer.buffer);
}


//--------------------------------------------------------------------------------------------------
// Called at each frame to update the frame information: camera matrices, light, clear color, ...
//
void VulkanSample::updateUniformBuffer(const VkCommandBuffer& cmdBuf, const VkExtent2D& size)
{
  // Prepare new UBO contents on host.
  const float aspectRatio = size.width / static_cast<float>(size.height);

  FrameInfo hostUBO;
  hostUBO.view       = CameraManip.getMatrix();
  hostUBO.proj       = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
  hostUBO.viewInv    = nvmath::invert(hostUBO.view);
  hostUBO.projInv    = nvmath::invert(hostUBO.proj);
  hostUBO.light[0]   = m_light[0];
  hostUBO.clearColor = m_clearColor.float32;
  hostUBO.light[0]   = m_light[0];

  // UBO on the device, and what stages access it.
  VkBuffer deviceUBO      = m_frameInfo.buffer;
  auto     uboUsageStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

  // Ensure that the modified UBO is not visible to previous frames.
  VkBufferMemoryBarrier beforeBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  beforeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  beforeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  beforeBarrier.buffer        = deviceUBO;
  beforeBarrier.offset        = 0;
  beforeBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, uboUsageStages, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &beforeBarrier, 0, nullptr);


  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_frameInfo.buffer, 0, sizeof(FrameInfo), &hostUBO);

  // Making sure the updated UBO will be visible.
  VkBufferMemoryBarrier afterBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  afterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  afterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  afterBarrier.buffer        = deviceUBO;
  afterBarrier.offset        = 0;
  afterBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, uboUsageStages, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &afterBarrier, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering the scene
//
void VulkanSample::createDescriptorSetLayout()
{
  // Frame Info (updated each frame)
  m_descSetLayoutBind.addBinding(SceneBindings::eFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR
                                     | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR);
  // Scene Description
  m_descSetLayoutBind.addBinding(SceneBindings::eSceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
  // Texture
  auto nbTextures = static_cast<uint32_t>(m_textures.size());
  m_descSetLayoutBind.addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nbTextures,
                                 VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR);

  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void VulkanSample::updateDescriptorSet()
{
  // Camera matrices and scene description
  VkDescriptorBufferInfo dbiUnif{m_frameInfo.buffer, 0, VK_WHOLE_SIZE};
  VkDescriptorBufferInfo sceneDesc{m_sceneDesc.buffer, 0, VK_WHOLE_SIZE};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eFrameInfo, &dbiUnif));
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eSceneDesc, &sceneDesc));

  // All texture samplers
  std::vector<VkDescriptorImageInfo> diit;
  for(auto& texture : m_textures)
    diit.emplace_back(texture.descriptor);
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, SceneBindings::eTextures, diit.data()));

  // Writing the information
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Creating the graphic (raster) pipeline layout
//
void VulkanSample::createGraphicsPipeline()
{
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(RasterPushConstant)};

  // Creating the Pipeline Layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_descSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_pipelineLayout);

  // Shader source (Spir-V)
  std::vector<uint32_t> vertexShader(std::begin(vert_shader_vert), std::end(vert_shader_vert));
  std::vector<uint32_t> fragShader(std::begin(frag_shader_frag), std::end(frag_shader_frag));

  // Creating the Pipeline
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_offscreenRenderPass);
  gpb.depthStencilState.depthTestEnable = true;
  gpb.addShader(vertexShader, VK_SHADER_STAGE_VERTEX_BIT);
  gpb.addShader(fragShader, VK_SHADER_STAGE_FRAGMENT_BIT);
  gpb.addBindingDescriptions({{0, sizeof(Vertex)}});
  gpb.addAttributeDescriptions({
      {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0},             // Position + texcoord U
      {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, sizeof(vec4)},  // Normal + texcoord V
  });
  m_graphicsPipeline = gpb.createPipeline();
  m_debug.setObjectName(m_graphicsPipeline, "Graphics");
}

//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices and other information changing at each frame
// - Buffer is host visible
//
void VulkanSample::createUniformBuffer()
{
  m_frameInfo = m_alloc.createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  NAME_VK(m_frameInfo.buffer);
}

//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//
void VulkanSample::createTextureImages(const VkCommandBuffer& cmdBuf, tinygltf::Model& gltfModel)
{
  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.maxLod     = FLT_MAX;

  VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;

  auto addDefaultTexture = [this]() {
    // Make dummy image(1,1), needed as we cannot have an empty array
    nvvk::ScopeCommandBuffer cmdBuf(m_device, m_graphicsQueueIndex);
    std::array<uint8_t, 4>   white = {255, 255, 255, 255};

    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_textures.emplace_back(m_alloc.createTexture(cmdBuf, 4, white.data(), nvvk::makeImage2DCreateInfo(VkExtent2D{1, 1}), sampler));
    m_debug.setObjectName(m_textures.back().image, "dummy");
  };

  if(gltfModel.images.empty())
  {
    addDefaultTexture();
    return;
  }

  m_textures.reserve(gltfModel.images.size());
  for(size_t i = 0; i < gltfModel.images.size(); i++)
  {
    auto& gltfimage = gltfModel.images[i];
    void* buffer    = gltfimage.image.data();
    ;
    VkDeviceSize bufferSize = gltfimage.image.size();
    auto         imgSize    = VkExtent2D{(uint32_t)gltfimage.width, (uint32_t)gltfimage.height};

    if(bufferSize == 0 || gltfimage.width == -1 || gltfimage.height == -1)
    {
      addDefaultTexture();
      continue;
    }

    VkImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);

    nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
    nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    m_textures.emplace_back(m_alloc.createTexture(image, ivInfo, samplerCreateInfo));

    m_debug.setObjectName(m_textures[i].image, std::string("Txt" + std::to_string(i)).c_str());
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void VulkanSample::destroyResources()
{
  vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);
  vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_recordedCmdBuffer);

  m_alloc.destroy(m_frameInfo);
  m_alloc.destroy(m_materialBuffer);
  m_alloc.destroy(m_primInfo);
  m_alloc.destroy(m_instInfoBuffer);
  m_alloc.destroy(m_sceneDesc);

  for(auto& v : m_vertices)
  {
    m_alloc.destroy(v);
  }

  for(auto& i : m_indices)
  {
    m_alloc.destroy(i);
  }

  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

  //#Post
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);
  vkDestroyPipeline(m_device, m_postPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_postPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_postDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_postDescSetLayout, nullptr);
  vkDestroyRenderPass(m_device, m_offscreenRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);

  // #VKRay
  m_rtBuilder.destroy();
  m_sbtWrapper.destroy();
  vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_rtDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_rtDescSetLayout, nullptr);

  m_picker.destroy();

  m_alloc.deinit();
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode: record all command and "play back"
//
void VulkanSample::rasterize(const VkCommandBuffer& cmdBuf)
{
  m_debug.beginLabel(cmdBuf, "Rasterize");


  // Recording the commands to draw the scene
  if(m_recordedCmdBuffer == VK_NULL_HANDLE)
  {
    nvh::Stopwatch              sw;
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool        = m_cmdPool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    allocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(m_device, &allocInfo, &m_recordedCmdBuffer);

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
    vkBeginCommandBuffer(m_recordedCmdBuffer, &beginInfo);

    // Dynamic Viewport
    if(m_inheritedViewport.inheritedViewportScissor2D == VK_FALSE)
      setViewport(m_recordedCmdBuffer);

    // Drawing all triangles
    vkCmdBindPipeline(m_recordedCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    vkCmdBindDescriptorSets(m_recordedCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);

    uint32_t     nodeId{0};
    VkDeviceSize offsets{0};
    for(auto& node : m_gltfScene.m_nodes)
    {
      auto& primitive = m_gltfScene.m_primMeshes[node.primMesh];

      m_pushConstant.materialId = primitive.materialIndex;
      m_pushConstant.instanceId = nodeId++;
      vkCmdPushConstants(m_recordedCmdBuffer, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(RasterPushConstant), &m_pushConstant);

      vkCmdBindVertexBuffers(m_recordedCmdBuffer, 0, 1, &m_vertices[node.primMesh].buffer, &offsets);
      vkCmdBindIndexBuffer(m_recordedCmdBuffer, m_indices[node.primMesh].buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(m_recordedCmdBuffer, primitive.indexCount, 1, 0, 0, 0);
    }
    vkEndCommandBuffer(m_recordedCmdBuffer);
    LOGI("Recoreded Command Buffer: %7.2fms\n", sw.elapsed());
  }


  // Executing the drawing of the recorded commands
  vkCmdExecuteCommands(cmdBuf, 1, &m_recordedCmdBuffer);


  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void VulkanSample::onResize(int /*w*/, int /*h*/)
{
  if(m_inheritedViewport.inheritedViewportScissor2D == VK_FALSE)
  {
    vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_recordedCmdBuffer);
    m_recordedCmdBuffer = VK_NULL_HANDLE;
  }

  createOffscreenRender();
  updatePostDescriptorSet();
  updateRtDescriptorSet();
  resetFrame();
}

//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void VulkanSample::createOffscreenRender()
{
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);

  // Creating the color image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
                                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                                           | VK_IMAGE_USAGE_STORAGE_BIT);


    nvvk::Image           image  = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    VkSamplerCreateInfo   sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_offscreenColor                        = m_alloc.createTexture(image, ivInfo, sampler);
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Creating the depth buffer
  auto depthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  {
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);


    VkImageViewCreateInfo depthStencilView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthStencilView.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format           = m_offscreenDepthFormat;
    depthStencilView.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    depthStencilView.image            = image.image;

    m_offscreenDepth = m_alloc.createTexture(image, depthStencilView);
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a renderpass for the offscreen
  if(!m_offscreenRenderPass)
  {
    m_offscreenRenderPass = nvvk::createRenderPass(m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat, 1, true,
                                                   true, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  }

  // Creating the frame buffer for offscreen
  std::vector<VkImageView> attachments = {m_offscreenColor.descriptor.imageView, m_offscreenDepth.descriptor.imageView};

  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);
  VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  info.renderPass      = m_offscreenRenderPass;
  info.attachmentCount = 2;
  info.pAttachments    = attachments.data();
  info.width           = m_size.width;
  info.height          = m_size.height;
  info.layers          = 1;
  vkCreateFramebuffer(m_device, &info, nullptr, &m_offscreenFramebuffer);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void VulkanSample::createPostPipeline()
{
  // Push constants in the fragment shader
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float)};

  // Creating the pipeline layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_postDescSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_postPipelineLayout);


  // Pipeline: completely generic, no vertices
  std::vector<uint32_t> vertexShader(std::begin(passthrough_vert), std::end(passthrough_vert));
  std::vector<uint32_t> fragShader(std::begin(post_frag), std::end(post_frag));

  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout, m_renderPass);
  pipelineGenerator.addShader(vertexShader, VK_SHADER_STAGE_VERTEX_BIT);
  pipelineGenerator.addShader(fragShader, VK_SHADER_STAGE_FRAGMENT_BIT);
  pipelineGenerator.rasterizationState.cullMode = VK_CULL_MODE_NONE;
  m_postPipeline                                = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void VulkanSample::createPostDescriptor()
{
  m_postDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
  m_postDescPool      = m_postDescSetLayoutBind.createPool(m_device);
  m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}


//--------------------------------------------------------------------------------------------------
// Update the output
//
void VulkanSample::updatePostDescriptorSet()
{
  VkWriteDescriptorSet writeDescriptorSets = m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor);
  vkUpdateDescriptorSets(m_device, 1, &writeDescriptorSets, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void VulkanSample::drawPost(VkCommandBuffer cmdBuf)
{
  m_debug.beginLabel(cmdBuf, "Post");

  setViewport(cmdBuf);

  auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  vkCmdPushConstants(cmdBuf, m_postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &aspectRatio);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout, 0, 1, &m_postDescSet, 0, nullptr);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);

  m_debug.endLabel(cmdBuf);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Initialize Vulkan ray tracing
//
void VulkanSample::initRayTracing()
{
  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext = &m_rtProperties;
  vkGetPhysicalDeviceProperties2(m_physicalDevice, &prop2);

  m_rtBuilder.setup(m_device, &m_alloc, m_graphicsQueueIndex);
  m_sbtWrapper.setup(m_device, m_graphicsQueueIndex, &m_alloc, m_rtProperties);
}

//--------------------------------------------------------------------------------------------------
// Converting a GLTF primitive in the Raytracing Geometry used for the BLAS
//
auto VulkanSample::primitiveToGeometry(const nvh::GltfPrimMesh& prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress)
{
  uint32_t maxPrimitiveCount = prim.indexCount / 3;

  // Describe buffer as array of VertexObj.
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat             = VK_FORMAT_R32G32B32A32_SFLOAT;  // vec3 vertex position data.
  triangles.vertexData.deviceAddress = vertexAddress;
  triangles.vertexStride             = sizeof(Vertex);
  triangles.indexType                = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress  = indexAddress;
  triangles.maxVertex                = prim.vertexCount;
  //triangles.transformData; // Identity

  // Identify the above data as containing opaque triangles.
  VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  asGeom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeom.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;
  asGeom.geometry.triangles = triangles;

  VkAccelerationStructureBuildRangeInfoKHR offset{};
  offset.firstVertex     = 0;
  offset.primitiveCount  = prim.indexCount / 3;
  offset.primitiveOffset = 0;
  offset.transformOffset = 0;

  // Our blas is made from only one geometry, but could be made of many geometries
  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);

  return input;
}

//--------------------------------------------------------------------------------------------------
// Create all bottom level acceleration structures (BLAS)
//
void VulkanSample::createBottomLevelAS()
{
  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
  allBlas.reserve(m_gltfScene.m_primMeshes.size());

  for(uint32_t p_idx = 0; p_idx < m_gltfScene.m_primMeshes.size(); p_idx++)
  {
    auto vertexAddress = nvvk::getBufferDeviceAddress(m_device, m_vertices[p_idx].buffer);
    auto indexAddress  = nvvk::getBufferDeviceAddress(m_device, m_indices[p_idx].buffer);

    auto geo = primitiveToGeometry(m_gltfScene.m_primMeshes[p_idx], vertexAddress, indexAddress);
    allBlas.push_back({geo});
  }
  m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
// Create the top level acceleration structures, referencing all BLAS
//
void VulkanSample::createTopLevelAS()
{
  std::vector<VkAccelerationStructureInstanceKHR> tlas;
  tlas.reserve(m_gltfScene.m_nodes.size());
  for(auto& node : m_gltfScene.m_nodes)
  {
    VkAccelerationStructureInstanceKHR rayInst{};
    rayInst.transform                      = nvvk::toTransformMatrixKHR(node.worldMatrix);
    rayInst.instanceCustomIndex            = node.primMesh;  // gl_InstanceCustomIndexEXT: to find which primitive
    rayInst.accelerationStructureReference = m_rtBuilder.getBlasDeviceAddress(node.primMesh);
    rayInst.flags                          = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
    rayInst.mask                                   = 0xFF;
    tlas.emplace_back(rayInst);
  }
  m_rtBuilder.buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);


  // Use the TLAS for picking element in the scene
  m_picker.setTlas(m_rtBuilder.getAccelerationStructure());
}

//--------------------------------------------------------------------------------------------------
// This descriptor set, holds the top level acceleration structure and the output image
//
void VulkanSample::createRtDescriptorSet()
{
  // Top-level acceleration structure, usable by both the ray generation and the closest hit (to shoot shadow rays)
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);  // TLAS
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);  // Output image (offscreen image)

  m_rtDescPool      = m_rtDescSetLayoutBind.createPool(m_device);
  m_rtDescSetLayout = m_rtDescSetLayoutBind.createLayout(m_device);

  VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocateInfo.descriptorPool     = m_rtDescPool;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts        = &m_rtDescSetLayout;
  vkAllocateDescriptorSets(m_device, &allocateInfo, &m_rtDescSet);

  VkAccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure();
  VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  descASInfo.accelerationStructureCount = 1;
  descASInfo.pAccelerationStructures    = &tlas;
  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eTlas, &descASInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void VulkanSample::updateRtDescriptorSet()
{
  // (1) Output buffer
  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet  wds = m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo);
  vkUpdateDescriptorSets(m_device, 1, &wds, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void VulkanSample::createRtPipeline()
{
  enum StageIndices
  {
    eRaygen,
    eMiss,
    eClosestHit,
    eShaderGroupCount
  };

  // All stages
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

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  m_rtShaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  m_rtShaderGroups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  m_rtShaderGroups.push_back(group);


  // Push constant: we want to be able to update constants used by the shaders
  VkPushConstantRange pushConstant{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                                   0, sizeof(RtxPushConstant)};


  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {m_rtDescSetLayout, m_descSetLayout};
  pipelineLayoutCreateInfo.setLayoutCount             = static_cast<uint32_t>(rtDescSetLayouts.size());
  pipelineLayoutCreateInfo.pSetLayouts                = rtDescSetLayouts.data();
  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_rtPipelineLayout);


  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  rayPipelineInfo.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
  rayPipelineInfo.pStages                      = stages.data();
  rayPipelineInfo.groupCount                   = static_cast<uint32_t>(m_rtShaderGroups.size());
  rayPipelineInfo.pGroups                      = m_rtShaderGroups.data();
  rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
  rayPipelineInfo.layout                       = m_rtPipelineLayout;
  vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_rtPipeline);

  // Creating the SBT
  m_sbtWrapper.create(m_rtPipeline, rayPipelineInfo);

  // Removing temp modules
  for(auto& s : stages)
    vkDestroyShaderModule(m_device, s.module, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void VulkanSample::raytrace(const VkCommandBuffer& cmdBuf)
{
  updateFrame();
  m_debug.beginLabel(cmdBuf, "Ray trace");


  std::vector<VkDescriptorSet> descSets{m_rtDescSet, m_descSet};
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_rtPipelineLayout,
                     VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR,
                     0, sizeof(RtxPushConstant), &m_rtxPushConstant);


  auto& regions = m_sbtWrapper.getRegions();
  vkCmdTraceRaysKHR(cmdBuf, &regions[0], &regions[1], &regions[2], &regions[3], m_size.width, m_size.height, 1);


  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// If the camera matrix has changed, resets the frame.
// otherwise, increments frame.
//
void VulkanSample::updateFrame()
{
  static nvmath::mat4f refCamMatrix;
  static float         refFov{CameraManip.getFov()};

  const auto& m   = CameraManip.getMatrix();
  const auto  fov = CameraManip.getFov();

  if(memcmp(&refCamMatrix.a00, &m.a00, sizeof(nvmath::mat4f)) != 0 || refFov != fov)
  {
    resetFrame();
    refCamMatrix = m;
    refFov       = fov;
  }
  m_rtxPushConstant.frame++;
}

//--------------------------------------------------------------------------------------------------
// To be call when renderer need to re-start
//
void VulkanSample::resetFrame()
{
  m_rtxPushConstant.frame = -1;
}


//--------------------------------------------------------------------------------------------------
// Send a ray under mouse coordinates, and retrieve the information
// - Set new camera interest point on hit position
//
void VulkanSample::screenPicking()
{
  double x, y;
  glfwGetCursorPos(m_window, &x, &y);

  nvvk::CommandPool sc(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = sc.createCommandBuffer();

  // Finding current camera matrices
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);
  const auto& view        = CameraManip.getMatrix();
  auto        proj        = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);

  // Setting up the data to do picking
  nvvk::RayPickerKHR::PickInfo pickInfo;
  pickInfo.pickX          = float(x) / float(m_size.width);
  pickInfo.pickY          = float(y) / float(m_size.height);
  pickInfo.modelViewInv   = nvmath::invert(view);
  pickInfo.perspectiveInv = nvmath::invert(proj);

  // Run and wait for result
  m_picker.run(cmdBuf, pickInfo);
  sc.submitAndWait(cmdBuf);

  // Retrieving picking information
  nvvk::RayPickerKHR::PickResult pr = m_picker.getResult();
  if(pr.instanceID == ~0)
  {
    LOGI("Nothing Hit\n");
    return;
  }

  // Find where the hit point is and set the interest position
  nvmath::vec3f worldPos = pr.worldRayOrigin + pr.worldRayDirection * pr.hitT;
  nvmath::vec3f eye, center, up;
  CameraManip.getLookat(eye, center, up);
  CameraManip.setLookat(eye, worldPos, up, false);

  // Logging picking info.
  auto& prim = m_gltfScene.m_primMeshes[pr.instanceCustomIndex];
  LOGI("Hit(%d): %s, PrimId: %d, ", pr.instanceCustomIndex, prim.name.c_str(), pr.primitiveID, pr.hitT);
  LOGI("{%3.2f, %3.2f, %3.2f}, Dist: %3.2f\n", worldPos.x, worldPos.y, worldPos.z, pr.hitT);
}


//--------------------------------------------------------------------------------------------------
// Display information in the title bar
//
void VulkanSample::titleBar()
{
  static float dirtyTimer = 0.0f;
  dirtyTimer += ImGui::GetIO().DeltaTime;
  if(dirtyTimer > 1.0f)  // Refresh every seconds
  {
    std::stringstream o;
    o << PROJECT_NAME;
    o << " | " << m_size.width << "x" << m_size.height;       // resolution
    o << " | " << static_cast<int>(ImGui::GetIO().Framerate)  // FPS / ms
      << " FPS / " << std::setprecision(3) << 1000.F / ImGui::GetIO().Framerate << "ms";
    if(m_renderMode == eRayTracer)
      o << " | #" << m_rtxPushConstant.frame;
    glfwSetWindowTitle(m_window, o.str().c_str());
    dirtyTimer = 0;
  }
}

//--------------------------------------------------------------------------------------------------
// Rendering UI
//
void VulkanSample::renderUI()
{
  if(showGui() == false)
    return;

  bool changed{false};

  ImGuiH::Panel::Begin();

  if(ImGui::CollapsingHeader("Render Mode", ImGuiTreeNodeFlags_DefaultOpen))
  {
    changed |= ImGui::RadioButton("Raster", (int*)&m_renderMode, eRaster);
    ImGui::SameLine();
    changed |= ImGui::RadioButton("Ray Tracing", (int*)&m_renderMode, eRayTracer);
  }

  if(ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGuiH::CameraWidget();
  }

  if(ImGui::CollapsingHeader("Environment", ImGuiTreeNodeFlags_DefaultOpen))
  {
    changed |= ImGui::ColorEdit3("Clear color", reinterpret_cast<float*>(&m_clearColor.float32[0]));

    //if(ImGui::CollapsingHeader("Light"), ImGuiTreeNodeFlags_DefaultOpen)
    {
      changed |= ImGui::RadioButton("Point", &m_light[0].type, 0);
      ImGui::SameLine();
      changed |= ImGui::RadioButton("Infinite", &m_light[0].type, 1);
      changed |= ImGui::DragFloat3("Position", &m_light[0].position.x, 0.1f);
      changed |= ImGui::DragFloat("Intensity", &m_light[0].intensity, 1.f, 0.0f, 1000.f, nullptr,
                                  ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
      changed |= ImGui::ColorEdit3("Color", reinterpret_cast<float*>(&m_light[0].color));
    }
  }
  if(ImGui::Checkbox("Inherited Viewport", (bool*)&m_inheritedViewport.inheritedViewportScissor2D))
  {
    vkDeviceWaitIdle(m_device);
    vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_recordedCmdBuffer);
    m_recordedCmdBuffer = VK_NULL_HANDLE;
  }
  ImGui::SliderFloat2("View Center", &m_viewCenter.x, 0.2f, 0.8f);

  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
  ImGuiH::Control::Info("", "", "(M) Toggle Render Mode", ImGuiH::Control::Flags::Disabled);
  ImGuiH::Control::Info("", "", "(F) Frame All", ImGuiH::Control::Flags::Disabled);
  ImGuiH::Control::Info("", "", "(R) Restart rendering", ImGuiH::Control::Flags::Disabled);
  ImGuiH::Control::Info("", "", "(SPACE) Set Eye position", ImGuiH::Control::Flags::Disabled);
  ImGuiH::Control::Info("", "", "(F10) Toggle Pane", ImGuiH::Control::Flags::Disabled);
  ImGuiH::Control::Info("", "", "(ESC) Quit", ImGuiH::Control::Flags::Disabled);
  ImGuiH::Panel::End();

  if(changed)
    resetFrame();
}

//--------------------------------------------------------------------------------------------------
// Override when mouse click, dealing only with double click
//
void VulkanSample::onMouseButton(int button, int action, int mods)
{
  AppBaseVk::onMouseButton(button, action, mods);
  if(ImGui::GetIO().MouseDownWasDoubleClick[0])
  {
    screenPicking();
  }
}

//--------------------------------------------------------------------------------------------------
// Overload keyboard hit
//
void VulkanSample::onKeyboard(int key, int scancode, int action, int mods)
{
  nvvk::AppBaseVk::onKeyboard(key, scancode, action, mods);
  if(action == GLFW_RELEASE)
    return;

  switch(key)
  {
    case GLFW_KEY_F:  // Set the camera as to see the model
      fitCamera(m_gltfScene.m_dimensions.min, m_gltfScene.m_dimensions.max, false);
      break;
    case GLFW_KEY_R:  // Restart rendering
      resetFrame();
      break;
    case GLFW_KEY_M:  // Toggling rendering mode
      m_renderMode = static_cast<RenderMode>((m_renderMode + 1) % 2);
      resetFrame();
      break;
    case GLFW_KEY_SPACE:  // Picking under mouse
      screenPicking();
      break;
  }
}


//--------------------------------------------------------------------------------------------------
// Rendering the scene in 4 different corners using always the same
// camera perspective (see: updateUniformBuffer())
//
void VulkanSample::fourViews(VkCommandBuffer cmdBuf)
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
