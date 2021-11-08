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

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#define VMA_IMPLEMENTATION


#include "vulkan_sample.hpp"

#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/buffers_vk.hpp"

// Glsl Shaders compiled to Spir-V (See Makefile)
#include "_autogen/raster.vert.h"
#include "_autogen/raster.frag.h"
#include "_autogen/passthrough.vert.h"
#include "_autogen/pathtrace.rahit.h"
#include "_autogen/pathtrace.rchit.h"
#include "_autogen/pathtrace.rgen.h"
#include "_autogen/pathtrace.rmiss.h"
#include "_autogen/post.frag.h"
#include <filesystem>


//--------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------
void VulkanSample::create(const nvvk::AppBaseVkCreateInfo& info)
{
  AppBaseVk::create(info);
  m_alloc.init(m_instance, m_device, m_physicalDevice);
  m_debug.setup(m_device);
  m_picker.setup(m_device, m_physicalDevice, info.queueIndices[0], &m_alloc);
  m_hdrEnv.setup(m_device, m_physicalDevice, info.queueIndices[0], &m_alloc);
  m_hdrDome.setup(m_device, m_physicalDevice, info.queueIndices[0], &m_alloc);
}

//--------------------------------------------------------------------------------------------------
// Load the glTF and create the resources and pipeline for the sample
//
void VulkanSample::createScene(const std::string& filename)
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
  {  // Ray tracing
    nvh::Stopwatch sw_;
    initRayTracing();
    createBottomLevelAS();
    createTopLevelAS();
    createRtPipeline();
    LOGI(" - %6.2fms: Ray tracing\n", sw_.elapsed());
  }
  {  // Post
    nvh::Stopwatch sw_;
    createPostPipeline();
    updatePostDescriptorSet(m_offscreenColor.descriptor);
    LOGI(" - %6.2fms: Post\n", sw_.elapsed());
  }

  m_hdrDome.setOutImage(m_offscreenColor.descriptor);

  LOGI("TOTAL: %7.2fms\n\n", sw.elapsed());
}

//--------------------------------------------------------------------------------------------------
//
//
void VulkanSample::createHdr(const std::string& hdrFilename)
{
  LOGI("- HDR section \n");
  {  // HDR
    nvh::Stopwatch sw_;
    m_hdrEnv.loadEnvironment(hdrFilename);
    m_hdrDome.create(m_hdrEnv.getDescSet(), m_hdrEnv.getDescLayout());
    m_pcRay.maxLuminance = m_hdrEnv.getIntegral();
    LOGI(" = Total HDR: %6.2fms\n", sw_.elapsed());

    // Forced to regenerate the raster recorded command buffer
    vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_recordedCmdBuffer);
    m_recordedCmdBuffer = VK_NULL_HANDLE;

    if(m_offscreenColor.image != VK_NULL_HANDLE)
      m_hdrDome.setOutImage(m_offscreenColor.descriptor);
  }
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

  LOGI("- Loading file:\n\t %s\n", filename.c_str());
  if(!tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename))
  {
    assert(!"Error while loading scene");
  }
  LOGW(warn.c_str());
  LOGE(error.c_str());

  m_gltfScene.importMaterials(tmodel);
  m_gltfScene.importDrawableNodes(tmodel, nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0 | nvh::GltfAttributes::Tangent);

  // Create the buffers, copy vertices, indices and materials
  nvvk::CommandPool cmdPool(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = cmdPool.createCommandBuffer();

  {
    LABEL_SCOPE_VK(cmdBuf);
    createMaterialBuffer(cmdBuf);
    createInstanceInfoBuffer(cmdBuf);
    createVertexBuffer(cmdBuf);
    createTextureImages(cmdBuf, tmodel);

    // Buffer references
    SceneDescription sceneDesc{};
    sceneDesc.materialAddress = nvvk::getBufferDeviceAddress(m_device, m_materialBuffer.buffer);
    sceneDesc.primInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_primInfo.buffer);
    sceneDesc.instInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_instInfoBuffer.buffer);
    m_sceneDesc               = m_alloc.createBuffer(cmdBuf, sizeof(SceneDescription), &sceneDesc,
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    NAME_VK(m_sceneDesc.buffer);
  }

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

  std::unordered_map<std::string, nvvk::Buffer> cachePrimitive;

  for(auto& primMesh : m_gltfScene.m_primMeshes)
  {
    // Create a key to find a primitive that is already uploaded
    std::stringstream o;
    o << primMesh.vertexOffset << ":" << primMesh.vertexCount;
    std::string key = o.str();

    nvvk::Buffer v_buffer;  // Vertex buffer result
    auto         it = cachePrimitive.find(key);
    if(it == cachePrimitive.end())
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
        v.tangent    = m_gltfScene.m_tangents[idx];
        vertex.emplace_back(v);
      }
      v_buffer = m_alloc.createBuffer(cmdBuf, vertex, usageFlag | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      NAME_IDX_VK(v_buffer.buffer, primIdx);
    }
    else
    {
      v_buffer = it->second;
    }
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
    PrimMeshInfo info{};
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
// Create a buffer of all materials, with only the elements we need
//
void VulkanSample::createMaterialBuffer(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  std::vector<GltfShadeMaterial> shadeMaterials;
  for(auto& m : m_gltfScene.m_materials)
  {
    GltfShadeMaterial s{};
    s.emissiveFactor               = m.emissiveFactor;
    s.emissiveTexture              = m.emissiveTexture;
    s.khrDiffuseFactor             = m.specularGlossiness.diffuseFactor;
    s.khrDiffuseTexture            = m.specularGlossiness.diffuseTexture;
    s.khrSpecularFactor            = m.specularGlossiness.specularFactor;
    s.khrGlossinessFactor          = m.specularGlossiness.glossinessFactor;
    s.khrSpecularGlossinessTexture = m.specularGlossiness.specularGlossinessTexture;
    s.normalTexture                = m.normalTexture;
    s.normalTextureScale           = m.normalTextureScale;
    s.pbrBaseColorFactor           = m.baseColorFactor;
    s.pbrBaseColorTexture          = m.baseColorTexture;
    s.pbrMetallicFactor            = m.metallicFactor;
    s.pbrMetallicRoughnessTexture  = m.metallicRoughnessTexture;
    s.pbrRoughnessFactor           = m.roughnessFactor;
    s.shadingModel                 = m.shadingModel;
    s.alphaMode                    = m.alphaMode;
    s.alphaCutoff                  = m.alphaCutoff;

    shadeMaterials.emplace_back(s);
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
  LABEL_SCOPE_VK(cmdBuf);
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
void VulkanSample::updateUniformBuffer(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  CameraManip.updateAnim();

  // Prepare new UBO contents on host.
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);

  constexpr float zNear = 0.1f;
  constexpr float zFar  = 10000.0f;

  FrameInfo hostUBO{};
  hostUBO.view       = CameraManip.getMatrix();
  hostUBO.proj       = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, zNear, zFar);
  hostUBO.viewInv    = nvmath::invert(hostUBO.view);
  hostUBO.projInv    = nvmath::invert(hostUBO.proj);
  hostUBO.light[0]   = m_lights[0];
  hostUBO.light[1]   = m_lights[1];
  hostUBO.clearColor = m_clearColor.float32;

  // Schedule the host-to-device upload. (hostUBO is copied into the cmd buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_frameInfo.buffer, 0, sizeof(FrameInfo), &hostUBO);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering the scene with rasterizer
// Note: the descriptors are used by the ray tracer too, as second set
void VulkanSample::createGraphicPipeline()
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
  VkDescriptorBufferInfo             dbiUnif{m_frameInfo.buffer, 0, VK_WHOLE_SIZE};
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
  std::vector<VkDescriptorSetLayout> layouts{p.dstLayout, m_hdrDome.getDescLayout()};
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

  // Creating the Pipeline
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, p.pipelineLayout, m_offscreenRenderPass);
  gpb.depthStencilState.depthTestEnable = true;
  gpb.addShader(vertexShader, VK_SHADER_STAGE_VERTEX_BIT);
  gpb.addShader(fragShader, VK_SHADER_STAGE_FRAGMENT_BIT);
  gpb.addBindingDescriptions({{0, sizeof(Vertex)}});
  gpb.addAttributeDescriptions({
      {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, position)},  // Position + texcoord U
      {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, normal)},    // Normal + texcoord V
      {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent)},   // Tangents
  });
  p.pipeline = gpb.createPipeline();
  m_debug.setObjectName(p.pipeline, "Graphics");
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
// Creating all images, textures and samplers
//
void VulkanSample::createTextureImages(VkCommandBuffer cmdBuf, tinygltf::Model& gltfModel)
{
  LABEL_SCOPE_VK(cmdBuf);
  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.maxLod     = FLT_MAX;

  VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

  // Make dummy image(1,1), needed as we cannot have an empty array
  auto addDefaultImage = [this, cmdBuf]() {
    std::array<uint8_t, 4> white           = {255, 255, 255, 255};
    VkImageCreateInfo      imageCreateInfo = nvvk::makeImage2DCreateInfo(VkExtent2D{1, 1});
    nvvk::Image            image           = m_alloc.createImage(cmdBuf, 4, white.data(), imageCreateInfo);
    m_images.emplace_back(image, imageCreateInfo);
    m_debug.setObjectName(m_images.back().first.image, "dummy");
  };

  // Make dummy texture/image(1,1), needed as we cannot have an empty array
  auto addDefaultTexture = [&]() {
    if(m_images.empty())
      addDefaultImage();

    std::pair<nvvk::Image, VkImageCreateInfo>& image  = m_images[0];
    VkImageViewCreateInfo                      ivInfo = nvvk::makeImageViewCreateInfo(image.first.image, image.second);
    m_textures.emplace_back(m_alloc.createTexture(image.first, ivInfo, samplerCreateInfo));
  };

  if(gltfModel.images.empty())
  {
    addDefaultTexture();
    return;
  }

  // First - create the images
  m_images.reserve(gltfModel.images.size());
  for(size_t i = 0; i < gltfModel.images.size(); i++)
  {
    auto& gltfimage = gltfModel.images[i];
    void* buffer    = gltfimage.image.data();
    ;
    VkDeviceSize bufferSize = gltfimage.image.size();
    auto         imgSize    = VkExtent2D{(uint32_t)gltfimage.width, (uint32_t)gltfimage.height};

    if(bufferSize == 0 || gltfimage.width == -1 || gltfimage.height == -1)
    {
      addDefaultImage();  // Image not present or incorrectly loaded (image.empty)
      continue;
    }

    VkImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);
    nvvk::Image       image           = m_alloc.createImage(cmdBuf, bufferSize, buffer, imageCreateInfo);
    nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
    m_images.emplace_back(image, imageCreateInfo);
    NAME_IDX_VK(image.image, i);
  }

  // Creating the textures using the above images
  m_textures.reserve(gltfModel.textures.size());
  for(size_t i = 0; i < gltfModel.textures.size(); i++)
  {
    int sourceImage = gltfModel.textures[i].source;
    if(sourceImage >= gltfModel.images.size() || sourceImage < 0)
    {
      addDefaultTexture();  // Incorrect source image
      continue;
    }

    std::pair<nvvk::Image, VkImageCreateInfo>& image  = m_images[sourceImage];
    VkImageViewCreateInfo                      ivInfo = nvvk::makeImageViewCreateInfo(image.first.image, image.second);
    m_textures.emplace_back(m_alloc.createTexture(image.first, ivInfo, samplerCreateInfo));
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void VulkanSample::freeResources()
{
  // All pipelines
  for(auto& p : m_pContainer)
  {
    vkDestroyPipeline(m_device, p.pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, p.pipelineLayout, nullptr);
    vkDestroyDescriptorPool(m_device, p.dstPool, nullptr);
    vkDestroyDescriptorSetLayout(m_device, p.dstLayout, nullptr);
    p.pipeline       = VK_NULL_HANDLE;
    p.pipelineLayout = VK_NULL_HANDLE;
    p.dstPool        = VK_NULL_HANDLE;
    p.dstLayout      = VK_NULL_HANDLE;
  }

  vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_recordedCmdBuffer);
  m_recordedCmdBuffer = VK_NULL_HANDLE;

  // Resources
  m_alloc.destroy(m_frameInfo);
  m_alloc.destroy(m_materialBuffer);
  m_alloc.destroy(m_primInfo);
  m_alloc.destroy(m_instInfoBuffer);
  m_alloc.destroy(m_sceneDesc);

  for(auto& v : m_vertices)
  {
    m_alloc.destroy(v);
  }
  m_vertices.clear();

  for(auto& i : m_indices)
  {
    m_alloc.destroy(i);
  }
  m_indices.clear();

  for(auto& i : m_images)
  {
    m_alloc.destroy(i.first);
  }
  m_images.clear();

  for(auto& t : m_textures)
  {
    vkDestroyImageView(m_device, t.descriptor.imageView, nullptr);
  }
  m_textures.clear();

  m_gltfScene.destroy();

  // Post
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);
  vkDestroyRenderPass(m_device, m_offscreenRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);
  m_offscreenRenderPass  = VK_NULL_HANDLE;
  m_offscreenFramebuffer = VK_NULL_HANDLE;

  // Utilities
  m_rtBuilder.destroy();
  m_sbt.destroy();
  m_picker.destroy();
}

//--------------------------------------------------------------------------------------------------
//
//
void VulkanSample::destroy()
{
  freeResources();
  m_hdrEnv.destroy();
  m_hdrDome.destroy();
  m_alloc.deinit();
  AppBaseVk::destroy();
}

//--------------------------------------------------------------------------------------------------
//
//
void VulkanSample::drawDome(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);
  auto&       view        = CameraManip.getMatrix();
  auto        proj        = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);

  m_hdrDome.draw(cmdBuf, view, proj, m_size, &m_clearColor.float32[0]);
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode: record all command and "play back"
//
void VulkanSample::rasterize(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);

  // Recording the commands to draw the scene if not done yet
  if(m_recordedCmdBuffer == VK_NULL_HANDLE)
  {
    nvh::Stopwatch sw;
    // Create the command buffer to record the drawing commands
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool        = m_cmdPool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
    allocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(m_device, &allocInfo, &m_recordedCmdBuffer);

    VkCommandBufferInheritanceInfo inheritInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO};
    inheritInfo.renderPass = m_offscreenRenderPass;

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
    beginInfo.pInheritanceInfo = &inheritInfo;
    vkBeginCommandBuffer(m_recordedCmdBuffer, &beginInfo);

    // Dynamic Viewport
    setViewport(m_recordedCmdBuffer);

    // Drawing all instances
    auto&                        p = m_pContainer[eGraphic];
    std::vector<VkDescriptorSet> dstSets{p.dstSet, m_hdrDome.getDescSet()};
    vkCmdBindPipeline(m_recordedCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipeline);
    vkCmdBindDescriptorSets(m_recordedCmdBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipelineLayout, 0,
                            static_cast<uint32_t>(dstSets.size()), dstSets.data(), 0, nullptr);

    uint32_t     nodeId{0};
    VkDeviceSize offsets{0};
    for(auto& node : m_gltfScene.m_nodes)
    {
      auto& primitive = m_gltfScene.m_primMeshes[node.primMesh];
      // Push constant information
      m_pcRaster.materialId = primitive.materialIndex;
      m_pcRaster.instanceId = nodeId++;
      vkCmdPushConstants(m_recordedCmdBuffer, p.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(RasterPushConstant), &m_pcRaster);

      vkCmdBindVertexBuffers(m_recordedCmdBuffer, 0, 1, &m_vertices[node.primMesh].buffer, &offsets);
      vkCmdBindIndexBuffer(m_recordedCmdBuffer, m_indices[node.primMesh].buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(m_recordedCmdBuffer, primitive.indexCount, 1, 0, 0, 0);
    }
    vkEndCommandBuffer(m_recordedCmdBuffer);
    LOGI("Recoreded Command Buffer: %7.2fms\n", sw.elapsed());
  }

  // Executing the drawing of the recorded commands
  vkCmdExecuteCommands(cmdBuf, 1, &m_recordedCmdBuffer);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
// All screen size resources are re-created
void VulkanSample::onResize(int /*w*/, int /*h*/)
{
  vkFreeCommandBuffers(m_device, m_cmdPool, 1, &m_recordedCmdBuffer);
  m_recordedCmdBuffer = VK_NULL_HANDLE;

  createOffscreenRender();
  updatePostDescriptorSet(m_offscreenColor.descriptor);
  updateRtDescriptorSet();
  resetFrame();
  m_hdrDome.setOutImage(m_offscreenColor.descriptor);
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
  vkDestroyRenderPass(m_device, m_offscreenRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);


  VkFormat colorFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat depthFormat = nvvk::findDepthFormat(m_physicalDevice);

  // Creating the color image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(
        m_size, colorFormat, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);

    nvvk::Image image = m_alloc.createImage(colorCreateInfo);
    NAME_VK(image.image);

    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    VkSamplerCreateInfo   sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_offscreenColor                        = m_alloc.createTexture(image, ivInfo, sampler);
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Creating the depth buffer
  auto depthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, depthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  {
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);
    NAME_VK(image.image);

    VkImageViewCreateInfo depthStencilView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthStencilView.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format           = depthFormat;
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
  m_offscreenRenderPass = nvvk::createRenderPass(m_device, {colorFormat}, depthFormat, 1, false, true,
                                                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  NAME_VK(m_offscreenRenderPass);

  // Creating the frame buffer for offscreen
  std::vector<VkImageView> attachments = {m_offscreenColor.descriptor.imageView, m_offscreenDepth.descriptor.imageView};

  VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  info.renderPass      = m_offscreenRenderPass;
  info.attachmentCount = 2;
  info.pAttachments    = attachments.data();
  info.width           = m_size.width;
  info.height          = m_size.height;
  info.layers          = 1;
  vkCreateFramebuffer(m_device, &info, nullptr, &m_offscreenFramebuffer);
  NAME_VK(m_offscreenFramebuffer);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void VulkanSample::createPostPipeline()
{
  auto& p = m_pContainer[ePost];

  // The descriptor layout is the description of the data that is passed to the vertex or the  fragment program.
  nvvk::DescriptorSetBindings bind;
  bind.addBinding(PostBindings::eImage, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  p.dstLayout = bind.createLayout(m_device);
  p.dstPool   = bind.createPool(m_device);
  p.dstSet    = nvvk::allocateDescriptorSet(m_device, p.dstPool, p.dstLayout);

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
// Update the output
//
void VulkanSample::updatePostDescriptorSet(const VkDescriptorImageInfo& descriptor)
{
  VkWriteDescriptorSet wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wds.dstSet          = m_pContainer[ePost].dstSet;
  wds.dstBinding      = PostBindings::eImage;
  wds.descriptorCount = 1;
  wds.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  wds.pImageInfo      = &m_offscreenColor.descriptor;
  vkUpdateDescriptorSets(m_device, 1, &wds, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void VulkanSample::drawPost(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);

  auto& p = m_pContainer[ePost];
  setViewport(cmdBuf);
  vkCmdPushConstants(cmdBuf, p.pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Tonemapper), &m_tonemapper);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, p.pipelineLayout, 0, 1, &p.dstSet, 0, nullptr);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);
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
  m_sbt.setup(m_device, m_graphicsQueueIndex, &m_alloc, m_rtProperties);
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
  asGeom.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
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
    VkGeometryInstanceFlagsKHR flags{};  // Geometry flag based on material
    nvh::GltfPrimMesh&         primMesh = m_gltfScene.m_primMeshes[node.primMesh];
    nvh::GltfMaterial&         mat      = m_gltfScene.m_materials[primMesh.materialIndex];

    // Always opaque, no need to use anyhit (faster)
    if(mat.alphaMode == 0 || (mat.baseColorFactor.w == 1.0f && mat.baseColorTexture == -1))
      flags |= VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    // Need to skip the cull flag in traceray_rtx for double sided materials
    if(mat.doubleSided == 1)
      flags |= VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

    VkAccelerationStructureInstanceKHR rayInst{};
    rayInst.transform                      = nvvk::toTransformMatrixKHR(node.worldMatrix);  // Position of the instance
    rayInst.instanceCustomIndex            = node.primMesh;                                 // gl_InstanceCustomIndexEXT
    rayInst.accelerationStructureReference = m_rtBuilder.getBlasDeviceAddress(node.primMesh);
    rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
    rayInst.flags                                  = flags;
    rayInst.mask                                   = 0xFF;
    tlas.emplace_back(rayInst);
  }
  m_rtBuilder.buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

  // Use the TLAS for picking element in the scene
  m_picker.setTlas(m_rtBuilder.getAccelerationStructure());
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void VulkanSample::createRtPipeline()
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
  group.anyHitShader     = VK_SHADER_UNUSED_KHR;
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
  std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {p.dstLayout, m_pContainer[eGraphic].dstLayout, m_hdrEnv.getDescLayout()};
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
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void VulkanSample::updateRtDescriptorSet()
{
  // (1) Output buffer
  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet  wds{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
  wds.dstSet          = m_pContainer[eRaytrace].dstSet;
  wds.dstBinding      = RtxBindings::eOutImage;
  wds.descriptorCount = 1;
  wds.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  wds.pImageInfo      = &imageInfo;

  vkUpdateDescriptorSets(m_device, 1, &wds, 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void VulkanSample::raytrace(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);

  if(!updateFrame())
    return;

  std::vector<VkDescriptorSet> descSets{m_pContainer[eRaytrace].dstSet, m_pContainer[eGraphic].dstSet, m_hdrEnv.getDescSet()};
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pContainer[eRaytrace].pipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pContainer[eRaytrace].pipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_pContainer[eRaytrace].pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(RtxPushConstant), &m_pcRay);

  const auto& regions = m_sbt.getRegions();
  vkCmdTraceRaysKHR(cmdBuf, &regions[0], &regions[1], &regions[2], &regions[3], m_size.width, m_size.height, 1);
}

//--------------------------------------------------------------------------------------------------
// If the camera matrix has changed, resets the frame.
// otherwise, increments frame.
//
bool VulkanSample::updateFrame()
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

  if(m_pcRay.frame >= m_maxFrames)
    return false;
  m_pcRay.frame++;
  return true;
}

//--------------------------------------------------------------------------------------------------
// To be call when renderer need to re-start
//
void VulkanSample::resetFrame()
{
  m_pcRay.frame = -1;
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
  auto&       view        = CameraManip.getMatrix();
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
    char buf[256];
    snprintf(buf, IM_ARRAYSIZE(buf), "%s | %dx%d | %d FPS / %.3fms", PROJECT_NAME, m_size.width, m_size.height,
             static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);
    if(m_renderMode == RenderMode::eRayTracer)
      snprintf(buf, IM_ARRAYSIZE(buf), "%s | #%d", buf, m_pcRay.frame);
    glfwSetWindowTitle(m_window, buf);
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
  float widgetWidth = std::min(std::max(ImGui::GetWindowWidth() - 150.0f, 100.0f), 300.0f);
  ImGui::PushItemWidth(widgetWidth);

  if(ImGui::CollapsingHeader("Render Mode", ImGuiTreeNodeFlags_DefaultOpen))
  {
    changed |= ImGui::RadioButton("Raster", (int*)&m_renderMode, (int)RenderMode::eRaster);
    ImGui::SameLine();
    changed |= ImGui::RadioButton("Ray Tracing", (int*)&m_renderMode, (int)RenderMode::eRayTracer);
    if(m_renderMode == RenderMode::eRayTracer && ImGui::TreeNode("Ray Tracing"))
    {
      changed |= ImGui::SliderFloat("Max Luminance", &m_pcRay.maxLuminance, 0.01f, 20.f);
      changed |= ImGui::SliderInt("Depth", (int*)&m_pcRay.maxDepth, 1, 15);
      changed |= ImGui::SliderInt("Samples", (int*)&m_pcRay.maxSamples, 1, 20);
      int before = m_maxFrames;
      ImGui::SliderInt("Max frames", &m_maxFrames, 1, 50000);
      if(m_maxFrames < m_pcRay.frame)
        changed = true;
      ImGui::TreePop();
    }
  }

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
  ImGui::PopItemWidth();
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
      m_renderMode = static_cast<RenderMode>(((int)m_renderMode + 1) % 2);
      resetFrame();
      break;
    case GLFW_KEY_SPACE:  // Picking under mouse
      screenPicking();
      break;
  }
}

//--------------------------------------------------------------------------------------------------
// Allow to drop .hdr files
//
void VulkanSample::onFileDrop(const char* filename)
{
  namespace fs = std::filesystem;
  vkDeviceWaitIdle(m_device);
  std::string extension = fs::path(filename).extension().string();
  if(extension == ".gltf")
  {
    freeResources();
    createScene(filename);
  }
  else if(extension == ".hdr")
  {
    createHdr(filename);
  }

  resetFrame();
}
