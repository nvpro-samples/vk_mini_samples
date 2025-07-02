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

//////////////////////////////////////////////////////////////////////////
/*

    This shows the use of Ray Query, or casting rays in a compute shader

*/
//////////////////////////////////////////////////////////////////////////

#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }


#include <array>
#include <glm/glm.hpp>
#include <vulkan/vulkan_core.h>


namespace shaderio {
using namespace glm;
#include "shaders/dh_bindings.h"
#include "shaders/shaderio.h"  // Shared between host and device
}  // namespace shaderio


#include "_autogen/ray_query.comp.glsl.h"
#include "_autogen/ray_query.slang.h"
#include "_autogen/tonemapper.slang.h"

#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/property_editor.hpp>
#include <nvgui/tonemapper.hpp>
#include <nvshaders_host/tonemapper.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/primitives.hpp>
#include <nvvk/acceleration_structures.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/specialization.hpp>
#include <nvvk/staging.hpp>
#include "common/utils.hpp"


std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};


/// </summary> Ray trace multiple primitives using Ray Query
class RayQuery : public nvapp::IAppElement
{
  enum
  {
    eImgTonemapped,
    eImgRendered
  };

public:
  RayQuery()           = default;
  ~RayQuery() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    // Create the Vulkan allocator (VMA)
    m_allocator.init({
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });  // Allocator


    // The texture sampler to use
    m_samplerPool.init(m_device);
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // GBuffer for the ray tracing and tonemapping
    m_gBuffers.init({.allocator      = &m_allocator,
                     .colorFormats   = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32G32B32A32_SFLOAT},
                     .imageSampler   = linearSampler,
                     .descriptorPool = m_app->getTextureDescriptorPool()});

    // Tonemapper
    {
      auto code = std::span<const uint32_t>(tonemapper_slang);
      m_tonemapper.init(&m_allocator, code);
    }

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    m_rtProperties.pNext = &m_asProperties;
    prop2.pNext          = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create resources
    createScene();
    createVkBuffers();
    createBottomLevelAS();
    createTopLevelAS();
    createCompPipelines();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    m_gBuffers.update(cmd, size);
    resetFrame();  // Reset frame to restart the rendering
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");

      nvgui::CameraWidget(g_cameraManip);

      namespace PE = nvgui::PropertyEditor;
      bool changed{false};
      if(ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
      {
        PE::begin();
        if(PE::treeNode("Light"))
        {
          changed |= PE::DragFloat3("Position", &m_light.position.x);
          changed |= PE::SliderFloat("Intensity", &m_light.intensity, 0.0F, 1000.0F, "%.3f", ImGuiSliderFlags_Logarithmic);
          changed |= PE::SliderFloat("Radius", &m_light.radius, 0.0F, 1.0F);
          PE::treePop();
        }
        if(PE::treeNode("Ray Tracer"))
        {
          changed |= PE::SliderInt("Depth", &m_pushConst.maxDepth, 0, 20);
          changed |= PE::SliderInt("Samples", &m_pushConst.maxSamples, 1, 10);
          PE::treePop();
        }
        PE::end();
      }

      if(ImGui::CollapsingHeader("Tonemapper"))
      {
        nvgui::tonemapperWidget(m_tonemapperData);
      }

      ImGui::End();
      if(changed)
        resetFrame();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer tonemapped image
      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(eImgTonemapped), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);

    if(!updateFrame())
    {
      return;
    }

    // Update Camera uniform buffer
    shaderio::CameraInfo cameraInfo{.projInv = glm::inverse(g_cameraManip->getPerspectiveMatrix()),
                                    .viewInv = glm::inverse(g_cameraManip->getViewMatrix())};
    vkCmdUpdateBuffer(cmd, m_bCameraInfo.buffer, 0, sizeof(shaderio::CameraInfo), &cameraInfo);

    m_pushConst.frame = m_frame;
    m_pushConst.light = m_light;

    // Make sure buffer is ready to be used
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Ray trace
    const VkExtent2D& size = m_app->getViewportSize();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    pushDescriptorSet(cmd);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);
    vkCmdDispatch(cmd, (size.width + (WORKGROUP_SIZE - 1)) / WORKGROUP_SIZE,
                  (size.height + (WORKGROUP_SIZE - 1)) / WORKGROUP_SIZE, 1);

    // Making sure the rendered image is ready to be used by tonemapper
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Tonemap
    m_tonemapper.runCompute(cmd, m_gBuffers.getSize(), m_tonemapperData, m_gBuffers.getDescriptorImageInfo(eImgRendered),
                            m_gBuffers.getDescriptorImageInfo(eImgTonemapped));
  }

private:
  void createScene()
  {
    m_materials.push_back({{0.985f, 0.862f, 0.405f}, 0.5f, 0.0f});
    m_materials.push_back({{0.9622f, 0.928f, 0.9728f}, 0.0f, 0.09f, 1.0f});
    m_materials.push_back({{.7F, .7F, .7F}, 0.3f, 0.0f});

    m_meshes.emplace_back(nvutils::createCube(1, 1, 1));
    m_meshes.emplace_back(nvutils::createSphereUv(0.5f, 100, 100));
    m_meshes.emplace_back(nvutils::createPlane(10, 100, 100));

    // Instance Cube
    {
      auto& node       = m_nodes.emplace_back();
      node.mesh        = 0;
      node.material    = 0;
      node.translation = {0.0f, 0.5f, 0.0F};
    }

    // Instance Sphere
    {
      auto& node       = m_nodes.emplace_back();
      node.mesh        = 1;
      node.material    = 1;
      node.translation = {1.0f, 1.5f, 1.0F};
    }

    // Adding a plane & material
    {
      auto& node       = m_nodes.emplace_back();
      node.mesh        = 2;
      node.material    = 2;
      node.translation = {0.0f, 0.0f, 0.0f};
    }

    // Adding a light
    m_light.intensity = 100.0f;
    m_light.position  = {2.0f, 7.0f, 2.0f};
    m_light.radius    = 0.2f;

    // Setting camera to see the scene
    g_cameraManip->setClipPlanes({0.1F, 100.0F});
    g_cameraManip->setLookat({-2.0F, 2.5F, 3.0f}, {0.4F, 0.3F, 0.2F}, {0.0F, 1.0F, 0.0F});

    // Default parameters for overall material
    m_pushConst.maxDepth              = 5;
    m_pushConst.frame                 = 0;
    m_pushConst.fireflyClampThreshold = 10;
    m_pushConst.maxSamples            = 2;
    m_pushConst.light                 = m_light;
  }

  // Create all Vulkan buffer data
  void createVkBuffers()
  {
    //    NVTX3_FUNC_RANGE();
    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_allocator, true);

    m_bMeshes.resize(m_meshes.size());

    VkBufferUsageFlags rtUsageFlag = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                     | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;


    // Create a buffer of Vertex and Index per mesh
    std::vector<shaderio::PrimMeshInfo> primInfo;
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      auto& m = m_bMeshes[i];

      NVVK_CHECK(m_allocator.createBuffer(m.vertices, std::span(m_meshes[i].vertices).size_bytes(), rtUsageFlag));
      NVVK_CHECK(m_allocator.createBuffer(m.indices, std::span(m_meshes[i].triangles).size_bytes(), rtUsageFlag));
      NVVK_CHECK(uploader.appendBuffer(m.vertices, 0, std::span(m_meshes[i].vertices)));
      NVVK_CHECK(uploader.appendBuffer(m.indices, 0, std::span(m_meshes[i].triangles)));
      NVVK_DBG_NAME(m.vertices.buffer);
      NVVK_DBG_NAME(m.indices.buffer);

      // To find the buffers of the mesh (buffer reference)
      shaderio::PrimMeshInfo info{
          .vertexAddress = m.vertices.address,
          .indexAddress  = m.indices.address,
      };
      primInfo.emplace_back(info);
    }

    // Creating the buffer of all primitive/mesh information
    NVVK_CHECK(m_allocator.createBuffer(m_bPrimInfo, std::span(primInfo).size_bytes(), rtUsageFlag));
    NVVK_CHECK(uploader.appendBuffer(m_bPrimInfo, 0, std::span(primInfo)));
    NVVK_DBG_NAME(m_bPrimInfo.buffer);

    // Create the buffer of the current camera transformation, changing at each frame
    NVVK_CHECK(m_allocator.createBuffer(m_bCameraInfo, sizeof(shaderio::CameraInfo),
                                        VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST));
    NVVK_DBG_NAME(m_bCameraInfo.buffer);

    // Primitive instance information
    std::vector<shaderio::InstanceInfo> instInfo;
    for(auto& node : m_nodes)
    {
      shaderio::InstanceInfo info{
          .transform  = node.localMatrix(),
          .materialID = node.material,
      };
      instInfo.emplace_back(info);
    }
    NVVK_CHECK(m_allocator.createBuffer(m_bInstInfoBuffer, std::span(instInfo).size_bytes(),
                                        VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(uploader.appendBuffer(m_bInstInfoBuffer, 0, std::span(instInfo)));
    NVVK_DBG_NAME(m_bInstInfoBuffer.buffer);

    NVVK_CHECK(m_allocator.createBuffer(m_bMaterials, std::span(m_materials).size_bytes(),
                                        VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(uploader.appendBuffer(m_bMaterials, 0, std::span(m_materials)));
    NVVK_DBG_NAME(m_bMaterials.buffer);

    // Buffer references of all scene elements
    shaderio::SceneInfo sceneDesc{
        .materialAddress = m_bMaterials.address,
        .instInfoAddress = m_bInstInfoBuffer.address,
        .primInfoAddress = m_bPrimInfo.address,
        .light           = m_light,
    };

    NVVK_CHECK(m_allocator.createBuffer(m_bSceneDesc, sizeof(shaderio::SceneInfo),
                                        VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(uploader.appendBuffer(m_bSceneDesc, 0, std::span(&sceneDesc, 1)));
    NVVK_DBG_NAME(m_bSceneDesc.buffer);

    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();
  }


  //--------------------------------------------------------------------------------------------------
  // Converting a PrimitiveMesh as input for BLAS
  //
  nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const nvutils::PrimitiveMesh& prim,
                                                              VkDeviceAddress               vertexAddress,
                                                              VkDeviceAddress               indexAddress)
  {
    nvvk::AccelerationStructureGeometryInfo result;

    uint32_t maxPrimitiveCount = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,  // vec3 vertex position data
        .vertexData   = {.deviceAddress = vertexAddress},
        .vertexStride = sizeof(nvutils::PrimitiveVertex),
        .maxVertex    = static_cast<uint32_t>(prim.vertices.size()) - 1,
        .indexType    = VK_INDEX_TYPE_UINT32,
        .indexData    = {.deviceAddress = indexAddress},
    };

    // Identify the above data as containing opaque triangles.
    result.geometry = VkAccelerationStructureGeometryKHR{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry     = {.triangles = triangles},
        .flags        = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR,  //| VK_GEOMETRY_OPAQUE_BIT_KHR,
    };

    result.rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = maxPrimitiveCount};

    return result;
  }

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  void createBottomLevelAS()
  {
    // NVTX3_FUNC_RANGE();
    std::vector<nvvk::AccelerationStructureBuildData> blasData;
    blasData.resize(m_meshes.size());    // Build Information for each BLAS
    m_bottomAs.resize(m_meshes.size());  // The actual BLAS

    // Convert all primitives to acceleration structures geometry
    VkDeviceSize maxScratchSize{0};
    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      auto vertexAddress = m_bMeshes[p_idx].vertices.address;
      auto indexAddress  = m_bMeshes[p_idx].indices.address;

      blasData[p_idx].asType = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      blasData[p_idx].addGeometry(primitiveToGeometry(m_meshes[p_idx], vertexAddress, indexAddress));
      VkAccelerationStructureBuildSizesInfoKHR sizeInfo =
          blasData[p_idx].finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
      maxScratchSize = std::max(maxScratchSize, sizeInfo.buildScratchSize);
    }

    // Scratch buffer for all BLAS
    nvvk::Buffer scratchBuffer;
    NVVK_CHECK(m_allocator.createBuffer(scratchBuffer, maxScratchSize,
                                        VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                        VMA_MEMORY_USAGE_AUTO, {}, m_asProperties.minAccelerationStructureScratchOffsetAlignment));
    NVVK_DBG_NAME(scratchBuffer.buffer);

    // Create and build all BLAS
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    for(size_t i = 0; i < m_bottomAs.size(); i++)
    {
      VkAccelerationStructureCreateInfoKHR createInfo = blasData[i].makeCreateInfo();
      NVVK_CHECK(m_allocator.createAcceleration(m_bottomAs[i], createInfo));
      NVVK_DBG_NAME(m_bottomAs[i].accel);

      blasData[i].cmdBuildAccelerationStructure(cmd, m_bottomAs[i].accel, scratchBuffer.address);
      // Because we will be reusing the scratch buffer, we need a barrier to ensure the BLAS is finished before next build
      nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                         VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    }
    m_app->submitAndWaitTempCmdBuffer(cmd);

    m_allocator.destroyBuffer(scratchBuffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    // NVTX3_FUNC_RANGE();

    nvvk::AccelerationStructureBuildData tlasData;
    tlasData.asType = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(m_nodes.size());

    for(auto& node : m_nodes)
    {
      VkAccelerationStructureInstanceKHR rayInst{.transform = nvvk::toTransformMatrixKHR(node.localMatrix()),  // Position of the instance
                                                 .instanceCustomIndex = static_cast<uint32_t>(node.mesh),  // gl_InstanceCustomIndexEX
                                                 .mask  = 0xFF,
                                                 .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV,
                                                 .accelerationStructureReference = m_bottomAs[node.mesh].address};
      instances.emplace_back(rayInst);
    }

    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_allocator);

    // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
    nvvk::Buffer instancesBuffer;
    NVVK_CHECK(m_allocator.createBuffer(instancesBuffer, std::span(instances).size_bytes(),
                                        VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                            | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
    NVVK_CHECK(uploader.appendBuffer(instancesBuffer, 0, std::span(instances)));
    NVVK_DBG_NAME(instancesBuffer.buffer);

    uploader.cmdUploadAppended(cmd);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

    auto instGeo = tlasData.makeInstanceGeometry(instances.size(), instancesBuffer.address);
    tlasData.addGeometry(instGeo);

    // Create the TLAS
    auto sizeInfo = tlasData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    nvvk::Buffer scratchBuffer;
    NVVK_CHECK(m_allocator.createBuffer(scratchBuffer, sizeInfo.buildScratchSize,
                                        VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                        VMA_MEMORY_USAGE_AUTO, {}, m_asProperties.minAccelerationStructureScratchOffsetAlignment));
    NVVK_DBG_NAME(scratchBuffer.buffer);
    VkAccelerationStructureCreateInfoKHR createInfo = tlasData.makeCreateInfo();
    NVVK_CHECK(m_allocator.createAcceleration(m_topAs, createInfo));
    NVVK_DBG_NAME(m_topAs.accel);

    tlasData.cmdBuildAccelerationStructure(cmd, m_topAs.accel, scratchBuffer.address);
    uploader.cmdUploadAppended(cmd);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();

    m_allocator.destroyBuffer(instancesBuffer);
    m_allocator.destroyBuffer(scratchBuffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Creating the pipeline: shader ...
  //
  void createCompPipelines()
  {
    // This descriptor set, holds the top level acceleration structure and the output image
    m_descriptorBinding.addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_descriptorBinding.addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_descriptorBinding.addBinding(B_cameraInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_descriptorBinding.addBinding(B_sceneDesc, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);

    NVVK_CHECK(m_descriptorBinding.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                                             &m_descriptorSetLayout));
    NVVK_DBG_NAME(m_descriptorSetLayout);


    // pushing time
    const VkPushConstantRange pushConstant{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(shaderio::PushConstant)};
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorSetLayout}, {pushConstant}));

#if USE_SLANG
    VkShaderModuleCreateInfo moduleInfo = nvsamples::getShaderModuleCreateInfo(ray_query_slang);
#else
    VkShaderModuleCreateInfo moduleInfo = nvsamples::getShaderModuleCreateInfo(ray_query_comp_glsl);
#endif

    VkPipelineShaderStageCreateInfo shaderStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = &moduleInfo,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .pName = "main",
    };

    VkComputePipelineCreateInfo cpCreateInfo{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = shaderStage,
        .layout = m_pipelineLayout,
    };

    NVVK_CHECK(vkCreateComputePipelines(m_device, {}, 1, &cpCreateInfo, nullptr, &m_pipeline));
    NVVK_DBG_NAME(m_pipeline);
  }


  void pushDescriptorSet(VkCommandBuffer cmd)
  {
    // Write to descriptors
    nvvk::WriteSetContainer writes{};
    writes.append(m_descriptorBinding.getWriteSet(B_tlas), m_topAs);
    writes.append(m_descriptorBinding.getWriteSet(B_outImage), m_gBuffers.getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL);
    writes.append(m_descriptorBinding.getWriteSet(B_cameraInfo), m_bCameraInfo);
    writes.append(m_descriptorBinding.getWriteSet(B_sceneDesc), m_bSceneDesc);

    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,
                              static_cast<uint32_t>(writes.size()), writes.data());
  }


  //--------------------------------------------------------------------------------------------------
  // To be call when renderer need to re-start
  //
  void resetFrame() { m_frame = -1; }

  //--------------------------------------------------------------------------------------------------
  // If the camera matrix has changed, resets the frame.
  // otherwise, increments frame.
  //
  bool updateFrame()
  {
    static float     ref_fov{0};
    static glm::mat4 ref_cam_matrix;

    const auto& m   = g_cameraManip->getViewMatrix();
    const auto  fov = g_cameraManip->getFov();

    if(ref_cam_matrix != m || ref_fov != fov)
    {
      resetFrame();
      ref_cam_matrix = m;
      ref_fov        = fov;
    }

    if(m_frame >= m_maxFrames)
    {
      return false;
    }
    m_frame++;
    return true;
  }


  void destroyResources()
  {
    for(auto& m : m_bMeshes)
    {
      m_allocator.destroyBuffer(m.vertices);
      m_allocator.destroyBuffer(m.indices);
    }
    m_allocator.destroyBuffer(m_bCameraInfo);
    m_allocator.destroyBuffer(m_bPrimInfo);
    m_allocator.destroyBuffer(m_bSceneDesc);
    m_allocator.destroyBuffer(m_bInstInfoBuffer);
    m_allocator.destroyBuffer(m_bMaterials);

    m_gBuffers.deinit();

    for(auto& b : m_bottomAs)
    {
      m_allocator.destroyAcceleration(b);
    }
    m_allocator.destroyAcceleration(m_topAs);


    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);


    m_samplerPool.deinit();
    m_tonemapper.deinit();
    m_allocator.deinit();
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers.getColorImage(), m_gBuffers.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvapp::Application*      m_app{};
  nvvk::ResourceAllocator  m_allocator;
  nvshaders::Tonemapper    m_tonemapper{};
  shaderio::TonemapperData m_tonemapperData;
  nvvk::GBuffer            m_gBuffers;  // G-Buffers: color + depth
  nvvk::SamplerPool        m_samplerPool{};

  VkDevice m_device{};  // Vulkan device

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
    nvvk::Buffer indices;   // Buffer of the indices
  };
  std::vector<PrimitiveMeshVk> m_bMeshes;
  nvvk::Buffer                 m_bCameraInfo;      // Camera information
  nvvk::Buffer                 m_bPrimInfo;        // Buffer of all PrimitiveMeshVk
  nvvk::Buffer                 m_bSceneDesc;       // Scene description with pointers to the buffers
  nvvk::Buffer                 m_bInstInfoBuffer;  // Transformation and material per instance
  nvvk::Buffer                 m_bMaterials;       // All materials

  // Data and setting
  std::vector<nvutils::PrimitiveMesh> m_meshes;
  std::vector<nvutils::Node>          m_nodes;
  std::vector<shaderio::Material>     m_materials;
  shaderio::Light                     m_light = {};

  // Pipeline
  shaderio::PushConstant m_pushConst{};  // Information sent to the shader
  int                    m_frame{0};
  int                    m_maxFrames{10000};

  // Pipeline
  nvvk::DescriptorBindings m_descriptorBinding{};
  VkDescriptorSetLayout    m_descriptorSetLayout{};
  VkPipeline               m_pipeline{};
  VkPipelineLayout         m_pipelineLayout{};

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_asProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

  // Acceleration Structures
  std::vector<nvvk::AccelerationStructure> m_bottomAs;
  nvvk::AccelerationStructure              m_topAs;
};

//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char** argv) -> int
{
  nvapp::ApplicationCreateInfo appInfo;

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless"}, &appInfo.headless, true);
  reg.add({"frames", "Number of frames to run in headless mode"}, &appInfo.headlessFrameCount);
  cli.add(reg);
  cli.parse(argc, argv);


  // #VKRay: Activate the ray tracing extension
  // Extension feature needed.
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
                             {VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME},
                             {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME},
                             {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature},
                             {VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature},
                             {VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjFeature},
                             {VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature}},
  };
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  // Create the Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  // Application settings
  appInfo.name           = fmt::format("{} ({})", nvutils::getExecutablePath().stem().string(), SHADER_LANGUAGE_STR);
  appInfo.vSync          = false;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();


  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  // Add all application elements
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  elemCamera->setCameraManipulator(g_cameraManip);
  app.addElement(elemCamera);
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());                         // Menu / Quit
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", appInfo.name));  // Window title info
  app.addElement(std::make_shared<RayQuery>());


  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
