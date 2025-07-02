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

#include <glm/glm.hpp>

namespace shaderio {
using namespace glm;
#include "shaders/dh_bindings.h"
}  // namespace shaderio


#include "_autogen/ray_query_pos_fetch.comp.glsl.h"
#include "_autogen/ray_query_pos_fetch.slang.h"
#include "_autogen/tonemapper.slang.h"


#include <nvapp/application.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvgui/tonemapper.hpp>
#include <nvshaders_host/tonemapper.hpp>
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
#include <nvvk/helpers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>
#include "common/utils.hpp"

/// </summary> Fetching position in ray tracing ray query
class RayQueryFetch : public nvapp::IAppElement
{
  enum
  {
    eImgTonemapped,
    eImgRendered
  };

public:
  RayQueryFetch()           = default;
  ~RayQueryFetch() override = default;

  void onAttach(nvapp::Application* app) override
  {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    m_app    = app;
    m_device = m_app->getDevice();

    m_allocator.init({
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    });  // Allocator

    m_uploader.init(&m_allocator, true);


    // The texture sampler to use
    m_samplerPool.init(m_device);
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);


    m_gBuffers.init({.allocator      = &m_allocator,
                     .colorFormats   = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32G32B32A32_SFLOAT},
                     .imageSampler   = linearSampler,
                     .descriptorPool = m_app->getTextureDescriptorPool()});

    // Requesting ray tracing properties
    m_rtProperties.pNext = &m_asProperties;
    VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m_rtProperties};
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    int32_t computeQueueIndex = m_app->getQueue(1).familyIndex;

    // Create resources
    createScene();
    createVkBuffers();
    createBottomLevelAS();
    createTopLevelAS();
    createCompPipelines();

    // Tonemapper
    {
      auto code = std::span<const uint32_t>(tonemapper_slang);
      m_tonemapper.init(&m_allocator, code);
    }
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override { m_gBuffers.update(cmd, size); }

  void onUIRender() override
  {

    {  // Setting menu
      ImGui::Begin("Settings");
      if(ImGui::CollapsingHeader("Tonemapper"))
        nvgui::tonemapperWidget(m_tonemapperData);
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(eImgTonemapped), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);

    animateTopLevelAS(cmd);  // <--- Animating the rotation at each frame

    VkMemoryBarrier memBarrier = {
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier,
                         0, nullptr, 0, nullptr);

    // Ray trace
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    pushDescriptorSet(cmd);

    const VkExtent2D& size = m_app->getViewportSize();
    vkCmdDispatch(cmd, (size.width + (WORKGROUP_SIZE - 1)) / WORKGROUP_SIZE,
                  (size.height + (WORKGROUP_SIZE - 1)) / WORKGROUP_SIZE, 1);

    // Making sure the rendered image is ready to be used
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Apply tonemapper
    m_tonemapper.runCompute(cmd, m_gBuffers.getSize(), m_tonemapperData, m_gBuffers.getDescriptorImageInfo(eImgRendered),
                            m_gBuffers.getDescriptorImageInfo(eImgTonemapped));
  }

private:
  // Creating the scene, a sphere with wobbled vertices.
  void createScene()
  {
    nvutils::PrimitiveMesh mesh = nvutils::createSphereMesh(1.0F, 2);
    mesh                        = nvutils::removeDuplicateVertices(mesh, false, false);
    m_meshes.emplace_back(wobblePrimitive(mesh, 0.1F));

    // Instance Ball
    nvutils::Node n{.mesh = 0};
    m_nodes.push_back(n);
  }


  // Create all Vulkan buffer data: vertices and indices of the scene
  void createVkBuffers()
  {
    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_allocator, true);


    m_bMeshes.resize(m_meshes.size());

    auto rtUsageFlag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex and Index per mesh
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      auto& m = m_bMeshes[i];

      NVVK_CHECK(m_allocator.createBuffer(m.vertices, std::span(m_meshes[i].vertices).size_bytes(), rtUsageFlag));
      NVVK_CHECK(m_allocator.createBuffer(m.indices, std::span(m_meshes[i].triangles).size_bytes(), rtUsageFlag));
      NVVK_CHECK(uploader.appendBuffer(m.vertices, 0, std::span(m_meshes[i].vertices)));
      NVVK_CHECK(uploader.appendBuffer(m.indices, 0, std::span(m_meshes[i].triangles)));
      NVVK_DBG_NAME(m.vertices.buffer);
      NVVK_DBG_NAME(m.indices.buffer);
    }

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
    uint32_t triangleCount = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,  // vec3 vertex position data.
        .vertexData   = {.deviceAddress = vertexAddress},
        .vertexStride = sizeof(nvutils::PrimitiveVertex),
        .maxVertex    = static_cast<uint32_t>(prim.vertices.size()),
        .indexType    = VK_INDEX_TYPE_UINT32,
        .indexData    = {.deviceAddress = indexAddress},
    };

    nvvk::AccelerationStructureGeometryInfo result;
    // Identify the above data as containing opaque triangles.
    result.geometry = VkAccelerationStructureGeometryKHR{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry     = {.triangles = triangles},
        .flags        = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | VK_GEOMETRY_OPAQUE_BIT_KHR,
    };

    result.rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = triangleCount};

    return result;
  }

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  void createBottomLevelAS()
  {
    // Initialize BLAS build data and resize BLAS array
    std::vector<nvvk::AccelerationStructureBuildData> blasBuildData(m_meshes.size());
    m_blas.resize(m_meshes.size());

    // Create all BLAS Build information and determine maximum scratch space needed for BLAS construction
    VkDeviceSize maxScratchSize{0};
    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      auto& mesh = m_bMeshes[p_idx];
      auto  geo  = primitiveToGeometry(m_meshes[p_idx], mesh.vertices.address, mesh.indices.address);

      blasBuildData[p_idx].asType = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      blasBuildData[p_idx].addGeometry(geo);
      VkAccelerationStructureBuildSizesInfoKHR sizeInfo =
          blasBuildData[p_idx].finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                              | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR);
      maxScratchSize = std::max(maxScratchSize, sizeInfo.buildScratchSize);
    }

    // Create scratch buffer for building BLAS
    // Scratch buffer for all BLAS
    nvvk::Buffer scratchBuffer;
    NVVK_CHECK(m_allocator.createBuffer(scratchBuffer, maxScratchSize,
                                        VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                        VMA_MEMORY_USAGE_AUTO, {}, m_asProperties.minAccelerationStructureScratchOffsetAlignment));
    NVVK_DBG_NAME(scratchBuffer.buffer);

    // Command buffer for BLAS construction commands
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      auto createInfo = blasBuildData[p_idx].makeCreateInfo();
      NVVK_CHECK(m_allocator.createAcceleration(m_blas[p_idx], createInfo));
      NVVK_DBG_NAME(m_blas[p_idx].accel);
      blasBuildData[p_idx].cmdBuildAccelerationStructure(cmd, m_blas[p_idx].accel, scratchBuffer.address);
    }

    // Submit command buffer and wait for completion
    m_app->submitAndWaitTempCmdBuffer(cmd);

    // Clean up scratch buffer after use
    m_allocator.destroyBuffer(scratchBuffer);
  }


  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    // Reserve space for instances equal to the number of nodes
    m_tlasInstances.reserve(m_nodes.size());

    // Create instances from node data
    for(auto& node : m_nodes)
    {
      VkGeometryInstanceFlagsKHR flags{VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV};

      VkAccelerationStructureInstanceKHR rayInst{
          .transform           = nvvk::toTransformMatrixKHR(node.localMatrix()),  // Position of the instance
          .instanceCustomIndex = static_cast<uint32_t>(node.mesh),                // gl_InstanceCustomIndexEXT
          .mask                = 0xFF,                                            // Mask to allow all ray types
          .instanceShaderBindingTableRecordOffset = 0,                            // Uniform hit group for all objects
          .flags                                  = flags,
          .accelerationStructureReference         = m_blas[node.mesh].address,  // Reference to BLAS
      };
      m_tlasInstances.emplace_back(rayInst);
    }

    // Create a command buffer for temporary commands
    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_allocator);


    // Create and fill the instance buffer
    NVVK_CHECK(m_allocator.createBuffer(m_instancesBuffer, std::span(m_tlasInstances).size_bytes(),
                                        VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                            | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
    NVVK_CHECK(uploader.appendBuffer(m_instancesBuffer, 0, std::span(m_tlasInstances)));

    // Upload the instance buffer to the GPU
    uploader.cmdUploadAppended(cmd);

    // Ensure buffer is ready for acceleration structure operations
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

    // Prepare for TLAS building
    auto geo = m_tlasBuildData.makeInstanceGeometry(m_tlasInstances.size(), m_instancesBuffer.address);
    m_tlasBuildData.addGeometry(geo);
    auto sizeInfo = m_tlasBuildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                                   | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);

    // Create scratch buffer for TLAS
    VkDeviceSize scratchSize = sizeInfo.buildScratchSize;  // Build size (larger) instead of update size
    NVVK_CHECK(m_allocator.createBuffer(m_tlasScratchBuffer, sizeInfo.buildScratchSize,
                                        VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT,
                                        VMA_MEMORY_USAGE_AUTO, {}, m_asProperties.minAccelerationStructureScratchOffsetAlignment));
    NVVK_DBG_NAME(m_tlasScratchBuffer.buffer);

    // Create the TLAS
    auto createInfo = m_tlasBuildData.makeCreateInfo();
    m_allocator.createAcceleration(m_tlas, createInfo);
    m_tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, m_tlasScratchBuffer.address);

    // Submit the command buffer and wait for execution to complete
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();
  }


  //--------------------------------------------------------------------------------------------------
  // At each frame, the instance is rotated
  void animateTopLevelAS(VkCommandBuffer cmd)
  {
    const float s_deg       = 30.0f;
    float       currentTime = static_cast<float>(ImGui::GetTime());

    uint32_t idx = 0;
    for(auto& node : m_nodes)
    {
      // Calculate rotation
      float angle = s_deg * currentTime;
      glm::quat rotation = glm::angleAxis(glm::radians(angle), glm::vec3(std::sin(currentTime), std::cos(currentTime), 0.0F));

      // Update node rotation and transform matrix
      node.rotation                    = rotation;
      m_tlasInstances[idx++].transform = nvvk::toTransformMatrixKHR(node.localMatrix());
    }

    m_uploader.appendBuffer(m_instancesBuffer, 0, std::span(m_tlasInstances));
    m_uploader.cmdUploadAppended(cmd);

    // Make sure the copy of the instance buffer are copied before triggering the acceleration structure build
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

    // Updating the buffer address in the instance
    m_tlasBuildData.asGeometry[0].geometry.instances.data.deviceAddress = m_instancesBuffer.address;
    m_tlasBuildData.cmdUpdateAccelerationStructure(cmd, m_tlas.accel, m_tlasScratchBuffer.address);
  }

  //--------------------------------------------------------------------------------------------------
  // Creating the pipeline: shader ...
  //
  void createCompPipelines()
  {
    // This descriptor set, holds the top level acceleration structure and the output image
    m_descriptorBinding.addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_descriptorBinding.addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    NVVK_CHECK(m_descriptorBinding.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                                             &m_descriptorSetLayout));
    NVVK_DBG_NAME(m_descriptorSetLayout);

    // pushing time
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorSetLayout}));

    VkShaderModuleCreateInfo moduleInfo = nvsamples::getShaderModuleCreateInfo(ray_query_pos_fetch_slang);

    VkPipelineShaderStageCreateInfo shaderStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = &moduleInfo,
        .stage = VK_SHADER_STAGE_COMPUTE_BIT,
        .pName = "computeMain",
    };

    VkComputePipelineCreateInfo cpCreateInfo{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = shaderStage,
        .layout = m_pipelineLayout,
    };

    vkCreateComputePipelines(m_device, {}, 1, &cpCreateInfo, nullptr, &m_pipeline);
  }


  void pushDescriptorSet(VkCommandBuffer cmd)
  {
    // Write to descriptors
    VkAccelerationStructureKHR                   tlas = m_tlas.accel;
    VkWriteDescriptorSetAccelerationStructureKHR descASInfo{
        .sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures    = &tlas,
    };
    VkDescriptorImageInfo imageInfo{
        .imageView   = m_gBuffers.getColorImageView(eImgRendered),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };

    // Write to descriptors
    nvvk::WriteSetContainer writes{};
    writes.append(m_descriptorBinding.getWriteSet(B_tlas), m_tlas);
    writes.append(m_descriptorBinding.getWriteSet(B_outImage), m_gBuffers.getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL);

    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,
                              static_cast<uint32_t>(writes.size()), writes.data());
  }


  void destroyResources()
  {
    for(auto& m : m_bMeshes)
    {
      m_allocator.destroyBuffer(m.vertices);
      m_allocator.destroyBuffer(m.indices);
    }

    for(auto& blas : m_blas)
    {
      m_allocator.destroyAcceleration(blas);
    }

    m_blas.clear();
    m_allocator.destroyAcceleration(m_tlas);
    m_allocator.destroyBuffer(m_instancesBuffer);
    m_allocator.destroyBuffer(m_tlasScratchBuffer);

    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);


    m_samplerPool.deinit();
    m_uploader.deinit();
    m_gBuffers.deinit();
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
  nvapp::Application*      m_app{nullptr};
  nvvk::ResourceAllocator  m_allocator;
  nvshaders::Tonemapper    m_tonemapper{};
  shaderio::TonemapperData m_tonemapperData;

  VkFormat              m_colorFormat = VK_FORMAT_R32G32B32A32_SFLOAT;  // Color format of the image
  VkFormat              m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkDevice              m_device      = VK_NULL_HANDLE;                 // Convenient
  nvvk::GBuffer         m_gBuffers;                                     // G-Buffers: color + depth
  nvvk::SamplerPool     m_samplerPool{};
  nvvk::StagingUploader m_uploader;

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
    nvvk::Buffer indices;   // Buffer of the indices
  };

  // Vulkan Ray Tracing Properties
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_asProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

  // Pipeline
  nvvk::DescriptorBindings m_descriptorBinding{};
  VkDescriptorSetLayout    m_descriptorSetLayout{};
  VkPipeline               m_pipeline{};
  VkPipelineLayout         m_pipelineLayout{};

  // Acceleration Structures
  std::vector<VkAccelerationStructureInstanceKHR> m_tlasInstances;  // Keeping for animation
  nvvk::AccelerationStructure                     m_tlas;
  nvvk::AccelerationStructureBuildData            m_tlasBuildData{VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
  std::vector<nvvk::AccelerationStructure>        m_blas;

  // Buffers for instances and scratch data
  nvvk::Buffer m_instancesBuffer;
  nvvk::Buffer m_tlasScratchBuffer;

  // Mesh data
  std::vector<PrimitiveMeshVk>        m_bMeshes;
  std::vector<nvutils::PrimitiveMesh> m_meshes;

  // Node and scene data
  std::vector<nvutils::Node> m_nodes;
};

//////////////////////////////////////////////////////////////////////////
///
///
auto main(int argc, char** argv) -> int
{
  nvapp::ApplicationCreateInfo appInfo;

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless"}, &appInfo.headless, true);
  cli.add(reg);
  cli.parse(argc, argv);


  // Specific extension features
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR fetchFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};

  // Config for Vulkan context creation
  nvvk::ContextInitInfo vkSetup;
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature});  // To build acceleration structures
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature});  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});  // Required by ray tracing pipeline
  vkSetup.deviceExtensions.push_back({VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, &fetchFeatures});
  vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjFeature});

  vkSetup.queues.push_back(VK_QUEUE_COMPUTE_BIT);  // Using other queue for SBT creation

  // Create the Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }


  // Set the information for the application
  appInfo.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  nvapp::Application app;
  app.init(appInfo);


  // Add all application elements
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());  // Menu / Quit
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app.addElement(std::make_shared<RayQueryFetch>());

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
