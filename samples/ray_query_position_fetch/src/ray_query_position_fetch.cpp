/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

//////////////////////////////////////////////////////////////////////////
/*

    This shows the use of Ray Query, or casting rays in a compute shader

*/
//////////////////////////////////////////////////////////////////////////


#define VMA_IMPLEMENTATION
#include "nvh/primitives.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"

#include "shaders/dh_bindings.h"

//#if USE_HLSL
//const auto& comp_shd = std::vector<char>{std::begin(ray_query_computeMain), std::end(ray_query_computeMain)};
//#else
#include "_autogen/ray_query.comp.h"
const auto& comp_shd = std::vector<uint32_t>{std::begin(ray_query_comp), std::end(ray_query_comp)};
//#endif

/// </summary> Fetching position in ray tracing ray query
class RayQueryFetch : public nvvkhl::IAppElement
{
  enum
  {
    eImgTonemapped,
    eImgRendered
  };

public:
  RayQueryFetch()           = default;
  ~RayQueryFetch() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    m_app    = app;
    m_device = m_app->getDevice();

    m_dutil      = std::make_unique<nvvk::DebugUtil>(m_device);                    // Debug utility
    m_alloc      = std::make_unique<nvvkhl::AllocVma>(m_app->getContext().get());  // Allocator
    m_rtSet      = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_tonemapper = std::make_unique<nvvkhl::TonemapperPostProcess>(m_app->getContext().get(), m_alloc.get());

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    int32_t computeQueueIndex = m_app->getContext()->m_queueC.familyIndex;
    m_rtBuilder.setup(m_device, m_alloc.get(), computeQueueIndex);
    m_sbt.setup(m_device, computeQueueIndex, m_alloc.get(), m_rtProperties);

    // Create resources
    createScene();
    createVkBuffers();
    createBottomLevelAS();
    createTopLevelAS();
    createCompPipelines();

    m_tonemapper->setSettings(
        {.isActive = 1, .exposure = 2.053f, .brightness = 1.119f, .contrast = 1.194f, .saturation = 1.478f, .vignette = 0.276f});
    m_tonemapper->createComputePipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    createGbuffers({width, height});
    m_tonemapper->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(eImgRendered),
                                              m_gBuffers->getDescriptorImageInfo(eImgTonemapped));
  }

  void onUIRender() override
  {
    animateTopLevelAS();  // <--- Animating the rotation at each frame

    {  // Setting menu
      ImGui::Begin("Settings");
      if(ImGui::CollapsingHeader("Tonemapper"))
        m_tonemapper->onUI();
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_gBuffers->getDescriptorSet(eImgTonemapped), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    auto sdbg = m_dutil->DBG_SCOPE(cmd);

    VkMemoryBarrier memBarrier = {
        .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memBarrier,
                         0, nullptr, 0, nullptr);

    // Ray trace
    std::vector<VkDescriptorSet> descSets{m_rtSet->getSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipe.plines[0]);
    pushDescriptorSet(cmd);

    const VkExtent2D& size = m_app->getViewportSize();
    vkCmdDispatch(cmd, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);

    // Making sure the rendered image is ready to be used
    VkImageMemoryBarrier image_memory_barrier =
        nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(eImgRendered), VK_ACCESS_SHADER_READ_BIT,
                                     VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &image_memory_barrier);

    // Apply tonemapper
    m_tonemapper->runCompute(cmd, size);
  }

private:
  // Creating the scene, a sphere with wobbled vertices.
  void createScene()
  {
    nvh::PrimitiveMesh mesh = nvh::createSphereMesh(1.0F, 2);
    mesh                    = nvh::removeDuplicateVertices(mesh, false, false);
    m_meshes.emplace_back(wobblePrimitive(mesh, 0.1F));

    // Instance Ball
    nvh::Node n{.mesh = 0};
    m_nodes.push_back(n);
  }

  // Creating 2 G-Buffers, the result RGBA32F and the tonemapped
  void createGbuffers(const VkExtent2D& size)
  {
    // Rendering image targets
    std::vector<VkFormat> color_buffers = {m_colorFormat, m_colorFormat};  // tonemapped, original
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), size, color_buffers, m_depthFormat);
  }

  // Create all Vulkan buffer data: vertices and indices of the scene
  void createVkBuffers()
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_bMeshes.resize(m_meshes.size());

    auto rtUsageFlag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex and Index per mesh
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      auto& m    = m_bMeshes[i];
      m.vertices = m_alloc->createBuffer(cmd, m_meshes[i].vertices, rtUsageFlag);
      m.indices  = m_alloc->createBuffer(cmd, m_meshes[i].triangles, rtUsageFlag);
      m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
      m_dutil->DBG_NAME_IDX(m.indices.buffer, i);
    }

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }


  //--------------------------------------------------------------------------------------------------
  // Converting a PrimitiveMesh as input for BLAS
  //
  nvvk::RaytracingBuilderKHR::BlasInput primitiveToGeometry(const nvh::PrimitiveMesh& prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress)
  {
    uint32_t maxPrimitiveCount = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .vertexFormat = VK_FORMAT_R32G32B32A32_SFLOAT,  // vec3 vertex position data.
        .vertexData   = {.deviceAddress = vertexAddress},
        .vertexStride = sizeof(nvh::PrimitiveVertex),
        .maxVertex    = static_cast<uint32_t>(prim.vertices.size()),
        .indexType    = VK_INDEX_TYPE_UINT32,
        .indexData    = {.deviceAddress = indexAddress},
    };

    // Identify the above data as containing opaque triangles.
    VkAccelerationStructureGeometryKHR asGeom{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry     = {.triangles = triangles},
        .flags        = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR,
    };

    VkAccelerationStructureBuildRangeInfoKHR offset{
        .primitiveCount  = maxPrimitiveCount,
        .primitiveOffset = 0,
        .firstVertex     = 0,
        .transformOffset = 0,
    };

    // Our BLAS is made from only one geometry, but could be made of many geometries
    nvvk::RaytracingBuilderKHR::BlasInput input;
    input.asGeometry.emplace_back(asGeom);
    input.asBuildOffsetInfo.emplace_back(offset);

    return input;
  }

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  void createBottomLevelAS()
  {
    // BLAS - Storing each primitive in a geometry
    std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
    allBlas.reserve(m_meshes.size());

    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      auto vertexAddress = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[p_idx].vertices.buffer);
      auto indexAddress  = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[p_idx].indices.buffer);

      auto geo = primitiveToGeometry(m_meshes[p_idx], vertexAddress, indexAddress);
      allBlas.push_back({geo});
    }
    // #FETCH
    m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                       | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    m_tlas.reserve(m_nodes.size());
    for(auto& node : m_nodes)
    {
      VkGeometryInstanceFlagsKHR flags{VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV};

      VkAccelerationStructureInstanceKHR rayInst{
          .transform           = nvvk::toTransformMatrixKHR(node.localMatrix()),  // Position of the instance
          .instanceCustomIndex = static_cast<uint32_t>(node.mesh),                // gl_InstanceCustomIndexEX
          .mask                = 0xFF,                                            // Allow everything
          .instanceShaderBindingTableRecordOffset = 0,  // We will use the same hit group for all object
          .flags                                  = flags,
          .accelerationStructureReference         = m_rtBuilder.getBlasDeviceAddress(node.mesh),
      };
      m_tlas.emplace_back(rayInst);
    }
    m_rtBuilder.buildTlas(m_tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                      | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);
  }

  //--------------------------------------------------------------------------------------------------
  // At each frame, the instance is rotated
  void animateTopLevelAS()
  {
    const float s_deg       = 30.0f;
    float       currentTime = ImGui::GetTime();

    for(auto& node : m_nodes)
    {
      // Calculate rotation
      float                     angle = s_deg * currentTime;
      nvmath::quaternion<float> rotation =
          nvmath::axis_to_quat(nvmath::vec3f(std::sin(currentTime), std::cos(currentTime), 0.0F), deg2rad(angle));
      rotation.normalize();

      // Update node rotation and transform matrix
      node.rotation       = rotation;
      m_tlas[0].transform = nvvk::toTransformMatrixKHR(node.localMatrix());
    }
    // Updating the top level acceleration structure
    m_rtBuilder.buildTlas(m_tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
                          true);
  }

  //--------------------------------------------------------------------------------------------------
  // Creating the pipeline: shader ...
  //
  void createCompPipelines()
  {
    m_rtPipe.destroy(m_device);
    m_rtSet->deinit();
    m_rtSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_rtPipe.plines.resize(1);

    // This descriptor set, holds the top level acceleration structure and the output image
    m_rtSet->addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

    // pushing time
    VkPipelineLayoutCreateInfo plCreateInfo{
        .sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1U,
        .pSetLayouts    = &m_rtSet->getLayout(),
    };
    vkCreatePipelineLayout(m_device, &plCreateInfo, nullptr, &m_rtPipe.layout);

    VkComputePipelineCreateInfo cpCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = nvvk::createShaderStageInfo(m_device, comp_shd, VK_SHADER_STAGE_COMPUTE_BIT, USE_HLSL ? "computeMain" : "main"),
        .layout = m_rtPipe.layout,
    };

    vkCreateComputePipelines(m_device, {}, 1, &cpCreateInfo, nullptr, &m_rtPipe.plines[0]);

    vkDestroyShaderModule(m_device, cpCreateInfo.stage.module, nullptr);
  }


  void pushDescriptorSet(VkCommandBuffer cmd)
  {
    // Write to descriptors
    VkAccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure();
    VkWriteDescriptorSetAccelerationStructureKHR descASInfo{
        .sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
        .accelerationStructureCount = 1,
        .pAccelerationStructures    = &tlas,
    };
    VkDescriptorImageInfo imageInfo{
        .imageView   = m_gBuffers->getColorImageView(eImgRendered),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtSet->makeWrite(0, B_tlas, &descASInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, B_outImage, &imageInfo));

    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipe.layout, 0,
                              static_cast<uint32_t>(writes.size()), writes.data());
  }


  void destroyResources()
  {
    for(auto& m : m_bMeshes)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }

    m_rtSet->deinit();
    m_gBuffers.reset();

    m_rtPipe.destroy(m_device);

    m_sbt.destroy();
    m_rtBuilder.destroy();
    m_tonemapper.reset();
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*                           m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>               m_dutil;
  std::unique_ptr<nvvkhl::AllocVma>              m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer>  m_rtSet;  // Descriptor set
  std::unique_ptr<nvvkhl::TonemapperPostProcess> m_tonemapper;

  VkFormat                         m_colorFormat = VK_FORMAT_R32G32B32A32_SFLOAT;  // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
    nvvk::Buffer indices;   // Buffer of the indices
  };
  std::vector<PrimitiveMeshVk>                    m_bMeshes;
  std::vector<VkAccelerationStructureInstanceKHR> m_tlas;  // Keeping for animation

  // Data and setting
  std::vector<nvh::PrimitiveMesh> m_meshes;
  std::vector<nvh::Node>          m_nodes;

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::SBTWrapper           m_sbt;  // Shading binding table wrapper
  nvvk::RaytracingBuilderKHR m_rtBuilder;
  nvvkhl::PipelineContainer  m_rtPipe;
};

//////////////////////////////////////////////////////////////////////////
///
///
auto main(int argc, char** argv) -> int
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name    = PROJECT_NAME " Example";
  spec.vSync   = true;
  spec.vkSetup = nvvk::ContextCreateInfo(false);  // Removing validation layer in debug
  spec.vkSetup.setVersion(1, 3);

  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accelFeature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rtPipelineFeature);  // To use vkCmdTraceRaysKHR
  spec.vkSetup.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
  spec.vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayqueryFeature);
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR fetchFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, false, &fetchFeatures);  // #FETCH

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());         // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>());  // Window title info
  app->addElement(std::make_shared<RayQueryFetch>());

  app->run();
  app.reset();

  return test->errorCode();
}
