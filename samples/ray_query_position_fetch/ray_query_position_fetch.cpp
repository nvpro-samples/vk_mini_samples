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


#define VMA_IMPLEMENTATION
#include "nvh/primitives.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"
#include "nvvk/acceleration_structures.hpp"

#include "shaders/dh_bindings.h"
#include "common/vk_context.hpp"
#include "nvvk/extensions_vk.hpp"

#if USE_SLANG
#include "_autogen/ray_query_slang.h"
const auto& comp_shd = std::vector<uint32_t>{std::begin(ray_querySlang), std::end(ray_querySlang)};
#else
#include "_autogen/ray_query.comp.glsl.h"
const auto& comp_shd = std::vector<uint32_t>{std::begin(ray_query_comp_glsl), std::end(ray_query_comp_glsl)};
#endif

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

    m_dutil      = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility
    m_alloc      = std::make_unique<nvvkhl::AllocVma>(VmaAllocatorCreateInfo{
             .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
             .physicalDevice = app->getPhysicalDevice(),
             .device         = app->getDevice(),
             .instance       = app->getInstance(),
    });  // Allocator
    m_rtSet      = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_tonemapper = std::make_unique<nvvkhl::TonemapperPostProcess>(m_device, m_alloc.get());

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &m_rtProperties};
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    int32_t computeQueueIndex = m_app->getQueue(1).familyIndex;
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

    animateTopLevelAS(cmd);  // <--- Animating the rotation at each frame

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
  nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const nvh::PrimitiveMesh& prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress)
  {
    uint32_t triangleCount = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,  // vec3 vertex position data.
        .vertexData   = {.deviceAddress = vertexAddress},
        .vertexStride = sizeof(nvh::PrimitiveVertex),
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
    VkDeviceSize maxScratch{0};
    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      auto& mesh = m_bMeshes[p_idx];
      auto  geo  = primitiveToGeometry(m_meshes[p_idx], mesh.vertices.address, mesh.indices.address);

      blasBuildData[p_idx].asType = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      blasBuildData[p_idx].addGeometry(geo);
      auto sizeInfo = blasBuildData[p_idx].finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                                          | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR);
      maxScratch = std::max(maxScratch, sizeInfo.buildScratchSize);
    }

    // Create scratch buffer for building BLAS
    nvvk::Buffer scratchBuffer =
        m_alloc->createBuffer(maxScratch, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    // Command buffer for BLAS construction commands
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      auto createInfo = blasBuildData[p_idx].makeCreateInfo();
      m_blas[p_idx]   = m_alloc->createAcceleration(createInfo);
      blasBuildData[p_idx].cmdBuildAccelerationStructure(cmd, m_blas[p_idx].accel, scratchBuffer.address);
    }

    // Submit command buffer and wait for completion
    m_app->submitAndWaitTempCmdBuffer(cmd);

    // Clean up scratch buffer after use
    m_alloc->destroy(scratchBuffer);
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
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    // Create and fill the instance buffer
    m_instancesBuffer = m_alloc->createBuffer(cmd, m_tlasInstances,
                                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                  | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

    // Ensure buffer is ready for acceleration structure operations
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

    // Prepare for TLAS building
    auto geo = m_tlasBuildData.makeInstanceGeometry(m_tlasInstances.size(), m_instancesBuffer.address);
    m_tlasBuildData.addGeometry(geo);
    auto sizeInfo = m_tlasBuildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                                                                   | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);

    // Create scratch buffer for TLAS
    VkDeviceSize scratchSize = sizeInfo.buildScratchSize;  // Build size (larger) instead of update size
    m_tlasScratchBuffer =
        m_alloc->createBuffer(scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    // Create the TLAS
    auto createInfo = m_tlasBuildData.makeCreateInfo();
    m_tlas          = m_alloc->createAcceleration(createInfo);
    m_tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, m_tlasScratchBuffer.address);

    // Submit the command buffer and wait for execution to complete
    m_app->submitAndWaitTempCmdBuffer(cmd);
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

    m_alloc->getStaging()->cmdToBuffer(cmd, m_instancesBuffer.buffer, 0,
                                       m_tlasInstances.size() * sizeof(VkAccelerationStructureInstanceKHR),
                                       m_tlasInstances.data());
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
        .stage = nvvk::createShaderStageInfo(m_device, comp_shd, VK_SHADER_STAGE_COMPUTE_BIT, USE_GLSL ? "main" : "computeMain"),
        .layout = m_rtPipe.layout,
    };

    vkCreateComputePipelines(m_device, {}, 1, &cpCreateInfo, nullptr, &m_rtPipe.plines[0]);

    vkDestroyShaderModule(m_device, cpCreateInfo.stage.module, nullptr);
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

    for(auto& blas : m_blas)
    {
      m_alloc->destroy(blas);
    }
    m_blas.clear();
    m_alloc->destroy(m_tlas);
    m_alloc->destroy(m_instancesBuffer);
    m_alloc->destroy(m_tlasScratchBuffer);

    m_sbt.destroy();
    m_tonemapper.reset();
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers->getColorImage(), m_gBuffers->getSize(),
                           nvh::getExecutablePath().replace_extension(".jpg").string());
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

  // Vulkan Ray Tracing Properties
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};

  // Pipeline and Shader Binding
  nvvkhl::PipelineContainer m_rtPipe;
  nvvk::SBTWrapper          m_sbt;  // Shading binding table wrapper

  // Acceleration Structures
  std::vector<VkAccelerationStructureInstanceKHR> m_tlasInstances;  // Keeping for animation
  nvvk::AccelKHR                                  m_tlas;
  nvvk::AccelerationStructureBuildData            m_tlasBuildData{VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
  std::vector<nvvk::AccelKHR>                     m_blas;

  // Buffers for instances and scratch data
  nvvk::Buffer m_instancesBuffer;
  nvvk::Buffer m_tlasScratchBuffer;

  // Mesh data
  std::vector<PrimitiveMeshVk>    m_bMeshes;
  std::vector<nvh::PrimitiveMesh> m_meshes;

  // Node and scene data
  std::vector<nvh::Node> m_nodes;
};

//////////////////////////////////////////////////////////////////////////
///
///
auto main(int argc, char** argv) -> int
{
  nvvkhl::ApplicationCreateInfo appInfo;

  nvh::CommandLineParser cli(PROJECT_NAME);
  cli.addArgument({"--headless"}, &appInfo.headless, "Run in headless mode");
  cli.parse(argc, argv);


  // Specific extension features
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR fetchFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};

  // Config for Vulkan context creation
  VkContextSettings vkSetup;
  if(!appInfo.headless)
  {
    nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature});  // To build acceleration structures
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature});  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});  // Required by ray tracing pipeline
  vkSetup.deviceExtensions.push_back({VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, &fetchFeatures});

  vkSetup.queues.push_back(VK_QUEUE_COMPUTE_BIT);  // Using other queue for SBT creation

  // Create the Vulkan context
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  if(!vkContext->isValid())
    std::exit(0);

  // Loading the Vulkan extension pointers
  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);

  // Set the information for the application
  appInfo.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext->getInstance();
  appInfo.device         = vkContext->getDevice();
  appInfo.physicalDevice = vkContext->getPhysicalDevice();
  appInfo.queues         = vkContext->getQueueInfos();

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appInfo);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app->addElement(std::make_shared<RayQueryFetch>());

  app->run();
  app.reset();
  vkContext.reset();

  return test->errorCode();
}
