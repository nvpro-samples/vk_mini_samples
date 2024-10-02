/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

//////////////////////////////////////////////////////////////////////////
/*

    This shows the use of Ray Query, or casting rays in a compute shader

*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <vulkan/vulkan_core.h>

#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvh/primitives.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/extensions_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/tonemap_postprocess.hpp"
#include "nvvkhl/shaders/dh_sky.h"
#include "common/vk_context.hpp"

#include "nvtx3/nvtx3.hpp"

#include "shaders/dh_bindings.h"

namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

#if USE_HLSL
#include "_autogen/ray_query_computeMain.spirv.h"
const auto& comp_shd = std::vector<char>{std::begin(ray_query_computeMain), std::end(ray_query_computeMain)};
#elif USE_SLANG
#include "_autogen/ray_query_slang.h"
const auto& comp_shd = std::vector<uint32_t>{std::begin(ray_querySlang), std::end(ray_querySlang)};
#else
#include "_autogen/ray_query.comp.glsl.h"
const auto& comp_shd = std::vector<uint32_t>{std::begin(ray_query_comp_glsl), std::end(ray_query_comp_glsl)};
#endif

#include "nvvk/specialization.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/acceleration_structures.hpp"

#define GROUP_SIZE 16  // Same group size as in compute shader


/// </summary> Ray trace multiple primitives using Ray Query
class RayQuery : public nvvkhl::IAppElement
{
  enum
  {
    eImgTonemapped,
    eImgRendered
  };

public:
  RayQuery()           = default;
  ~RayQuery() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    m_dutil      = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility
    m_alloc      = std::make_unique<nvvk::ResourceAllocatorDma>(m_device, m_app->getPhysicalDevice());
    m_rtSet      = std::make_unique<nvvk::DescriptorSetContainer>(m_device);
    m_tonemapper = std::make_unique<nvvkhl::TonemapperPostProcess>(m_device, m_alloc.get());

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    int32_t gctQueueIndex = m_app->getQueue(0).familyIndex;
    //m_rtBuilder.setup(m_device, m_alloc.get(), gctQueueIndex);
    m_sbt.setup(m_device, gctQueueIndex, m_alloc.get(), m_rtProperties);

    // Create resources
    createScene();
    createVkBuffers();
    createBottomLevelAS();
    createTopLevelAS();
#if USE_RTX
    createRtxPipeline();
#else
    createCompPipelines();
#endif

    m_tonemapper->createComputePipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    // Create two G-Buffers: the tonemapped image and the original rendered image
    std::vector<VkFormat> color_buffers = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R32G32B32A32_SFLOAT};  // tonemapped, original
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), VkExtent2D{width, height}, color_buffers);

    // Update the tonemapper with the information of the new G-Buffers
    m_tonemapper->updateComputeDescriptorSets(m_gBuffers->getDescriptorImageInfo(eImgRendered),
                                              m_gBuffers->getDescriptorImageInfo(eImgTonemapped));
    resetFrame();  // Reset frame to restart the rendering
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");

      ImGuiH::CameraWidget();

      using namespace ImGuiH;
      bool changed{false};
      if(ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
      {
        PropertyEditor::begin();
        if(PropertyEditor::treeNode("Light"))
        {
          changed |= PropertyEditor::entry("Position", [&] { return ImGui::DragFloat3("#1", &m_light.position.x); });

          changed |= PropertyEditor::entry("Intensity", [&] {
            return ImGui::SliderFloat("#1", &m_light.intensity, 0.0F, 1000.0F, "%.3f", ImGuiSliderFlags_Logarithmic);
          });
          changed |=
              PropertyEditor::entry("Radius", [&] { return ImGui::SliderFloat("#1", &m_light.radius, 0.0F, 1.0F); });
          PropertyEditor::treePop();
        }
        if(PropertyEditor::treeNode("Ray Tracer"))
        {
          changed |= PropertyEditor::entry("Depth", [&] { return ImGui::SliderInt("#1", &m_pushConst.maxDepth, 0, 20); });
          changed |=
              PropertyEditor::entry("Samples", [&] { return ImGui::SliderInt("#1", &m_pushConst.maxSamples, 1, 10); });
          PropertyEditor::treePop();
        }
        PropertyEditor::end();
      }

      if(ImGui::CollapsingHeader("Tonemapper"))
      {
        changed |= m_tonemapper->onUI();
      }

      ImGui::End();
      if(changed)
        resetFrame();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer tonemapped image
      ImGui::Image(m_gBuffers->getDescriptorSet(eImgTonemapped), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void memoryBarrier(VkCommandBuffer cmd)
  {
    VkMemoryBarrier mb{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
    };
    VkPipelineStageFlags srcDstStage{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    vkCmdPipelineBarrier(cmd, srcDstStage, srcDstStage, 0, 1, &mb, 0, nullptr, 0, nullptr);
  }

  void onRender(VkCommandBuffer cmd) override
  {
    auto sdbg = m_dutil->DBG_SCOPE(cmd);

    if(!updateFrame())
    {
      return;
    }

    // Update Camera uniform buffer
    const auto& clip = CameraManip.getClipPlanes();
    glm::mat4 proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), CameraManip.getAspectRatio(), clip.x, clip.y);
    proj[1][1] *= -1;
    DH::CameraInfo finfo{.projInv = glm::inverse(proj), .viewInv = glm::inverse(CameraManip.getMatrix())};
    vkCmdUpdateBuffer(cmd, m_bCameraInfo.buffer, 0, sizeof(DH::CameraInfo), &finfo);

    m_pushConst.frame = m_frame;
    m_pushConst.light = m_light;

    // Make sure buffer is ready to be used
    memoryBarrier(cmd);

    // Ray trace
    const VkExtent2D& size = m_app->getViewportSize();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipe.plines[0]);
    pushDescriptorSet(cmd);
    vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant), &m_pushConst);
    vkCmdDispatch(cmd, (size.width + (GROUP_SIZE - 1)) / GROUP_SIZE, (size.height + (GROUP_SIZE - 1)) / GROUP_SIZE, 1);

    // Making sure the rendered image is ready to be used by tonemapper
    memoryBarrier(cmd);

    m_tonemapper->runCompute(cmd, size);
  }

private:
  void createScene()
  {
    m_materials.push_back({{0.985f, 0.862f, 0.405f}, 0.5f, 0.0f});
    m_materials.push_back({{0.9622f, 0.928f, 0.9728f}, 0.0f, 0.09f, 1.0f});
    m_materials.push_back({{.7F, .7F, .7F}, 0.3f, 0.0f});

    m_meshes.emplace_back(nvh::createCube(1, 1, 1));
    m_meshes.emplace_back(nvh::createSphereUv(0.5f, 100, 100));
    m_meshes.emplace_back(nvh::createPlane(10, 100, 100));

    // Instance Cube
    {
      auto& n       = m_nodes.emplace_back();
      n.mesh        = 0;
      n.material    = 0;
      n.translation = {0.0f, 0.5f, 0.0F};
    }

    // Instance Sphere
    {
      auto& n       = m_nodes.emplace_back();
      n.mesh        = 1;
      n.material    = 1;
      n.translation = {1.0f, 1.5f, 1.0F};
    }

    // Adding a plane & material
    {
      auto& n       = m_nodes.emplace_back();
      n.mesh        = 2;
      n.material    = 2;
      n.translation = {0.0f, 0.0f, 0.0f};
    }

    m_light.intensity = 100.0f;
    m_light.position  = {2.0f, 7.0f, 2.0f};
    m_light.radius    = 0.2f;

    // Setting camera to see the scene
    CameraManip.setClipPlanes({0.1F, 100.0F});
    CameraManip.setLookat({-2.0F, 2.5F, 3.0f}, {0.4F, 0.3F, 0.2F}, {0.0F, 1.0F, 0.0F});

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
    NVTX3_FUNC_RANGE();
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_bMeshes.resize(m_meshes.size());

    VkBufferUsageFlags rtUsageFlag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                     | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;


    // Create a buffer of Vertex and Index per mesh
    std::vector<DH::PrimMeshInfo> primInfo;
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      auto& m    = m_bMeshes[i];
      m.vertices = m_alloc->createBuffer(cmd, m_meshes[i].vertices, rtUsageFlag);
      m.indices  = m_alloc->createBuffer(cmd, m_meshes[i].triangles, rtUsageFlag);
      m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
      m_dutil->DBG_NAME_IDX(m.indices.buffer, i);

      // To find the buffers of the mesh (buffer reference)
      DH::PrimMeshInfo info{
          .vertexAddress = m.vertices.address,
          .indexAddress  = m.indices.address,
      };
      primInfo.emplace_back(info);
    }

    // Creating the buffer of all primitive/mesh information
    m_bPrimInfo = m_alloc->createBuffer(cmd, primInfo, rtUsageFlag);
    m_dutil->DBG_NAME(m_bPrimInfo.buffer);

    // Create the buffer of the current camera transformation, changing at each frame
    m_bCameraInfo = m_alloc->createBuffer(sizeof(DH::CameraInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bCameraInfo.buffer);

    // Primitive instance information
    std::vector<DH::InstanceInfo> instInfo;
    for(auto& node : m_nodes)
    {
      DH::InstanceInfo info{
          .transform  = node.localMatrix(),
          .materialID = node.material,
      };
      instInfo.emplace_back(info);
    }
    m_bInstInfoBuffer =
        m_alloc->createBuffer(cmd, instInfo, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bInstInfoBuffer.buffer);

    m_bMaterials = m_alloc->createBuffer(cmd, m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bMaterials.buffer);

    // Buffer references of all scene elements
    DH::SceneInfo sceneDesc{
        .materialAddress = m_bMaterials.address,
        .instInfoAddress = m_bInstInfoBuffer.address,
        .primInfoAddress = m_bPrimInfo.address,
        .light           = m_light,
    };

    m_bSceneDesc = m_alloc->createBuffer(cmd, sizeof(DH::SceneInfo), &sceneDesc,
                                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bSceneDesc.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }


  //--------------------------------------------------------------------------------------------------
  // Converting a PrimitiveMesh as input for BLAS
  //
  nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const nvh::PrimitiveMesh& prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress)
  {
    nvvk::AccelerationStructureGeometryInfo result;

    uint32_t maxPrimitiveCount = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,  // vec3 vertex position data
        .vertexData   = {.deviceAddress = vertexAddress},
        .vertexStride = sizeof(nvh::PrimitiveVertex),
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
    NVTX3_FUNC_RANGE();
    std::vector<nvvk::AccelerationStructureBuildData> blasData;
    blasData.resize(m_meshes.size());  // Build Information for each BLAS
    m_blas.resize(m_meshes.size());    // The actual BLAS

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
    nvvk::Buffer scratchBuffer =
        m_alloc->createBuffer(maxScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    VkDeviceAddress scratchAddress = scratchBuffer.address;

    // Create and build all BLAS
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    for(size_t i = 0; i < m_blas.size(); i++)
    {
      VkAccelerationStructureCreateInfoKHR createInfo = blasData[i].makeCreateInfo();
      m_blas[i]                                       = m_alloc->createAcceleration(createInfo);
      m_dutil->DBG_NAME(m_blas[i].accel);
      blasData[i].cmdBuildAccelerationStructure(cmd, m_blas[i].accel, scratchAddress);
      // Because we will be reusing the scratch buffer, we need a barrier to ensure the BLAS is finished before next build
      nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                         VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    }
    m_app->submitAndWaitTempCmdBuffer(cmd);

    m_alloc->finalizeAndReleaseStaging();
    m_alloc->destroy(scratchBuffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    NVTX3_FUNC_RANGE();

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
                                                 .accelerationStructureReference = m_blas[node.mesh].address};
      instances.emplace_back(rayInst);
    }

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    // Create a buffer holding the actual instance data (matrices++) for use by the AS builder
    nvvk::Buffer instancesBuffer = m_alloc->createBuffer(
        cmd, instances, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

    auto instGeo = tlasData.makeInstanceGeometry(instances.size(), instancesBuffer.address);
    tlasData.addGeometry(instGeo);

    // Create the TLAS
    auto sizeInfo = tlasData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    nvvk::Buffer scratchBuffer =
        m_alloc->createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    VkDeviceAddress scratchAddress = scratchBuffer.address;

    VkAccelerationStructureCreateInfoKHR createInfo = tlasData.makeCreateInfo();
    m_tlas                                          = m_alloc->createAcceleration(createInfo);
    m_dutil->DBG_NAME(m_tlas.accel);
    tlasData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, scratchAddress);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    m_app->submitAndWaitTempCmdBuffer(cmd);

    m_alloc->destroy(instancesBuffer);
    m_alloc->destroy(scratchBuffer);
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
    m_rtSet->addBinding(B_cameraInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_sceneDesc, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

    // pushing time
    VkPushConstantRange        pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant)};
    VkPipelineLayoutCreateInfo plCreateInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1U,
        .pSetLayouts            = &m_rtSet->getLayout(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstant,
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
    VkDescriptorImageInfo  imageInfo{{}, m_gBuffers->getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo dbi_unif{m_bCameraInfo.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo sceneDesc{m_bSceneDesc.buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtSet->makeWrite(0, B_tlas, &descASInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, B_outImage, &imageInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, B_cameraInfo, &dbi_unif));
    writes.emplace_back(m_rtSet->makeWrite(0, B_sceneDesc, &sceneDesc));

    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipe.layout, 0,
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

    const auto& m   = CameraManip.getMatrix();
    const auto  fov = CameraManip.getFov();

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
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }
    m_alloc->destroy(m_bCameraInfo);
    m_alloc->destroy(m_bPrimInfo);
    m_alloc->destroy(m_bSceneDesc);
    m_alloc->destroy(m_bInstInfoBuffer);
    m_alloc->destroy(m_bMaterials);

    m_rtSet->deinit();
    m_gBuffers.reset();

    m_rtPipe.destroy(m_device);

    m_sbt.destroy();
    //m_rtBuilder.destroy();
    for(auto& b : m_blas)
    {
      m_alloc->destroy(b);
    }
    m_alloc->destroy(m_tlas);


    m_tonemapper.reset();
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*                           m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>               m_dutil;
  std::unique_ptr<nvvk::ResourceAllocatorDma>    m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer>  m_rtSet;  // Descriptor set
  std::unique_ptr<nvvkhl::TonemapperPostProcess> m_tonemapper;
  std::unique_ptr<nvvkhl::GBuffer>               m_gBuffers;  // G-Buffers: color + depth

  VkDevice m_device = VK_NULL_HANDLE;  // Convenient

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
  std::vector<nvh::PrimitiveMesh> m_meshes;
  std::vector<nvh::Node>          m_nodes;
  std::vector<DH::Material>       m_materials;
  DH::Light                       m_light = {};

  // Pipeline
  DH::PushConstant m_pushConst{};  // Information sent to the shader
  int              m_frame{0};
  int              m_maxFrames{10000};

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::SBTWrapper m_sbt;  // Shading binding table wrapper

  std::vector<nvvk::AccelKHR> m_blas;
  nvvk::AccelKHR              m_tlas;

  nvvkhl::PipelineContainer m_rtPipe;
};

//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char** argv) -> int
{
  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};

  // Config for Vulkan context creation
  VkContextSettings vkSetup;
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature});  // To build acceleration structures
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature});  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});  // Required by ray tracing pipeline
  vkSetup.deviceExtensions.push_back({VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature});

  // Vulkan context creation
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  if(!vkContext->isValid())
    std::exit(0);

  // Loading the Vulkan extension pointers
  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);

  // Application settings
  nvvkhl::ApplicationCreateInfo appSetup;
  appSetup.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appSetup.vSync          = false;
  appSetup.instance       = vkContext->getInstance();
  appSetup.device         = vkContext->getDevice();
  appSetup.physicalDevice = vkContext->getPhysicalDevice();
  appSetup.queues         = vkContext->getQueueInfos();


  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appSetup);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app->addElement(std::make_shared<RayQuery>());

  app->run();
  app.reset();
  vkContext.reset();

  return test->errorCode();
}
