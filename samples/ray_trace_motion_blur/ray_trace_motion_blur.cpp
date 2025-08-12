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

  This sample raytrace a scene with motion information

*/
//////////////////////////////////////////////////////////////////////////

#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#include <imgui/imgui.h>

#include <array>
#include <vulkan/vulkan_core.h>

#include <glm/glm.hpp>


#define VMA_IMPLEMENTATION
#define VMA_LEAK_LOG_FORMAT(format, ...)                                                                               \
  {                                                                                                                    \
    printf((format), __VA_ARGS__);                                                                                     \
    printf("\n");                                                                                                      \
  }

#include "shaders/dh_bindings.h"
#include "shaders/shaderio.h"  // Shared between host and device

#include "nvutils/primitives.hpp"


struct MatPrimitiveMesh : public nvutils::PrimitiveMesh
{
  std::vector<int> triMat;  // Material per triangle
};

#include "_autogen/raytrace_motion_blur.rchit.glsl.h"
#include "_autogen/raytrace_motion_blur.rgen.glsl.h"
#include "_autogen/raytrace_motion_blur.rmiss.glsl.h"
#include "_autogen/raytrace_motion_blur.slang.h"

#include "nvvk/validation_settings.hpp"
#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvgui/camera.hpp>
#include <nvutils/file_operations.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/timers.hpp>
#include <nvvk/acceleration_structures.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/staging.hpp>
#include "common/utils.hpp"

#define MAXRAYRECURSIONDEPTH 5

std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};


//////////////////////////////////////////////////////////////////////////
/// </summary> Ray trace multiple primitives
class RtMotionBlur : public nvapp::IAppElement
{
public:
  RtMotionBlur()           = default;
  ~RtMotionBlur() override = default;

  void onAttach(nvapp::Application* app) override
  {
    SCOPED_TIMER(__FUNCTION__);

    m_app    = app;
    m_device = m_app->getDevice();

    m_alloc.init({
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

    // GBuffer for the ray tracing
    m_gBuffers.init({.allocator      = &m_alloc,
                     .colorFormats   = {m_colorFormat},
                     .depthFormat    = m_depthFormat,
                     .imageSampler   = linearSampler,
                     .descriptorPool = m_app->getTextureDescriptorPool()});


    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    m_rtProperties.pNext = &m_asProperties;
    prop2.pNext          = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    const uint32_t gct_queue_index = m_app->getQueue(0).familyIndex;
    m_sbt.init(m_device, m_rtProperties);

    // Create resources
    createScene();
    createVulkanBuffers();
    createBottomLevelAS();
    createTopLevelAS();
    createRtxPipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    m_gBuffers.update(cmd, size);
    writeRtDesc();
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");
      nvgui::CameraWidget(g_cameraManip);
      ImGui::ColorEdit3("Clear Color", &m_pushConst.clearColor.x);
      ImGui::SliderInt("Num Samples", &m_pushConst.numSamples, 1, 1000);
      ImGui::SliderFloat3("Light Position", &m_pushConst.lightPosition.x, -10, 10);
      ImGui::SliderFloat("Light Intensity", &m_pushConst.lightIntensity, 0, 200);
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");
      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());  // Display the G-Buffer image
      ImGui::End();
      ImGui::PopStyleVar();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);

    ++m_pushConst.frame;

    // Update Frame buffer uniform buffer
    const glm::vec2&    clip = g_cameraManip->getClipPlanes();
    shaderio::FrameInfo finfo{};
    finfo.view    = g_cameraManip->getViewMatrix();
    finfo.proj    = g_cameraManip->getPerspectiveMatrix();
    finfo.projInv = glm::inverse(finfo.proj);
    finfo.viewInv = glm::inverse(finfo.view);
    finfo.camPos  = g_cameraManip->getEye();
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &finfo);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);


    // Ray trace
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 0, 1,
                            &m_descriptorPack.sets[0], 0, nullptr);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);

    const nvvk::SBTGenerator::Regions& regions = m_sbt.getSBTRegions();
    const VkExtent2D&                  size    = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, &regions.raygen, &regions.miss, &regions.hit, &regions.callable, size.width, size.height, 1);
  }

private:
  void createScene()
  {
    SCOPED_TIMER(__FUNCTION__);

    // Materials
    m_materials.push_back({{0.7, 0.7, 0.7, 1.0}});                 // gray
    m_materials.push_back({{0.982062, 0.857638, 0.400811, 1.0}});  // yellow
    m_materials.push_back({{0.982062, 0.1, 0.1, 1.0}});            // red
    m_materials.push_back({{0.1, 0.9, 0.1, 1.0}});                 // green
    m_materials.push_back({{0.1, 0.1, 0.982062, 1.0}});            // blue
    m_materials.push_back({{0.982062, 0.1, 0.982062, 1.0}});       // magenta

    // Meshes (cube)
    m_meshes.emplace_back(nvutils::createCube());
    m_meshes.emplace_back(nvutils::createPlane(10, 100, 100));
    m_meshes[0].triMat = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};        // Cube
    m_meshes.back().triMat.resize(m_meshes.back().triangles.size());  // Plane all tri to gray
    // Modified Cube: one vertex is moved, this will be used to have motion between Cube and the Modified Cube
    m_meshes.emplace_back(m_meshes[0]);     // Copy of cube
    m_meshes.back().vertices[6].pos *= 2;   // Modifying the +x,+y,+z position (appearing 3 times)
    m_meshes.back().vertices[11].pos *= 2;  // Modifying the +x,+y,+z position
    m_meshes.back().vertices[22].pos *= 2;  // Modifying the +x,+y,+z position

    // Instances
    m_instances.resize(4);
    m_instances[0].mesh        = 0;           // cube 0
    m_instances[0].translation = {2, 0, 2};   //
    m_instances[1].mesh        = 0;           // cube 1
    m_instances[1].translation = {0, 0, 2};   //
    m_instances[2].mesh        = 2;           // cube 2
    m_instances[2].translation = {0, 0, 0};   //
    m_instances[3].mesh        = 1;           // plane
    m_instances[3].translation = {0, -1, 0};  //

    // Setting camera to see the scene
    g_cameraManip->setClipPlanes({0.1F, 100.0F});
    g_cameraManip->setLookat({3.91698, 2.65970, -0.42755}, {0.71716, 0.03205, 1.36345}, {0.00000, 1.00000, 0.00000});

    // Default parameters for overall material
    m_pushConst.clearColor     = {1, 1, 1, 1};
    m_pushConst.lightPosition  = {9.5f, 5.5f, -6.5f};
    m_pushConst.lightIntensity = 100;
    m_pushConst.frame          = 0;
    m_pushConst.numSamples     = 100;
  }

  // Create all Vulkan buffer data
  void createVulkanBuffers()
  {
    SCOPED_TIMER(__FUNCTION__);

    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);

    // Usage flags for the different buffers
    const VkBufferUsageFlags rt_usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                             | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex and Index per mesh
    m_bMeshes.resize(m_meshes.size());
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& m = m_bMeshes[i];
      NVVK_CHECK(m_alloc.createBuffer(m.vertices, std::span(m_meshes[i].vertices).size_bytes(), rt_usage_flag));
      NVVK_CHECK(m_alloc.createBuffer(m.indices, std::span(m_meshes[i].triangles).size_bytes(), rt_usage_flag));
      NVVK_CHECK(m_alloc.createBuffer(m.triMaterial, std::span(m_meshes[i].triMat).size_bytes(), rt_usage_flag));
      NVVK_CHECK(uploader.appendBuffer(m.vertices, 0, std::span(m_meshes[i].vertices)));
      NVVK_CHECK(uploader.appendBuffer(m.indices, 0, std::span(m_meshes[i].triangles)));
      NVVK_CHECK(uploader.appendBuffer(m.triMaterial, 0, std::span(m_meshes[i].triMat)));
      NVVK_DBG_NAME(m.vertices.buffer);
      NVVK_DBG_NAME(m.indices.buffer);
      NVVK_DBG_NAME(m.triMaterial.buffer);
    }

    // Create the buffer of the current frame, changing at each frame
    NVVK_CHECK(m_alloc.createBuffer(m_bFrameInfo, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    VMA_MEMORY_USAGE_AUTO_PREFER_HOST));
    NVVK_DBG_NAME(m_bFrameInfo.buffer);

    // Primitive instance information
    std::vector<shaderio::InstanceInfo> inst_info;
    inst_info.reserve(m_instances.size());
    for(const nvutils::Node& node : m_instances)
    {
      shaderio::InstanceInfo info{};
      info.transform = node.localMatrix();
      inst_info.push_back(info);
    }

    NVVK_CHECK(m_alloc.createBuffer(m_bInstInfoBuffer, std::span(inst_info).size_bytes(),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(uploader.appendBuffer(m_bInstInfoBuffer, 0, std::span(inst_info)));
    NVVK_DBG_NAME(m_bInstInfoBuffer.buffer);


    NVVK_CHECK(m_alloc.createBuffer(m_bMaterials, std::span(m_materials).size_bytes(),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(uploader.appendBuffer(m_bMaterials, 0, std::span(m_materials)));
    NVVK_DBG_NAME(m_bMaterials.buffer);

    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();
  }


  //--------------------------------------------------------------------------------------------------
  // Converting a PrimitiveMesh as input for BLAS
  //
  nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const MatPrimitiveMesh& prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress)
  {
    const auto max_primitive_count = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride             = sizeof(nvutils::PrimitiveVertex);
    triangles.indexType                = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress  = indexAddress;
    triangles.maxVertex                = static_cast<uint32_t>(prim.vertices.size()) - 1;
    //triangles.transformData; // Identity

    // Identify the above data as containing opaque triangles.
    VkAccelerationStructureGeometryKHR as_geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    as_geom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    as_geom.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;
    as_geom.geometry.triangles = triangles;

    VkAccelerationStructureBuildRangeInfoKHR offset{};
    offset.firstVertex     = 0;
    offset.primitiveCount  = max_primitive_count;
    offset.primitiveOffset = 0;
    offset.transformOffset = 0;

    // Our BLAS is made from only one geometry, but could be made of many geometries
    nvvk::AccelerationStructureGeometryInfo result;
    result.geometry  = as_geom;
    result.rangeInfo = offset;

    return result;
  }

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  void createBottomLevelAS()
  {
    SCOPED_TIMER(__FUNCTION__);

    std::vector<nvvk::AccelerationStructureBuildData> blasBuildData;

    // BLAS - Storing each primitive in a geometry
    blasBuildData.resize(m_meshes.size());
    m_blas.resize(m_meshes.size());
    uint32_t meshIndex = 0;

    VkDeviceSize maxScratchSize{0};
    // Adding Cube and Plane
    for(uint32_t p_idx = 0; p_idx < 2; p_idx++)
    {
      const VkDeviceAddress vertex_address = m_bMeshes[p_idx].vertices.address;
      const VkDeviceAddress index_address  = m_bMeshes[p_idx].indices.address;
      const nvvk::AccelerationStructureGeometryInfo geo = primitiveToGeometry(m_meshes[p_idx], vertex_address, index_address);
      blasBuildData[meshIndex].asType = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      blasBuildData[meshIndex].addGeometry(geo);
      auto sizeInfo = blasBuildData[meshIndex].finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
      maxScratchSize = std::max(maxScratchSize, sizeInfo.buildScratchSize);
      meshIndex++;
    }

    VkAccelerationStructureGeometryMotionTrianglesDataNV motionTriangles{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MOTION_TRIANGLES_DATA_NV};
    motionTriangles.vertexData.deviceAddress = m_bMeshes[2].vertices.address;  // cube_modif

    {                                                              // #NV_Motion_blur
      uint32_t                                p_idx          = 0;  // Using the original Cube
      const VkDeviceAddress                   vertex_address = m_bMeshes[p_idx].vertices.address;
      const VkDeviceAddress                   index_address  = m_bMeshes[p_idx].indices.address;
      nvvk::AccelerationStructureGeometryInfo geo = primitiveToGeometry(m_meshes[p_idx], vertex_address, index_address);
      geo.geometry.geometry.triangles.pNext       = &motionTriangles;

      blasBuildData[meshIndex].asType = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      blasBuildData[meshIndex].addGeometry(geo);
      auto sizeInfo = blasBuildData[meshIndex].finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV);
      maxScratchSize = std::max(maxScratchSize, sizeInfo.buildScratchSize);
      meshIndex++;
    }

    nvvk::Buffer scratchBuffer;
    NVVK_CHECK(m_alloc.createBuffer(scratchBuffer, maxScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                    VMA_MEMORY_USAGE_AUTO, {}, m_asProperties.minAccelerationStructureScratchOffsetAlignment));
    NVVK_DBG_NAME(scratchBuffer.buffer);

    // Create the acceleration structures
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    {
      for(size_t p_idx = 0; p_idx < blasBuildData.size(); p_idx++)
      {
        NVVK_CHECK(m_alloc.createAcceleration(m_blas[p_idx], blasBuildData[p_idx].makeCreateInfo()));
        NVVK_DBG_NAME(m_blas[p_idx].accel);
        blasBuildData[p_idx].cmdBuildAccelerationStructure(cmd, m_blas[p_idx].accel, scratchBuffer.address);
      }
    }
    m_app->submitAndWaitTempCmdBuffer(cmd);


    m_alloc.destroyBuffer(scratchBuffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    // VkAccelerationStructureMotionInstanceNV must have a stride of 160 bytes.
    // See https://www.khronos.org/registry/vulkan/specs/1.2-extensions/man/html/VkAccelerationStructureGeometryInstancesDataKHR.html
    struct VkAccelerationStructureMotionInstanceNVPad : VkAccelerationStructureMotionInstanceNV
    {
      uint64_t _pad{0};
    };
    static_assert(sizeof(VkAccelerationStructureMotionInstanceNVPad) == 160);
    SCOPED_TIMER(__FUNCTION__);

    // #NV_Motion_blur
    std::vector<VkAccelerationStructureMotionInstanceNVPad> tlas;
    tlas.reserve(m_instances.size());

    nvvk::AccelerationStructureBuildData tlasBuildData{VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};

    // #NV_Motion_blur
    // CUBE-0 : Matrix transformation animation
    {
      // Position of the instance at T0 and T1
      glm::mat4 matT0 = m_instances[0].localMatrix();
      glm::mat4 matT1 = glm::translate(glm::mat4(1), glm::vec3(0.30f, 0.0f, 0.0f)) * matT0;

      VkAccelerationStructureMatrixMotionInstanceNV data{};
      data.transformT0                            = nvvk::toTransformMatrixKHR(matT0);
      data.transformT1                            = nvvk::toTransformMatrixKHR(matT1);
      data.instanceCustomIndex                    = 0;  // gl_InstanceCustomIndexEXT
      data.accelerationStructureReference         = m_blas[m_instances[0].mesh].address;
      data.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
      data.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      data.mask                                   = 0xFF;
      VkAccelerationStructureMotionInstanceNVPad rayInst;
      rayInst.type                      = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_MATRIX_MOTION_NV;
      rayInst.data.matrixMotionInstance = data;
      tlas.emplace_back(rayInst);
    }

    // #NV_Motion_blur
    // CUBE-1 : SRT transformation animation
    {
      glm::quat rot = {1, 0, 0, 0};

      // Position of the instance at T0 and T1
      VkSRTDataNV matT0{};  // Translated to 0,0,2
      matT0.sx = 1.0f;
      matT0.sy = 1.0f;
      matT0.sz = 1.0f;
      matT0.tz = 2.0f;
      matT0.qx = rot.x;
      matT0.qy = rot.y;
      matT0.qz = rot.z;
      matT0.qw = rot.w;

      VkSRTDataNV matT1 = matT0;  // Setting a rotation
      rot               = glm::quat(glm::vec3(glm::radians(10.0f), glm::radians(30.0f), 0.0f));
      matT1.qx          = rot.x;
      matT1.qy          = rot.y;
      matT1.qz          = rot.z;
      matT1.qw          = rot.w;

      VkAccelerationStructureSRTMotionInstanceNV data{};
      data.transformT0                            = matT0;
      data.transformT1                            = matT1;
      data.instanceCustomIndex                    = 0;  // gl_InstanceCustomIndexEXT
      data.accelerationStructureReference         = m_blas[m_instances[1].mesh].address;
      data.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
      data.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
      data.mask                                   = 0xFF;
      VkAccelerationStructureMotionInstanceNVPad rayInst;
      rayInst.type                   = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_SRT_MOTION_NV;
      rayInst.data.srtMotionInstance = data;
      tlas.emplace_back(rayInst);
    }

    // Static instances: cube (morph) and plane
    const VkGeometryInstanceFlagsKHR flags{VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV};
    for(int i = 2; i < (int)m_instances.size(); i++)
    {
      auto&                              node = m_instances[i];
      VkAccelerationStructureInstanceKHR ray_inst{};
      ray_inst.transform           = nvvk::toTransformMatrixKHR(node.localMatrix());  // Position of the instance
      ray_inst.instanceCustomIndex = node.mesh;                                       // gl_InstanceCustomIndexEXT
      ray_inst.accelerationStructureReference         = m_blas[node.mesh].address;
      ray_inst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
      ray_inst.flags                                  = flags;
      ray_inst.mask                                   = 0xFF;
      VkAccelerationStructureMotionInstanceNVPad rayInst;
      rayInst.type                = VK_ACCELERATION_STRUCTURE_MOTION_INSTANCE_TYPE_STATIC_NV;
      rayInst.data.staticInstance = ray_inst;
      tlas.emplace_back(rayInst);
    }

    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);


    // Create the instances buffer, add a barrier to ensure the data is copied before the TLAS build
    nvvk::Buffer instancesBuffer;
    NVVK_CHECK(m_alloc.createBuffer(instancesBuffer, std::span(tlas).size_bytes(),
                                    VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                        | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(uploader.appendBuffer(instancesBuffer, 0, std::span(tlas)));
    NVVK_DBG_NAME(instancesBuffer.buffer);
    uploader.cmdUploadAppended(cmd);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);


    nvvk::AccelerationStructureGeometryInfo geometryInfo =
        tlasBuildData.makeInstanceGeometry(tlas.size(), instancesBuffer.address);
    tlasBuildData.addGeometry(geometryInfo);
    // Get the size of the TLAS
    auto sizeInfo = tlasBuildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV);

    // Create the scratch buffer
    nvvk::Buffer scratchBuffer;
    NVVK_CHECK(m_alloc.createBuffer(scratchBuffer, sizeInfo.buildScratchSize,
                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    VMA_MEMORY_USAGE_AUTO, {}, m_asProperties.minAccelerationStructureScratchOffsetAlignment));

    // Create the TLAS with motionblur support
    VkAccelerationStructureCreateInfoKHR createInfo = tlasBuildData.makeCreateInfo();
#ifdef VK_NV_ray_tracing_motion_blur
    VkAccelerationStructureMotionInfoNV motionInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MOTION_INFO_NV};
    motionInfo.maxInstances = uint32_t(tlas.size());
    createInfo.createFlags  = VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV;
    createInfo.pNext        = &motionInfo;
#endif

    NVVK_CHECK(m_alloc.createAcceleration(m_tlas, createInfo));
    NVVK_DBG_NAME(m_tlas.accel);
    tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, scratchBuffer.address);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();

    m_alloc.destroyBuffer(scratchBuffer);
    m_alloc.destroyBuffer(instancesBuffer);
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    SCOPED_TIMER(__FUNCTION__);

    // This descriptor set, holds the top level acceleration structure and the output image
    // Create Binding Set

    m_descriptorPack.bindings.addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_descriptorPack.bindings.addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_descriptorPack.bindings.addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_descriptorPack.bindings.addBinding(B_materials, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_descriptorPack.bindings.addBinding(B_instances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_descriptorPack.bindings.addBinding(B_vertex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
    m_descriptorPack.bindings.addBinding(B_index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
    m_descriptorPack.bindings.addBinding(B_triMat, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);

    // Create Layout, Pool, and Sets
    NVVK_CHECK(m_descriptorPack.initFromBindings(m_device, 1));
    NVVK_DBG_NAME(m_descriptorPack.layout);
    NVVK_DBG_NAME(m_descriptorPack.pool);
    NVVK_DBG_NAME(m_descriptorPack.sets[0]);


    // Creating all shaders
    enum StageIndices
    {
      eRaygen,
      eMiss,
      eClosestHit,
      eShaderGroupCount
    };
    std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
    for(auto& s : stages)
      s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

#if USE_SLANG
    const VkShaderModuleCreateInfo createInfo = nvsamples::getShaderModuleCreateInfo(raytrace_motion_blur_slang);
    stages[eRaygen].pNext                     = &createInfo;
    stages[eRaygen].pName                     = "rgenMain";
    stages[eRaygen].stage                     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eMiss].pNext                       = &createInfo;
    stages[eMiss].pName                       = "rmissMain";
    stages[eMiss].stage                       = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eClosestHit].pNext                 = &createInfo;
    stages[eClosestHit].pName                 = "rchitMain";
    stages[eClosestHit].stage                 = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
#else
    // Raygen
    const VkShaderModuleCreateInfo rgenCreateInfo = nvsamples::getShaderModuleCreateInfo(raytrace_motion_blur_rgen_glsl);
    stages[eRaygen].pNext = &rgenCreateInfo;
    stages[eRaygen].pName = "main";
    stages[eRaygen].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Miss
    const VkShaderModuleCreateInfo rmissCreateInfo = nvsamples::getShaderModuleCreateInfo(raytrace_motion_blur_rmiss_glsl);
    stages[eMiss].pNext = &rmissCreateInfo;
    stages[eMiss].pName = "main";
    stages[eMiss].stage = VK_SHADER_STAGE_MISS_BIT_KHR;

    // Closest hit
    const VkShaderModuleCreateInfo rchitCreateInfo = nvsamples::getShaderModuleCreateInfo(raytrace_motion_blur_rchit_glsl);
    stages[eClosestHit].pNext = &rchitCreateInfo;
    stages[eClosestHit].pName = "main";
    stages[eClosestHit].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
#endif


    // Shader groups
    VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
    group.anyHitShader       = VK_SHADER_UNUSED_KHR;
    group.closestHitShader   = VK_SHADER_UNUSED_KHR;
    group.generalShader      = VK_SHADER_UNUSED_KHR;
    group.intersectionShader = VK_SHADER_UNUSED_KHR;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;
    // Raygen
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eRaygen;
    shader_groups.push_back(group);

    // Miss
    group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    group.generalShader = eMiss;
    shader_groups.push_back(group);

    // closest hit shader
    group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    group.generalShader    = VK_SHADER_UNUSED_KHR;
    group.closestHitShader = eClosestHit;
    shader_groups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    const VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant)};

    // Pipeline layout
    nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorPack.layout}, {push_constant});
    NVVK_DBG_NAME(m_pipelineLayout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    ray_pipeline_info.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    ray_pipeline_info.pStages                      = stages.data();
    ray_pipeline_info.groupCount                   = static_cast<uint32_t>(shader_groups.size());
    ray_pipeline_info.pGroups                      = shader_groups.data();
    ray_pipeline_info.maxPipelineRayRecursionDepth = MAXRAYRECURSIONDEPTH;  // Ray depth
    ray_pipeline_info.layout                       = m_pipelineLayout;
    ray_pipeline_info.flags = VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV;  // #NV_Motion_blur
    NVVK_CHECK(vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, &m_pipeline));
    NVVK_DBG_NAME(m_pipeline);

    // Creating the SBT
    // Prepare SBT data from ray pipeline
    size_t bufferSize = m_sbt.calculateSBTBufferSize(m_pipeline, ray_pipeline_info);

    // Create SBT buffer using the size from above
    NVVK_CHECK(m_alloc.createBuffer(m_sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                                    m_sbt.getBufferAlignment()));
    NVVK_DBG_NAME(m_sbtBuffer.buffer);

    // Pass the manual mapped pointer to fill the SBT data
    NVVK_CHECK(m_sbt.populateSBTBuffer(m_sbtBuffer.address, bufferSize, m_sbtBuffer.mapping));
  }

  void writeRtDesc()
  {
    // Write to descriptors
    std::vector<VkDescriptorBufferInfo> vertex_desc;
    std::vector<VkDescriptorBufferInfo> index_desc;
    std::vector<VkDescriptorBufferInfo> trimat_desc;
    vertex_desc.reserve(m_bMeshes.size());
    index_desc.reserve(m_bMeshes.size());
    trimat_desc.reserve(m_bMeshes.size());
    for(auto& m : m_bMeshes)
    {
      vertex_desc.push_back({m.vertices.buffer, 0, VK_WHOLE_SIZE});
      index_desc.push_back({m.indices.buffer, 0, VK_WHOLE_SIZE});
      trimat_desc.push_back({m.triMaterial.buffer, 0, VK_WHOLE_SIZE});
    }

    nvvk::WriteSetContainer         writes{};
    const nvvk::DescriptorBindings& bindings = m_descriptorPack.bindings;
    const VkDescriptorSet           set      = m_descriptorPack.sets[0];
    writes.append(bindings.getWriteSet(B_tlas, set), m_tlas);
    writes.append(bindings.getWriteSet(B_outImage, set), m_gBuffers.getColorImageView(), VK_IMAGE_LAYOUT_GENERAL);
    writes.append(bindings.getWriteSet(B_frameInfo, set), m_bFrameInfo);
    writes.append(bindings.getWriteSet(B_materials, set), m_bMaterials);
    writes.append(bindings.getWriteSet(B_instances, set), m_bInstInfoBuffer);
    writes.append(bindings.getWriteSet(B_vertex, set), vertex_desc.data());
    writes.append(bindings.getWriteSet(B_index, set), index_desc.data());
    writes.append(bindings.getWriteSet(B_triMat, set), trimat_desc.data());
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void destroyResources()
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    m_descriptorPack.deinit();

    for(PrimitiveMeshVk& m : m_bMeshes)
    {
      m_alloc.destroyBuffer(m.vertices);
      m_alloc.destroyBuffer(m.indices);
      m_alloc.destroyBuffer(m.triMaterial);
    }
    m_alloc.destroyBuffer(m_bFrameInfo);
    m_alloc.destroyBuffer(m_bInstInfoBuffer);
    m_alloc.destroyBuffer(m_bMaterials);
    m_alloc.destroyBuffer(m_sbtBuffer);

    m_sbt.deinit();

    for(auto& blas : m_blas)
      m_alloc.destroyAcceleration(blas);
    m_alloc.destroyAcceleration(m_tlas);

    m_samplerPool.deinit();
    m_gBuffers.deinit();
    m_alloc.deinit();
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers.getColorImage(), m_gBuffers.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string(), 95);
  }


  //--------------------------------------------------------------------------------------------------
  //
  //
  nvapp::Application*     m_app{};
  nvvk::ResourceAllocator m_alloc;
  nvvk::GBuffer           m_gBuffers;  // G-Buffers: color + depth
  nvvk::SamplerPool       m_samplerPool{};

  VkFormat m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkDevice m_device      = VK_NULL_HANDLE;                 // Convenient

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;     // Buffer of the vertices
    nvvk::Buffer indices;      // Buffer of the indices
    nvvk::Buffer triMaterial;  // Buffer of the material per triangle
  };
  std::vector<PrimitiveMeshVk> m_bMeshes;
  nvvk::Buffer                 m_bFrameInfo;
  nvvk::Buffer                 m_bPrimInfo;
  nvvk::Buffer                 m_bSceneDesc;  // SceneDescription
  nvvk::Buffer                 m_bInstInfoBuffer;
  nvvk::Buffer                 m_bMaterials;

  std::vector<VkSampler> m_samplers;

  // Data and setting
  struct Material
  {
    glm::vec4 color{1.F};
  };
  std::vector<MatPrimitiveMesh> m_meshes;
  std::vector<nvutils::Node>    m_instances;
  std::vector<Material>         m_materials;

  // Pipeline
  shaderio::PushConstant m_pushConst{};  // Information sent to the shader

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_asProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

  nvvk::SBTGenerator m_sbt;        // Shading binding table wrapper
  nvvk::Buffer       m_sbtBuffer;  // Buffer for the SBT

  std::vector<nvvk::AccelerationStructure> m_blas;
  nvvk::AccelerationStructure              m_tlas;

  nvvk::DescriptorPack m_descriptorPack;  // Bindings, pool, layout, and sets
  VkPipeline           m_pipeline{};
  VkPipelineLayout     m_pipelineLayout{};
};


//////////////////////////////////////////////////////////////////////////
///
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless"}, &appInfo.headless, true);
  reg.add({"frames", "Number of frames to run in headless mode"}, &appInfo.headlessFrameCount);
  cli.add(reg);
  cli.parse(argc, argv);


  // #NV_Motion_blur
  // Validation Layer, filtering message: unexpected VkStructureType VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MOTION_TRIANGLES_DATA_NV
  nvvk::ValidationSettings validationLayer{/*.message_id_filter = {0xf69d66f5}*/};

  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  // #NV_Motion_blur
  VkPhysicalDeviceRayTracingMotionBlurFeaturesNV rtMotionBlurFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV};

  // Configure Vulkan context creation
  nvvk::ContextInitInfo vkSetup;
  vkSetup.instanceCreateInfoExt = validationLayer.buildPNextChain();
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature});  // To build acceleration structures
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature});  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});  // Required by ray tracing pipeline
  vkSetup.deviceExtensions.push_back({VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME, &rtMotionBlurFeatures});  // Required for motion blur


  // Create the Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }


  appInfo.name           = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();


  // Exit nicely if the extension isn't present
  if(rtMotionBlurFeatures.rayTracingMotionBlur == VK_FALSE)
  {
    LOGE("VK_NV_ray_tracing_motion_blur is not present");
    exit(0);
  }

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
  app.addElement(std::make_shared<RtMotionBlur>());                                      // Sample

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
