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

#include <array>
#include <vulkan/vulkan_core.h>


#define VMA_IMPLEMENTATION
#include "common/vk_context.hpp"                    // Vulkan context creation
#include "imgui/imgui_camera_widget.h"              // Camera UI
#include "nvh/primitives.hpp"                       // Various primitives
#include "nvvk/acceleration_structures.hpp"         // BLAS & TLAS creation helper
#include "nvvk/descriptorsets_vk.hpp"               // Descriptor set creation helper
#include "nvvk/extensions_vk.hpp"                   // Vulkan extension declaration
#include "nvvk/sbtwrapper_vk.hpp"                   // Shading binding table creation helper
#include "nvvk/shaders_vk.hpp"                      // Shader module creation wrapper
#include "nvvkhl/alloc_vma.hpp"                     // VMA memory allocator
#include "nvvkhl/element_benchmark_parameters.hpp"  // For benchmark and tests
#include "nvvkhl/element_camera.hpp"                // To manipulate the camera
#include "nvvkhl/element_gui.hpp"                   // Application Menu / titlebar
#include "nvvkhl/gbuffer.hpp"                       // G-Buffer creation helper
#include "nvvkhl/pipeline_container.hpp"            // Container to hold pipelines

#include "common/utils.hpp"

namespace DH {
using namespace glm;
#include "shaders/dh_bindings.h"
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

//#undef USE_HLSL

struct MatPrimitiveMesh : public nvh::PrimitiveMesh
{
  std::vector<int> triMat;  // Material per triangle
};


#if USE_HLSL
#include "_autogen/raytrace_rgenMain.spirv.h"
#include "_autogen/raytrace_rchitMain.spirv.h"
#include "_autogen/raytrace_rmissMain.spirv.h"
const auto& rgen_shd  = std::vector<char>{std::begin(raytrace_rgenMain), std::end(raytrace_rgenMain)};
const auto& rchit_shd = std::vector<char>{std::begin(raytrace_rchitMain), std::end(raytrace_rchitMain)};
const auto& rmiss_shd = std::vector<char>{std::begin(raytrace_rmissMain), std::end(raytrace_rmissMain)};
#elif USE_SLANG
#include "_autogen/raytrace_slang.h"
#else
#include "_autogen/raytrace.rchit.glsl.h"
#include "_autogen/raytrace.rgen.glsl.h"
#include "_autogen/raytrace.rmiss.glsl.h"
const auto& rgen_shd  = std::vector<uint32_t>{std::begin(raytrace_rgen_glsl), std::end(raytrace_rgen_glsl)};
const auto& rchit_shd = std::vector<uint32_t>{std::begin(raytrace_rchit_glsl), std::end(raytrace_rchit_glsl)};
const auto& rmiss_shd = std::vector<uint32_t>{std::begin(raytrace_rmiss_glsl), std::end(raytrace_rmiss_glsl)};
#endif


#define MAXRAYRECURSIONDEPTH 5

//////////////////////////////////////////////////////////////////////////
/// </summary> Ray trace multiple primitives
class RtMotionBlur : public nvvkhl::IAppElement
{
public:
  RtMotionBlur()           = default;
  ~RtMotionBlur() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    nvh::ScopedTimer st(__FUNCTION__);

    m_app    = app;
    m_device = m_app->getDevice();

    m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility
    m_alloc = std::make_unique<nvvkhl::AllocVma>(VmaAllocatorCreateInfo{
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    });  // Allocator
    m_rtSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    const uint32_t gct_queue_index = m_app->getQueue(0).familyIndex;
    m_sbt.setup(m_device, gct_queue_index, m_alloc.get(), m_rtProperties);

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

  void onResize(uint32_t width, uint32_t height) override
  {
    nvh::ScopedTimer st(__FUNCTION__);
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), VkExtent2D{width, height}, m_colorFormat, m_depthFormat);
    writeRtDesc();
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");
      ImGuiH::CameraWidget();
      ImGui::ColorEdit3("Clear Color", &m_pushConst.clearColor.x);
      ImGui::SliderInt("Num Samples", &m_pushConst.numSamples, 1, 1000);
      ImGui::SliderFloat3("Light Position", &m_pushConst.lightPosition.x, -10, 10);
      ImGui::SliderFloat("Light Intensity", &m_pushConst.lightIntensity, 0, 200);
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");
      ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());  // Display the G-Buffer image
      ImGui::End();
      ImGui::PopStyleVar();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    ++m_pushConst.frame;

    // Update Frame buffer uniform buffer
    const glm::vec2& clip = CameraManip.getClipPlanes();
    DH::FrameInfo    finfo{};
    finfo.view = CameraManip.getMatrix();
    finfo.proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), m_gBuffers->getAspectRatio(), clip.x, clip.y);
    finfo.proj[1][1] *= -1;
    finfo.projInv = glm::inverse(finfo.proj);
    finfo.viewInv = glm::inverse(finfo.view);
    finfo.camPos  = CameraManip.getEye();
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);
    nvvk::memoryBarrier(cmd);


    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_rtSet->getSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.layout, 0, (uint32_t)desc_sets.size(),
                            desc_sets.data(), 0, nullptr);
    vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant), &m_pushConst);

    const std::array<VkStridedDeviceAddressRegionKHR, 4>& regions = m_sbt.getRegions();
    const VkExtent2D&                                     size    = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, &regions[0], &regions[1], &regions[2], &regions[3], size.width, size.height, 1);
  }

private:
  void createScene()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    // Materials
    m_materials.push_back({{0.7, 0.7, 0.7, 1.0}});                 // gray
    m_materials.push_back({{0.982062, 0.857638, 0.400811, 1.0}});  // yellow
    m_materials.push_back({{0.982062, 0.1, 0.1, 1.0}});            // red
    m_materials.push_back({{0.1, 0.9, 0.1, 1.0}});                 // green
    m_materials.push_back({{0.1, 0.1, 0.982062, 1.0}});            // blue
    m_materials.push_back({{0.982062, 0.1, 0.982062, 1.0}});       // magenta

    // Meshes (cube)
    m_meshes.emplace_back(nvh::createCube());
    m_meshes.emplace_back(nvh::createPlane(10, 100, 100));
    m_meshes[0].triMat = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};        // Cube
    m_meshes.back().triMat.resize(m_meshes.back().triangles.size());  // Plane all tri to gray
    // Modified Cube: one vertex is moved, this will be used to have motion between Cube and the Modified Cube
    m_meshes.emplace_back(m_meshes[0]);   // Copy of cube
    m_meshes.back().vertices[6].p *= 2;   // Modifying the +x,+y,+z position (appearing 3 times)
    m_meshes.back().vertices[11].p *= 2;  // Modifying the +x,+y,+z position
    m_meshes.back().vertices[22].p *= 2;  // Modifying the +x,+y,+z position

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
    CameraManip.setClipPlanes({0.1F, 100.0F});
    CameraManip.setLookat({3.91698, 2.65970, -0.42755}, {0.71716, 0.03205, 1.36345}, {0.00000, 1.00000, 0.00000});

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
    nvh::ScopedTimer st(__FUNCTION__);

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_bMeshes.resize(m_meshes.size());

    const VkBufferUsageFlags rt_usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                             | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex and Index per mesh
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& m = m_bMeshes[i];
      m.vertices         = m_alloc->createBuffer(cmd, m_meshes[i].vertices, rt_usage_flag);
      m.indices          = m_alloc->createBuffer(cmd, m_meshes[i].triangles, rt_usage_flag);
      m.triMaterial      = m_alloc->createBuffer(cmd, m_meshes[i].triMat, rt_usage_flag);
      m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
      m_dutil->DBG_NAME_IDX(m.indices.buffer, i);
      m_dutil->DBG_NAME_IDX(m.triMaterial.buffer, i);
    }

    // Create the buffer of the current frame, changing at each frame
    m_bFrameInfo = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    // Primitive instance information
    std::vector<DH::InstanceInfo> inst_info;
    inst_info.reserve(m_instances.size());
    for(const nvh::Node& node : m_instances)
    {
      DH::InstanceInfo info{};
      info.transform = node.localMatrix();
      inst_info.push_back(info);
    }
    m_bInstInfoBuffer =
        m_alloc->createBuffer(cmd, inst_info, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bInstInfoBuffer.buffer);

    m_bMaterials = m_alloc->createBuffer(cmd, m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bMaterials.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
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
    triangles.vertexStride             = sizeof(nvh::PrimitiveVertex);
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
    nvh::ScopedTimer st(__FUNCTION__);

    std::vector<nvvk::AccelerationStructureBuildData> blasBuildData;

    // BLAS - Storing each primitive in a geometry
    //std::vector<nvvk::RaytracingBuilderKHR::BlasInput> all_blas;
    //all_blas.reserve(m_meshes.size());
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
      //all_blas.push_back({geo});
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
      // Add the modified geometry
      //
      //geo.asGeometry[0].geometry.triangles.pNext = &motionTriangles;
      //geo.flags                                  = VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV;
      //all_blas.push_back({geo});
    }

    nvvk::Buffer scratchBuffer =
        m_alloc->createBuffer(maxScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    // Create the acceleration structures
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    {
      for(size_t p_idx = 0; p_idx < blasBuildData.size(); p_idx++)
      {
        m_blas[p_idx] = m_alloc->createAcceleration(blasBuildData[p_idx].makeCreateInfo());
        blasBuildData[p_idx].cmdBuildAccelerationStructure(cmd, m_blas[p_idx].accel, scratchBuffer.address);
      }
    }
    m_app->submitAndWaitTempCmdBuffer(cmd);


    m_alloc->destroy(scratchBuffer);
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
    nvh::ScopedTimer st(__FUNCTION__);

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

      VkAccelerationStructureMatrixMotionInstanceNV data;
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

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    // Create the instances buffer, add a barrier to ensure the data is copied before the TLAS build
    nvvk::Buffer instancesBuffer = m_alloc->createBuffer(cmd, tlas,
                                                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                                                             | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);


    nvvk::AccelerationStructureGeometryInfo geometryInfo =
        tlasBuildData.makeInstanceGeometry(tlas.size(), instancesBuffer.address);
    tlasBuildData.addGeometry(geometryInfo);
    // Get the size of the TLAS
    auto sizeInfo = tlasBuildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_MOTION_BIT_NV);

    // Create the scratch buffer
    nvvk::Buffer scratchBuffer = m_alloc->createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                                                      | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    // Create the TLAS with motionblur support
    VkAccelerationStructureCreateInfoKHR createInfo = tlasBuildData.makeCreateInfo();
#ifdef VK_NV_ray_tracing_motion_blur
    VkAccelerationStructureMotionInfoNV motionInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_MOTION_INFO_NV};
    motionInfo.maxInstances = uint32_t(tlas.size());
    createInfo.createFlags  = VK_ACCELERATION_STRUCTURE_CREATE_MOTION_BIT_NV;
    createInfo.pNext        = &motionInfo;
#endif

    m_tlas = m_alloc->createAcceleration(createInfo);
    tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, scratchBuffer.address);
    m_app->submitAndWaitTempCmdBuffer(cmd);

    m_alloc->destroy(scratchBuffer);
    m_alloc->destroy(instancesBuffer);
    m_alloc->finalizeAndReleaseStaging();
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    nvh::ScopedTimer st(__FUNCTION__);

    nvvkhl::PipelineContainer& p = m_rtPipe;
    p.plines.resize(1);

    // This descriptor set, holds the top level acceleration structure and the output image
    // Create Binding Set
    m_rtSet->addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_materials, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_instances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_vertex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_triMat, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
    m_rtSet->initLayout();
    m_rtSet->initPool(1);

    m_dutil->DBG_NAME(m_rtSet->getLayout());
    m_dutil->DBG_NAME(m_rtSet->getSet(0));

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
    VkShaderModule shaderModule = nvvk::createShaderModule(m_device, &raytraceSlang[0], sizeof(raytraceSlang));
    stages[eRaygen].module      = shaderModule;
    stages[eRaygen].pName       = "rgenMain";
    stages[eRaygen].stage       = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eMiss].module        = shaderModule;
    stages[eMiss].pName         = "rmissMain";
    stages[eMiss].stage         = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eClosestHit].module  = shaderModule;
    stages[eClosestHit].pName   = "rchitMain";
    stages[eClosestHit].stage   = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
#else
    stages[eRaygen].module     = nvvk::createShaderModule(m_device, rgen_shd);
    stages[eRaygen].pName      = USE_HLSL ? "rgenMain" : "main";
    stages[eRaygen].stage      = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eMiss].module       = nvvk::createShaderModule(m_device, rmiss_shd);
    stages[eMiss].pName        = USE_HLSL ? "rmissMain" : "main";
    stages[eMiss].stage        = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eClosestHit].module = nvvk::createShaderModule(m_device, rchit_shd);
    stages[eClosestHit].pName  = USE_HLSL ? "rchitMain" : "main";
    stages[eClosestHit].stage  = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
#endif

    m_dutil->setObjectName(stages[eRaygen].module, "Raygen");
    m_dutil->setObjectName(stages[eMiss].module, "Miss");
    m_dutil->setObjectName(stages[eClosestHit].module, "Closest Hit");


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
    const VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant)};

    VkPipelineLayoutCreateInfo pipeline_layout_create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_create_info.pushConstantRangeCount = 1;
    pipeline_layout_create_info.pPushConstantRanges    = &push_constant;

    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::vector<VkDescriptorSetLayout> rt_desc_set_layouts = {m_rtSet->getLayout()};  // , m_pContainer[eGraphic].dstLayout};
    pipeline_layout_create_info.setLayoutCount = static_cast<uint32_t>(rt_desc_set_layouts.size());
    pipeline_layout_create_info.pSetLayouts    = rt_desc_set_layouts.data();
    vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &p.layout);
    m_dutil->DBG_NAME(p.layout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    ray_pipeline_info.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    ray_pipeline_info.pStages                      = stages.data();
    ray_pipeline_info.groupCount                   = static_cast<uint32_t>(shader_groups.size());
    ray_pipeline_info.pGroups                      = shader_groups.data();
    ray_pipeline_info.maxPipelineRayRecursionDepth = MAXRAYRECURSIONDEPTH;  // Ray depth
    ray_pipeline_info.layout                       = p.layout;
    ray_pipeline_info.flags = VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV;  // #NV_Motion_blur
    NVVK_CHECK(vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, &p.plines[0]));
    m_dutil->DBG_NAME(p.plines[0]);

    // Creating the SBT
    m_sbt.create(p.plines[0], ray_pipeline_info);

    // Removing temp modules
#if USE_SLANG
    vkDestroyShaderModule(m_device, shaderModule, nullptr);
#else
    for(const VkPipelineShaderStageCreateInfo& s : stages)
      vkDestroyShaderModule(m_device, s.module, nullptr);
#endif
  }

  void writeRtDesc()
  {
    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_tlas.accel;
    VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    desc_as_info.accelerationStructureCount = 1;
    desc_as_info.pAccelerationStructures    = &tlas;
    const VkDescriptorImageInfo  image_info{{}, m_gBuffers->getColorImageView(), VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo mat_desc{m_bMaterials.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo inst_desc{m_bInstInfoBuffer.buffer, 0, VK_WHOLE_SIZE};

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

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtSet->makeWrite(0, B_tlas, &desc_as_info));
    writes.emplace_back(m_rtSet->makeWrite(0, B_outImage, &image_info));
    writes.emplace_back(m_rtSet->makeWrite(0, B_frameInfo, &dbi_unif));
    writes.emplace_back(m_rtSet->makeWrite(0, B_materials, &mat_desc));
    writes.emplace_back(m_rtSet->makeWrite(0, B_instances, &inst_desc));
    writes.emplace_back(m_rtSet->makeWriteArray(0, B_vertex, vertex_desc.data()));
    writes.emplace_back(m_rtSet->makeWriteArray(0, B_index, index_desc.data()));
    writes.emplace_back(m_rtSet->makeWriteArray(0, B_triMat, trimat_desc.data()));

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void destroyResources()
  {
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    for(PrimitiveMeshVk& m : m_bMeshes)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
      m_alloc->destroy(m.triMaterial);
    }
    m_alloc->destroy(m_bFrameInfo);
    m_alloc->destroy(m_bInstInfoBuffer);
    m_alloc->destroy(m_bMaterials);

    m_rtSet->deinit();
    m_gBuffers.reset();

    m_rtPipe.destroy(m_device);

    m_sbt.destroy();

    for(auto& b : m_blas)
      m_alloc->destroy(b);
    m_alloc->destroy(m_tlas);
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers->getColorImage(), m_gBuffers->getSize(),
                           nvh::getExecutablePath().replace_extension(".jpg").string(), 95);
  }


  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*                          m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>              m_dutil;
  std::unique_ptr<nvvkhl::AllocVma>             m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtSet;  // Descriptor set

  VkFormat                         m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth

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
  std::vector<nvh::Node>        m_instances;
  std::vector<Material>         m_materials;

  // Pipeline
  DH::PushConstant m_pushConst{};                        // Information sent to the shader
  VkPipelineLayout m_pipelineLayout   = VK_NULL_HANDLE;  // The description of the pipeline
  VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::SBTWrapper          m_sbt;  // Shading binding table wrapper
  nvvkhl::PipelineContainer m_rtPipe;

  std::vector<nvvk::AccelKHR> m_blas;
  nvvk::AccelKHR              m_tlas;
};


//////////////////////////////////////////////////////////////////////////
///
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo appInfo;

  nvh::CommandLineParser cli(PROJECT_NAME);
  cli.addArgument({"--headless"}, &appInfo.headless, "Run in headless mode");
  cli.addArgument({"--frames"}, &appInfo.headlessFrameCount, "Number of frames to render in headless mode");
  cli.parse(argc, argv);

  // #NV_Motion_blur
  // Validation Layer, filtering message: unexpected VkStructureType VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_MOTION_TRIANGLES_DATA_NV
  ValidationSettings validationLayer{.message_id_filter = {0xf69d66f5}};

  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  // #NV_Motion_blur
  VkPhysicalDeviceRayTracingMotionBlurFeaturesNV rtMotionBlurFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV};

  // Configure Vulkan context creation
  VkContextSettings vkSetup;
  vkSetup.instanceCreateInfoExt = validationLayer.buildPNextChain();
  if(!appInfo.headless)
  {
    nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature});  // To build acceleration structures
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature});  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});  // Required by ray tracing pipeline
  vkSetup.deviceExtensions.push_back({VK_NV_RAY_TRACING_MOTION_BLUR_EXTENSION_NAME, &rtMotionBlurFeatures});  // Required for motion blur

#if USE_HLSL  // DXC is automatically adding the extension
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature});
#endif  // USE_HLSL

  // Create Vulkan context
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  if(!vkContext->isValid())
    std::exit(0);

  // Loading the Vulkan extension pointers
  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);

  appInfo.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext->getInstance();
  appInfo.device         = vkContext->getDevice();
  appInfo.physicalDevice = vkContext->getPhysicalDevice();
  appInfo.queues         = vkContext->getQueueInfos();


  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appInfo);

  // Exit nicely if the extension isn't present
  if(rtMotionBlurFeatures.rayTracingMotionBlur == VK_FALSE)
  {
    app.reset();
    LOGE("VK_NV_ray_tracing_motion_blur is not present");
    exit(0);
  }


  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());              // Camera manipulation
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());         // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>());  // Window title info
  app->addElement(std::make_shared<RtMotionBlur>());                       // Sample

  app->run();
  app.reset();
  vkContext.reset();

  return test->errorCode();
}
