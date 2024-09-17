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

  This sample raytrace a scene and uses the built-in functions to retrieve the positions of each triangle and compute the geometric normal.. 
  - Minimizing data uploaded to the GPU
  - Rough visualization


*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#include "common/vk_context.hpp"                    // Vulkan context creation
#include "imgui/imgui_camera_widget.h"              // Camera UI
#include "imgui/imgui_helper.h"                     // Property Editor
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
#include "nvvkhl/shaders/dh_sky.h"                  // Sun & Sky

namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
#include "shaders/dh_bindings.h"
}  // namespace DH


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

#include "teapot_tris.h"

#define MAXRAYRECURSIONDEPTH 5

using TriangulatedMesh = std::vector<glm::vec3>;

//////////////////////////////////////////////////////////////////////////
/// </summary> Ray trace multiple primitives
class RaytracePositionFetch : public nvvkhl::IAppElement
{
public:
  RaytracePositionFetch()           = default;
  ~RaytracePositionFetch() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
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

    VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feat2.pNext = &m_rtPosFetch;
    vkGetPhysicalDeviceFeatures2(m_app->getPhysicalDevice(), &feat2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    const uint32_t gct_queue_index = m_app->getQueue(0).familyIndex;
    m_sbt.setup(m_device, gct_queue_index, m_alloc.get(), m_rtProperties);

    // Create resources
    createScene();
    createVkBuffers();
    createBottomLevelAS();
    createTopLevelAS();
    createRtxPipeline();

    // Since geometry is in the BLAS, no need to keep the buffers
    destroyGeometries();
  }

  void destroyGeometries()
  {
    // Destroy all GPU Geometry
    for(PrimitiveMeshVk& m : m_bMeshes)
    {
      m_alloc->destroy(m.vertices);
      m = {};
    }
  }

  void onDetach() override { destroyResources(); }

  void onResize(uint32_t width, uint32_t height) override
  {
    createGbuffers({width, height});
    writeRtDesc();
  }

  void onUIRender() override
  {
    if(m_rtPosFetch.rayTracingPositionFetch == VK_FALSE)
    {
      ImGui::Begin("Viewport");
      ImGui::TextColored({1, 0, 0, 1}, "Ray Tracing Position Fetch is not supported");
      ImGui::End();
      return;
    }

    {  // Setting menu
      ImGui::Begin("Settings");
      ImGuiH::CameraWidget();

      using namespace ImGuiH;
      PropertyEditor::begin();
      PropertyEditor::entry("Metallic", [&] { return ImGui::SliderFloat("#1", &m_pushConst.metallic, 0.0F, 1.0F); });
      PropertyEditor::entry("Roughness", [&] { return ImGui::SliderFloat("#1", &m_pushConst.roughness, 0.0F, 1.0F); });
      PropertyEditor::entry("Intensity", [&] { return ImGui::SliderFloat("#1", &m_pushConst.intensity, 0.0F, 10.0F); });
      PropertyEditor::entry("Depth",
                            [&] { return ImGui::SliderInt("#1", &m_pushConst.maxDepth, 0, MAXRAYRECURSIONDEPTH); });
      PropertyEditor::end();
      ImGui::Separator();
      ImGui::Text("Sun Orientation");
      PropertyEditor::begin();
      glm::vec3 dir = m_skyParams.directionToLight;
      ImGuiH::azimuthElevationSliders(dir, false);
      m_skyParams.directionToLight = dir;
      PropertyEditor::end();
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    const float view_aspect_ratio = m_viewSize.x / m_viewSize.y;
    glm::vec3   eye;
    glm::vec3   center;
    glm::vec3   up;
    CameraManip.getLookat(eye, center, up);

    // Update Frame buffer uniform buffer
    DH::FrameInfo    finfo{};
    const glm::vec2& clip = CameraManip.getClipPlanes();
    finfo.view            = CameraManip.getMatrix();
    finfo.proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), view_aspect_ratio, clip.x, clip.y);
    finfo.proj[1][1] *= -1;
    finfo.projInv = glm::inverse(finfo.proj);
    finfo.viewInv = glm::inverse(finfo.view);
    finfo.camPos  = eye;
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);

    // Update the sky
    vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(nvvkhl_shaders::SimpleSkyParameters), &m_skyParams);

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
    // Adding Teapot
    m_meshes.emplace_back(triangulatedTeapot);
    m_materials.push_back({{0.5F, 0.5F, 0.5F, 1.0F}});

    // Teapot instance
    {
      nvh::Node& n  = m_nodes.emplace_back();
      n.mesh        = 0;
      n.material    = 0;
      n.translation = {0.0F, 0.F, 0.F};
    }

    // Adding Plane
    {
      TriangulatedMesh planeMesh;
      const float      planeSize = 50.0F;
      planeMesh.push_back({-planeSize, 0, planeSize});
      planeMesh.push_back({planeSize, 0, planeSize});
      planeMesh.push_back({-planeSize, 0, -planeSize});
      planeMesh.push_back({planeSize, 0, planeSize});
      planeMesh.push_back({planeSize, 0, -planeSize});
      planeMesh.push_back({-planeSize, 0, -planeSize});
      m_meshes.emplace_back(planeMesh);
    }
    m_materials.push_back({{0.6F, 0.6F, 0.6F, 1.0F}});
    // Plane Instance
    {
      nvh::Node& n  = m_nodes.emplace_back();
      n.mesh        = static_cast<int>(m_meshes.size()) - 1;
      n.material    = static_cast<int>(m_materials.size()) - 1;
      n.translation = {0, -1, 0};
    }

    // Setting camera to see the scene
    CameraManip.setClipPlanes({0.1F, 100.0F});
    CameraManip.setLookat({-2.0F, 4.0F, 8.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});

    // Default parameters for overall material
    m_pushConst.intensity = 5.0F;
    m_pushConst.maxDepth  = 1;
    m_pushConst.metallic  = 0.2F;
    m_pushConst.roughness = 1.0F;

    // Default Sky values
    m_skyParams = nvvkhl_shaders::initSimpleSkyParameters();
  }


  void createGbuffers(const glm::vec2& size)
  {
    // Rendering image targets
    m_viewSize = size;
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(),
                                                   VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)},
                                                   m_colorFormat, m_depthFormat);
  }

  // Create all Vulkan buffer data
  void createVkBuffers()
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_bMeshes.resize(m_meshes.size());

    const VkBufferUsageFlags rt_usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                             | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex per mesh
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& m = m_bMeshes[i];
      m.vertices         = m_alloc->createBuffer(cmd, m_meshes[i], rt_usage_flag);
      m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
    }

    // Create the buffer of the current frame, changing at each frame
    m_bFrameInfo = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    // Create the buffer of sky parameters, updated at each frame
    m_bSkyParams = m_alloc->createBuffer(sizeof(nvvkhl_shaders::SimpleSkyParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bSkyParams.buffer);

    // Primitive instance information
    std::vector<DH::InstanceInfo> inst_info;
    for(const nvh::Node& node : m_nodes)
    {
      DH::InstanceInfo info{};
      info.materialID = node.material;
      inst_info.emplace_back(info);
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
  nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const TriangulatedMesh& prim, VkDeviceAddress vertexAddress)
  {
    const auto max_primitive_count = static_cast<uint32_t>(prim.size() / 3);

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride             = sizeof(glm::vec3);
    // No indices, all 3 vertex is a triangle
    triangles.indexType = VK_INDEX_TYPE_NONE_NV;
    //triangles.indexData.deviceAddress  = indexAddress;
    triangles.maxVertex = static_cast<uint32_t>(prim.size()) - 1;

    nvvk::AccelerationStructureGeometryInfo result;
    result.geometry = VkAccelerationStructureGeometryKHR{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry     = {triangles},
        .flags        = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR,
    };

    result.rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = max_primitive_count};

    return result;
  }

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  void createBottomLevelAS()
  {
    // BLAS - Storing each primitive in a geometry
    std::vector<nvvk::AccelerationStructureBuildData> buildData(m_meshes.size());
    m_blas.resize(m_meshes.size());
    VkDeviceSize maxScratchSize = 0;
    // #FETCH
    VkBuildAccelerationStructureFlagsKHR flags =
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      const VkDeviceAddress vertex_address = m_bMeshes[p_idx].vertices.address;

      const nvvk::AccelerationStructureGeometryInfo geo = primitiveToGeometry(m_meshes[p_idx], vertex_address);
      buildData[p_idx].asType                           = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      buildData[p_idx].addGeometry(geo);
      auto sizeInfo  = buildData[p_idx].finalizeGeometry(m_device, flags);
      maxScratchSize = std::max(maxScratchSize, sizeInfo.buildScratchSize);
    }
    // Scratch Buffer
    nvvk::Buffer scratchBuffer =
        m_alloc->createBuffer(maxScratchSize, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      auto createInfo = buildData[p_idx].makeCreateInfo();
      m_blas[p_idx]   = m_alloc->createAcceleration(createInfo);
      buildData[p_idx].cmdBuildAccelerationStructure(cmd, m_blas[p_idx].accel, scratchBuffer.address);
    }

    m_app->submitAndWaitTempCmdBuffer(cmd);

    m_alloc->destroy(scratchBuffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    nvvk::AccelerationStructureBuildData            tlasBuildData{VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(m_nodes.size());
    for(const nvh::Node& node : m_nodes)
    {
      const VkGeometryInstanceFlagsKHR flags{VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV};

      VkAccelerationStructureInstanceKHR ray_inst{};
      ray_inst.transform           = nvvk::toTransformMatrixKHR(node.localMatrix());  // Position of the instance
      ray_inst.instanceCustomIndex = node.mesh;                                       // gl_InstanceCustomIndexEXT
      ray_inst.accelerationStructureReference         = m_blas[node.mesh].address;
      ray_inst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
      ray_inst.flags                                  = flags;
      ray_inst.mask                                   = 0xFF;
      tlasInstances.emplace_back(ray_inst);
    }

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();

    nvvk::Buffer instanceBuffer = m_alloc->createBuffer(cmd, tlasInstances,
                                                        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    auto geo = tlasBuildData.makeInstanceGeometry(tlasInstances.size(), instanceBuffer.address);
    tlasBuildData.addGeometry(geo);
    auto sizeInfo = tlasBuildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    nvvk::Buffer scratchBuffer =
        m_alloc->createBuffer(sizeInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);


    m_tlas = m_alloc->createAcceleration(tlasBuildData.makeCreateInfo());
    tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, scratchBuffer.address);

    m_app->submitAndWaitTempCmdBuffer(cmd);

    m_alloc->destroy(scratchBuffer);
    m_alloc->destroy(instanceBuffer);
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    nvvkhl::PipelineContainer& p = m_rtPipe;
    p.plines.resize(1);

    // This descriptor set, holds the top level acceleration structure and the output image
    // Create Binding Set
    m_rtSet->addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_sceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_skyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_materials, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_instances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
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

#if(USE_SLANG)
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
    m_dutil->setObjectName(shaderModule, "Raygen");
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
    m_dutil->setObjectName(stages[eRaygen].module, "Raygen");
    m_dutil->setObjectName(stages[eMiss].module, "Miss");
    m_dutil->setObjectName(stages[eClosestHit].module, "Closest Hit");
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
    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr, &p.plines[0]);
    m_dutil->DBG_NAME(p.plines[0]);

    // Creating the SBT
    m_sbt.create(p.plines[0], ray_pipeline_info);

    // Removing temp modules
#if(USE_SLANG)
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
    const VkDescriptorBufferInfo dbi_sky{m_bSkyParams.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo mat_desc{m_bMaterials.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo inst_desc{m_bInstInfoBuffer.buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtSet->makeWrite(0, B_tlas, &desc_as_info));
    writes.emplace_back(m_rtSet->makeWrite(0, B_outImage, &image_info));
    writes.emplace_back(m_rtSet->makeWrite(0, B_frameInfo, &dbi_unif));
    writes.emplace_back(m_rtSet->makeWrite(0, B_skyParam, &dbi_sky));
    writes.emplace_back(m_rtSet->makeWrite(0, B_materials, &mat_desc));
    writes.emplace_back(m_rtSet->makeWrite(0, B_instances, &inst_desc));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void destroyResources()
  {
    vkDeviceWaitIdle(m_device);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    // destroyGeometries(); // already done
    m_alloc->destroy(m_bFrameInfo);
    m_alloc->destroy(m_bSceneDesc);
    m_alloc->destroy(m_bInstInfoBuffer);
    m_alloc->destroy(m_bMaterials);
    m_alloc->destroy(m_bSkyParams);

    m_rtSet->deinit();
    m_gBuffers.reset();

    m_rtPipe.destroy(m_device);

    m_sbt.destroy();

    for(auto& b : m_blas)
      m_alloc->destroy(b);
    m_alloc->destroy(m_tlas);
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*                          m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>              m_dutil;
  std::unique_ptr<nvvkhl::AllocVma>             m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtSet;  // Descriptor set

  glm::vec2                        m_viewSize    = {1, 1};
  VkFormat                         m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth
  nvvkhl_shaders::SimpleSkyParameters m_skyParams{};

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
  };
  std::vector<PrimitiveMeshVk> m_bMeshes;
  nvvk::Buffer                 m_bFrameInfo;
  nvvk::Buffer                 m_bSceneDesc;  // SceneDescription
  nvvk::Buffer                 m_bInstInfoBuffer;
  nvvk::Buffer                 m_bMaterials;
  nvvk::Buffer                 m_bSkyParams;

  std::vector<VkSampler> m_samplers;

  // Data and setting
  struct Material
  {
    glm::vec4 color{1.F};
  };
  std::vector<TriangulatedMesh> m_meshes;
  std::vector<nvh::Node>        m_nodes;
  std::vector<Material>         m_materials;

  // Pipeline
  DH::PushConstant m_pushConst{};                        // Information sent to the shader
  VkPipelineLayout m_pipelineLayout   = VK_NULL_HANDLE;  // The description of the pipeline
  VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR m_rtPosFetch{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};
  nvvk::SBTWrapper          m_sbt;  // Shading binding table wrapper
  nvvkhl::PipelineContainer m_rtPipe;

  std::vector<nvvk::AccelKHR> m_blas;
  nvvk::AccelKHR              m_tlas;
};
//////////////////////////////////////////////////////////////////////////
///
///
///
int main(int argc, char** argv)
{
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR fetchFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};

  VkContextSettings vkSetup;
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accel_feature});  // To build acceleration structures
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rt_pipeline_feature});  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});  // Required by ray tracing pipeline
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, &fetchFeatures});  // #FETCH
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature});

  // Create Vulkan context
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  if(!vkContext->isValid())
    std::exit(0);

  // Loading the Vulkan extension pointers
  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);

  nvvkhl::ApplicationCreateInfo appSetup;
  appSetup.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appSetup.vSync          = true;
  appSetup.instance       = vkContext->getInstance();
  appSetup.device         = vkContext->getDevice();
  appSetup.physicalDevice = vkContext->getPhysicalDevice();
  appSetup.queues         = vkContext->getQueueInfos();


  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(appSetup);

  if(fetchFeatures.rayTracingPositionFetch == VK_FALSE)
  {
    LOGE("ERROR: Position Fetch not supported");
    exit(1);
  }

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app->addElement(std::make_shared<RaytracePositionFetch>());

  app->run();
  app.reset();
  vkContext.reset();

  return test->errorCode();
}
