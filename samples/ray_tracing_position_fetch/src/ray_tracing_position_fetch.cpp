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

  This sample raytrace a scene and uses the built-in functions to retrieve the positions of each triangle and compute the geometric normal.. 
  - Minimizing data uploaded to the GPU
  - Rough visualization


*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvh/primitives.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"

#include "shaders/dh_bindings.h"
#include "shaders/device_host.h"
#include "nvvkhl/shaders/dh_sky.h"

#if USE_HLSL
#include "_autogen/raytrace_rgenMain.spirv.h"
#include "_autogen/raytrace_rchitMain.spirv.h"
#include "_autogen/raytrace_rmissMain.spirv.h"
const auto& rgen_shd  = std::vector<char>{std::begin(raytrace_rgenMain), std::end(raytrace_rgenMain)};
const auto& rchit_shd = std::vector<char>{std::begin(raytrace_rchitMain), std::end(raytrace_rchitMain)};
const auto& rmiss_shd = std::vector<char>{std::begin(raytrace_rmissMain), std::end(raytrace_rmissMain)};
#elif USE_SLANG
#include "_autogen/raytrace_rgenMain.spirv.h"
#include "_autogen/raytrace_rchitMain.spirv.h"
#include "_autogen/raytrace_rmissMain.spirv.h"
const auto& rgen_shd  = std::vector<uint32_t>{std::begin(raytrace_rgenMain), std::end(raytrace_rgenMain)};
const auto& rchit_shd = std::vector<uint32_t>{std::begin(raytrace_rchitMain), std::end(raytrace_rchitMain)};
const auto& rmiss_shd = std::vector<uint32_t>{std::begin(raytrace_rmissMain), std::end(raytrace_rmissMain)};
#else
#include "_autogen/raytrace.rchit.h"
#include "_autogen/raytrace.rgen.h"
#include "_autogen/raytrace.rmiss.h"
const auto& rgen_shd  = std::vector<uint32_t>{std::begin(raytrace_rgen), std::end(raytrace_rgen)};
const auto& rchit_shd = std::vector<uint32_t>{std::begin(raytrace_rchit), std::end(raytrace_rchit)};
const auto& rmiss_shd = std::vector<uint32_t>{std::begin(raytrace_rmiss), std::end(raytrace_rmiss)};
#endif

#include "teapot_tris.h"

#define MAXRAYRECURSIONDEPTH 5

using TriangulatedMesh = std::vector<nvmath::vec3f>;

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

    m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);                    // Debug utility
    m_alloc = std::make_unique<nvvkhl::AllocVma>(m_app->getContext().get());  // Allocator
    m_rtSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    // Requesting ray tracing properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    prop2.pNext = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feat2.pNext = &m_rtPosFetch;
    vkGetPhysicalDeviceFeatures2(m_app->getPhysicalDevice(), &feat2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    const uint32_t gct_queue_index = m_app->getContext()->m_queueGCT.familyIndex;
    m_rtBuilder.setup(m_device, m_alloc.get(), gct_queue_index);
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
      nvmath::vec3f dir = m_skyParams.directionToLight;
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

    const float   view_aspect_ratio = m_viewSize.x / m_viewSize.y;
    nvmath::vec3f eye;
    nvmath::vec3f center;
    nvmath::vec3f up;
    CameraManip.getLookat(eye, center, up);

    // Update Frame buffer uniform buffer
    FrameInfo            finfo{};
    const nvmath::vec2f& clip = CameraManip.getClipPlanes();
    finfo.view                = CameraManip.getMatrix();
    finfo.proj                = nvmath::perspectiveVK(CameraManip.getFov(), view_aspect_ratio, clip.x, clip.y);
    finfo.projInv             = nvmath::inverse(finfo.proj);
    finfo.viewInv             = nvmath::inverse(finfo.view);
    finfo.camPos              = eye;
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(FrameInfo), &finfo);

    // Update the sky
    vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(ProceduralSkyShaderParameters), &m_skyParams);

    // Ray trace
    std::vector<VkDescriptorSet> desc_sets{m_rtSet->getSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.plines[0]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.layout, 0, (uint32_t)desc_sets.size(),
                            desc_sets.data(), 0, nullptr);
    vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant), &m_pushConst);

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
    m_pushConst.maxDepth  = 5;
    m_pushConst.roughness = 1.0F;
    m_pushConst.metallic  = 0.5F;

    // Default Sky values
    m_skyParams = initSkyShaderParameters();
  }


  void createGbuffers(const nvmath::vec2f& size)
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
    m_bFrameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    // Create the buffer of sky parameters, updated at each frame
    m_bSkyParams = m_alloc->createBuffer(sizeof(ProceduralSkyShaderParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bSkyParams.buffer);

    // Primitive instance information
    std::vector<InstanceInfo> inst_info;
    for(const nvh::Node& node : m_nodes)
    {
      InstanceInfo info{};
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
  nvvk::RaytracingBuilderKHR::BlasInput primitiveToGeometry(const TriangulatedMesh& prim, VkDeviceAddress vertexAddress)
  {
    const auto max_primitive_count = static_cast<uint32_t>(prim.size() / 3);

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat             = VK_FORMAT_R32G32B32A32_SFLOAT;  // vec3 vertex position data.
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride             = sizeof(nvmath::vec3f);
    // No indices, all 3 vertex is a triangle
    //triangles.indexType                = VK_INDEX_TYPE_UINT32;
    //triangles.indexData.deviceAddress  = indexAddress;
    triangles.maxVertex = static_cast<uint32_t>(prim.size()) - 1;

    // Identify the above data as containing opaque triangles.
    VkAccelerationStructureGeometryKHR as_geom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    as_geom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    as_geom.flags              = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
    as_geom.geometry.triangles = triangles;

    VkAccelerationStructureBuildRangeInfoKHR offset{};
    offset.firstVertex     = 0;
    offset.primitiveCount  = max_primitive_count;
    offset.primitiveOffset = 0;
    offset.transformOffset = 0;

    // Our BLAS is made from only one geometry, but could be made of many geometries
    nvvk::RaytracingBuilderKHR::BlasInput input;
    input.asGeometry.emplace_back(as_geom);
    input.asBuildOffsetInfo.emplace_back(offset);

    return input;
  }

  //--------------------------------------------------------------------------------------------------
  // Create all bottom level acceleration structures (BLAS)
  //
  void createBottomLevelAS()
  {
    // BLAS - Storing each primitive in a geometry
    std::vector<nvvk::RaytracingBuilderKHR::BlasInput> all_blas;
    all_blas.reserve(m_meshes.size());

    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      const VkDeviceAddress vertex_address = nvvk::getBufferDeviceAddress(m_device, m_bMeshes[p_idx].vertices.buffer);

      const nvvk::RaytracingBuilderKHR::BlasInput geo = primitiveToGeometry(m_meshes[p_idx], vertex_address);
      all_blas.push_back({geo});
    }

    // #FETCH
    VkBuildAccelerationStructureFlagsKHR flags =
        VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    m_rtBuilder.buildBlas(all_blas, flags);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    std::vector<VkAccelerationStructureInstanceKHR> tlas;
    tlas.reserve(m_nodes.size());
    for(const nvh::Node& node : m_nodes)
    {
      const VkGeometryInstanceFlagsKHR flags{VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV};

      VkAccelerationStructureInstanceKHR ray_inst{};
      ray_inst.transform           = nvvk::toTransformMatrixKHR(node.localMatrix());  // Position of the instance
      ray_inst.instanceCustomIndex = node.mesh;                                       // gl_InstanceCustomIndexEXT
      ray_inst.accelerationStructureReference         = m_rtBuilder.getBlasDeviceAddress(node.mesh);
      ray_inst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
      ray_inst.flags                                  = flags;
      ray_inst.mask                                   = 0xFF;
      tlas.emplace_back(ray_inst);
    }
    m_rtBuilder.buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
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
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.pName = "main";  // All the same entry point
    // Raygen
    stage.module    = nvvk::createShaderModule(m_device, rgen_shd);
    stage.pName     = USE_HLSL ? "rgenMain" : "main";
    stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eRaygen] = stage;
    m_dutil->setObjectName(stage.module, "Raygen");
    // Miss
    stage.module  = nvvk::createShaderModule(m_device, rmiss_shd);
    stage.pName   = USE_HLSL ? "rmissMain" : "main";
    stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eMiss] = stage;
    m_dutil->setObjectName(stage.module, "Miss");
    // Hit Group - Closest Hit
    stage.module        = nvvk::createShaderModule(m_device, rchit_shd);
    stage.pName         = USE_HLSL ? "rchitMain" : "main";
    stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[eClosestHit] = stage;
    m_dutil->setObjectName(stage.module, "Closest Hit");


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
    const VkPushConstantRange push_constant{VK_SHADER_STAGE_ALL, 0, sizeof(PushConstant)};

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
    for(const VkPipelineShaderStageCreateInfo& s : stages)
      vkDestroyShaderModule(m_device, s.module, nullptr);
  }

  void writeRtDesc()
  {
    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_rtBuilder.getAccelerationStructure();
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
    m_rtBuilder.destroy();
  }

  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*                          m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>              m_dutil;
  std::unique_ptr<nvvkhl::AllocVma>             m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_rtSet;  // Descriptor set

  nvmath::vec2f                    m_viewSize    = {1, 1};
  VkFormat                         m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue                m_clearColor  = {{0.3F, 0.3F, 0.3F, 1.0F}};     // Clear color
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth
  ProceduralSkyShaderParameters    m_skyParams{};

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
    vec4 color{1.F};
  };
  std::vector<TriangulatedMesh> m_meshes;
  std::vector<nvh::Node>        m_nodes;
  std::vector<Material>         m_materials;

  // Pipeline
  PushConstant     m_pushConst{};                        // Information sent to the shader
  VkPipelineLayout m_pipelineLayout   = VK_NULL_HANDLE;  // The description of the pipeline
  VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
  int              m_frame{0};

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR m_rtPosFetch{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};
  nvvk::SBTWrapper           m_sbt;  // Shading binding table wrapper
  nvvk::RaytracingBuilderKHR m_rtBuilder;
  nvvkhl::PipelineContainer  m_rtPipe;
};
//////////////////////////////////////////////////////////////////////////
///
///
///
int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name             = PROJECT_NAME " Example";
  spec.vSync            = true;
  spec.vkSetup.apiMajor = 1;
  spec.vkSetup.apiMinor = 3;

  spec.vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  // #VKRay: Activate the ray tracing extension
  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &accel_feature);  // To build acceleration structures
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false, &rt_pipeline_feature);  // To use vkCmdTraceRaysKHR
  spec.vkSetup.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR fetchFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, false, &fetchFeatures);  // #FETCH
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayqueryFeature);

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());         // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>());  // Window title info
  app->addElement(std::make_shared<RaytracePositionFetch>());


  app->run();
  app.reset();

  return test->errorCode();
}
