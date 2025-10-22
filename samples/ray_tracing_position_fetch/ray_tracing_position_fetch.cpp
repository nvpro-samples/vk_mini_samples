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

  This sample raytrace a scene and uses the built-in functions to retrieve the positions of each triangle and compute the geometric normal.. 
  - Minimizing data uploaded to the GPU
  - Rough visualization


*/
//////////////////////////////////////////////////////////////////////////

#include <array>
#include <vulkan/vulkan_core.h>

#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION

#include <glm/glm.hpp>

#include "shaders/dh_bindings.h"
#include "shaders/shaderio.h"  // Shared between host and device


#include "_autogen/raytrace_position_fetch.rchit.glsl.h"
#include "_autogen/raytrace_position_fetch.rgen.glsl.h"
#include "_autogen/raytrace_position_fetch.rmiss.glsl.h"
#include "_autogen/raytrace_position_fetch.slang.h"


#include "nvapp/application.hpp"
#include "nvapp/elem_camera.hpp"
#include "nvapp/elem_default_menu.hpp"
#include "nvapp/elem_default_title.hpp"
#include "nvgui/azimuth_sliders.hpp"
#include "nvgui/camera.hpp"
#include "nvgui/property_editor.hpp"
#include "nvgui/sky.hpp"
#include "nvutils/camera_manipulator.hpp"
#include "nvutils/logger.hpp"
#include "nvutils/parameter_parser.hpp"
#include "nvutils/primitives.hpp"
#include "nvvk/check_error.hpp"
#include "nvvk/context.hpp"
#include "nvvk/debug_util.hpp"
#include "nvvk/descriptors.hpp"
#include "nvvk/gbuffers.hpp"
#include "nvvk/resource_allocator.hpp"
#include "nvvk/sampler_pool.hpp"
#include "nvvk/sbt_generator.hpp"

#define MAXRAYRECURSIONDEPTH 5

#include "common/utils.hpp"
#include "nvvk/acceleration_structures.hpp"
#include "nvvk/helpers.hpp"
#include "nvvk/staging.hpp"
#include "teapot_tris.h"
#include <nvvk/formats.hpp>

using TriangulatedMesh = std::vector<glm::vec3>;
std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};


//////////////////////////////////////////////////////////////////////////
/// </summary> Ray trace multiple primitives
class RaytracePositionFetch : public nvapp::IAppElement
{
public:
  RaytracePositionFetch()           = default;
  ~RaytracePositionFetch() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();

    m_alloc.init({
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    });  // Allocator

    // Uploader for staging data to GPU
    m_stagingUploader.init(&m_alloc);

    // The texture sampler to use
    m_samplerPool.init(m_device);
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    m_depthFormat = nvvk::findDepthFormat(m_app->getPhysicalDevice());

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

    VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feat2.pNext = &m_rtPosFetch;
    vkGetPhysicalDeviceFeatures2(m_app->getPhysicalDevice(), &feat2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    m_sbt.init(m_device, m_rtProperties);

    m_asHelper.init(&m_alloc, &m_stagingUploader, m_app->getQueue(0));

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
      m_alloc.destroyBuffer(m.vertices);
      m = {};
    }
  }

  void onDetach() override { destroyResources(); }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) override
  {
    m_gBuffers.update(cmd, size);
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
      nvgui::CameraWidget(g_cameraManip);

      using namespace nvgui;
      PropertyEditor::begin();
      PropertyEditor::SliderFloat("Metallic", &m_pushConst.metallic, 0.0F, 1.0F);
      PropertyEditor::SliderFloat("Roughness", &m_pushConst.roughness, 0.0F, 1.0F);
      PropertyEditor::SliderFloat("Intensity", &m_pushConst.intensity, 0.0F, 10.0F);
      PropertyEditor::SliderInt("Depth", &m_pushConst.maxDepth, 0, MAXRAYRECURSIONDEPTH);
      PropertyEditor::end();
      ImGui::Separator();
      ImGui::Text("Sun Orientation");
      PropertyEditor::begin();
      glm::vec3 dir = m_skyParams.sunDirection;
      nvgui::azimuthElevationSliders(dir, false, true);
      m_skyParams.sunDirection = dir;
      PropertyEditor::end();
      ImGui::End();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);

    // Update Frame buffer uniform buffer
    shaderio::FrameInfo finfo{};
    finfo.view    = g_cameraManip->getViewMatrix();
    finfo.proj    = g_cameraManip->getPerspectiveMatrix();
    finfo.projInv = glm::inverse(finfo.proj);
    finfo.viewInv = glm::inverse(finfo.view);
    finfo.camPos  = g_cameraManip->getEye();
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(shaderio::FrameInfo), &finfo);

    // Update the sky
    vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(shaderio::SkySimpleParameters), &m_skyParams);

    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);


    // Ray trace
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 0, 1,
                            m_descriptorPack.getSetPtr(0), 0, nullptr);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);

    // Ray trace
    const nvvk::SBTGenerator::Regions& regions = m_sbt.getSBTRegions();
    const VkExtent2D&                  size    = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, &regions.raygen, &regions.miss, &regions.hit, &regions.callable, size.width, size.height, 1);
  }

private:
  void createScene()
  {

    // Adding Plane
    {
      TriangulatedMesh planeMesh;
      const float      planeSize = 50.0F;
      planeMesh.emplace_back(-planeSize, 0, planeSize);
      planeMesh.emplace_back(planeSize, 0, planeSize);
      planeMesh.emplace_back(-planeSize, 0, -planeSize);
      planeMesh.emplace_back(planeSize, 0, planeSize);
      planeMesh.emplace_back(planeSize, 0, -planeSize);
      planeMesh.emplace_back(-planeSize, 0, -planeSize);
      m_meshes.emplace_back(planeMesh);
    }
    m_materials.push_back({{0.6F, 0.6F, 0.6F, 1.0F}});
    // Plane Instance
    {
      nvutils::Node& node = m_nodes.emplace_back();
      node.mesh           = static_cast<int>(m_meshes.size()) - 1;
      node.material       = static_cast<int>(m_materials.size()) - 1;
      node.translation    = {0, -1, 0};
    }

    // Adding Teapot
    m_meshes.emplace_back(triangulatedTeapot);
    m_materials.push_back({{0.5F, 0.5F, 0.5F, 1.0F}});

    // Teapot instance
    {
      nvutils::Node& node = m_nodes.emplace_back();
      node.mesh           = static_cast<int>(m_meshes.size()) - 1;
      node.material       = static_cast<int>(m_materials.size()) - 1;
      node.translation    = {0.0F, 0.F, 0.F};
    }


    // Setting camera to see the scene
    g_cameraManip->setClipPlanes({0.1F, 100.0F});
    g_cameraManip->setLookat({-2.0F, 4.0F, 8.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});

    // Default parameters for overall material
    m_pushConst.intensity = 5.0F;
    m_pushConst.maxDepth  = 1;
    m_pushConst.metallic  = 0.2F;
    m_pushConst.roughness = 1.0F;

    // Default Sky values
    m_skyParams = {};
  }


  // Create all Vulkan buffer data
  void createVkBuffers()
  {
    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);

    const VkBufferUsageFlags rt_usage_flag = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
                                             | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex per mesh
    m_bMeshes.resize(m_meshes.size());
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& m = m_bMeshes[i];
      NVVK_CHECK(m_alloc.createBuffer(m.vertices, std::span(m_meshes[i]).size_bytes(), rt_usage_flag));
      NVVK_CHECK(uploader.appendBuffer(m.vertices, 0, std::span(m_meshes[i])));
      NVVK_DBG_NAME(m.vertices.buffer);
    }

    // Create the buffer of the current frame, changing at each frame
    NVVK_CHECK(m_alloc.createBuffer(m_bFrameInfo, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT));
    NVVK_DBG_NAME(m_bFrameInfo.buffer);

    // Create the buffer of sky parameters, updated at each frame
    NVVK_CHECK(m_alloc.createBuffer(m_bSkyParams, sizeof(shaderio::SkySimpleParameters), VK_BUFFER_USAGE_2_UNIFORM_BUFFER_BIT));
    NVVK_DBG_NAME(m_bSkyParams.buffer);

    // Primitive instance information
    std::vector<shaderio::InstanceInfo> inst_info;
    for(const nvutils ::Node& node : m_nodes)
    {
      shaderio::InstanceInfo info{};
      info.materialID = node.material;
      inst_info.emplace_back(info);
    }
    NVVK_CHECK(m_alloc.createBuffer(m_bInstInfoBuffer, std::span(inst_info).size_bytes(),
                                    VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(uploader.appendBuffer(m_bInstInfoBuffer, 0, std::span(inst_info)));
    NVVK_DBG_NAME(m_bInstInfoBuffer.buffer);

    NVVK_CHECK(m_alloc.createBuffer(m_bMaterials, std::span(m_materials).size_bytes(),
                                    VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(uploader.appendBuffer(m_bMaterials, 0, std::span(m_materials)));
    NVVK_DBG_NAME(m_bMaterials.buffer);

    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();
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
    std::vector<nvvk::AccelerationStructureGeometryInfo> geoInfos(m_meshes.size());
    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      geoInfos[p_idx] = primitiveToGeometry(m_meshes[p_idx], m_bMeshes[p_idx].vertices.address);
    }

    m_asHelper.blasSubmitBuildAndWait(geoInfos, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR
                                                    | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(m_nodes.size());
    const VkGeometryInstanceFlagsKHR flags{VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV};
    for(const nvutils ::Node& node : m_nodes)
    {
      VkAccelerationStructureInstanceKHR ray_inst{};
      ray_inst.transform           = nvvk::toTransformMatrixKHR(node.localMatrix());  // Position of the instance
      ray_inst.instanceCustomIndex = node.mesh;                                       // gl_InstanceCustomIndexEXT
      ray_inst.accelerationStructureReference         = m_asHelper.blasSet[node.mesh].address;
      ray_inst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
      ray_inst.flags                                  = flags;
      ray_inst.mask                                   = 0xFF;
      tlasInstances.emplace_back(ray_inst);
    }

    m_asHelper.tlasSubmitBuildAndWait(tlasInstances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    // This descriptor set, holds the top level acceleration structure and the output image
    // Create Binding Set
    nvvk::DescriptorBindings bindings;
    bindings.addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_sceneDesc, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_skyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_materials, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_instances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);

    // Create descriptor layout, pool, and 1 set
    NVVK_CHECK(m_descriptorPack.init(bindings, m_device, 1));
    NVVK_DBG_NAME(m_descriptorPack.getLayout());
    NVVK_DBG_NAME(m_descriptorPack.getPool());
    NVVK_DBG_NAME(m_descriptorPack.getSet(0));

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

#if (USE_SLANG)
    const VkShaderModuleCreateInfo createInfo = nvsamples::getShaderModuleCreateInfo(raytrace_position_fetch_slang);
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
    const VkShaderModuleCreateInfo rgenCreateInfo = nvsamples::getShaderModuleCreateInfo(raytrace_position_fetch_rgen_glsl);
    stages[eRaygen].pNext = &rgenCreateInfo;
    stages[eRaygen].pName = "main";
    stages[eRaygen].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Miss
    const VkShaderModuleCreateInfo rmissCreateInfo = nvsamples::getShaderModuleCreateInfo(raytrace_position_fetch_rmiss_glsl);
    stages[eMiss].pNext = &rmissCreateInfo;
    stages[eMiss].pName = "main";
    stages[eMiss].stage = VK_SHADER_STAGE_MISS_BIT_KHR;

    // Closest hit
    const VkShaderModuleCreateInfo rchitCreateInfo = nvsamples::getShaderModuleCreateInfo(raytrace_position_fetch_rchit_glsl);
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
    shaderGroups.push_back(group);

    // Push constant: we want to be able to update constants used by the shaders
    const VkPushConstantRange pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant)};

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts    = m_descriptorPack.getLayoutPtr();
    vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_pipelineLayout);
    NVVK_DBG_NAME(m_pipelineLayout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    rayPipelineInfo.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    rayPipelineInfo.pStages                      = stages.data();
    rayPipelineInfo.groupCount                   = static_cast<uint32_t>(shaderGroups.size());
    rayPipelineInfo.pGroups                      = shaderGroups.data();
    rayPipelineInfo.maxPipelineRayRecursionDepth = MAXRAYRECURSIONDEPTH;  // Ray depth
    rayPipelineInfo.layout                       = m_pipelineLayout;
    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_pipeline);
    NVVK_DBG_NAME(m_pipeline);

    // Creating the SBT
    // Prepare SBT data from ray pipeline
    size_t bufferSize = m_sbt.calculateSBTBufferSize(m_pipeline, rayPipelineInfo);

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
    nvvk::WriteSetContainer writes{};
    writes.append(m_descriptorPack.makeWrite(B_tlas), m_asHelper.tlas);
    writes.append(m_descriptorPack.makeWrite(B_outImage), m_gBuffers.getColorImageView(), VK_IMAGE_LAYOUT_GENERAL);
    writes.append(m_descriptorPack.makeWrite(B_frameInfo), m_bFrameInfo);
    writes.append(m_descriptorPack.makeWrite(B_materials), m_bMaterials);
    writes.append(m_descriptorPack.makeWrite(B_instances), m_bInstInfoBuffer);
    writes.append(m_descriptorPack.makeWrite(B_skyParam), m_bSkyParams);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void destroyResources()
  {
    vkDeviceWaitIdle(m_device);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    m_descriptorPack.deinit();


    // destroyGeometries(); // already done
    m_alloc.destroyBuffer(m_bFrameInfo);
    m_alloc.destroyBuffer(m_bSceneDesc);
    m_alloc.destroyBuffer(m_bInstInfoBuffer);
    m_alloc.destroyBuffer(m_bMaterials);
    m_alloc.destroyBuffer(m_bSkyParams);
    m_alloc.destroyBuffer(m_sbtBuffer);

    m_asHelper.deinitAccelerationStructures();
    m_asHelper.deinit();

    m_gBuffers.deinit();
    m_samplerPool.deinit();
    m_sbt.deinit();
    m_stagingUploader.deinit();
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
  nvapp::Application*     m_app{nullptr};
  nvvk::ResourceAllocator m_alloc;
  nvvk::GBuffer           m_gBuffers;  // G-Buffers: color + depth
  nvvk::SamplerPool       m_samplerPool{};
  nvvk::StagingUploader   m_stagingUploader{};


  VkFormat                      m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;  // Color format of the image
  VkFormat                      m_depthFormat = VK_FORMAT_UNDEFINED;       // Depth format of the depth buffer
  VkDevice                      m_device      = VK_NULL_HANDLE;            // Convenient
  shaderio::SkySimpleParameters m_skyParams{};

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
  std::vector<nvutils::Node>    m_nodes;
  std::vector<Material>         m_materials;

  // Pipeline
  shaderio::PushConstant m_pushConst{};     // Information sent to the shader
  nvvk::DescriptorPack   m_descriptorPack;  // Descriptor bindings, layout, pool, and set
  VkPipeline             m_pipeline{};
  VkPipelineLayout       m_pipelineLayout{};

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR m_rtPosFetch{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};
  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_asProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};


  nvvk::SBTGenerator m_sbt;        // Shading binding table wrapper
  nvvk::Buffer       m_sbtBuffer;  // Buffer for the SBT


  nvvk::AccelerationStructureHelper m_asHelper;  // Helper to build acceleration structures
};
//////////////////////////////////////////////////////////////////////////
///
///
///
int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless"}, &appInfo.headless, true);
  reg.add({"frames", "Number of frames to run in headless mode"}, &appInfo.headlessFrameCount);
  cli.add(reg);
  cli.parse(argc, argv);

  VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR fetchFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR};
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};

  nvvk::ContextInitInfo vkSetup;
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accel_feature});  // To build acceleration structures
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rt_pipeline_feature});  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});  // Required by ray tracing pipeline
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, &fetchFeatures});  // #FETCH
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature});

  // Create Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  if(fetchFeatures.rayTracingPositionFetch == VK_FALSE)
  {
    LOGE("ERROR: Position Fetch not supported");
    exit(1);
  }

  // Application information
  appInfo.name           = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
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
  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());  // Menu / Quit
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app.addElement(std::make_shared<RaytracePositionFetch>());

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
