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

    This shows the use of Shading Execution Reorder (SER) through a simple
    path tracer. The reorder code is in the rayGen shader, and adds an 
    indication if the ray has hit something or not.

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

#include "_autogen/ser_pathtrace.rchit.glsl.h"
#include "_autogen/ser_pathtrace.rgen.glsl.h"
#include "_autogen/ser_pathtrace.rmiss.glsl.h"
#include "_autogen/ser_pathtrace.slang.h"
#include "_autogen/tonemapper.slang.h"


#include <common/utils.hpp>
#include <nvapp/application.hpp>
#include <nvapp/elem_camera.hpp>
#include <nvapp/elem_dbgprintf.hpp>
#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvgui/camera.hpp>
#include <nvgui/sky.hpp>
#include <nvgui/tonemapper.hpp>
#include <nvshaders_host/tonemapper.hpp>
#include <nvutils/camera_manipulator.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/primitives.hpp>
#include <nvvk/acceleration_structures.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/resource_allocator.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/specialization.hpp>
#include <nvvk/staging.hpp>
#include <nvvk/validation_settings.hpp>

// The camera for the scene
std::shared_ptr<nvutils::CameraManipulator> g_cameraManip{};


/// </summary> Ray trace multiple primitives using SER
class SerPathtrace : public nvapp::IAppElement
{
  enum
  {
    eImgTonemapped,
    eImgRendered,
    eImgHeatmap,
  };

public:
  SerPathtrace()           = default;
  ~SerPathtrace() override = default;

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

    // Tonemapper
    {
      auto code = std::span<const uint32_t>(tonemapper_slang);
      m_tonemapper.init(&m_alloc, code);
    }

    // Acquiring the sampler which will be used for displaying the GBuffer
    m_samplerPool.init(app->getDevice());
    VkSampler linearSampler;
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // Create the G-Buffer
    nvvk::GBufferInitInfo gBufferInit{
        .allocator      = &m_alloc,
        .colorFormats   = {m_colorFormat, m_colorFormat, m_colorFormat},  // tonemap, color and heatmap
        .depthFormat    = nvvk::findDepthFormat(m_app->getPhysicalDevice()),
        .imageSampler   = linearSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    };
    m_gBuffers.init(gBufferInit);


    // Requesting ray tracing properties and reorder properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    m_rtProperties.pNext      = &m_reorderProperties;
    m_reorderProperties.pNext = &m_asProperties;
    prop2.pNext               = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    m_sbt.init(m_device, m_rtProperties);

    // Create resources
    createScene();
    createVkBuffers();
    createBottomLevelAS();
    createTopLevelAS();
    createRtxPipeline();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size)
  {
    NVVK_CHECK(m_gBuffers.update(cmd, size));

    resetFrame();
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");

      static float col[3] = {0, 0, 0};
      ImGui::ColorEdit3("Color Clear", col);

      nvgui::CameraWidget(g_cameraManip);

      using namespace nvgui;
      bool changed{false};
      if(ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
      {
        PropertyEditor::begin();
        {
          PropertyEditor::entry("SER Mode", fmt::format("{}", m_reorderProperties.rayTracingInvocationReorderReorderingHint
                                                                      == VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_NV ?
                                                                  "Active" :
                                                                  "Not Available"));

          if(PropertyEditor::Checkbox("Use SER", (bool*)&m_useSER))
          {
            changed = true;
            vkDeviceWaitIdle(m_device);
            createRtxPipeline();
          }

          if(PropertyEditor::entry("Heatmap", [&] {
               static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
               ImVec4 selectedColor = m_showHeatmap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
               ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
               ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
               ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
               ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));
               bool result = ImGui::ImageButton("##but", (ImTextureID)m_gBuffers.getDescriptorSet(eImgHeatmap),
                                                ImVec2(100 * m_gBuffers.getAspectRatio(), 100));
               ImGui::PopStyleColor(2);
               ImGui::PopStyleVar();
               return result;
             }))
          {
            m_showHeatmap = !m_showHeatmap;
          }
        }
        PropertyEditor::end();

        PropertyEditor::begin();
        if(PropertyEditor::treeNode("Material"))
        {
          changed |= PropertyEditor::SliderFloat("Metallic", &m_pushConst.metallic, 0.0F, 1.0F);
          changed |= PropertyEditor::SliderFloat("Roughness", &m_pushConst.roughness, 0.001F, 1.0F);
          PropertyEditor::treePop();
        }
        if(PropertyEditor::treeNode("Sun"))
        {
          changed |= PropertyEditor::SliderFloat("Intensity", &m_pushConst.intensity, 0.0F, 10.0F);
          glm::vec3 dir = m_skyParams.sunDirection;
          changed |= nvgui::azimuthElevationSliders(dir, false, true);
          m_skyParams.sunDirection = dir;
          PropertyEditor::treePop();
        }
        if(PropertyEditor::treeNode("Ray Tracer"))
        {
          changed |= PropertyEditor::SliderInt("Depth", &m_pushConst.maxDepth, 0, 20);
          changed |= PropertyEditor::SliderInt("Samples", &m_pushConst.maxSamples, 1, 10);
          PropertyEditor::treePop();
        }
        PropertyEditor::end();
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

      // Display the G-Buffer image
      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(m_showHeatmap ? eImgHeatmap : eImgTonemapped),
                   ImGui::GetContentRegionAvail());

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

    // Reset maximum for current frame
    vkCmdFillBuffer(cmd, m_bHeatStats.buffer, (uint32_t(m_frame) & 1) * sizeof(uint32_t), sizeof(uint32_t), 1);

    m_pushConst.frame      = m_frame;
    m_pushConst.mouseCoord = nvapp::ElementDbgPrintf::getMouseCoord();
    //m_pushConst.useSER = m_useSER;

    // Make sure all buffer are transfered
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);

    // Ray trace
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline);
    pushDescriptorSet(cmd);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PushConstant), &m_pushConst);

    const nvvk::SBTGenerator::Regions& regions = m_sbt.getSBTRegions();
    const VkExtent2D&                  size    = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, &regions.raygen, &regions.miss, &regions.hit, &regions.callable, size.width, size.height, 1);

    // Barrier to make sure the image is ready for Tonemapping
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Tonemap the image
    m_tonemapper.runCompute(cmd, m_gBuffers.getSize(), m_tonemapperData, m_gBuffers.getDescriptorImageInfo(eImgRendered),
                            m_gBuffers.getDescriptorImageInfo(eImgTonemapped));
  }

private:
  void createScene()
  {
    constexpr int   num_obj       = 20;
    constexpr float obj_size      = 1.0F;
    constexpr float obj_spacing   = 2.0f;
    constexpr int   num_materials = 128;

    srand(12312412);

    // Meshes
    //m_meshes.emplace_back(nvutils::cube(obj_size, obj_size, obj_size));
    m_meshes.emplace_back(nvutils::createSphereUv(obj_size));
    //m_meshes.emplace_back(nvutils::tetrahedron());
    //m_meshes.emplace_back(nvutils::octahedron());
    //m_meshes.emplace_back(nvutils::icosahedron());
    //m_meshes.emplace_back(nvutils::cone());
    // int num_meshes = static_cast<int>(m_meshes.size());

    // Materials (colorful)
    m_materials.reserve(num_materials);
    for(int i = 0; i < num_materials; i++)
    {
      const glm::vec3 freq = glm::vec3(1.33333F, 2.33333F, 3.33333F) * static_cast<float>(i);
      glm::vec3       v    = static_cast<glm::vec3>(sin(freq) * 0.5F + 0.5F);
      m_materials.push_back({glm::vec4(v, 1)});
    }

    auto inRange = [](int a, int b, int v0 = 10 - 3, int v1 = 10 + 2) {
      return (a >= v0 && a <= v1) && (b >= v0 && b <= v1);
    };

    // Instances
    m_nodes.reserve(static_cast<size_t>(num_obj * num_obj) * num_obj);
    bool skip{false};
    for(int k = 0; k < num_obj; k++)
    {
      for(int j = 0; j < num_obj; j++)
      {
        for(int i = 0; i < num_obj; i++)
        {
          bool center = inRange(i, j);
          center |= inRange(i, k);
          center |= inRange(k, j);
          if(!skip && !center)
          {
            auto& n       = m_nodes.emplace_back();
            n.mesh        = 0;
            n.material    = rand() % num_materials;  // (i * num_obj * num_obj) + (j * num_obj) + (k);
            n.translation = glm::vec3(-(static_cast<float>(num_obj) / 2.F) + static_cast<float>(i),
                                      -(static_cast<float>(num_obj) / 2.F) + static_cast<float>(j),
                                      -(static_cast<float>(num_obj) / 2.F) + static_cast<float>(k));

            n.translation *= obj_spacing;
          }
          skip = !skip;
        }
        skip = !skip;
      }
      skip = !skip;
    }
    m_nodes.shrink_to_fit();

    // Adding a plane & material
    m_materials.push_back({glm::vec4(.7F, .7F, .7F, 1.0F)});
    m_meshes.emplace_back(nvutils::createPlane(10, 100, 100));
    auto& n       = m_nodes.emplace_back();
    n.mesh        = static_cast<int>(m_meshes.size()) - 1;
    n.material    = static_cast<int>(m_materials.size()) - 1;
    n.translation = {0.0f, static_cast<float>(-num_obj / 2 - 1) * obj_spacing, 0.0f};

    // Setting camera to see the scene
    g_cameraManip->setClipPlanes({0.1F, 100.0F});
    g_cameraManip->setLookat({0.0F, 2.0F, static_cast<float>(num_obj) * obj_spacing * 1.5f}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});

    // Default parameters for overall material
    m_pushConst.intensity             = 1.0F;
    m_pushConst.maxDepth              = 5;
    m_pushConst.roughness             = 0.05F;
    m_pushConst.metallic              = 0.5F;
    m_pushConst.frame                 = 0;
    m_pushConst.fireflyClampThreshold = 10;
    m_pushConst.maxSamples            = 2;

    // Default Sky values
    m_skyParams = {};
  }


  // Create all Vulkan buffer data
  void createVkBuffers()
  {
    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);

    auto rtUsageFlag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex and Index per mesh
    m_bMeshes.resize(m_meshes.size());
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& m = m_bMeshes[i];
      NVVK_CHECK(m_alloc.createBuffer(m.vertices, std::span(m_meshes[i].vertices).size_bytes(), rtUsageFlag));
      NVVK_CHECK(m_alloc.createBuffer(m.indices, std::span(m_meshes[i].triangles).size_bytes(), rtUsageFlag));
      NVVK_CHECK(uploader.appendBuffer(m.vertices, 0, std::span(m_meshes[i].vertices)));
      NVVK_CHECK(uploader.appendBuffer(m.indices, 0, std::span(m_meshes[i].triangles)));
      NVVK_DBG_NAME(m.vertices.buffer);
      NVVK_DBG_NAME(m.indices.buffer);
    }

    // Create the buffer of the current frame, changing at each frame
    NVVK_CHECK(m_alloc.createBuffer(m_bFrameInfo, sizeof(shaderio::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    NVVK_DBG_NAME(m_bFrameInfo.buffer);

    // Create the buffer of sky parameters, updated at each frame
    NVVK_CHECK(m_alloc.createBuffer(m_bSkyParams, sizeof(shaderio::SkySimpleParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    NVVK_DBG_NAME(m_bSkyParams.buffer);

    // Create the buffer for the heatmap statistics
    NVVK_CHECK(m_alloc.createBuffer(m_bHeatStats, sizeof(uint32_t) * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT));
    NVVK_DBG_NAME(m_bHeatStats.buffer);

    // Primitive instance information
    std::vector<shaderio::InstanceInfo> instInfo;
    for(auto& node : m_nodes)
    {
      shaderio::InstanceInfo info{};
      info.transform  = node.localMatrix();
      info.materialID = node.material;
      instInfo.emplace_back(info);
    }

    NVVK_CHECK(m_alloc.createBuffer(m_bInstInfoBuffer, std::span(instInfo).size_bytes(),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT));
    NVVK_CHECK(uploader.appendBuffer(m_bInstInfoBuffer, 0, std::span(instInfo)));
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
  nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const nvutils::PrimitiveMesh& prim,
                                                              VkDeviceAddress               vertexAddress,
                                                              VkDeviceAddress               indexAddress)
  {
    uint32_t maxPrimitiveCount = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride             = sizeof(nvutils::PrimitiveVertex);
    triangles.indexType                = VK_INDEX_TYPE_UINT32;
    triangles.indexData.deviceAddress  = indexAddress;
    triangles.maxVertex                = static_cast<uint32_t>(prim.vertices.size()) - 1;
    //triangles.transformData; // Identity

    nvvk::AccelerationStructureGeometryInfo result;

    result.geometry = VkAccelerationStructureGeometryKHR{
        .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry     = {triangles},
        .flags        = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR,
    };

    result.rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = maxPrimitiveCount};
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
    VkDeviceSize maxScratch{0};

    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      auto vertexAddress = m_bMeshes[p_idx].vertices.address;
      auto indexAddress  = m_bMeshes[p_idx].indices.address;

      auto geo                = primitiveToGeometry(m_meshes[p_idx], vertexAddress, indexAddress);
      buildData[p_idx].asType = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      buildData[p_idx].addGeometry(geo);
      auto sizeInfo = buildData[p_idx].finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
      maxScratch = std::max(maxScratch, sizeInfo.buildScratchSize);
    }
    // Scratch buffer
    nvvk::Buffer scratchBuffer;
    NVVK_CHECK(m_alloc.createBuffer(scratchBuffer, maxScratch, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                    VMA_MEMORY_USAGE_AUTO, {}, m_asProperties.minAccelerationStructureScratchOffsetAlignment));
    NVVK_DBG_NAME(scratchBuffer.buffer);

    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    for(uint32_t p_idx = 0; p_idx < m_meshes.size(); p_idx++)
    {
      auto createInfo = buildData[p_idx].makeCreateInfo();
      NVVK_CHECK(m_alloc.createAcceleration(m_blas[p_idx], createInfo));
      NVVK_DBG_NAME(m_blas[p_idx].accel);
      buildData[p_idx].cmdBuildAccelerationStructure(cmd, m_blas[p_idx].accel, scratchBuffer.address);
    }
    m_app->submitAndWaitTempCmdBuffer(cmd);

    m_alloc.destroyBuffer(scratchBuffer);
  }

  //--------------------------------------------------------------------------------------------------
  // Create the top level acceleration structures, referencing all BLAS
  //
  void createTopLevelAS()
  {
    nvvk::AccelerationStructureBuildData            tlasBuildData{VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR};
    std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
    tlasInstances.reserve(m_nodes.size());
    for(auto& node : m_nodes)
    {
      VkGeometryInstanceFlagsKHR flags{VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV};

      VkAccelerationStructureInstanceKHR rayInst{};
      rayInst.transform           = nvvk::toTransformMatrixKHR(node.localMatrix());  // Position of the instance
      rayInst.instanceCustomIndex = node.mesh;                                       // gl_InstanceCustomIndexEXT
      rayInst.accelerationStructureReference         = m_blas[node.mesh].address;
      rayInst.instanceShaderBindingTableRecordOffset = 0;  // We will use the same hit group for all objects
      rayInst.flags                                  = flags;
      rayInst.mask                                   = 0xFF;
      tlasInstances.emplace_back(rayInst);
    }

    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);

    nvvk::Buffer instanceBuffer;
    NVVK_CHECK(m_alloc.createBuffer(instanceBuffer, std::span(tlasInstances).size_bytes(),
                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                        | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR));
    NVVK_CHECK(uploader.appendBuffer(instanceBuffer, 0, std::span(tlasInstances)));
    NVVK_DBG_NAME(instanceBuffer.buffer);
    uploader.cmdUploadAppended(cmd);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

    auto geo = tlasBuildData.makeInstanceGeometry(tlasInstances.size(), instanceBuffer.address);
    tlasBuildData.addGeometry(geo);
    auto sizeInfo = tlasBuildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    // Allocate the scratch memory
    VkDeviceSize scratchSize = sizeInfo.buildScratchSize;
    nvvk::Buffer scratchBuffer;
    NVVK_CHECK(m_alloc.createBuffer(scratchBuffer, scratchSize, VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT,
                                    VMA_MEMORY_USAGE_AUTO, {}, m_asProperties.minAccelerationStructureScratchOffsetAlignment));
    NVVK_DBG_NAME(scratchBuffer.buffer);

    auto createInfo = tlasBuildData.makeCreateInfo();
    NVVK_CHECK(m_alloc.createAcceleration(m_tlas, createInfo));
    NVVK_DBG_NAME(m_tlas.accel);
    tlasBuildData.cmdBuildAccelerationStructure(cmd, m_tlas.accel, scratchBuffer.address);

    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();

    m_alloc.destroyBuffer(scratchBuffer);
    m_alloc.destroyBuffer(instanceBuffer);
  }


  //--------------------------------------------------------------------------------------------------
  // Pipeline for the ray tracer: all shaders, raygen, chit, miss
  //
  void createRtxPipeline()
  {
    // Clean for re-entry
    m_alloc.destroyBuffer(m_sbtBuffer);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    m_descriptorPack.deinit();


    // Create Binding Set
    nvvk::DescriptorBindings& bindings = m_descriptorPack.bindings;
    bindings.addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_outHeatmap, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_skyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_heatStats, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_materials, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_instances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_vertex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
    bindings.addBinding(B_index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);

    NVVK_CHECK(m_descriptorPack.initFromBindings(m_device, 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR));
    NVVK_DBG_NAME(m_descriptorPack.layout);


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
    const VkShaderModuleCreateInfo moduleInfo = nvsamples::getShaderModuleCreateInfo(ser_pathtrace_slang);
    stages[eRaygen].pNext                     = &moduleInfo;
    stages[eRaygen].pName                     = "rgenMain";
    stages[eRaygen].stage                     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eMiss].pNext                       = &moduleInfo;
    stages[eMiss].pName                       = "rmissMain";
    stages[eMiss].stage                       = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eClosestHit].pNext                 = &moduleInfo;
    stages[eClosestHit].pName                 = "rchitMain";
    stages[eClosestHit].stage                 = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
#else
    // Raygen
    const VkShaderModuleCreateInfo rgenInfo = nvsamples::getShaderModuleCreateInfo(ser_pathtrace_rgen_glsl);
    stages[eRaygen].pNext                   = &rgenInfo;
    stages[eRaygen].pName                   = "main";
    stages[eRaygen].stage                   = VK_SHADER_STAGE_RAYGEN_BIT_KHR;

    // Miss
    const VkShaderModuleCreateInfo missInfo = nvsamples::getShaderModuleCreateInfo(ser_pathtrace_rmiss_glsl);
    stages[eMiss].pNext                     = &missInfo;
    stages[eMiss].pName                     = "main";
    stages[eMiss].stage                     = VK_SHADER_STAGE_MISS_BIT_KHR;

    // Closest hit
    const VkShaderModuleCreateInfo chitInfo = nvsamples::getShaderModuleCreateInfo(ser_pathtrace_rchit_glsl);
    stages[eClosestHit].pNext               = &chitInfo;
    stages[eClosestHit].pName               = "main";
    stages[eClosestHit].stage               = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
#endif

    nvvk::Specialization specialization;
    specialization.add(0, m_useSER ? 1 : 0);
    stages[eRaygen].pSpecializationInfo = specialization.getSpecializationInfo();

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
    const VkPushConstantRange pushConstant{.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(shaderio::PushConstant)};
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorPack.layout}, {pushConstant}));
    NVVK_DBG_NAME(m_pipelineLayout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    rayPipelineInfo.flags                        = VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;
    rayPipelineInfo.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    rayPipelineInfo.pStages                      = stages.data();
    rayPipelineInfo.groupCount                   = static_cast<uint32_t>(shaderGroups.size());
    rayPipelineInfo.pGroups                      = shaderGroups.data();
    rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
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


  void pushDescriptorSet(VkCommandBuffer cmd)
  {

    std::vector<VkDescriptorBufferInfo> vertex_desc;
    std::vector<VkDescriptorBufferInfo> index_desc;
    vertex_desc.reserve(m_bMeshes.size());
    index_desc.reserve(m_bMeshes.size());
    for(auto& m : m_bMeshes)
    {
      vertex_desc.push_back({m.vertices.buffer, 0, VK_WHOLE_SIZE});
      index_desc.push_back({m.indices.buffer, 0, VK_WHOLE_SIZE});
    }

    nvvk::WriteSetContainer         writeContainer;
    const nvvk::DescriptorBindings& bindings = m_descriptorPack.bindings;
    writeContainer.append(bindings.getWriteSet(B_tlas), m_tlas);
    writeContainer.append(bindings.getWriteSet(B_outImage), m_gBuffers.getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(bindings.getWriteSet(B_outHeatmap), m_gBuffers.getColorImageView(eImgHeatmap), VK_IMAGE_LAYOUT_GENERAL);
    writeContainer.append(bindings.getWriteSet(B_frameInfo), m_bFrameInfo);
    writeContainer.append(bindings.getWriteSet(B_skyParam), m_bSkyParams);
    writeContainer.append(bindings.getWriteSet(B_heatStats), m_bHeatStats);
    writeContainer.append(bindings.getWriteSet(B_materials), m_bMaterials);
    writeContainer.append(bindings.getWriteSet(B_instances), m_bInstInfoBuffer);

    writeContainer.append(bindings.getWriteSet(B_vertex), vertex_desc.data());
    writeContainer.append(bindings.getWriteSet(B_index), index_desc.data());
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 0,
                              static_cast<uint32_t>(writeContainer.size()), writeContainer.data());
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
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    m_descriptorPack.deinit();

    for(auto& m : m_bMeshes)
    {
      m_alloc.destroyBuffer(m.vertices);
      m_alloc.destroyBuffer(m.indices);
    }
    m_alloc.destroyBuffer(m_bFrameInfo);
    m_alloc.destroyBuffer(m_bInstInfoBuffer);
    m_alloc.destroyBuffer(m_bMaterials);
    m_alloc.destroyBuffer(m_bSkyParams);
    m_alloc.destroyBuffer(m_bHeatStats);
    m_alloc.destroyBuffer(m_sbtBuffer);

    for(auto& blas : m_blas)
      m_alloc.destroyAcceleration(blas);
    m_alloc.destroyAcceleration(m_tlas);

    m_gBuffers.deinit();
    m_sbt.deinit();
    m_tonemapper.deinit();
    m_samplerPool.deinit();
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
  nvapp::Application*      m_app{nullptr};
  nvvk::ResourceAllocator  m_alloc;
  nvshaders::Tonemapper    m_tonemapper{};
  shaderio::TonemapperData m_tonemapperData;
  nvvk::GBuffer            m_gBuffers;       // G-Buffers: color + depth
  nvvk::SamplerPool        m_samplerPool{};  // The sampler pool, used to create a sampler for the texture


  VkFormat                      m_colorFormat = VK_FORMAT_R32G32B32A32_SFLOAT;  // Color format of the image
  VkFormat                      m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkDevice                      m_device      = VK_NULL_HANDLE;                 // Convenient
  shaderio::SkySimpleParameters m_skyParams{};

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
    nvvk::Buffer indices;   // Buffer of the indices
  };
  std::vector<PrimitiveMeshVk> m_bMeshes;
  nvvk::Buffer                 m_bFrameInfo;
  nvvk::Buffer                 m_bInstInfoBuffer;
  nvvk::Buffer                 m_bMaterials;
  nvvk::Buffer                 m_bSkyParams;
  nvvk::Buffer                 m_bHeatStats;

  bool m_useSER      = true;
  bool m_showHeatmap = false;

  // Data and setting
  std::vector<nvutils::PrimitiveMesh> m_meshes;
  std::vector<nvutils::Node>          m_nodes;
  std::vector<shaderio::Material>     m_materials;

  // Pipeline
  shaderio::PushConstant m_pushConst{};       // Information sent to the shader
  VkPipelineLayout       m_pipelineLayout{};  // The description of the pipeline
  VkPipeline             m_pipeline{};        // The graphic pipeline to render
  nvvk::DescriptorPack   m_descriptorPack{};  // Descriptor bindings, layout, pool, and set
  int                    m_frame{0};
  int                    m_maxFrames{10000};

  VkPhysicalDeviceAccelerationStructurePropertiesKHR m_asProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceRayTracingInvocationReorderPropertiesNV m_reorderProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_NV};

  nvvk::SBTGenerator m_sbt;  // Shader binding table wrapper
  nvvk::Buffer       m_sbtBuffer;

  std::vector<nvvk::AccelerationStructure> m_blas;  // Bottom-level AS
  nvvk::AccelerationStructure              m_tlas;  // Top-level AS
};

//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char** argv) -> int
{
  nvapp::Application           application;  // The application
  nvapp::ApplicationCreateInfo appInfo;      // Information to create the application
  nvvk::Context                vkContext;    // The Vulkan context
  nvvk::ContextInitInfo        vkSetup;      // Information to create the Vulkan context

  // Parse the command line to get the application information
  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;

  reg.add({"headless"}, &appInfo.headless, true);
  reg.add({"frames", "Number of frames to run in headless mode"}, &appInfo.headlessFrameCount);
  cli.add(reg);
  cli.parse(argc, argv);


  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceShaderClockFeaturesKHR clockFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  VkPhysicalDeviceShaderSMBuiltinsFeaturesNV smBuiltinFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV};
  VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV reorderFeature{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV};
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};


  // Vulkan context creation
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature});  // To build acceleration structures
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});  // Required by ray tracing pipeline
  vkSetup.deviceExtensions.push_back({VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature});  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.push_back({VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockFeature});
  vkSetup.deviceExtensions.push_back({VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, &reorderFeature});
  vkSetup.deviceExtensions.push_back({VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME, &smBuiltinFeature});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature});
  vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjectFeatures});

  // Validation layers settings
  nvvk::ValidationSettings vvlInfo{};
  vvlInfo.setPreset(nvvk::ValidationSettings::LayerPresets::eDebugPrintf);
  vvlInfo.printf_to_stdout      = true;
  vvlInfo.printf_buffer_size    = 4096;
  vkSetup.instanceCreateInfoExt = vvlInfo.buildPNextChain();

  // Create the Vulkan context
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  // Application setup
  appInfo.name           = TARGET_NAME;
  appInfo.vSync          = false;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create application
  application.init(appInfo);

  g_cameraManip   = std::make_shared<nvutils::CameraManipulator>();
  auto elemCamera = std::make_shared<nvapp::ElementCamera>();
  elemCamera->setCameraManipulator(g_cameraManip);
  application.addElement(elemCamera);                                     // Camera manipulator
  application.addElement(std::make_shared<nvapp::ElementDefaultMenu>());  // Menu / Quit
  application.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  application.addElement(std::make_shared<SerPathtrace>());

  application.run();
  application.deinit();
  vkContext.deinit();

  return 0;
}
