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

#include <array>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#include "common/vk_context.hpp"
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvh/primitives.hpp"
#include "nvvk/acceleration_structures.hpp"
#include "nvvk/extensions_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/tonemap_postprocess.hpp"


namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
#include "shaders/dh_bindings.h"
}  // namespace DH


#if USE_HLSL
#include "_autogen/pathtrace_rgenMain.spirv.h"
#include "_autogen/pathtrace_rchitMain.spirv.h"
#include "_autogen/pathtrace_rmissMain.spirv.h"
const auto& rgen_shd  = std::vector<char>{std::begin(pathtrace_rgenMain), std::end(pathtrace_rgenMain)};
const auto& rchit_shd = std::vector<char>{std::begin(pathtrace_rchitMain), std::end(pathtrace_rchitMain)};
const auto& rmiss_shd = std::vector<char>{std::begin(pathtrace_rmissMain), std::end(pathtrace_rmissMain)};
#elif USE_SLANG
#include "_autogen/pathtrace_slang.h"
#else
#include "_autogen/pathtrace.rchit.glsl.h"
#include "_autogen/pathtrace.rgen.glsl.h"
#include "_autogen/pathtrace.rmiss.glsl.h"
const auto& rgen_shd  = std::vector<uint32_t>{std::begin(pathtrace_rgen_glsl), std::end(pathtrace_rgen_glsl)};
const auto& rchit_shd = std::vector<uint32_t>{std::begin(pathtrace_rchit_glsl), std::end(pathtrace_rchit_glsl)};
const auto& rmiss_shd = std::vector<uint32_t>{std::begin(pathtrace_rmiss_glsl), std::end(pathtrace_rmiss_glsl)};
#endif

#include "nvvk/specialization.hpp"
#include "nvvk/images_vk.hpp"
#include "common/utils.hpp"


/// </summary> Ray trace multiple primitives using SER
class SerPathtrace : public nvvkhl::IAppElement
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

  void onAttach(nvvkhl::Application* app) override
  {
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

    // Requesting ray tracing properties and reorder properties
    VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    m_rtProperties.pNext = &m_reorderProperties;
    prop2.pNext          = &m_rtProperties;
    vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

    // Create utilities to create BLAS/TLAS and the Shading Binding Table (SBT)
    int32_t gctQueueIndex = m_app->getQueue(0).familyIndex;
    m_sbt.setup(m_device, gctQueueIndex, m_alloc.get(), m_rtProperties);

    // Create resources
    createScene();
    createVkBuffers();
    createBottomLevelAS();
    createTopLevelAS();
    createRtxPipeline();
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
    resetFrame();
  }

  void onUIRender() override
  {
    {  // Setting menu
      ImGui::Begin("Settings");

      static float col[3] = {0, 0, 0};
      ImGui::ColorEdit3("Color Clear", col);

      ImGuiH::CameraWidget();

      using namespace ImGuiH;
      bool changed{false};
      if(ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen))
      {
        PropertyEditor::begin();
        {
          PropertyEditor::entry("SER Mode", fmt::format("{}", m_reorderProperties.rayTracingInvocationReorderReorderingHint
                                                                      == VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_NV ?
                                                                  "Active" :
                                                                  "Not Available"));

          if(PropertyEditor::entry("Use SER", [&] { return ImGui::Checkbox("", (bool*)&m_useSER); }))
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
               bool result = ImGui::ImageButton("##but", m_gBuffers->getDescriptorSet(eImgHeatmap),
                                                ImVec2(100 * m_gBuffers->getAspectRatio(), 100));
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
          changed |= PropertyEditor::entry("Metallic",
                                           [&] { return ImGui::SliderFloat("#1", &m_pushConst.metallic, 0.0F, 1.0F); });
          changed |= PropertyEditor::entry("Roughness",
                                           [&] { return ImGui::SliderFloat("#1", &m_pushConst.roughness, 0.001F, 1.0F); });
          PropertyEditor::treePop();
        }
        if(PropertyEditor::treeNode("Sun"))
        {
          changed |= PropertyEditor::entry("Intensity",
                                           [&] { return ImGui::SliderFloat("#1", &m_pushConst.intensity, 0.0F, 10.0F); });
          glm::vec3 dir = m_skyParams.directionToLight;
          changed |= ImGuiH::azimuthElevationSliders(dir, false);
          m_skyParams.directionToLight = dir;
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
        m_tonemapper->onUI();
      }


      ImGui::End();
      if(changed)
        resetFrame();
    }

    {  // Rendering Viewport
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      // Display the G-Buffer image
      ImGui::Image(m_gBuffers->getDescriptorSet(m_showHeatmap ? eImgHeatmap : eImgTonemapped), ImGui::GetContentRegionAvail());

      ImGui::End();
      ImGui::PopStyleVar();
    }
  }


  void onRender(VkCommandBuffer cmd) override
  {
    auto sdbg = m_dutil->DBG_SCOPE(cmd);

    if(!updateFrame())
    {
      return;
    }

    // Update Frame buffer uniform buffer
    DH::FrameInfo finfo{};
    const auto&   clip = CameraManip.getClipPlanes();
    finfo.view         = CameraManip.getMatrix();
    finfo.proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), m_gBuffers->getAspectRatio(), clip.x, clip.y);
    finfo.proj[1][1] *= -1;
    finfo.projInv = glm::inverse(finfo.proj);
    finfo.viewInv = glm::inverse(finfo.view);
    finfo.camPos  = CameraManip.getEye();
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);

    // Update the sky
    vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(nvvkhl_shaders::SimpleSkyParameters), &m_skyParams);

    // Reset maximum for current frame
    vkCmdFillBuffer(cmd, m_bHeatStats.buffer, (uint32_t(m_frame) & 1) * sizeof(uint32_t), sizeof(uint32_t), 1);

    m_pushConst.frame  = m_frame;
    m_pushConst.useSER = m_useSER;

    nvvk::memoryBarrier(cmd);


    // Ray trace
    std::vector<VkDescriptorSet> descSets{m_rtSet->getSet()};
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.plines[0]);
    pushDescriptorSet(cmd);
    vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant), &m_pushConst);

    const auto& regions = m_sbt.getRegions();
    const auto& size    = m_app->getViewportSize();
    vkCmdTraceRaysKHR(cmd, &regions[0], &regions[1], &regions[2], &regions[3], size.width, size.height, 1);

    // Making sure the rendered image is ready to be used
    auto image_memory_barrier =
        nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(eImgRendered), VK_ACCESS_SHADER_READ_BIT,
                                     VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &image_memory_barrier);

    m_tonemapper->runCompute(cmd, size);
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
    //m_meshes.emplace_back(nvh::cube(obj_size, obj_size, obj_size));
    m_meshes.emplace_back(nvh::createSphereUv(obj_size));
    //m_meshes.emplace_back(nvh::tetrahedron());
    //m_meshes.emplace_back(nvh::octahedron());
    //m_meshes.emplace_back(nvh::icosahedron());
    //m_meshes.emplace_back(nvh::cone());
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
    m_meshes.emplace_back(nvh::createPlane(10, 100, 100));
    auto& n       = m_nodes.emplace_back();
    n.mesh        = static_cast<int>(m_meshes.size()) - 1;
    n.material    = static_cast<int>(m_materials.size()) - 1;
    n.translation = {0.0f, static_cast<float>(-num_obj / 2 - 1) * obj_spacing, 0.0f};

    // Setting camera to see the scene
    CameraManip.setClipPlanes({0.1F, 100.0F});
    CameraManip.setLookat({0.0F, 2.0F, static_cast<float>(num_obj) * obj_spacing * 1.5f}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});

    // Default parameters for overall material
    m_pushConst.intensity             = 1.0F;
    m_pushConst.maxDepth              = 5;
    m_pushConst.roughness             = 0.05F;
    m_pushConst.metallic              = 0.5F;
    m_pushConst.frame                 = 0;
    m_pushConst.fireflyClampThreshold = 10;
    m_pushConst.maxSamples            = 2;

    // Default Sky values
    m_skyParams = nvvkhl_shaders::initSimpleSkyParameters();
  }


  void createGbuffers(const glm::vec2& size)
  {
    // Rendering image targets
    m_viewSize = size;
    std::vector<VkFormat> color_buffers = {m_colorFormat, m_colorFormat, m_colorFormat};  // tonemapped, original, heatmap
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(),
                                                   VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)},
                                                   color_buffers, m_depthFormat);
  }

  // Create all Vulkan buffer data
  void createVkBuffers()
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_bMeshes.resize(m_meshes.size());

    auto rtUsageFlag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                       | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

    // Create a buffer of Vertex and Index per mesh
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& m = m_bMeshes[i];
      m.vertices         = m_alloc->createBuffer(cmd, m_meshes[i].vertices, rtUsageFlag);
      m.indices          = m_alloc->createBuffer(cmd, m_meshes[i].triangles, rtUsageFlag);
      m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
      m_dutil->DBG_NAME_IDX(m.indices.buffer, i);
    }

    // Create the buffer of the current frame, changing at each frame
    m_bFrameInfo = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    // Create the buffer of sky parameters, updated at each frame
    m_bSkyParams = m_alloc->createBuffer(sizeof(nvvkhl_shaders::SimpleSkyParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bSkyParams.buffer);

    // Create the buffer for the heatmap statistics
    m_bHeatStats = m_alloc->createBuffer(sizeof(uint32_t) * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_dutil->DBG_NAME(m_bHeatStats.buffer);

    // Primitive instance information
    std::vector<DH::InstanceInfo> instInfo;
    for(auto& node : m_nodes)
    {
      DH::InstanceInfo info{};
      info.transform  = node.localMatrix();
      info.materialID = node.material;
      instInfo.emplace_back(info);
    }
    m_bInstInfoBuffer =
        m_alloc->createBuffer(cmd, instInfo, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bInstInfoBuffer.buffer);

    m_bMaterials = m_alloc->createBuffer(cmd, m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_dutil->DBG_NAME(m_bMaterials.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }


  //--------------------------------------------------------------------------------------------------
  // Converting a PrimitiveMesh as input for BLAS
  //
  nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const nvh::PrimitiveMesh& prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress)
  {
    uint32_t maxPrimitiveCount = static_cast<uint32_t>(prim.triangles.size());

    // Describe buffer as array of VertexObj.
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data.
    triangles.vertexData.deviceAddress = vertexAddress;
    triangles.vertexStride             = sizeof(nvh::PrimitiveVertex);
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
    nvvk::Buffer scratchBuffer =
        m_alloc->createBuffer(maxScratch, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

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

    VkCommandBuffer cmd            = m_app->createTempCmdBuffer();
    nvvk::Buffer    instanceBuffer = m_alloc->createBuffer(cmd, tlasInstances,
                                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                               | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    nvvk::accelerationStructureBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

    auto geo = tlasBuildData.makeInstanceGeometry(tlasInstances.size(), instanceBuffer.address);
    tlasBuildData.addGeometry(geo);
    auto sizeInfo = tlasBuildData.finalizeGeometry(m_device, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    // Allocate the scratch memory
    VkDeviceSize scratchSize = sizeInfo.buildScratchSize;
    nvvk::Buffer scratchBuffer =
        m_alloc->createBuffer(scratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    auto createInfo = tlasBuildData.makeCreateInfo();
    m_tlas          = m_alloc->createAcceleration(createInfo);
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
    m_rtPipe.destroy(m_device);
    m_rtSet->deinit();
    m_rtSet = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    m_rtPipe.plines.resize(1);

    // This descriptor set, holds the top level acceleration structure and the output image
    // Create Binding Set
    m_rtSet->addBinding(B_tlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_outImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_outHeatmap, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_frameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_skyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_heatStats, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_materials, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_instances, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_vertex, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
    m_rtSet->addBinding(B_index, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (uint32_t)m_bMeshes.size(), VK_SHADER_STAGE_ALL);
    m_rtSet->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);

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
    VkShaderModule shaderModule = nvvk::createShaderModule(m_device, &pathtraceSlang[0], sizeof(pathtraceSlang));
    stages[eRaygen].module      = shaderModule;
    stages[eRaygen].pName       = "rgenMain";
    stages[eRaygen].stage       = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[eMiss].module        = shaderModule;
    stages[eMiss].pName         = "rmissMain";
    stages[eMiss].stage         = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[eClosestHit].module  = shaderModule;
    stages[eClosestHit].pName   = "rchitMain";
    stages[eClosestHit].stage   = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    m_dutil->setObjectName(shaderModule, "pathtraceSlang");
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

#if !USE_SLANG
    nvvk::Specialization specialization;
    specialization.add(0, m_useSER ? 1 : 0);
    stages[eRaygen].pSpecializationInfo = specialization.getSpecialization();
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
    VkPushConstantRange pushConstant{VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant)};

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

    // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
    std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {m_rtSet->getLayout()};  // , m_pContainer[eGraphic].dstLayout};
    pipelineLayoutCreateInfo.setLayoutCount = static_cast<uint32_t>(rtDescSetLayouts.size());
    pipelineLayoutCreateInfo.pSetLayouts    = rtDescSetLayouts.data();
    vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_rtPipe.layout);
    m_dutil->DBG_NAME(m_rtPipe.layout);

    // Assemble the shader stages and recursion depth info into the ray tracing pipeline
    VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    rayPipelineInfo.stageCount                   = static_cast<uint32_t>(stages.size());  // Stages are shaders
    rayPipelineInfo.pStages                      = stages.data();
    rayPipelineInfo.groupCount                   = static_cast<uint32_t>(shaderGroups.size());
    rayPipelineInfo.pGroups                      = shaderGroups.data();
    rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
    rayPipelineInfo.layout                       = m_rtPipe.layout;
    vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_rtPipe.plines[0]);
    m_dutil->DBG_NAME(m_rtPipe.plines[0]);

    // Creating the SBT
    m_sbt.create(m_rtPipe.plines[0], rayPipelineInfo);

    // Removing temp modules

#if(USE_SLANG)
    vkDestroyShaderModule(m_device, shaderModule, nullptr);
#else
    for(auto& s : stages)
      vkDestroyShaderModule(m_device, s.module, nullptr);
#endif
  }


  void pushDescriptorSet(VkCommandBuffer cmd)
  {
    // Write to descriptors
    VkAccelerationStructureKHR tlas = m_tlas.accel;
    VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    descASInfo.accelerationStructureCount = 1;
    descASInfo.pAccelerationStructures    = &tlas;
    VkDescriptorImageInfo  imageInfo{{}, m_gBuffers->getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo  heatInfo{{}, m_gBuffers->getColorImageView(eImgHeatmap), VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo dbi_sky{m_bSkyParams.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo dbi_heatstats{m_bHeatStats.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo mat_desc{m_bMaterials.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo inst_desc{m_bInstInfoBuffer.buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkDescriptorBufferInfo> vertex_desc;
    std::vector<VkDescriptorBufferInfo> index_desc;
    vertex_desc.reserve(m_bMeshes.size());
    index_desc.reserve(m_bMeshes.size());
    for(auto& m : m_bMeshes)
    {
      vertex_desc.push_back({m.vertices.buffer, 0, VK_WHOLE_SIZE});
      index_desc.push_back({m.indices.buffer, 0, VK_WHOLE_SIZE});
    }

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_rtSet->makeWrite(0, B_tlas, &descASInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, B_outImage, &imageInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, B_outHeatmap, &heatInfo));
    writes.emplace_back(m_rtSet->makeWrite(0, B_frameInfo, &dbi_unif));
    writes.emplace_back(m_rtSet->makeWrite(0, B_skyParam, &dbi_sky));
    writes.emplace_back(m_rtSet->makeWrite(0, B_heatStats, &dbi_heatstats));
    writes.emplace_back(m_rtSet->makeWrite(0, B_materials, &mat_desc));
    writes.emplace_back(m_rtSet->makeWrite(0, B_instances, &inst_desc));
    writes.emplace_back(m_rtSet->makeWriteArray(0, B_vertex, vertex_desc.data()));
    writes.emplace_back(m_rtSet->makeWriteArray(0, B_index, index_desc.data()));
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.layout, 0,
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
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    for(auto& m : m_bMeshes)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }
    m_alloc->destroy(m_bFrameInfo);
    m_alloc->destroy(m_bInstInfoBuffer);
    m_alloc->destroy(m_bMaterials);
    m_alloc->destroy(m_bSkyParams);
    m_alloc->destroy(m_bHeatStats);

    for(auto& b : m_blas)
      m_alloc->destroy(b);
    m_alloc->destroy(m_tlas);

    m_rtSet->deinit();
    m_gBuffers.reset();

    m_rtPipe.destroy(m_device);

    m_sbt.destroy();
    m_tonemapper.reset();
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers->getColorImage(), m_gBuffers->getSize(),
                           nvh::getExecutablePath().replace_extension(".jpg").string(), 95);
  }


  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*                           m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>               m_dutil;
  std::unique_ptr<nvvkhl::AllocVma>              m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer>  m_rtSet;  // Descriptor set
  std::unique_ptr<nvvkhl::TonemapperPostProcess> m_tonemapper;

  glm::vec2                        m_viewSize    = {1, 1};
  VkFormat                         m_colorFormat = VK_FORMAT_R32G32B32A32_SFLOAT;  // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth
  nvvkhl_shaders::SimpleSkyParameters m_skyParams{};

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
  std::vector<nvh::PrimitiveMesh> m_meshes;
  std::vector<nvh::Node>          m_nodes;
  std::vector<DH::Material>       m_materials;

  // Pipeline
  DH::PushConstant m_pushConst{};                        // Information sent to the shader
  VkPipelineLayout m_pipelineLayout   = VK_NULL_HANDLE;  // The description of the pipeline
  VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
  int              m_frame{0};
  int              m_maxFrames{10000};

  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  VkPhysicalDeviceRayTracingInvocationReorderPropertiesNV m_reorderProperties{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_PROPERTIES_NV};

  nvvk::SBTWrapper          m_sbt;  // Shading binding table wrapper
  nvvkhl::PipelineContainer m_rtPipe;

  std::vector<nvvk::AccelKHR> m_blas;  // Bottom-level AS
  nvvk::AccelKHR              m_tlas;  // Top-level AS
};

//////////////////////////////////////////////////////////////////////////
///
///
///
auto main(int argc, char** argv) -> int
{
  nvvkhl::ApplicationCreateInfo appInfo;

  nvh::CommandLineParser cli(PROJECT_NAME);
  cli.addArgument({"--headless"}, &appInfo.headless, "Run in headless mode");
  cli.addArgument({"--frames"}, &appInfo.headlessFrameCount, "Number of frames to render in headless mode");
  cli.parse(argc, argv);

  VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
  VkPhysicalDeviceShaderClockFeaturesKHR clockFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
  VkPhysicalDeviceShaderSMBuiltinsFeaturesNV smBuiltinFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SM_BUILTINS_FEATURES_NV};
  VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV reorderFeature{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_INVOCATION_REORDER_FEATURES_NV};

  // Vulkan context creation
  VkContextSettings vkSetup;
  if(!appInfo.headless)
  {
    nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature});  // To build acceleration structures
  vkSetup.deviceExtensions.push_back({VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME});  // Required by ray tracing pipeline
  vkSetup.deviceExtensions.push_back({VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature});  // To use vkCmdTraceRaysKHR
  vkSetup.deviceExtensions.push_back({VK_KHR_SHADER_CLOCK_EXTENSION_NAME, &clockFeature});
  vkSetup.deviceExtensions.push_back({VK_NV_RAY_TRACING_INVOCATION_REORDER_EXTENSION_NAME, &reorderFeature});
  vkSetup.deviceExtensions.push_back({VK_NV_SHADER_SM_BUILTINS_EXTENSION_NAME, &smBuiltinFeature});

#if USE_HLSL || USE_SLANG  // DXC is automatically adding the extension
  VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
  vkSetup.deviceExtensions.push_back({VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature});
#endif  // USE_HLSL

  // Create Vulkan context
  auto vkContext = std::make_unique<VulkanContext>(vkSetup);
  if(!vkContext->isValid())
    std::exit(0);

  // Loading the Vulkan extension pointers
  load_VK_EXTENSIONS(vkContext->getInstance(), vkGetInstanceProcAddr, vkContext->getDevice(), vkGetDeviceProcAddr);

  // Application setup
  appInfo.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = false;
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
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // Menu / Quit
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app->addElement(std::make_shared<SerPathtrace>());

  app->run();
  app.reset();
  vkContext.reset();

  return test->errorCode();
}
