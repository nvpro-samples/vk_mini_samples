/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


/*

This sample demonstrates the usage of many application elements.
The fluid simulation is an implementation of: https://www.youtube.com/watch?v=rSKMYc1CQHE

Inspector: #INSPECTOR
 - The inspector element is used for visualizing in realtime data that are on the GPU.
   Data that are stored in buffers, images, or just for inspecting a variable in fragment or compute shader.
 
Profiler: #PROFILER
- The profiler element gets the time it took for some operations to run on the GPU. 
  The profiler is scoped in functions and can be nested.

NVML Monitor: 
- This is independent of the application and returns in realtime the status of the GPU.
  Information like the memory it uses, the load of the GPU, and many other metrics.

*/

#include <random>
#include <glm/gtc/matrix_transform.hpp>

// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
// clang-format on

#define VMA_IMPLEMENTATION

#include "imgui/imgui_icon.h"
#include "nvh/primitives.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_inspector.hpp"
#include "nvvkhl/element_nvml.hpp"
#include "nvvkhl/element_profiler.hpp"
#include "nvvkhl/gbuffer.hpp"


namespace DH {
using namespace glm;
#include "shaders/device_host.h"
}  // namespace DH

#include "realtime_analysis.h"


// Adding the compiled Vulkan shaders
#if USE_GLSL
#include "_autogen/raster.frag.h"
#include "_autogen/raster.vert.h"
#include "_autogen/calculate_densities.comp.h"
#include "_autogen/calculate_pressure_force.comp.h"
#include "_autogen/calculate_viscosity.comp.h"
#include "_autogen/external_forces.comp.h"
#include "_autogen/update_positions.comp.h"
#include "_autogen/update_spatial_hash.comp.h"
#include "_autogen/bitonic_sort.comp.h"
#include "_autogen/bitonic_sort_offsets.comp.h"
const auto& vert_shd = std::vector<uint32_t>{std::begin(raster_vert), std::end(raster_vert)};
const auto& frag_shd = std::vector<uint32_t>{std::begin(raster_frag), std::end(raster_frag)};
const auto& calculateDensities_shd =
    std::vector<uint32_t>{std::begin(calculate_densities_comp), std::end(calculate_densities_comp)};
const auto& calculatePressureForce_shd =
    std::vector<uint32_t>{std::begin(calculate_pressure_force_comp), std::end(calculate_pressure_force_comp)};
const auto& calculateViscosity_shd =
    std::vector<uint32_t>{std::begin(calculate_viscosity_comp), std::end(calculate_viscosity_comp)};
const auto& externalForces_shd = std::vector<uint32_t>{std::begin(external_forces_comp), std::end(external_forces_comp)};
const auto& updatePositions_shd = std::vector<uint32_t>{std::begin(update_positions_comp), std::end(update_positions_comp)};
const auto& updateSpatialHash_shd =
    std::vector<uint32_t>{std::begin(update_spatial_hash_comp), std::end(update_spatial_hash_comp)};
const auto& bitonicSort_shd = std::vector<uint32_t>{std::begin(bitonic_sort_comp), std::end(bitonic_sort_comp)};
const auto& bitonicSortOffsets_shd =
    std::vector<uint32_t>{std::begin(bitonic_sort_offsets_comp), std::end(bitonic_sort_offsets_comp)};
#elif USE_SLANG
#include "_autogen/fluid_sim_2D_slang.h"
#include "_autogen/raster_slang.h"
#endif
#include "nvvk/shaders_vk.hpp"


// Elements used by the sample
std::shared_ptr<nvvkhl::ElementProfiler>  g_profiler;          // #PROFILER
std::shared_ptr<nvvkhl::ElementInspector> g_inspectorElement;  // #INSPECTOR

//-------------------------------------------------------------------------------------------------
// This sample implement a simple fluid simulation.
//
// It derives from IAppElement, which is attached to nvvkh::Application. The Application is
// calling the initialization of Vulkan, GLFW and ImGUI. It runs in an infinite loop and
// call the function from the IAppElement interface, such as onRender, onUIRender, onResize, ..
//
class RealtimeAnalysis : public nvvkhl::IAppElement
{
  struct AppSettings
  {
    float     particleRadius      = 0.03f;  // Visual size of the particle
    bool      play                = true;   // Simulation running?
    bool      runOnce             = false;  // Simulation step-once
    glm::vec2 mouseWindowCoord    = {};     // Mouse coord in screen space
    glm::vec2 mouseCoord          = {};     // Mouse coord in simulation space
    bool      pushInteraction     = false;  // Pushing active?
    bool      pullInteraction     = false;  // Pulling active?
    float     interactionStrength = 25.f;   // Pull/push strength
    float     interactionRadius   = 0.2f;   // Pull/push radius
  } m_settings;

  DH::ParticleSetting m_particleSetting = TestA;  // Initialized with Test-A


public:
  RealtimeAnalysis()           = default;
  ~RealtimeAnalysis() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app         = app;
    m_device      = app->getDevice();
    m_alloc       = std::make_unique<nvvkhl::AllocVma>(app->getContext().get());  // Vulkan memory allocator
    m_dutil       = std::make_unique<nvvk::DebugUtil>(m_device);                  // Debug utility
    m_dsetRaster  = std::make_unique<nvvk::DescriptorSetContainer>(m_device);     // Descriptor Set helper
    m_dsetCompute = std::make_unique<nvvk::DescriptorSetContainer>(m_device);     // Descriptor Set helper

    // #INSPECTOR
    nvvkhl::ElementInspector::InitInfo inspectInfo{
        .device                   = m_device,
        .graphicsQueueFamilyIndex = m_app->getQueueGCT().familyIndex,
        .allocator                = m_alloc.get(),
        .imageCount               = 1u,
        .bufferCount              = 2u,
        .computeCount             = 1u,
        .fragmentCount            = 1u,
    };
    g_inspectorElement->init(inspectInfo);

    initParticles();
    createScene();
    createVkBuffers();
    createRasterPipeline();
    createComputeShaderObjectAndLayout();
  }

  void onDetach() override
  {
    NVVK_CHECK(vkDeviceWaitIdle(m_device));
    destroyResources();
  }

  void onResize(uint32_t width, uint32_t height) override
  {
    VkExtent2D size = {width, height};
    m_gBuffers      = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), size, m_colorFormat, m_depthFormat);

    inspectorViewportResize(size);  // #INSPECTOR
  }

  void onUIRender() override
  {
    using PE                = ImGuiH::PropertyEditor;
    DH::ParticleSetting& pS = m_particleSetting;
    ImGui::Begin("Settings");
    ImGui::TextDisabled("%d FPS / %.3fms", static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);
    PE::begin();
    ImGui::SeparatorText("Visual");
    PE::entry("Radius", [&] { return ImGui::SliderFloat("##1", &m_settings.particleRadius, 0.005f, 0.05f); });
    PE::entry("Volume", [&] { return ImGui::SliderFloat("##1", (float*)&pS.boundsMultiplier, 1.0f, 15.0f); });
    ImGui::SeparatorText("Physics");
    PE::entry("Gravity", [&] { return ImGui::SliderFloat("##1", &pS.gravity, -10.0, 0); });
    PE::entry("Collision Damping", [&] {
      return ImGui::SliderFloat("##1", &pS.collisionDamping, 0.0, 1, "%.5f", ImGuiSliderFlags_Logarithmic);
    });
    PE::entry("Smoothing Radius", [&] {
      return ImGui::SliderFloat("##1", &pS.smoothingRadius, 0.2f, 2, "%.5f", ImGuiSliderFlags_Logarithmic);
    });
    PE::entry("Target Density", [&] { return ImGui::SliderFloat("##1", &pS.targetDensity, 0.0, 500); });
    PE::entry("Pressure Multiplier", [&] { return ImGui::SliderFloat("##1", &pS.pressureMultiplier, 0.0, 100); });
    PE::entry("Near Pressure Multiplier",
              [&] { return ImGui::SliderFloat("##1", &pS.nearPressureMultiplier, 0.0f, 100); });
    PE::entry("Viscosity Strength", [&] {
      return ImGui::SliderFloat("##1", &pS.viscosityStrength, 0.0f, 0.5f, "%.5f", ImGuiSliderFlags_Logarithmic);
    });
    ImGui::SeparatorText("Interaction");
    PE::entry("Interaction Strength",
              [&] { return ImGui::SliderFloat("##1", &m_settings.interactionStrength, 0.0f, 100); });
    PE::entry("Interaction radius", [&] { return ImGui::SliderFloat("##1", &m_settings.interactionRadius, 0.0f, .5f); });
    PE::end();

    ImGui::SeparatorText("Test");
    if(ImGui::SmallButton("A"))
    {
      pS = TestA;
      initParticles();
    }
    ImGui::SameLine();
    if(ImGui::SmallButton("B"))
    {
      pS = TestB;
      initParticles();
    }
    ImGui::SameLine();
    if(ImGui::SmallButton("C"))
    {
      pS = TestC;
      initParticles();
    }
    ImGui::SameLine();
    if(ImGui::SmallButton("D"))
    {
      pS = TestD;
      initParticles();
    }

    ImGui::SeparatorText("Controls");
    ImGui::PushFont(ImGuiH::getIconicFont());
    if(ImGui::Button(m_settings.play ? ImGuiH::icon_media_pause : ImGuiH::icon_media_play) || ImGui::IsKeyPressed(ImGuiKey_Space))
      m_settings.play = !m_settings.play;
    ImGui::SameLine();
    if(ImGui::Button(ImGuiH::icon_media_step_forward) || ImGui::IsKeyPressed(ImGuiKey_RightArrow))
    {
      m_settings.runOnce = true;
      m_settings.play    = false;
    }
    ImGui::SameLine();
    if(ImGui::Button(ImGuiH::icon_media_skip_backward) || ImGui::IsKeyPressed(ImGuiKey_R))
    {
      initParticles();
    }
    ImGui::PopFont();
    ImGui::End();

    // Rendered image displayed fully in 'Viewport' window
    ImGui::Begin("Viewport");

    // Retrieving mouse information
    if(ImGui::IsWindowHovered(ImGuiHoveredFlags_None))
    {
      const glm::vec2 mouse_pos   = ImGui::GetMousePos();         // Current mouse pos in window
      const glm::vec2 corner      = ImGui::GetCursorScreenPos();  // Corner of the viewport
      m_settings.mouseWindowCoord = (mouse_pos - corner);
      float     aspectRatio       = ImGui::GetWindowWidth() / ImGui::GetWindowHeight();
      glm::vec2 mouseCoord        = (mouse_pos - corner) / glm::vec2(ImGui::GetWindowSize());
      mouseCoord += glm::vec2(-0.5f, -0.5f);
      mouseCoord.x *= aspectRatio;
      m_settings.mouseCoord = (mouseCoord * glm::vec2(m_particleSetting.boundsMultiplier, -m_particleSetting.boundsMultiplier));
      m_settings.pushInteraction = ImGui::IsMouseDown(ImGuiMouseButton_Left);
      m_settings.pullInteraction = ImGui::IsMouseDown(ImGuiMouseButton_Right);
    }

    // Display of the rendered GBuffer
    ImGui::Image(m_gBuffers->getDescriptorSet(0U), ImGui::GetContentRegionAvail());
    ImGui::End();
  }

  void onRender(VkCommandBuffer cmd)
  {
    auto _scope = m_dutil->DBG_SCOPE(cmd);
    auto sec    = g_profiler->timeRecurring(__FUNCTION__, cmd);  // #PROFILER

    computeSimulation(cmd);
    renderParticles(cmd);

    // #INSPECTOR
    {
      auto _scopeInspection = m_dutil->scopeLabel(cmd, "Inspection");
      g_inspectorElement->inspectBuffer(cmd, 0);
      g_inspectorElement->inspectBuffer(cmd, 1);
      g_inspectorElement->inspectImage(cmd, 0, VK_IMAGE_LAYOUT_GENERAL);
      g_inspectorElement->inspectFragmentVariables(cmd, 0);
      g_inspectorElement->inspectComputeVariables(cmd, 0);
    }
  }


private:
  void createScene()
  {
    // Square for the display of the particle in fragment shader
    m_rasterParticle.vertices  = {{{-.5, -.5, 0}, {0, 0, 1}, {0, 0}},
                                  {{-.5, 0.5, 0}, {0, 0, 1}, {0, 1}},
                                  {{0.5, 0.5, 0}, {0, 0, 1}, {1, 1}},
                                  {{0.5, -.5, 0}, {0, 0, 1}, {1, 0}}};
    m_rasterParticle.triangles = {{{0, 2, 1}}, {{0, 3, 2}}};
  }

  int getNumBlocks()
  {
    int threadsPerBlock = WORKGROUP_SIZE;
    return (NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;
  }

  float randomFloat(float value)
  {  // Return a random value between -value and value
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff   = 2 * value;
    float r      = random * diff - value;
    return r;
  }

  void initParticles()
  {
    float distribution = 1.5f;
    m_particles.resize(NUM_PARTICLES);
    for(int i = 0; i < NUM_PARTICLES; i++)
    {
      m_particles[i]                   = {};
      m_particles[i].position          = {randomFloat(distribution), randomFloat(distribution)};
      m_particles[i].predictedPosition = m_particles[i].position;
    }

    // Copy the particles to the device
    if(m_bParticles.buffer != NULL)
    {
      auto cmd = m_app->createTempCmdBuffer();
      m_alloc->getStaging()->cmdToBuffer(cmd, m_bParticles.buffer, 0, m_particles.size() * sizeof(DH::Particle),
                                         m_particles.data());
      m_app->submitAndWaitTempCmdBuffer(cmd);
      m_alloc->finalizeAndReleaseStaging();
    }
  }


  void createRasterPipeline()
  {
    m_dsetRaster->addBinding(DH::eFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dsetRaster->addBinding(DH::eParticles, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dsetRaster->addBinding(DH::eFragInspectorData, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);  // #INSPECTOR
    m_dsetRaster->addBinding(DH::eFragInspectorMeta, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);  // #INSPECTOR
    m_dsetRaster->initLayout();
    m_dsetRaster->initPool(1);

    const VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                                      sizeof(DH::PushConstant)};
    m_dsetRaster->initPipeLayout(1, &push_constant_ranges);

    // Writing to descriptors
    const VkDescriptorBufferInfo      dbi_frameinfo{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo      dbi_particles{m_bParticles.buffer, 0, VK_WHOLE_SIZE};
    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_dsetRaster->makeWrite(0, DH::eFrameInfo, &dbi_frameinfo));
    writes.emplace_back(m_dsetRaster->makeWrite(0, DH::eParticles, &dbi_particles));
    // #INSPECTOR : Inspector bindings are done in inspectorViewportResize()
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkPipelineRenderingCreateInfo prend_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    prend_info.colorAttachmentCount    = 1;
    prend_info.pColorAttachmentFormats = &m_colorFormat;
    prend_info.depthAttachmentFormat   = m_depthFormat;

    // Creating the Pipeline
    const VkColorComponentFlags allBits =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    nvvk::GraphicsPipelineState pipelineState;
    pipelineState.rasterizationState.cullMode        = VK_CULL_MODE_NONE;
    pipelineState.depthStencilState.depthWriteEnable = false;
    pipelineState.setBlendAttachmentState(0,  // Attachment
                                          nvvk::GraphicsPipelineState::makePipelineColorBlendAttachmentState(
                                              allBits, VK_TRUE,                     //
                                              VK_BLEND_FACTOR_SRC_ALPHA,            // Source color blend factor
                                              VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,  // Destination color blend factor
                                              VK_BLEND_OP_ADD,                      // Color blend operation
                                              VK_BLEND_FACTOR_SRC_ALPHA,            // Source alpha blend factor
                                              VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,  // Destination alpha blend factor
                                              VK_BLEND_OP_ADD                       // Alpha blend operation
                                              ));


    pipelineState.addBindingDescriptions({{0, sizeof(nvh::PrimitiveVertex)}});
    pipelineState.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, p))},  // Position
        {1, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, t))},     // TexCoords
    });

    nvvk::GraphicsPipelineGenerator pgen(m_device, m_dsetRaster->getPipeLayout(), prend_info, pipelineState);
#if USE_SLANG
    VkShaderModule shaderModule = nvvk::createShaderModule(m_device, &rasterSlang[0], sizeof(rasterSlang));
    pgen.addShader(shaderModule, VK_SHADER_STAGE_VERTEX_BIT, "vertexMain");
    pgen.addShader(shaderModule, VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain");
#else
    pgen.addShader(vert_shd, VK_SHADER_STAGE_VERTEX_BIT, "main");
    pgen.addShader(frag_shd, VK_SHADER_STAGE_FRAGMENT_BIT, "main");
#endif

    m_graphicsPipeline = pgen.createPipeline();
    m_dutil->setObjectName(m_graphicsPipeline, "Graphics");
    pgen.clearShaders();
#if USE_SLANG
    vkDestroyShaderModule(m_device, shaderModule, nullptr);
#endif
  }

  //-------------------------------------------------------------------------------------------------
  // Creating the descriptor set and all compute shaders
  void createComputeShaderObjectAndLayout()
  {
    VkPushConstantRange push_constant_ranges = {.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(DH::PushConstant)};

    // Create the layout used by the shader
    m_dsetCompute->addBinding(DH::eCompParticles, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dsetCompute->addBinding(DH::eCompSort, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dsetCompute->addBinding(DH::eCompSetting, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    m_dsetCompute->addBinding(DH::eThreadInspection, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);  // #INSPECTOR
    m_dsetCompute->addBinding(DH::eThreadMetadata, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);  // #INSPECTOR

    m_dsetCompute->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
    m_dsetCompute->initPipeLayout(1, &push_constant_ranges);

    auto shdInfo = VkShaderCreateInfoEXT{
        .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .pNext                  = NULL,
        .flags                  = VK_SHADER_CREATE_DISPATCH_BASE_BIT_EXT,
        .stage                  = VK_SHADER_STAGE_COMPUTE_BIT,
        .nextStage              = 0,
        .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
        .pName                  = "main",
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_dsetCompute->getLayout(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &push_constant_ranges,
        .pSpecializationInfo    = NULL,
    };

    // Compute shader description
    std::vector<VkShaderCreateInfoEXT> shaderCreateInfos(numCompShaders);
#if USE_GLSL
    shdInfo.codeSize                              = getShaderSize(calculateDensities_shd);
    shdInfo.pCode                                 = calculateDensities_shd.data();
    shaderCreateInfos[eCalculateDensitiesShd]     = shdInfo;
    shdInfo.codeSize                              = getShaderSize(calculatePressureForce_shd);
    shdInfo.pCode                                 = calculatePressureForce_shd.data();
    shaderCreateInfos[eCalculatePressureForceShd] = shdInfo;
    shdInfo.codeSize                              = getShaderSize(calculateViscosity_shd);
    shdInfo.pCode                                 = calculateViscosity_shd.data();
    shaderCreateInfos[eCalculateViscosityShd]     = shdInfo;
    shdInfo.codeSize                              = getShaderSize(externalForces_shd);
    shdInfo.pCode                                 = externalForces_shd.data();
    shaderCreateInfos[eExternalForcesShd]         = shdInfo;
    shdInfo.codeSize                              = getShaderSize(updatePositions_shd);
    shdInfo.pCode                                 = updatePositions_shd.data();
    shaderCreateInfos[eUpdatePositionsShd]        = shdInfo;
    shdInfo.codeSize                              = getShaderSize(updateSpatialHash_shd);
    shdInfo.pCode                                 = updateSpatialHash_shd.data();
    shaderCreateInfos[eUpdateSpatialHashShd]      = shdInfo;
    shdInfo.codeSize                              = getShaderSize(bitonicSort_shd);
    shdInfo.pCode                                 = bitonicSort_shd.data();
    shaderCreateInfos[eBitonicSort]               = shdInfo;
    shdInfo.codeSize                              = getShaderSize(bitonicSortOffsets_shd);
    shdInfo.pCode                                 = bitonicSortOffsets_shd.data();
    shaderCreateInfos[eBitonicSortOffsets]        = shdInfo;
#elif USE_SLANG
    shdInfo.codeSize                              = fluid_sim_2DSlang_sizeInBytes;
    shdInfo.pCode                                 = fluid_sim_2DSlang;
    shdInfo.pName                                 = "calculateDensity";
    shaderCreateInfos[eCalculateDensitiesShd]     = shdInfo;
    shdInfo.pName                                 = "calculatePressureForce";
    shaderCreateInfos[eCalculatePressureForceShd] = shdInfo;
    shdInfo.pName                                 = "calculateViscosity";
    shaderCreateInfos[eCalculateViscosityShd]     = shdInfo;
    shdInfo.pName                                 = "externalForces";
    shaderCreateInfos[eExternalForcesShd]         = shdInfo;
    shdInfo.pName                                 = "updatePositions";
    shaderCreateInfos[eUpdatePositionsShd]        = shdInfo;
    shdInfo.pName                                 = "updateSpatialHash";
    shaderCreateInfos[eUpdateSpatialHashShd]      = shdInfo;
    shdInfo.pName                                 = "bitonicSort";
    shaderCreateInfos[eBitonicSort]               = shdInfo;
    shdInfo.pName                                 = "bitonicSortOffset";
    shaderCreateInfos[eBitonicSortOffsets]        = shdInfo;
#endif

    // Create the shaders
    NVVK_CHECK(vkCreateShadersEXT(m_app->getDevice(), static_cast<int>(shaderCreateInfos.size()),
                                  shaderCreateInfos.data(), NULL, m_shaders.data()));
  }

  void createVkBuffers()
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    {
      // Buffer for the raster particles (square)
      m_bParticle.vertices = m_alloc->createBuffer(cmd, m_rasterParticle.vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      m_bParticle.indices  = m_alloc->createBuffer(cmd, m_rasterParticle.triangles, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
      m_dutil->DBG_NAME(m_bParticle.vertices.buffer);
      m_dutil->DBG_NAME(m_bParticle.indices.buffer);
    }

    // Buffer used by raster with updated information at each frame
    m_bFrameInfo = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bFrameInfo.buffer);

    // Buffer holding the particle settings
    m_bParticleSetting = m_alloc->createBuffer(sizeof(DH::ParticleSetting),
                                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bParticleSetting.buffer);

    // Buffer of the particles, used for the simulation
    m_bParticles = m_alloc->createBuffer(cmd, m_particles, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    m_dutil->DBG_NAME(m_bParticles.buffer);

    // Buffer used for sorting particles spatially
    m_bSpatialInfo = m_alloc->createBuffer(m_particles.size() * sizeof(DH::SpatialInfo),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    m_dutil->DBG_NAME(m_bSpatialInfo.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);

    // Updating the inspector with the buffers
    inspectorBufferInit();   // #INSPECTOR
    inspectorComputeInit();  // #INSPECTOR
  }

  void computeSimulation(VkCommandBuffer cmd)
  {
    auto _scope = m_dutil->DBG_SCOPE(cmd);
    auto sec    = g_profiler->timeRecurring(__FUNCTION__, cmd);  // #PROFILER

    if((m_settings.play || m_settings.runOnce))
    {
      DH::ParticleSetting& pSetting             = m_particleSetting;
      pSetting.poly6ScalingFactor               = 4.f / (glm::pi<float>() * glm::pow(pSetting.smoothingRadius, 8.f));
      pSetting.spikyPow3ScalingFactor           = 10.f / (glm::pi<float>() * glm::pow(pSetting.smoothingRadius, 5.f));
      pSetting.spikyPow2ScalingFactor           = 6.f / (glm::pi<float>() * glm::pow(pSetting.smoothingRadius, 4.f));
      pSetting.spikyPow3DerivativeScalingFactor = 30.f / (glm::pow(pSetting.smoothingRadius, 5.f) * glm::pi<float>());
      pSetting.spikyPow2DerivativeScalingFactor = 12.f / (glm::pow(pSetting.smoothingRadius, 4.f) * glm::pi<float>());
      pSetting.numParticles                     = NUM_PARTICLES;
      pSetting.deltaTime                        = std::min(1.f / 60.0f, ImGui::GetIO().DeltaTime);
      pSetting.boundsSize               = glm::vec2(m_gBuffers->getAspectRatio(), 1) * pSetting.boundsMultiplier;
      pSetting.interactionInputStrength = 0;
      pSetting.interactionInputRadius   = m_settings.interactionRadius * pSetting.boundsMultiplier;
      if(m_settings.pullInteraction || m_settings.pushInteraction)
        pSetting.interactionInputStrength = m_settings.pushInteraction ? -m_settings.interactionStrength : m_settings.interactionStrength;
      pSetting.interactionInputPoint = m_settings.mouseCoord;

      m_settings.runOnce = false;

      // Push descriptor set
      const VkDescriptorBufferInfo in_desc0{m_bParticles.buffer, 0, VK_WHOLE_SIZE};
      const VkDescriptorBufferInfo in_desc1{m_bSpatialInfo.buffer, 0, VK_WHOLE_SIZE};
      const VkDescriptorBufferInfo in_desc2{m_bParticleSetting.buffer, 0, VK_WHOLE_SIZE};
      const VkDescriptorBufferInfo in_desc3{g_inspectorElement->getComputeInspectionBuffer(0), 0, VK_WHOLE_SIZE};
      const VkDescriptorBufferInfo in_desc4{g_inspectorElement->getComputeMetadataBuffer(0), 0, VK_WHOLE_SIZE};

      std::vector<VkWriteDescriptorSet> writes;
      writes.push_back(m_dsetCompute->makeWrite(0, DH::eCompParticles, &in_desc0));
      writes.push_back(m_dsetCompute->makeWrite(0, DH::eCompSort, &in_desc1));
      writes.push_back(m_dsetCompute->makeWrite(0, DH::eCompSetting, &in_desc2));
      writes.push_back(m_dsetCompute->makeWrite(0, DH::eThreadInspection, &in_desc3));
      writes.push_back(m_dsetCompute->makeWrite(0, DH::eThreadMetadata, &in_desc4));
      vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_dsetCompute->getPipeLayout(), 0,
                                static_cast<uint32_t>(writes.size()), writes.data());

      // Bind compute shader
      const VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};

      int numBlocks = getNumBlocks();

      vkCmdUpdateBuffer(cmd, m_bParticleSetting.buffer, 0, sizeof(DH::ParticleSetting), &m_particleSetting);
      memoryBarrier(cmd);  // Make sure the buffer is ready before executing any dispatch shader


      dispatch(cmd, eExternalForcesShd, numBlocks, "ExternalForce");
      {
        auto _scopeSort = m_dutil->scopeLabel(cmd, "Sort");
        auto tsort      = g_profiler->timeRecurring("Sort", cmd);  // #PROFILER

        dispatch(cmd, eUpdateSpatialHashShd, numBlocks, "Hash");

        // Sorting: https://en.wikipedia.org/wiki/Bitonic_sorter
        // Launch each step of the sorting algorithm (once the previous step is complete)
        // Number of steps = [log2(n) * (log2(n) + 1)] / 2
        // where n = nearest power of 2 that is greater or equal to the number of inputs
        int numStages = static_cast<int>(std::log2(nextPowerOfTwo(NUM_PARTICLES))) + 1;

        for(int stageIndex = 0; stageIndex < numStages; stageIndex++)
        {
          for(int stepIndex = 0; stepIndex < stageIndex + 1; stepIndex++)
          {
            m_pushConst.groupWidth  = 1 << (stageIndex - stepIndex);
            m_pushConst.groupHeight = 2 * m_pushConst.groupWidth - 1;
            m_pushConst.stepIndex   = stepIndex;

            // Run the sorting step on the GPU
            dispatch(cmd, eBitonicSort, nextPowerOfTwo(NUM_PARTICLES) / 2);
          }
        }
        dispatch(cmd, eBitonicSortOffsets, NUM_PARTICLES, "Offsets");  // Calculate offset
      }
      dispatch(cmd, eCalculateDensitiesShd, numBlocks, "Density");
      dispatch(cmd, eCalculatePressureForceShd, numBlocks, "Pressure");
      dispatch(cmd, eCalculateViscosityShd, numBlocks, "Viscosity");
      dispatch(cmd, eUpdatePositionsShd, numBlocks, "Position");
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

  void renderParticles(VkCommandBuffer cmd)
  {
    auto _scope = m_dutil->DBG_SCOPE(cmd);
    auto sec    = g_profiler->timeRecurring(__FUNCTION__, cmd);  // #PROFILER

    // Update Frame buffer uniform buffer
    DH::FrameInfo finfo{};
    finfo.proj = glm::ortho(-1.F * m_gBuffers->getAspectRatio(), 1.F * m_gBuffers->getAspectRatio(), 1.F, -1.F, -1.F, 1.F);
    finfo.radius = m_settings.particleRadius;
    finfo.scale  = 1 / m_particleSetting.boundsMultiplier;
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);

    // Drawing the primitives in a G-Buffer
    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView()},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, {{0, 0, 0, 0}});
    r_info.pStencilAttachment = nullptr;


    // #INSPECTOR
    g_inspectorElement->clearFragmentVariables(cmd, 0);
    g_inspectorElement->updateMinMaxFragmentInspection(cmd, 0, m_settings.mouseWindowCoord - glm::vec2(1, 1),
                                                       m_settings.mouseWindowCoord + glm::vec2(1, 1));

    vkCmdBeginRendering(cmd, &r_info);  // Begin rendering commands
    m_app->setViewport(cmd);            // Set the viewport and scissor rectangle

    // Bind the graphics pipeline and descriptor sets, which contain shader resources
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dsetRaster->getPipeLayout(), 0, 1,
                            m_dsetRaster->getSets(), 0, nullptr);

    // Push constant information to the shaders
    vkCmdPushConstants(cmd, m_dsetRaster->getPipeLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(DH::PushConstant), &m_pushConst);

    // Bind the vertex buffer and its offset, and the index buffer used for indexed drawing.
    const VkDeviceSize offsets{0};
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_bParticle.vertices.buffer, &offsets);
    vkCmdBindIndexBuffer(cmd, m_bParticle.indices.buffer, 0, VK_INDEX_TYPE_UINT32);

    // Draw all particles in a single draw call, 2 triangles: always 6 indices
    vkCmdDrawIndexed(cmd, 6, static_cast<int>(m_particles.size()), 0, 0, 0);

    vkCmdEndRendering(cmd);  // End the rendering
  }


  // #INSPECTOR
  void inspectorBufferInit()
  {
    using EI = nvvkhl::ElementInspector;
    {
      EI::BufferInspectionInfo info{
          .name         = "Particles",
          .sourceBuffer = m_bParticles.buffer,
          .format =
              {
                  {EI::eF32Vec2, "position"},
                  {EI::eF32Vec2, "predictedPosition"},
                  {EI::eF32Vec2, "velocity"},
                  {EI::eF32Vec2, "density"},
              },
          .entryCount = NUM_PARTICLES,
      };
      g_inspectorElement->initBufferInspection(0, info);
    }
    {
      EI::BufferInspectionInfo info{
          .name         = "HashInfo",
          .sourceBuffer = m_bSpatialInfo.buffer,
          .format =
              {
                  {EI::eUint32, "originalIndex"},
                  {EI::eUint32, "hash"},
                  {EI::eUint32, "key"},
                  {EI::eUint32, "offset"},
              },
          .entryCount = NUM_PARTICLES,
      };
      g_inspectorElement->initBufferInspection(1, info);
    }
  }

  // #INSPECTOR
  void inspectorComputeInit()
  {
    using EI = nvvkhl::ElementInspector;
    EI::ComputeInspectionInfo info{
        .name             = "Pressure",
        .format           = {{EI::eF32Vec2, "pressure"}, {EI::eUint32, "Num Elem"}},
        .gridSizeInBlocks = {getNumBlocks(), 1, 1},  // What dispatch receives
        .blockSize        = {WORKGROUP_SIZE, 1, 1},  // Workgroup size, as defined in the shader
        .minBlock         = {0, 0, 0},
        .maxBlock         = {2, 0, 0},
        .minWarp          = {0u},
        .maxWarp          = {~0u},
    };

    g_inspectorElement->initComputeInspection(0, info);
  }

  // #INSPECTOR
  void inspectorViewportResize(VkExtent2D& size)
  {
    using EI = nvvkhl::ElementInspector;

    // Inspection of the image stored in m_gBuffers
    VkImageUsageFlags flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    VkImageCreateInfo createInfo = nvvk::makeImage2DCreateInfo(size, m_colorFormat, flags);

    EI::ImageInspectionInfo imageInspectInfo{
        .name        = "GBuffer-0",
        .createInfo  = createInfo,
        .sourceImage = m_gBuffers->getColorImage(0),
        .format      = EI::formatRGBA8(),  //VK_FORMAT_R8G8B8A8_UNORM
    };
    g_inspectorElement->initImageInspection(0, imageInspectInfo);

    // Inspection of a variable in the fragment shader within an area of the frame
    EI::FragmentInspectionInfo fragInspectInfo{
        .name        = "My Fragment Inspection",
        .format      = {{EI::eF32Vec2, "velocity"}},
        .renderSize  = glm::uvec2(size.width, size.height),
        .minFragment = glm::uvec2(0, 0),  // Inspection min/max corner, will be updated at rendering
        .maxFragment = glm::uvec2(2, 2),  // time, using the mouse position.
    };
    g_inspectorElement->initFragmentInspection(0, fragInspectInfo);

    // initFragmentInspection creates buffers, write the info in descriptor set
    const VkDescriptorBufferInfo inspectorInspection{g_inspectorElement->getFragmentInspectionBuffer(0), 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo inspectorMetadata{g_inspectorElement->getFragmentMetadataBuffer(0), 0, VK_WHOLE_SIZE};
    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_dsetRaster->makeWrite(0, DH::eFragInspectorData, &inspectorInspection));
    writes.emplace_back(m_dsetRaster->makeWrite(0, DH::eFragInspectorMeta, &inspectorMetadata));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  // Dispatch computer shader
  void dispatch(VkCommandBuffer cmd, int shaderID, int numBlocks, const std::string& label = "")
  {
    VkDebugUtilsLabelEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f}};
    if(!label.empty())
      vkCmdBeginDebugUtilsLabelEXT(cmd, &s);

    const VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};
    vkCmdBindShadersEXT(cmd, 1, stages, &m_shaders[shaderID]);
    vkCmdPushConstants(cmd, m_dsetCompute->getPipeLayout(), VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant), &m_pushConst);
    vkCmdDispatch(cmd, numBlocks, 1, 1);
    if(!label.empty())
      vkCmdEndDebugUtilsLabelEXT(cmd);
  }

  uint32_t nextPowerOfTwo(uint32_t n)
  {
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
  }

  void destroyResources()
  {
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    m_alloc->destroy(m_bParticle.vertices);
    m_alloc->destroy(m_bParticle.indices);
    m_alloc->destroy(m_bFrameInfo);
    m_alloc->destroy(m_bParticles);
    m_alloc->destroy(m_bSpatialInfo);
    m_alloc->destroy(m_bParticleSetting);

    for(auto shader : m_shaders)
      vkDestroyShaderEXT(m_app->getDevice(), shader, NULL);

    m_dsetRaster->deinit();
    m_dsetCompute->deinit();
    m_gBuffers.reset();
  }


  //-------------------------------------------------------------------------------------------------
  //
  std::unique_ptr<nvvkhl::AllocVma> m_alloc    = {};
  std::unique_ptr<nvvkhl::GBuffer>  m_gBuffers = {};
  std::unique_ptr<nvvk::DebugUtil>  m_dutil    = {};
  nvvkhl::Application*              m_app      = {nullptr};
  VkDevice                          m_device   = VK_NULL_HANDLE;

  VkFormat m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer

  // Scene resources
  nvh::PrimitiveMesh        m_rasterParticle;
  std::vector<DH::Particle> m_particles;
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;                                     // Buffer of the vertices
    nvvk::Buffer indices;                                      // Buffer of the indices
  } m_bParticle;                                               // Geometry
  nvvk::Buffer                            m_bParticles;        // All Positions
  nvvk::Buffer                            m_bSpatialInfo;      // Hash Spatial Info
  nvvk::Buffer                            m_bParticleSetting;  // Particle Settings
  nvvk::Buffer                            m_bFrameInfo;        // Raster frame info
  std::array<VkShaderEXT, numCompShaders> m_shaders = {};

  // Pipeline
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dsetRaster;                         // Descriptor set
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dsetCompute;                        // Descriptor set
  DH::PushConstant                              m_pushConst        = {};              // Information sent to the shader
  VkPipeline                                    m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
};


int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name = "Fluid Example";
  spec.vkSetup.setVersion(1, 3);
  spec.vSync  = true;
  spec.width  = 1700;
  spec.height = 900;

  // Setting up the layout of the application
  spec.dockSetup = [](ImGuiID viewportID) {
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.2F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(settingID, ImGuiDir_Down, 0.35F, nullptr, &settingID);
    ImGui::DockBuilderDockWindow("Profiler", profilerID);
    ImGuiID nvmlID = ImGui::DockBuilderSplitNode(profilerID, ImGuiDir_Down, 0.5F, nullptr, &profilerID);
    ImGui::DockBuilderDockWindow("NVML Monitor", profilerID /*nvmlID*/);
    ImGuiID inspectorID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.3F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Inspector", inspectorID);
  };

  // Required extra extensions
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  spec.vkSetup.addDeviceExtension(VK_EXT_SHADER_OBJECT_EXTENSION_NAME, false, &shaderObjFeature);
  spec.vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // Global element used in sample
  g_profiler         = std::make_shared<nvvkhl::ElementProfiler>();   // #PROFILER
  g_inspectorElement = std::make_shared<nvvkhl::ElementInspector>();  // #INSPECTOR

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

  app->addElement(test);                                            // Command line Testing application
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());  // File,  Help
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info  // Window title
  app->addElement(std::make_shared<nvvkhl::ElementNvml>(true));  // NVML monitor
  app->addElement(g_profiler);                                   // #PROFILER GPU profiler
  app->addElement(g_inspectorElement);                           // #INSPECTOR Vk object inspector
  app->addElement(std::make_shared<RealtimeAnalysis>());         // This sample

  g_profiler->setLabelUsage(false);  // Not using the "auto debug scope naming", using our own instead

  app->run();

  return test->errorCode();
}
