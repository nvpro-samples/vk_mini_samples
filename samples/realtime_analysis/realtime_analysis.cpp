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

#include <glm/gtc/matrix_transform.hpp>
#include <random>

#define USE_SLANG 0  // SLANG is not implemented yet.
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
// clang-format on

#define VMA_IMPLEMENTATION
#define USE_GLSL 1

#include <fmt/format.h>

namespace DH {
using namespace glm;
#include "shaders/device_host.h"
}  // namespace DH

#include "realtime_analysis.h"


// Adding the compiled Vulkan shaders
#if USE_GLSL
#include "_autogen/bitonic_sort.comp.glsl.h"
#include "_autogen/bitonic_sort_offsets.comp.glsl.h"
#include "_autogen/calculate_densities.comp.glsl.h"
#include "_autogen/calculate_pressure_force.comp.glsl.h"
#include "_autogen/calculate_viscosity.comp.glsl.h"
#include "_autogen/external_forces.comp.glsl.h"
#include "_autogen/raster.frag.glsl.h"
#include "_autogen/raster.vert.glsl.h"
#include "_autogen/update_positions.comp.glsl.h"
#include "_autogen/update_spatial_hash.comp.glsl.h"
const auto& vert_shd = std::vector<uint32_t>{std::begin(raster_vert_glsl), std::end(raster_vert_glsl)};
const auto& frag_shd = std::vector<uint32_t>{std::begin(raster_frag_glsl), std::end(raster_frag_glsl)};
const auto& calculateDensities_shd =
    std::vector<uint32_t>{std::begin(calculate_densities_comp_glsl), std::end(calculate_densities_comp_glsl)};
const auto& calculatePressureForce_shd =
    std::vector<uint32_t>{std::begin(calculate_pressure_force_comp_glsl), std::end(calculate_pressure_force_comp_glsl)};
const auto& calculateViscosity_shd =
    std::vector<uint32_t>{std::begin(calculate_viscosity_comp_glsl), std::end(calculate_viscosity_comp_glsl)};
const auto& externalForces_shd =
    std::vector<uint32_t>{std::begin(external_forces_comp_glsl), std::end(external_forces_comp_glsl)};
const auto& updatePositions_shd =
    std::vector<uint32_t>{std::begin(update_positions_comp_glsl), std::end(update_positions_comp_glsl)};
const auto& updateSpatialHash_shd =
    std::vector<uint32_t>{std::begin(update_spatial_hash_comp_glsl), std::end(update_spatial_hash_comp_glsl)};
const auto& bitonicSort_shd = std::vector<uint32_t>{std::begin(bitonic_sort_comp_glsl), std::end(bitonic_sort_comp_glsl)};
const auto& bitonicSortOffsets_shd =
    std::vector<uint32_t>{std::begin(bitonic_sort_offsets_comp_glsl), std::end(bitonic_sort_offsets_comp_glsl)};
#elif USE_SLANG
#include "_autogen/fluid_sim_2D_slang.h"
#include "_autogen/raster_slang.h"
#endif

#include "common/utils.hpp"


#include <nvapp/elem_default_menu.hpp>
#include <nvapp/elem_default_title.hpp>
#include <nvapp/elem_inspector.hpp>
#include <nvapp/elem_profiler.hpp>
#include <nvgpu_monitor/elem_gpu_monitor.hpp>
#include <nvgui/fonts.hpp>
#include <nvgui/property_editor.hpp>
#include <nvutils/logger.hpp>
#include <nvutils/parameter_parser.hpp>
#include <nvutils/primitives.hpp>
#include <nvvk/check_error.hpp>
#include <nvvk/context.hpp>
#include <nvvk/debug_util.hpp>
#include <nvvk/default_structs.hpp>
#include <nvvk/descriptors.hpp>
#include <nvvk/formats.hpp>
#include <nvvk/gbuffers.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/helpers.hpp>
#include <nvvk/profiler_vk.hpp>
#include <nvvk/sampler_pool.hpp>
#include <nvvk/staging.hpp>


// Elements used by the sample
nvutils::ProfilerManager                 g_profilerManager;   // #PROFILER
std::shared_ptr<nvapp::ElementInspector> g_inspectorElement;  // #INSPECTOR

//-------------------------------------------------------------------------------------------------
// This sample implement a simple fluid simulation.
//
// It derives from IAppElement, which is attached to nvvkh::Application. The Application is
// calling the initialization of Vulkan, GLFW and ImGUI. It runs in an infinite loop and
// call the function from the IAppElement interface, such as onRender, onUIRender, onResize, ..
//
class RealtimeAnalysis : public nvapp::IAppElement
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

  struct PipelineResources
  {
    VkPipeline           pipeline{};        // Graphic pipeline to render
    VkPipelineLayout     pipelineLayout{};  // Pipeline layout
    nvvk::DescriptorPack descriptorPack{};  // Descriptor bindings, layout, pool, and set
  };

  DH::ParticleSetting m_particleSetting = TestA;  // Initialized with Test-A


public:
  RealtimeAnalysis()           = default;
  ~RealtimeAnalysis() override = default;

  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = app->getDevice();
    m_alloc.init(VmaAllocatorCreateInfo{
        .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = app->getPhysicalDevice(),
        .device         = app->getDevice(),
        .instance       = app->getInstance(),
    });  // Allocator


    // Acquiring the sampler which will be used for displaying the GBuffer
    m_samplerPool.init(app->getDevice());
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // GBuffer
    m_depthFormat = nvvk::findDepthFormat(app->getPhysicalDevice());
    m_gBuffers.init({
        .allocator      = &m_alloc,
        .colorFormats   = {m_colorFormat},  // Only one GBuffer color attachment
        .depthFormat    = m_depthFormat,
        .imageSampler   = linearSampler,
        .descriptorPool = m_app->getTextureDescriptorPool(),
    });

    // #INSPECTOR
    nvapp::ElementInspector::InitInfo inspectInfo{
        .device        = m_device,
        .queueInfo     = m_app->getQueue(0),
        .allocator     = &m_alloc,
        .imageCount    = 1u,
        .bufferCount   = 2u,
        .computeCount  = 1u,
        .fragmentCount = 1u,
    };
    g_inspectorElement->init(inspectInfo);

    m_profilerTimeline = g_profilerManager.createTimeline({.name = "Primary Timeline"});
    m_profilerGpuTimer.init(m_profilerTimeline, m_app->getDevice(), m_app->getPhysicalDevice(), m_app->getQueue(0).familyIndex, false);


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

  void onResize(VkCommandBuffer cmd, const VkExtent2D& viewportSize) override
  {
    m_gBuffers.update(cmd, viewportSize);

    inspectorViewportResize(viewportSize);  // #INSPECTOR
  }

  void onUIRender() override
  {
    namespace PE            = nvgui::PropertyEditor;
    DH::ParticleSetting& pS = m_particleSetting;
    ImGui::Begin("Settings");
    ImGui::TextDisabled("%d FPS / %.3fms", static_cast<int>(ImGui::GetIO().Framerate), 1000.F / ImGui::GetIO().Framerate);
    PE::begin();
    ImGui::SeparatorText("Visual");
    PE::SliderFloat("Radius", &m_settings.particleRadius, 0.005f, 0.05f);
    PE::SliderFloat("Volume", (float*)&pS.boundsMultiplier, 1.0f, 15.0f);
    ImGui::SeparatorText("Physics");
    PE::SliderFloat("Gravity", &pS.gravity, -10.0, 0);
    PE::SliderFloat("Collision Damping", &pS.collisionDamping, 0.0, 1, "%.5f", ImGuiSliderFlags_Logarithmic);
    PE::SliderFloat("Smoothing Radius", &pS.smoothingRadius, 0.2f, 2, "%.5f", ImGuiSliderFlags_Logarithmic);
    PE::SliderFloat("Target Density", &pS.targetDensity, 0.0, 500);
    PE::SliderFloat("Pressure Multiplier", &pS.pressureMultiplier, 0.0, 100);
    PE::SliderFloat("Near Pressure Multiplier", &pS.nearPressureMultiplier, 0.0f, 100);
    PE::SliderFloat("Viscosity Strength", &pS.viscosityStrength, 0.0f, 0.5f, "%.5f", ImGuiSliderFlags_Logarithmic);
    ImGui::SeparatorText("Interaction");
    PE::SliderFloat("Interaction Strength", &m_settings.interactionStrength, 0.0f, 100);
    PE::SliderFloat("Interaction radius", &m_settings.interactionRadius, 0.0f, .5f);
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
    ImGui::PushFont(nvgui::getIconicFont());
    if(ImGui::Button(m_settings.play ? nvgui::icon_media_pause : nvgui::icon_media_play) || ImGui::IsKeyPressed(ImGuiKey_Space))
      m_settings.play = !m_settings.play;
    ImGui::SameLine();
    if(ImGui::Button(nvgui::icon_media_step_forward) || ImGui::IsKeyPressed(ImGuiKey_RightArrow))
    {
      m_settings.runOnce = true;
      m_settings.play    = false;
    }
    ImGui::SameLine();
    if(ImGui::Button(nvgui::icon_media_skip_backward) || ImGui::IsKeyPressed(ImGuiKey_R))
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
    ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(0U), ImGui::GetContentRegionAvail());
    ImGui::End();
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);
    m_profilerTimeline->frameAdvance();

    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

    computeSimulation(cmd);
    renderParticles(cmd);

    // #INSPECTOR
    {
      nvvk::DebugUtil::ScopedCmdLabel _scopeInspection(cmd, "Inspection");

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
    if(m_bParticles.buffer != nullptr)
    {
      auto                  cmd = m_app->createTempCmdBuffer();
      nvvk::StagingUploader uploader;
      uploader.init(&m_alloc);

      uploader.appendBuffer(m_bParticles, 0, std::span(m_particles));
      uploader.cmdUploadAppended(cmd);

      m_app->submitAndWaitTempCmdBuffer(cmd);
      uploader.deinit();
    }
  }


  void createRasterPipeline()
  {
    nvvk::DescriptorBindings& bindings = m_rasterPipeline.descriptorPack.bindings;
    bindings.addBinding(DH::eFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(DH::eParticles, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(DH::eFragInspectorData, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);  // #INSPECTOR
    bindings.addBinding(DH::eFragInspectorMeta, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);  // #INSPECTOR

    NVVK_CHECK(m_rasterPipeline.descriptorPack.initFromBindings(m_device, 1));
    NVVK_DBG_NAME(m_rasterPipeline.descriptorPack.layout);
    NVVK_DBG_NAME(m_rasterPipeline.descriptorPack.pool);
    NVVK_DBG_NAME(m_rasterPipeline.descriptorPack.sets[0]);

    const VkPushConstantRange push_constant_ranges{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                                   .offset = 0,
                                                   .size   = sizeof(DH::PushConstant)};
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_rasterPipeline.pipelineLayout,
                                          {m_rasterPipeline.descriptorPack.layout}, {push_constant_ranges}));
    NVVK_DBG_NAME(m_rasterPipeline.pipelineLayout);

    // Writing to descriptors
    const VkDescriptorBufferInfo dbi_frameinfo{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo dbi_particles{m_bParticles.buffer, 0, VK_WHOLE_SIZE};
    // #INSPECTOR : Inspector bindings are done in inspectorViewportResize()
    nvvk::WriteSetContainer writeContainer;
    VkDescriptorSet         set = m_rasterPipeline.descriptorPack.sets[0];
    writeContainer.append(bindings.getWriteSet(DH::eFrameInfo, set), m_bFrameInfo);
    writeContainer.append(bindings.getWriteSet(DH::eParticles, set), m_bParticles);
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);

    VkPipelineRenderingCreateInfo prend_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    prend_info.colorAttachmentCount    = 1;
    prend_info.pColorAttachmentFormats = &m_colorFormat;
    prend_info.depthAttachmentFormat   = m_depthFormat;

    // Creating the Pipeline
    const VkColorComponentFlags allBits =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    nvvk::GraphicsPipelineState pipelineState;
    pipelineState.rasterizationState.cullMode                = VK_CULL_MODE_NONE;
    pipelineState.depthStencilState.depthWriteEnable         = false;
    pipelineState.colorBlendEnables[0]                       = VK_TRUE;
    pipelineState.colorBlendEquations[0].alphaBlendOp        = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].colorBlendOp        = VK_BLEND_OP_ADD;
    pipelineState.colorBlendEquations[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    pipelineState.colorBlendEquations[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;


    pipelineState.vertexBindings   = {{.sType   = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
                                       .stride  = sizeof(nvutils::PrimitiveVertex),
                                       .divisor = 1}};
    pipelineState.vertexAttributes = {{.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                       .location = 0,
                                       .format   = VK_FORMAT_R32G32B32_SFLOAT,
                                       .offset   = offsetof(nvutils::PrimitiveVertex, pos)},
                                      {.sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                       .location = 1,
                                       .format   = VK_FORMAT_R32G32_SFLOAT,
                                       .offset   = offsetof(nvutils::PrimitiveVertex, tex)}};


    nvvk::GraphicsPipelineCreator creator;
    creator.pipelineInfo.layout                  = m_rasterPipeline.pipelineLayout;
    creator.colorFormats                         = {m_colorFormat};
    creator.renderingState.depthAttachmentFormat = m_depthFormat;

#if USE_SLANG
    assert("Not implemented");
#else
    creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", vert_shd);
    creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", frag_shd);
#endif

    NVVK_CHECK(creator.createGraphicsPipeline(m_device, nullptr, pipelineState, &m_rasterPipeline.pipeline));
    NVVK_DBG_NAME(m_rasterPipeline.pipeline);
  }

  //-------------------------------------------------------------------------------------------------
  // Creating the descriptor set and all compute shaders
  void createComputeShaderObjectAndLayout()
  {
    // Create the layout used by the shader
    nvvk::DescriptorBindings& bindings = m_computePipeline.descriptorPack.bindings;
    bindings.addBinding(DH::eCompParticles, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(DH::eCompSort, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(DH::eCompSetting, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
    bindings.addBinding(DH::eThreadInspection, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);  // #INSPECTOR
    bindings.addBinding(DH::eThreadMetadata, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL);  // #INSPECTOR

    NVVK_CHECK(m_computePipeline.descriptorPack.initFromBindings(m_device, 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR));
    NVVK_DBG_NAME(m_computePipeline.descriptorPack.layout);


    const VkPushConstantRange push_constant_ranges = {.stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(DH::PushConstant)};
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_computePipeline.pipelineLayout,
                                          {m_computePipeline.descriptorPack.layout}, {push_constant_ranges}));
    NVVK_DBG_NAME(m_computePipeline.pipelineLayout);


    auto shdInfo = VkShaderCreateInfoEXT{
        .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .pNext                  = NULL,
        .flags                  = VK_SHADER_CREATE_DISPATCH_BASE_BIT_EXT,
        .stage                  = VK_SHADER_STAGE_COMPUTE_BIT,
        .nextStage              = 0,
        .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
        .pName                  = "main",
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_computePipeline.descriptorPack.layout,
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
                                  shaderCreateInfos.data(), nullptr, m_shaders.data()));
  }

  void createVkBuffers()
  {
    VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
    nvvk::StagingUploader uploader;
    uploader.init(&m_alloc);
    {
      // Buffer for the raster particles (square)
      NVVK_CHECK(m_alloc.createBuffer(m_bParticle.vertices, std::span(m_rasterParticle.vertices).size_bytes(),
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
      NVVK_CHECK(m_alloc.createBuffer(m_bParticle.indices, std::span(m_rasterParticle.triangles).size_bytes(),
                                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT));
      NVVK_CHECK(uploader.appendBuffer(m_bParticle.vertices, 0, std::span(m_rasterParticle.vertices)));
      NVVK_CHECK(uploader.appendBuffer(m_bParticle.indices, 0, std::span(m_rasterParticle.triangles)));
      NVVK_DBG_NAME(m_bParticle.vertices.buffer);
      NVVK_DBG_NAME(m_bParticle.indices.buffer);
    }

    // Buffer used by raster with updated information at each frame
    NVVK_CHECK(m_alloc.createBuffer(m_bFrameInfo, sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT));
    NVVK_DBG_NAME(m_bFrameInfo.buffer);

    // Buffer holding the particle settings
    NVVK_CHECK(m_alloc.createBuffer(m_bParticleSetting, sizeof(DH::ParticleSetting),
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT));
    NVVK_DBG_NAME(m_bParticleSetting.buffer);

    // Buffer of the particles, used for the simulation
    NVVK_CHECK(m_alloc.createBuffer(m_bParticles, std::span(m_particles).size_bytes(),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT));
    NVVK_CHECK(uploader.appendBuffer(m_bParticles, 0, std::span(m_particles)));
    NVVK_DBG_NAME(m_bParticles.buffer);

    // Buffer used for sorting particles spatially
    NVVK_CHECK(m_alloc.createBuffer(m_bSpatialInfo, m_particles.size() * sizeof(DH::SpatialInfo),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT));
    NVVK_DBG_NAME(m_bSpatialInfo.buffer);

    uploader.cmdUploadAppended(cmd);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    uploader.deinit();

    // Updating the inspector with the buffers
    inspectorBufferInit();   // #INSPECTOR
    inspectorComputeInit();  // #INSPECTOR
  }

  void computeSimulation(VkCommandBuffer cmd)
  {
    NVVK_DBG_SCOPE(cmd);

    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

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
      pSetting.boundsSize                       = glm::vec2(m_gBuffers.getAspectRatio(), 1) * pSetting.boundsMultiplier;
      pSetting.interactionInputStrength         = 0;
      pSetting.interactionInputRadius           = m_settings.interactionRadius * pSetting.boundsMultiplier;
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

      nvvk::WriteSetContainer         writeContainer;
      const nvvk::DescriptorBindings& computeBindings = m_computePipeline.descriptorPack.bindings;
      writeContainer.append(computeBindings.getWriteSet(DH::eCompParticles), m_bParticles);
      writeContainer.append(computeBindings.getWriteSet(DH::eCompSort), m_bSpatialInfo);
      writeContainer.append(computeBindings.getWriteSet(DH::eCompSetting), m_bParticleSetting);
      writeContainer.append(computeBindings.getWriteSet(DH::eThreadInspection), g_inspectorElement->getComputeInspectionBuffer(0));
      writeContainer.append(computeBindings.getWriteSet(DH::eThreadMetadata), g_inspectorElement->getComputeMetadataBuffer(0));

      vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline.pipelineLayout, 0,
                                static_cast<uint32_t>(writeContainer.size()), writeContainer.data());


      int numBlocks = getNumBlocks();  // Number of working blocks

      vkCmdUpdateBuffer(cmd, m_bParticleSetting.buffer, 0, sizeof(DH::ParticleSetting), &m_particleSetting);
      memoryBarrier(cmd);  // Make sure the buffer is ready before executing any dispatch shader


      dispatch(cmd, eExternalForcesShd, numBlocks, "ExternalForce");
      {
        nvvk::DebugUtil::ScopedCmdLabel _scopeSort(cmd, "Sort");

        auto timerSection2 = m_profilerGpuTimer.cmdFrameSection(cmd, "Sort");

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
    NVVK_DBG_SCOPE(cmd);

    auto timerSection = m_profilerGpuTimer.cmdFrameSection(cmd, __FUNCTION__);

    // Update Frame buffer uniform buffer
    DH::FrameInfo finfo{};
    finfo.proj = glm::ortho(-1.F * m_gBuffers.getAspectRatio(), 1.F * m_gBuffers.getAspectRatio(), 1.F, -1.F, -1.F, 1.F);
    finfo.radius = m_settings.particleRadius;
    finfo.scale  = 1 / m_particleSetting.boundsMultiplier;
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);
    nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);


    // Drawing the primitives in a G-Buffer
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers.getColorImageView();
    VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
    depthAttachment.imageView                 = m_gBuffers.getDepthImageView();
    depthAttachment.clearValue                = {.depthStencil = DEFAULT_VkClearDepthStencilValue};

    // Create the rendering info
    VkRenderingInfo renderingInfo      = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea           = DEFAULT_VkRect2D(m_gBuffers.getSize());
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments    = &colorAttachment;
    renderingInfo.pDepthAttachment     = &depthAttachment;


    // #INSPECTOR
    g_inspectorElement->clearFragmentVariables(cmd, 0);
    g_inspectorElement->updateMinMaxFragmentInspection(cmd, 0, m_settings.mouseWindowCoord - glm::vec2(1, 1),
                                                       m_settings.mouseWindowCoord + glm::vec2(1, 1));

    vkCmdBeginRendering(cmd, &renderingInfo);  // Begin rendering commands
    nvvk::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, m_gBuffers.getSize());

    // Bind the graphics pipeline and descriptor sets, which contain shader resources
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_rasterPipeline.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_rasterPipeline.pipelineLayout, 0, 1,
                            &m_rasterPipeline.descriptorPack.sets[0], 0, nullptr);

    // Push constant information to the shaders
    vkCmdPushConstants(cmd, m_rasterPipeline.pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(DH::PushConstant), &m_pushConst);

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
    using EI = nvapp::ElementInspector;
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
    using EI = nvapp::ElementInspector;
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
  void inspectorViewportResize(const VkExtent2D& size)
  {
    using EI = nvapp::ElementInspector;

    // Inspection of the image stored in m_gBuffers
    VkImageUsageFlags flags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT
                              | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VkImageCreateInfo createInfo = DEFAULT_VkImageCreateInfo;
    createInfo.format            = m_colorFormat;
    createInfo.extent            = {size.width, size.height, 1};
    createInfo.usage             = flags;

    EI::ImageInspectionInfo imageInspectInfo{
        .name        = "GBuffer-0",
        .createInfo  = createInfo,
        .sourceImage = m_gBuffers.getColorImage(0),
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

    nvvk::WriteSetContainer         writeContainer;
    const nvvk::DescriptorBindings& rasterBindings = m_rasterPipeline.descriptorPack.bindings;
    writeContainer.append(rasterBindings.getWriteSet(DH::eFragInspectorData, m_rasterPipeline.descriptorPack.sets[0]),
                          g_inspectorElement->getFragmentInspectionBuffer(0));
    writeContainer.append(rasterBindings.getWriteSet(DH::eFragInspectorMeta, m_rasterPipeline.descriptorPack.sets[0]),
                          g_inspectorElement->getFragmentMetadataBuffer(0));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writeContainer.size()), writeContainer.data(), 0, nullptr);
  }

  // Dispatch computer shader
  void dispatch(VkCommandBuffer cmd, int shaderID, int numBlocks, const std::string& label = "")
  {
    VkDebugUtilsLabelEXT s{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT, nullptr, label.c_str(), {1.0f, 1.0f, 1.0f, 1.0f}};
    if(!label.empty())
      vkCmdBeginDebugUtilsLabelEXT(cmd, &s);

    const VkShaderStageFlagBits stages[1] = {VK_SHADER_STAGE_COMPUTE_BIT};
    vkCmdBindShadersEXT(cmd, 1, stages, &m_shaders[shaderID]);
    vkCmdPushConstants(cmd, m_computePipeline.pipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(DH::PushConstant), &m_pushConst);

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
    vkDestroyPipeline(m_device, m_rasterPipeline.pipeline, nullptr);

    m_alloc.destroyBuffer(m_bParticle.vertices);
    m_alloc.destroyBuffer(m_bParticle.indices);
    m_alloc.destroyBuffer(m_bFrameInfo);
    m_alloc.destroyBuffer(m_bParticles);
    m_alloc.destroyBuffer(m_bSpatialInfo);
    m_alloc.destroyBuffer(m_bParticleSetting);

    vkDestroyPipelineLayout(m_device, m_rasterPipeline.pipelineLayout, nullptr);
    m_rasterPipeline.descriptorPack.deinit();

    vkDestroyPipelineLayout(m_device, m_computePipeline.pipelineLayout, nullptr);
    m_computePipeline.descriptorPack.deinit();


    for(auto shader : m_shaders)
      vkDestroyShaderEXT(m_app->getDevice(), shader, NULL);

    // #PROFILER
    m_profilerGpuTimer.deinit();
    g_profilerManager.destroyTimeline(m_profilerTimeline);

    m_gBuffers.deinit();
    m_samplerPool.deinit();
    m_alloc.deinit();
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers.getColorImage(), m_gBuffers.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  //-------------------------------------------------------------------------------------------------
  //
  nvapp::Application*     m_app      = {nullptr};
  nvvk::ResourceAllocator m_alloc    = {};
  nvvk::GBuffer           m_gBuffers = {};
  nvvk::SamplerPool       m_samplerPool{};  // The sampler pool, used to create a sampler for the texture

  VkDevice m_device = VK_NULL_HANDLE;

  VkFormat m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer

  // Scene resources
  nvutils::PrimitiveMesh    m_rasterParticle;
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
  PipelineResources m_rasterPipeline;
  PipelineResources m_computePipeline;

  // Profiler
  nvutils::ProfilerTimeline* m_profilerTimeline{};
  nvvk::ProfilerGpuTimer     m_profilerGpuTimer;

  DH::PushConstant m_pushConst        = {};              // Information sent to the shader
  VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
};


int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;
  appInfo.name                  = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);
  appInfo.windowSize            = {1700, 900};
  appInfo.hasUndockableViewport = true;

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  reg.add({"frames", "Number of frames to render in headless mode"}, &appInfo.headlessFrameCount, true);
  reg.addVector({"winSize", "Width and height of the window"}, &appInfo.windowSize);
  cli.add(reg);
  cli.parse(argc, argv);

  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};

  nvvk::ContextInitInfo vkSetup;
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  }
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjFeature});
  vkSetup.deviceExtensions.push_back({VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME});

  // Create Vulkan context
  nvvk::Context vkContext;
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }

  // Setting up the Vulkan part of the application
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Setting up the layout of the application
  appInfo.dockSetup = [](ImGuiID viewportID) {
    ImGuiID settingID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.2F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Settings", settingID);
    ImGuiID profilerID = ImGui::DockBuilderSplitNode(settingID, ImGuiDir_Down, 0.35F, nullptr, &settingID);
    ImGui::DockBuilderDockWindow("Profiler", profilerID);
    ImGuiID nvmlID = ImGui::DockBuilderSplitNode(profilerID, ImGuiDir_Down, 0.5F, nullptr, &profilerID);
    ImGui::DockBuilderDockWindow("NVML Monitor", profilerID /*nvmlID*/);
    ImGuiID inspectorID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Left, 0.3F, nullptr, &viewportID);
    ImGui::DockBuilderDockWindow("Inspector", inspectorID);
  };

  // Create the application
  nvapp::Application app;
  app.init(appInfo);


  // #PROFILER
  // setup the profiler element and views
  nvapp::ElementProfiler::ViewSettings viewSettings{.name       = "Profiler",
                                                    .defaultTab = nvapp::ElementProfiler::TABLE,
                                                    .pieChart   = {.cpuTotal = false, .levels = 2},
                                                    .lineChart  = {.cpuLine = false}};
  // setting are optional, but can be used to expose to sample code (like hiding views for benchmark)

  auto elemProfiler = std::make_shared<nvapp::ElementProfiler>(
      &g_profilerManager, std::make_shared<nvapp::ElementProfiler::ViewSettings>(std::move(viewSettings)));


  g_inspectorElement = std::make_shared<nvapp::ElementInspector>();  // #INSPECTOR


  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());  // File,  Help
  app.addElement(std::make_shared<nvapp::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info  // Window title
  app.addElement(std::make_shared<nvgpu_monitor::ElementGpuMonitor>());  // NVML monitor
  app.addElement(elemProfiler);                                          // #PROFILER GPU profiler
  app.addElement(g_inspectorElement);                                    // #INSPECTOR Vk object inspector
  app.addElement(std::make_shared<RealtimeAnalysis>());                  // This sample

  // g_profiler->setLabelUsage(false);  // Not using the "auto debug scope naming", using our own instead

  app.run();
  app.deinit();
  vkContext.deinit();

  return 0;
}
