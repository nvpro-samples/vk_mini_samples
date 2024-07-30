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

 This sample creates a 3D cube and render using the builtin camera

*/
//////////////////////////////////////////////////////////////////////////


#define VMA_IMPLEMENTATION
#include "common/vk_context.hpp"
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvh/primitives.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/extensions_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/gbuffer.hpp"

#include "common/alloc_dma.hpp"

namespace DH {
using namespace glm;
#include "shaders/device_host.h"  // Shared between host and device
}  // namespace DH

#if USE_HLSL
#include "_autogen/raster_vertexMain.spirv.h"
#include "_autogen/raster_fragmentMain.spirv.h"
const auto& vert_shd = std::vector<uint8_t>{std::begin(raster_vertexMain), std::end(raster_vertexMain)};
const auto& frag_shd = std::vector<uint8_t>{std::begin(raster_fragmentMain), std::end(raster_fragmentMain)};
#elif USE_SLANG
#include "_autogen/raster_slang.h"
#else
#include "_autogen/raster.frag.glsl.h"
#include "_autogen/raster.vert.glsl.h"
const auto& vert_shd = std::vector<uint32_t>{std::begin(raster_vert_glsl), std::end(raster_vert_glsl)};
const auto& frag_shd = std::vector<uint32_t>{std::begin(raster_frag_glsl), std::end(raster_frag_glsl)};
#endif  // USE_HLSL

//////////////////////////////////////////////////////////////////////////
/// </summary> Display an image on a quad.
class ShaderObject : public nvvkhl::IAppElement
{
  // Application settings
  struct Settings
  {
    float           lineWidth        = 1.0F;
    VkFrontFace     frontFace        = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkCompareOp     depthCompare     = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkBool32        depthTestEnable  = VK_TRUE;
    VkBool32        depthWriteEnable = VK_TRUE;
    VkPolygonMode   polygonMode      = VK_POLYGON_MODE_FILL;
    VkCullModeFlags cullMode         = VK_CULL_MODE_BACK_BIT;
  };


public:
  ShaderObject() { m_pushConst.pointSize = 1.0F; }
  ~ShaderObject() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();
    m_dutil  = std::make_unique<nvvk::DebugUtil>(m_device);                                 // Debug utility
    m_alloc  = std::make_unique<AllocDma>(m_app->getDevice(), m_app->getPhysicalDevice());  // Allocator
    //m_alloc  = std::make_unique<nvvkhl::AllocVma>(VmaAllocatorCreateInfo{
    //     .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
    //     .physicalDevice = app->getPhysicalDevice(),
    //     .device         = app->getDevice(),
    //     .instance       = app->getInstance(),
    //});  // Allocator
    m_dset = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    createScene();
    createSceneDataBuffers();
    createFrameInfoBuffer();
    createDescriptorSet();
    createShaderObjects();  // #SHADER_OBJECT
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(uint32_t width, uint32_t height) override { createGbuffers({width, height}); }

  void onUIRender() override
  {
    if(!m_gBuffers)
      return;

    {  // Setting menu
      namespace PE = ImGuiH::PropertyEditor;

      ImGui::Begin("Settings");
      ImGuiH::CameraWidget();

      ImGui::Text("Menger Sponge");
      {
        static int   seed        = 1;
        static float probability = 0.7F;
        bool         update      = false;
        PE::begin();
        update |= PE::entry("Probability", [&] { return ImGui::SliderFloat("##1", &probability, 0, 1); });
        update |= PE::entry("Seed", [&] { return ImGui::SliderInt("##1", &seed, 1, 100); });
        if(update)
        {
          probability = (probability < 0.025F ? -1 : probability);
          vkDeviceWaitIdle(m_device);
          destroyPrimitiveMeshResources();
          createMesh(probability, seed);
          createSceneDataBuffers();
        }
        PE::end();
      }

      ImGui::Text("Dynamic Pipeline");
      PE::begin();
      PE::entry("Point Size", [&] { return ImGui::SliderFloat("##1", &m_pushConst.pointSize, 1, 10); });
      PE::entry("Line Width", [&] { return ImGui::SliderFloat("##1", &m_settings.lineWidth, 1, 10); });
      PE::entry("Depth Test", [&] { return ImGui::Checkbox("##1", (bool*)&m_settings.depthTestEnable); });
      PE::entry("Depth Write", [&] { return ImGui::Checkbox("##1", (bool*)&m_settings.depthWriteEnable); });
      {
        const char* items[] = {"Counter Clockwise", "Clockwise"};
        PE::entry("Front Face",
                  [&] { return ImGui::Combo("combo", (int*)&m_settings.frontFace, items, IM_ARRAYSIZE(items)); });
      }
      {
        const char* items[] = {"None", "Front", "Back", "Front&Back"};
        PE::entry("Cull Mode",
                  [&] { return ImGui::Combo("combo", (int*)&m_settings.cullMode, items, IM_ARRAYSIZE(items)); });
      }
      {
        const char* items[] = {"Fill", "Line", "Point"};
        PE::entry("Polygon Mode",
                  [&] { return ImGui::Combo("combo", (int*)&m_settings.polygonMode, items, IM_ARRAYSIZE(items)); });
      }
      {
        const char* items[] = {"Never", "Less", "Equal", "Less or Equal", "Greater", "Not Equal", "Greater or Equal",
                               "Always"};
        PE::entry("Depth Compare",
                  [&] { return ImGui::Combo("combo", (int*)&m_settings.depthCompare, items, IM_ARRAYSIZE(items)); });
      }
      PE::end();

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
    if(!m_gBuffers)
      return;

    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);
    const glm::vec2&                      clip = CameraManip.getClipPlanes();

    // Update Frame buffer uniform buffer
    DH::FrameInfo finfo{};
    finfo.view = CameraManip.getMatrix();
    finfo.proj = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), m_gBuffers->getAspectRatio(), clip.x, clip.y);
    finfo.proj[1][1] *= -1;
    finfo.camPos = CameraManip.getEye();
    vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);

    // Drawing the primitives in G-Buffer 0
    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView(0)},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    vkCmdBeginRendering(cmd, &r_info);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dset->getPipeLayout(), 0, 1, m_dset->getSets(), 0, nullptr);

    // #SHADER_OBJECT
    setupShaderObjectPipeline(cmd);

    const VkShaderStageFlagBits stages[2]       = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
    const VkShaderStageFlagBits unusedStages[3] = {VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                                                   VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_SHADER_STAGE_GEOMETRY_BIT};
    // Bind linked shaders
    vkCmdBindShadersEXT(cmd, 2, stages, m_shaders.data());
    vkCmdBindShadersEXT(cmd, 3, unusedStages, NULL);

    const VkDeviceSize offsets{0};
    for(const nvh::Node& n : m_nodes)
    {
      PrimitiveMeshVk& m           = m_meshVk[n.mesh];
      auto             num_indices = static_cast<uint32_t>(m_meshes[n.mesh].triangles.size() * 3);

      // Push constant information
      m_pushConst.transfo = n.localMatrix();
      m_pushConst.color   = m_materials[n.material].color;
      vkCmdPushConstants(cmd, m_dset->getPipeLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(DH::PushConstant), &m_pushConst);

      vkCmdBindVertexBuffers(cmd, 0, 1, &m.vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
      vkCmdDrawIndexed(cmd, num_indices, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
  }

private:
  void createScene()
  {

    createMesh(0.7F, 4080);                             // Meshes
    m_materials.push_back({.color = {.88, .88, .88}});  // Material
    m_nodes.push_back({.mesh = 0});                     // Instances

    CameraManip.setClipPlanes({0.01F, 100.0F});  // Default camera
    CameraManip.setLookat({-1.24282, 0.28388, 1.24613}, {-0.07462, -0.08036, -0.02502}, {0.00000, 1.00000, 0.00000});
  }

  void createMesh(float probability, int seed)
  {
    m_meshes                            = {};
    nvh::PrimitiveMesh     cube         = nvh::createCube();
    std::vector<nvh::Node> mengerNodes  = nvh::mengerSpongeNodes(MENGER_SUBDIV, probability, seed);
    nvh::PrimitiveMesh     mengerSponge = nvh::mergeNodes(mengerNodes, {cube});
    if(mengerSponge.triangles.empty())  // Don't allow empty result
      mengerSponge = cube;
    m_meshes.push_back(mengerSponge);
  }

  template <typename T>
  size_t getBaseTypeSize(const std::vector<T>& vec)
  {
    // Use std::remove_reference to remove any reference qualifiers
    using BaseType = typename std::remove_reference<T>::type;

    // Use sizeof to get the size of the base type
    return sizeof(BaseType);
  }

  VkPushConstantRange getPushConstantRange()
  {
    return VkPushConstantRange{.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               .offset     = 0,
                               .size       = sizeof(DH::PushConstant)};
  }

  //-------------------------------------------------------------------------------------------------
  // Descriptor Set contains only the access to the Frame Buffer
  //
  void createDescriptorSet()
  {
    m_dset->addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_FRAGMENT_BIT);
    m_dset->initLayout();
    m_dset->initPool(1);

    VkPushConstantRange push_constant_ranges = getPushConstantRange();
    m_dset->initPipeLayout(1, &push_constant_ranges);

    // Writing to descriptors
    const VkDescriptorBufferInfo      dbi_unif{m_frameInfo.buffer, 0, VK_WHOLE_SIZE};
    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_dset->makeWrite(0, 0, &dbi_unif));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  //-------------------------------------------------------------------------------------------------
  // Creating all Shader Objects
  // #SHADER_OBJECT
  void createShaderObjects()
  {
    VkPushConstantRange push_constant_ranges = getPushConstantRange();

    // Vertex
    std::vector<VkShaderCreateInfoEXT> shaderCreateInfos;
    shaderCreateInfos.push_back(VkShaderCreateInfoEXT {
      .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, .pNext = NULL, .flags = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT,
      .stage = VK_SHADER_STAGE_VERTEX_BIT, .nextStage = VK_SHADER_STAGE_FRAGMENT_BIT, .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
#if(USE_SLANG)
      .codeSize = sizeof(rasterSlang), .pCode = &rasterSlang[0], .pName = "vertexMain",
#else
        .codeSize = vert_shd.size() * getBaseTypeSize(vert_shd),
        .pCode    = vert_shd.data(),
        .pName    = USE_HLSL ? "vertexMain" : "main",
#endif  // USE_SLANG
      .setLayoutCount             = 1,
      .pSetLayouts                = &m_dset->getLayout(),  // Descriptor set layout compatible with the shaders
          .pushConstantRangeCount = 1, .pPushConstantRanges = &push_constant_ranges, .pSpecializationInfo = NULL,
    });

    // Fragment
    shaderCreateInfos.push_back(VkShaderCreateInfoEXT {
      .sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT, .pNext = NULL, .flags = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .nextStage = 0, .codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
#if(USE_SLANG)
      .codeSize = sizeof(rasterSlang), .pCode = &rasterSlang[0], .pName = "fragmentMain",
#else
        .codeSize = frag_shd.size() * getBaseTypeSize(frag_shd),
        .pCode    = frag_shd.data(),
        .pName    = USE_HLSL ? "fragmentMain" : "main",
#endif  // USE_SLANG
      .setLayoutCount             = 1,
      .pSetLayouts                = &m_dset->getLayout(),  // Descriptor set layout compatible with the shaders
          .pushConstantRangeCount = 1, .pPushConstantRanges = &push_constant_ranges, .pSpecializationInfo = NULL,
    });

    // Create the shaders
    NVVK_CHECK(vkCreateShadersEXT(m_device, 2, shaderCreateInfos.data(), NULL, m_shaders.data()));
  }


  //-------------------------------------------------------------------------------------------------
  // Setting up the first states of the frame
  // #SHADER_OBJECT
  //
  void setupShaderObjectPipeline(VkCommandBuffer cmd)
  {
    VkExtent2D viewportSize = m_app->getViewportSize();

    VkViewport viewport{
        .x        = 0.0F,
        .y        = 0.0F,
        .width    = static_cast<float>(viewportSize.width),
        .height   = static_cast<float>(viewportSize.height),
        .minDepth = 0.0F,
        .maxDepth = 1.0F,
    };

    VkRect2D scissor{
        .offset = {0, 0},
        .extent = {viewportSize.width, viewportSize.height},
    };

    VkVertexInputBindingDescription2EXT vertexInputBindingDescription{
        .sType     = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT,
        .binding   = 0,
        .stride    = sizeof(nvh::PrimitiveVertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        .divisor   = 1,
    };

    VkVertexInputAttributeDescription2EXT vertexInputAttributeDescription[2]{
        {
            // Position
            .sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            .location = 0,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32_SFLOAT,
            .offset   = static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, p)),
        },
        {
            // Normal
            .sType    = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
            .location = 1,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32_SFLOAT,
            .offset   = static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, n)),
        },
    };

    VkColorBlendEquationEXT colorBlendEquation = {.srcColorBlendFactor = VK_BLEND_FACTOR_ZERO,
                                                  .dstColorBlendFactor = VK_BLEND_FACTOR_ZERO,
                                                  .colorBlendOp        = VK_BLEND_OP_ADD,
                                                  .srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                                                  .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                                                  .alphaBlendOp        = VK_BLEND_OP_ADD};

    float                 blendConstants[4]{0.0F, 0.0F, 0.0F, 0.0F};
    VkSampleMask          sampleMask{~0U};
    VkBool32              colorBlendEnables = VK_FALSE;
    VkColorComponentFlags colorWriteMasks =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    vkCmdSetViewportWithCount(cmd, 1, &viewport);
    vkCmdSetScissorWithCount(cmd, 1, &scissor);
    vkCmdSetLineWidth(cmd, m_settings.lineWidth);  // 1.0F
    vkCmdSetDepthBias(cmd, 0.0F, 0.0F, 1.0F);
    vkCmdSetBlendConstants(cmd, blendConstants);
    vkCmdSetDepthBounds(cmd, 0.0F, 1.0F);
    vkCmdSetCullMode(cmd, m_settings.cullMode);  // VK_CULL_MODE_NONE
    vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
    vkCmdSetDepthCompareOp(cmd, m_settings.depthCompare);        // VK_COMPARE_OP_LESS_OR_EQUAL
    vkCmdSetDepthTestEnable(cmd, m_settings.depthTestEnable);    // TRUE
    vkCmdSetDepthWriteEnable(cmd, m_settings.depthWriteEnable);  // TRUE
    vkCmdSetFrontFace(cmd, m_settings.frontFace);                // VK_FRONT_FACE_COUNTER_CLOCKWISE
    vkCmdSetPrimitiveTopology(cmd, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vkCmdSetStencilTestEnable(cmd, VK_FALSE);
    vkCmdSetDepthBiasEnable(cmd, VK_TRUE);
    vkCmdSetPrimitiveRestartEnable(cmd, VK_FALSE);
    vkCmdSetRasterizerDiscardEnable(cmd, VK_FALSE);
    vkCmdSetVertexInputEXT(cmd, 1, &vertexInputBindingDescription, 2, vertexInputAttributeDescription);
    vkCmdSetDepthClampEnableEXT(cmd, VK_TRUE);
    vkCmdSetPolygonModeEXT(cmd, m_settings.polygonMode);  // VK_POLYGON_MODE_FILL
    vkCmdSetRasterizationSamplesEXT(cmd, VK_SAMPLE_COUNT_1_BIT);
    vkCmdSetColorBlendEquationEXT(cmd, 0, 1, &colorBlendEquation);
    vkCmdSetSampleMaskEXT(cmd, VK_SAMPLE_COUNT_1_BIT, &sampleMask);
    vkCmdSetAlphaToCoverageEnableEXT(cmd, VK_FALSE);
    vkCmdSetAlphaToOneEnableEXT(cmd, VK_FALSE);
    vkCmdSetLogicOpEnableEXT(cmd, VK_FALSE);
    vkCmdSetColorBlendEnableEXT(cmd, 0, 1, &colorBlendEnables);
    vkCmdSetColorWriteMaskEXT(cmd, 0, 1, &colorWriteMasks);

    // --- Unused Shader Object functions ---
    // See https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_EXT_shader_object.html
  }


  //-------------------------------------------------------------------------------------------------
  // G-Buffers, a color and a depth, which are used for rendering. The result color will be displayed
  // and an image filling the ImGui Viewport window.
  void createGbuffers(const VkExtent2D& size)
  {
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), size, m_colorFormat, m_depthFormat);
  }

  //-------------------------------------------------------------------------------------------------
  // Creating the Vulkan buffers that are holding the data for:
  // - vertices and indices, one of each for each object
  //
  void createSceneDataBuffers()
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_meshVk.resize(m_meshes.size());
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& m = m_meshVk[i];
      m.vertices         = m_alloc->createBuffer(cmd, m_meshes[i].vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      m.indices          = m_alloc->createBuffer(cmd, m_meshes[i].triangles, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
      m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
      m_dutil->DBG_NAME_IDX(m.indices.buffer, i);
    }

    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_alloc->finalizeAndReleaseStaging();  // The create buffer uses staging and at this point, we know the buffers are uploaded
  }

  //-------------------------------------------------------------------------------------------------
  // Creating the Vulkan buffer that is holding the data for Frame information
  // The frame info contains the camera and other information changing at each frame.
  void createFrameInfoBuffer()
  {
    VkCommandBuffer cmd = m_app->createTempCmdBuffer();
    m_frameInfo         = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_app->submitAndWaitTempCmdBuffer(cmd);
    m_dutil->DBG_NAME(m_frameInfo.buffer);
  }

  void destroyPrimitiveMeshResources()
  {
    for(PrimitiveMeshVk& m : m_meshVk)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }
    m_meshVk.clear();
  }

  void destroyResources()
  {
    // #SHADER_OBJECT
    for(auto shader : m_shaders)
      vkDestroyShaderEXT(m_device, shader, NULL);

    destroyPrimitiveMeshResources();

    m_alloc->destroy(m_frameInfo);

    m_dset->deinit();
    m_gBuffers.reset();
  }


  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*             m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil> m_dutil;
  std::shared_ptr<AllocDma>        m_alloc;
  //std::shared_ptr<nvvkhl::AllocVma> m_alloc;

  VkFormat                         m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue                m_clearColor  = {{0.7F, 0.7F, 0.7F, 1.0F}};     // Clear color
  VkDevice                         m_device      = VK_NULL_HANDLE;                 // Convenient
  std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;                                     // G-Buffers: color + depth
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;                            // Descriptor set

  // Resources
  struct PrimitiveMeshVk
  {
    nvvk::Buffer vertices;  // Buffer of the vertices
    nvvk::Buffer indices;   // Buffer of the indices
  };
  std::vector<PrimitiveMeshVk> m_meshVk;
  nvvk::Buffer                 m_frameInfo;

  // Data and setting
  struct Material
  {
    glm::vec3 color{1.F};
  };
  std::vector<nvh::PrimitiveMesh> m_meshes;
  std::vector<nvh::Node>          m_nodes;
  std::vector<Material>           m_materials;

  // Pipeline
  DH::PushConstant           m_pushConst{};  // Information sent to the shader
  std::array<VkShaderEXT, 2> m_shaders{};

  Settings m_settings;
};


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  // #SHADER_OBJECT
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};

  VkContextSettings vkSetup;
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);
  vkSetup.instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  vkSetup.deviceExtensions.push_back({VK_KHR_SWAPCHAIN_EXTENSION_NAME});
  vkSetup.deviceExtensions.push_back({VK_EXT_SHADER_OBJECT_EXTENSION_NAME, &shaderObjFeature});

  // Create Vulkan context
  auto vkContext = std::make_unique<VkContext>(vkSetup);
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

  if(shaderObjFeature.shaderObject == VK_FALSE)
  {
    nvprintf("ERROR: Shader Object is not supported");
    std::exit(1);
  }

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app->addElement(std::make_shared<ShaderObject>());

  app->run();
  app.reset();
  vkContext.reset();

  return test->errorCode();
}
