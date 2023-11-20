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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//////////////////////////////////////////////////////////////////////////
/*

 This sample creates a 3D cube and render using the builtin camera

*/
//////////////////////////////////////////////////////////////////////////


#define VMA_IMPLEMENTATION
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "nvh/primitives.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "shaders/device_host.h"

#if USE_HLSL
#include "_autogen/raster_vertexMain.spirv.h"
#include "_autogen/raster_fragmentMain.spirv.h"
const auto& vert_shd = std::vector<uint8_t>{std::begin(raster_vertexMain), std::end(raster_vertexMain)};
const auto& frag_shd = std::vector<uint8_t>{std::begin(raster_fragmentMain), std::end(raster_fragmentMain)};
#elif USE_SLANG
#include "_autogen/raster_vertexMain.spirv.h"
#include "_autogen/raster_fragmentMain.spirv.h"
const auto& vert_shd = std::vector<uint32_t>{std::begin(raster_vertexMain), std::end(raster_vertexMain)};
const auto& frag_shd = std::vector<uint32_t>{std::begin(raster_fragmentMain), std::end(raster_fragmentMain)};
#else
#include "_autogen/raster.frag.h"
#include "_autogen/raster.vert.h"
const auto& vert_shd = std::vector<uint32_t>{std::begin(raster_vert), std::end(raster_vert)};
const auto& frag_shd = std::vector<uint32_t>{std::begin(raster_frag), std::end(raster_frag)};
#endif  // USE_HLSL


//-------------------------------------------------
// Using Device Memory Allocator, similar to VMA
// but using a slightly different approach
class AllocDma : public nvvk::ResourceAllocator
{
public:
  explicit AllocDma(const nvvk::Context* context) { init(context); }
  ~AllocDma() override { deinit(); }

private:
  void init(const nvvk::Context* context)
  {
    m_deviceMemoryAllocator = std::make_unique<nvvk::DeviceMemoryAllocator>(context->m_device, context->m_physicalDevice);
    m_dma = std::make_unique<nvvk::DMAMemoryAllocator>(m_deviceMemoryAllocator.get());
    nvvk::ResourceAllocator::init(context->m_device, context->m_physicalDevice, m_dma.get());
  }

  void deinit()
  {
    releaseStaging();
    m_deviceMemoryAllocator->deinit();
    m_dma->deinit();
    nvvk::ResourceAllocator::deinit();
  }

  std::unique_ptr<nvvk::DMAMemoryAllocator>    m_dma;  // The memory allocator
  std::unique_ptr<nvvk::DeviceMemoryAllocator> m_deviceMemoryAllocator;
};


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

    m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);            // Debug utility
    m_alloc = std::make_unique<AllocDma>(m_app->getContext().get());  // Allocator
    //m_alloc = std::make_unique<nvvkhl::AllocVma>(m_app->getContext().get());  // Allocator
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
      using PE = ImGuiH::PropertyEditor;

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

    const float aspect_ratio = m_gBuffers->getAspectRatio();

    // Update Frame buffer uniform buffer
    FrameInfo        finfo{};
    const glm::vec2& clip = CameraManip.getClipPlanes();
    finfo.view            = CameraManip.getMatrix();
    finfo.proj            = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspect_ratio, clip.x, clip.y);
    finfo.proj[1][1] *= -1;
    finfo.camPos = CameraManip.getEye();
    vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(FrameInfo), &finfo);

    // Drawing the primitives in a G-Buffer
    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView()},
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
      PrimitiveMeshVk& m = m_meshVk[n.mesh];
      // Push constant information
      m_pushConst.transfo = n.localMatrix();
      m_pushConst.color   = m_materials[n.material].color;
      vkCmdPushConstants(cmd, m_dset->getPipeLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(PushConstant), &m_pushConst);

      vkCmdBindVertexBuffers(cmd, 0, 1, &m.vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
      auto num_indices = static_cast<uint32_t>(m_meshes[n.mesh].triangles.size() * 3);
      vkCmdDrawIndexed(cmd, num_indices, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
  }

private:
  void createScene()
  {
    // Meshes
    createMesh(0.7F, 4080);

    // Material
    m_materials.push_back({.color = {.88, .88, .88}});

    // Instances
    m_nodes.push_back({.mesh = 0});

    CameraManip.setClipPlanes({0.01F, 100.0F});
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
                               .size       = sizeof(PushConstant)};
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
    shaderCreateInfos.push_back(VkShaderCreateInfoEXT{
        .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .pNext                  = NULL,
        .flags                  = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT,
        .stage                  = VK_SHADER_STAGE_VERTEX_BIT,
        .nextStage              = VK_SHADER_STAGE_FRAGMENT_BIT,
        .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
        .codeSize               = vert_shd.size() * getBaseTypeSize(vert_shd),
        .pCode                  = vert_shd.data(),
        .pName                  = USE_HLSL ? "vertexMain" : "main",
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_dset->getLayout(),  // Descriptor set layout compatible with the shaders
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &push_constant_ranges,
        .pSpecializationInfo    = NULL,
    });

    // Fragment
    shaderCreateInfos.push_back(VkShaderCreateInfoEXT{
        .sType                  = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
        .pNext                  = NULL,
        .flags                  = VK_SHADER_CREATE_LINK_STAGE_BIT_EXT,
        .stage                  = VK_SHADER_STAGE_FRAGMENT_BIT,
        .nextStage              = 0,
        .codeType               = VK_SHADER_CODE_TYPE_SPIRV_EXT,
        .codeSize               = frag_shd.size() * getBaseTypeSize(frag_shd),
        .pCode                  = frag_shd.data(),
        .pName                  = USE_HLSL ? "fragmentMain" : "main",
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_dset->getLayout(),  // Descriptor set layout compatible with the shaders
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &push_constant_ranges,
        .pSpecializationInfo    = NULL,
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
  void createGbuffers(const glm::vec2& size)
  {
    m_viewSize = size;
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(),
                                                   VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)},
                                                   m_colorFormat, m_depthFormat);
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

    m_frameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_frameInfo.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
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

  glm::vec2                        m_viewSize    = {0, 0};
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

  std::vector<VkSampler> m_samplers;

  // Data and setting
  struct Material
  {
    glm::vec3 color{1.F};
  };
  std::vector<nvh::PrimitiveMesh> m_meshes;
  std::vector<nvh::Node>          m_nodes;
  std::vector<Material>           m_materials;

  // Pipeline
  PushConstant               m_pushConst{};  // Information sent to the shader
  std::array<VkShaderEXT, 2> m_shaders{};

  Settings m_settings;
};


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name             = PROJECT_NAME " Example";
  spec.vSync            = true;
  spec.vkSetup          = nvvk::ContextCreateInfo(false);
  spec.vkSetup.apiMajor = 1;
  spec.vkSetup.apiMinor = 3;

  // #SHADER_OBJECT
  VkPhysicalDeviceShaderObjectFeaturesEXT shaderObjFeature{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  spec.vkSetup.addDeviceExtension(VK_EXT_SHADER_OBJECT_EXTENSION_NAME, false, &shaderObjFeature);

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  if(shaderObjFeature.shaderObject == VK_FALSE)
  {
    nvprintf("ERROR: Shader Object is not supported");
    std::exit(1);
  }

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>());
  app->addElement(std::make_shared<ShaderObject>());

  app->run();
  app.reset();

  return test->errorCode();
}
