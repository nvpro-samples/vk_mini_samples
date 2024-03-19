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

 This sample shows how to draw wireframe on top of solid geometry in a 
 single pass, using gl_BaryCoordEXT (same as gl_BaryCoordNV)

 https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_fragment_shader_barycentric.html

*/
//////////////////////////////////////////////////////////////////////////

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }

// clang-format on
#include <array>
#include <vulkan/vulkan_core.h>

#define VMA_IMPLEMENTATION
#define IMGUI_DEFINE_MATH_OPERATORS

#include "imgui/imgui_camera_widget.h"
#include "nvh/primitives.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvk/shaders_vk.hpp"

namespace DH {
using namespace glm;
#include "shaders/device_host.h"
}  // namespace DH

#include "nvvk/images_vk.hpp"
#include "imgui/imgui_helper.h"

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


// thickness;color;thicknessVar;smoothing;screenSpace;backFaceColor;enableDash;dashRepeats;dashLength;onlyWire;
std::vector<DH::WireframeSettings> presets{
    {1.0F, {0.8F, 0.F, 0.F}, {1.0F, 1.0F}, 1.0F, 1, {0.5F, 0.5F, 0.5F}, 0, 5, 0.5F, 0},       // default
    {1.0F, {0.F, 0.8F, 0.F}, {1.0F, 1.0F}, 0.5F, 1, {0.5F, 0.5F, 0.5F}, 0, 5, 0.5F, 1},       // Wire dot
    {0.1F, {0.9F, 0.9F, 0.F}, {0.0F, 1.0F}, 0.1F, 0, {0.5F, 0.5F, 0.5F}, 0, 5, 0.5F, 0},      // Star
    {0.1F, {0.7F, 0.0F, 0.01F}, {0.7F, 0.0F}, 0.1F, 0, {0.07F, 0.0F, 0.0F}, 1, 10, 0.8F, 1},  // Flake
    {0.3F, {1.F, 1.0F, 1.F}, {1.F, 1.F}, 2.F, 1, {0.06F, 0.06F, 0.06F}, 0, 8, 0.4F, 0},       // Thin
    {0.5F, {.8F, .8F, .8F}, {1.F, 1.F}, 1.F, 1, {0.1F, 0.1F, 0.1F}, 1, 1, 1.F, 1},            // Wire line
    {0.5F, {.8F, .8F, .8F}, {1.F, 1.F}, 1.F, 1, {0.1F, 0.1F, 0.1F}, 1, 20, 0.5F, 1},          // Stipple
};


//////////////////////////////////////////////////////////////////////////
/// </summary> Display an image on a quad.
class BaryWireframe : public nvvkhl::IAppElement
{
public:
  BaryWireframe()           = default;
  ~BaryWireframe() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    nvh::ScopedTimer st(__FUNCTION__);

    m_app    = app;
    m_device = m_app->getDevice();

    m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);                    // Debug utility
    m_alloc = std::make_unique<nvvkhl::AllocVma>(m_app->getContext().get());  // Allocator
    m_dset  = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    m_settings = presets[0];

    createScene();
    createVkBuffers();
    createPipeline();
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
      ImGui::Begin("Settings");
      ImGuiH::CameraWidget();

      // Objects
      const char* items[] = {"Sphere", "Cube", "Tetrahedron", "Octahedron", "Icosahedron", "Cone"};
      int         flag    = ImGuiSliderFlags_Logarithmic;
      float       maxT    = m_settings.screenSpace ? 10.0F : 0.3f;
      using PE            = ImGuiH::PropertyEditor;
      PE::begin();
      PE::entry("Geometry", [&] { return ImGui::Combo("##1", &m_currentObject, items, IM_ARRAYSIZE(items)); });
      PE::entry("Screen Space", [&] { return ImGui::Checkbox("##2", (bool*)&m_settings.screenSpace); });
      PE::entry("Only Wire", [&] { return ImGui::Checkbox("##8", (bool*)&m_settings.onlyWire); });
      PE::entry("Color", [&] { return ImGui::ColorEdit3("##3", &m_settings.color.x); });
      PE::entry("Back Color", [&] { return ImGui::ColorEdit3("##4", &m_settings.backFaceColor.x); });
      PE::entry("Thickness", [&] { return ImGui::SliderFloat("##5", &m_settings.thickness, 0.F, maxT, "%.3f", flag); });
      PE::entry("Edge Variation", [&] { return ImGui::SliderFloat2("##6", &m_settings.thicknessVar.x, 0.F, 1.F); });
      PE::entry("Smoothing", [&] { return ImGui::SliderFloat("##7", &m_settings.smoothing, 0.F, 2.F); });
      PE::entry("Stipple", [&] { return ImGui::Checkbox("", (bool*)&m_settings.enableStipple); });
      PE::entry("Repeats", [&] { return ImGui::SliderInt("##1", &m_settings.stippleRepeats, 0, 20); });
      PE::entry("Length", [&] { return ImGui::SliderFloat("##2", &m_settings.stippleLength, 0.F, 1.F); });
      PE::end();
      PE::begin();
      {
        static int  item_current = 0;
        const char* items[]      = {"Default", "Wireframe dot", "Star", "Flake", " Thin", "Wireframe line", "Stipple"};
        if(PE::entry("Preset", [&] { return ImGui::Combo("##1", &item_current, items, IM_ARRAYSIZE(items)); }))
          m_settings = presets[item_current];
      }
      PE::end();

      ImGui::End();  // "Settings"
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
    glm::vec3   eye;
    glm::vec3   center;
    glm::vec3   up;
    CameraManip.getLookat(eye, center, up);

    // Update Frame buffer uniform buffer
    DH::FrameInfo    finfo{};
    const glm::vec2& clip = CameraManip.getClipPlanes();
    finfo.view            = CameraManip.getMatrix();
    finfo.proj            = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), aspect_ratio, clip.x, clip.y);
    finfo.proj[1][1] *= -1;
    finfo.camPos = eye;
    vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(DH::FrameInfo), &finfo);

    vkCmdUpdateBuffer(cmd, m_bSettings.buffer, 0, sizeof(DH::WireframeSettings), &m_settings);

    // Drawing the primitives in a G-Buffer
    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView()},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    vkCmdBeginRendering(cmd, &r_info);
    m_app->setViewport(cmd);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dset->getPipeLayout(), 0, 1, m_dset->getSets(), 0, nullptr);
    const VkDeviceSize offsets{0};
    auto&              n = m_nodes[m_currentObject];
    {
      PrimitiveMeshVk& m = m_meshVk[n.mesh];
      // Push constant information
      m_pushConst.transfo    = n.localMatrix();
      m_pushConst.color      = m_materials[n.material].color;
      m_pushConst.clearColor = glm::make_vec4(m_clearColor.float32);
      vkCmdPushConstants(cmd, m_dset->getPipeLayout(), VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(DH::PushConstant), &m_pushConst);

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
    nvh::ScopedTimer st(__FUNCTION__);
    // Meshes
    m_meshes.emplace_back(nvh::createSphereUv());
    m_meshes.emplace_back(nvh::createCube());
    m_meshes.emplace_back(nvh::createTetrahedron());
    m_meshes.emplace_back(nvh::createOctahedron());
    m_meshes.emplace_back(nvh::createIcosahedron());
    m_meshes.emplace_back(nvh::createConeMesh());
    const int num_meshes = static_cast<int>(m_meshes.size());

    // Materials (colorful)
    for(int i = 0; i < num_meshes; i++)
    {
      const glm::vec3 freq = glm::vec3(1.33333F, 2.33333F, 3.33333F) * static_cast<float>(i);
      const glm::vec3 v    = static_cast<glm::vec3>(sin(freq) * 0.5F + 0.5F);
      m_materials.push_back({glm::vec4(v, 1)});
    }

    // Instances
    for(int i = 0; i < num_meshes; i++)
    {
      nvh::Node& n = m_nodes.emplace_back();
      n.mesh       = i;
      n.material   = i;
    }

    CameraManip.setClipPlanes({0.1F, 100.0F});
    CameraManip.setLookat({0.0F, 1.0F, 2.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F});
  }

  void createPipeline()
  {
    nvh::ScopedTimer st(__FUNCTION__);
    m_dset->addBinding(BIND_FRAME_INFO, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_FRAGMENT_BIT);
    m_dset->addBinding(BIND_SETTINGS, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_FRAGMENT_BIT);
    m_dset->initLayout();
    m_dset->initPool(1);

    const VkPushConstantRange push_constant_ranges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                                      sizeof(DH::PushConstant)};
    m_dset->initPipeLayout(1, &push_constant_ranges);

    // Writing to descriptors
    const VkDescriptorBufferInfo      dbi_unif{m_frameInfo.buffer, 0, VK_WHOLE_SIZE};
    const VkDescriptorBufferInfo      dbi_setting{m_bSettings.buffer, 0, VK_WHOLE_SIZE};
    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(m_dset->makeWrite(0, BIND_FRAME_INFO, &dbi_unif));
    writes.emplace_back(m_dset->makeWrite(0, BIND_SETTINGS, &dbi_setting));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    VkPipelineRenderingCreateInfo prend_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    prend_info.colorAttachmentCount    = 1;
    prend_info.pColorAttachmentFormats = &m_colorFormat;
    prend_info.depthAttachmentFormat   = m_depthFormat;

    // Creating the Pipeline
    nvvk::GraphicsPipelineState pstate;
    pstate.rasterizationState.cullMode = VK_CULL_MODE_NONE;
    pstate.addBindingDescriptions({{0, sizeof(nvh::PrimitiveVertex)}});
    pstate.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, p))},  // Position
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, n))},  // Normal
    });

    nvvk::GraphicsPipelineGenerator pgen(m_device, m_dset->getPipeLayout(), prend_info, pstate);
#if USE_SLANG
    VkShaderModule shaderModule = nvvk::createShaderModule(m_device, &rasterSlang[0], sizeof(rasterSlang));
    pgen.addShader(shaderModule, VK_SHADER_STAGE_VERTEX_BIT, "vertexMain");
    pgen.addShader(shaderModule, VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain");
#else
    pgen.addShader(vert_shd, VK_SHADER_STAGE_VERTEX_BIT, USE_HLSL ? "vertexMain" : "main");
    pgen.addShader(frag_shd, VK_SHADER_STAGE_FRAGMENT_BIT, USE_HLSL ? "fragmentMain" : "main");
#endif

    m_graphicsPipeline = pgen.createPipeline();
    m_dutil->setObjectName(m_graphicsPipeline, "Graphics");
    pgen.clearShaders();

#if USE_SLANG
    vkDestroyShaderModule(m_device, shaderModule, nullptr);
#endif
  }

  void createGbuffers(const VkExtent2D& size)
  {
    nvh::ScopedTimer st(std::string(__FUNCTION__) + std::string(": ") + std::to_string(size.width) + std::string(", ")
                        + std::to_string(size.height));
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), size, m_colorFormat, m_depthFormat);
  }

  void createVkBuffers()
  {
    nvh::ScopedTimer st(__FUNCTION__);
    VkCommandBuffer  cmd = m_app->createTempCmdBuffer();
    m_meshVk.resize(m_meshes.size());
    for(size_t i = 0; i < m_meshes.size(); i++)
    {
      PrimitiveMeshVk& m = m_meshVk[i];
      m.vertices         = m_alloc->createBuffer(cmd, m_meshes[i].vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      m.indices          = m_alloc->createBuffer(cmd, m_meshes[i].triangles, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
      m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
      m_dutil->DBG_NAME_IDX(m.indices.buffer, i);
    }

    m_frameInfo = m_alloc->createBuffer(sizeof(DH::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_frameInfo.buffer);

    m_bSettings = m_alloc->createBuffer(sizeof(DH::WireframeSettings), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    m_dutil->DBG_NAME(m_bSettings.buffer);

    m_app->submitAndWaitTempCmdBuffer(cmd);
  }


  void destroyResources()
  {
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

    for(PrimitiveMeshVk& m : m_meshVk)
    {
      m_alloc->destroy(m.vertices);
      m_alloc->destroy(m.indices);
    }
    m_alloc->destroy(m_frameInfo);
    m_alloc->destroy(m_bSettings);

    m_dset->deinit();
    m_gBuffers.reset();
  }


  //--------------------------------------------------------------------------------------------------
  //
  //
  nvvkhl::Application*              m_app{nullptr};
  std::unique_ptr<nvvk::DebugUtil>  m_dutil;
  std::shared_ptr<nvvkhl::AllocVma> m_alloc;

  VkFormat                         m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
  VkFormat                         m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
  VkClearColorValue                m_clearColor  = {{0.3F, 0.3F, 0.3F, 1.0F}};     // Clear color
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
    glm::vec4 color{1.F};
  };
  std::vector<nvh::PrimitiveMesh> m_meshes;
  std::vector<nvh::Node>          m_nodes;
  std::vector<Material>           m_materials;

  // Pipeline
  DH::PushConstant m_pushConst{};                        // Information sent to the shader
  VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
  int              m_currentObject    = 0;

  DH::WireframeSettings m_settings = {};
  nvvk::Buffer          m_bSettings;
};

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name             = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  spec.vSync            = true;
  spec.vkSetup.apiMajor = 1;
  spec.vkSetup.apiMinor = 3;

  static VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR baryFeat{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
  spec.vkSetup.addDeviceExtension(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME, false, &baryFeat);

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>("", fmt::format("({})", SHADER_LANGUAGE_STR)));  // Window title info
  app->addElement(std::make_shared<BaryWireframe>());

  app->run();
  app.reset();

  return test->errorCode();
}
