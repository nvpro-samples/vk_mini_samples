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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


/*
 This sample shows how shaders can be loaded and reloaded from disk
*/
// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const nvmath::vec2f& f) {x = f.x; y = f.y;} operator nvmath::vec2f() const { return nvmath::vec2f(x, y); }
// clang-format on

#include <array>
#include <vulkan/vulkan_core.h>
#include "nvmath/nvmath.h"
#include <imgui.h>
#include <shaderc/shaderc.hpp>

#define VMA_IMPLEMENTATION
#include "imgui/backends/imgui_impl_vulkan.h"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/glsl_compiler.hpp"
#include "nvvkhl/pipeline_container.hpp"


namespace nvvkhl {

// ShaderToy inputs
struct InputUniforms
{
  nvmath::vec3f iResolution{0.0F};
  float         iTime{0};
  nvmath::vec4f iMouse{0.0F};
  float         iTimeDelta{0};
  int           iFrame{0};
  int           iFrameRate{1};
  nvmath::vec3f iChannelResolution[1]{};
  float         iChannelTime[1]{};
};


class TinyShaderToy : public nvvkhl::IAppElement
{
  enum GbufItems
  {
    eColor,
    eImage,
    eBufA0,
    eBufA1
  };


public:
  TinyShaderToy()           = default;
  ~TinyShaderToy() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();
    m_dutil  = std::make_unique<nvvk::DebugUtil>(m_device);            // Debug utility
    m_alloc  = std::make_unique<AllocVma>(m_app->getContext().get());  // Allocator
    m_dset   = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    // glsl compiler
    m_glslC = std::make_unique<nvvkhl::GlslCompiler>();
    m_glslC->addInclude(NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY) + std::string("shaders"));
    m_glslC->addInclude(NVPSystem::exePath());
    m_glslC->addInclude(NVPSystem::exePath() + std::string(PROJECT_NAME) + std::string("/shaders"));

    auto err_msg = compileShaders();
    if(!err_msg.empty())
    {
      LOGE("%s\n", err_msg.c_str());
      exit(1);
    }

    m_depthFormat = nvvk::findDepthFormat(m_app->getPhysicalDevice());  // Not all depth are supported

    createPipelineLayout();
    createPipelines();
    createGeometryBuffers();
  }

  void onDetach() override { detroyResources(); }

  void onResize(uint32_t width, uint32_t height) override { createGbuffers({width, height}); }

  void onUIRender() override
  {
    static std::string error_msg;
    {  // Setting panel
      ImGui::Begin("Settings");
      ImGui::Text("Edit the fragment shader, then click:");
      if(ImGui::Button("Reload Shaders"))
      {
        error_msg = compileShaders();
        if(error_msg.empty())
        {
          createPipelines();
        }
      }

      if(!error_msg.empty())
      {
        ImGui::TextColored({1, 0, 0, 1}, "ERROR");
        ImGui::Separator();
        ImGui::TextWrapped(error_msg.c_str());
        ImGui::Separator();
      }

      if(ImGui::Button(m_pause ? "Play" : "Pause"))
      {
        m_pause = !m_pause;
      }

      if(ImGui::Button("Reset"))
      {
        m_frame        = 0;
        m_time         = 0;
        m_inputUniform = {};
      }
      ImGui::Separator();
      ImGui::Text("Resolution: %.0f, %.0f", m_inputUniform.iResolution.x, m_inputUniform.iResolution.y);
      ImGui::Text("Time: %.2f", m_inputUniform.iTime);
      ImGui::Text("Mouse: %.0f, %.0f, %.0f", m_inputUniform.iMouse.x, m_inputUniform.iMouse.y, m_inputUniform.iMouse.z);
      ImGui::Text("Delta Time: %.3f", m_inputUniform.iTimeDelta);
      ImGui::Text("Frame: %d", m_inputUniform.iFrame);
      ImGui::Text("Frame Rate: %d", m_inputUniform.iFrameRate);
      ImGui::End();
    }

    {
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");

      updateUniforms();

      // Display the G-Buffer image
      ImGui::Image(m_gBuffers->getDescriptorSet(eImage), ImGui::GetContentRegionAvail());
      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    auto sdbg = m_dutil->DBG_SCOPE(cmd);

    // Ping-Pong double buffer
    static int double_buffer{0};
    GbufItems  in_image{eBufA0};
    GbufItems  out_image{eBufA1};
    if(double_buffer != 0)
    {
      in_image  = eBufA1;
      out_image = eBufA0;
    }

    renderToBuffer(cmd, m_pipelineBufA, in_image, out_image);

    // Barrier - making sure the rendered image from BufferA is ready to be used
    auto image_memory_barrier =
        nvvk::makeImageMemoryBarrier(m_gBuffers->getColorImage(out_image), VK_ACCESS_SHADER_READ_BIT,
                                     VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &image_memory_barrier);

    renderToBuffer(cmd, m_pipelineImg, out_image, eImage);

    double_buffer = ((double_buffer == 1) ? 0 : 1);
  }


private:
  struct Vertex
  {
    nvmath::vec2f pos;
  };

  void createPipelineLayout()
  {
    VkPushConstantRange push_constants = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(InputUniforms)};

    m_dset->addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_dset->addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_dset->initLayout(VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR);
    m_dset->initPipeLayout(1, &push_constants);
    m_dutil->DBG_NAME(m_dset->getLayout());
    m_dutil->DBG_NAME(m_dset->getPipeLayout());
  }

  void createPipelines()
  {
    nvvk::GraphicsPipelineState pstate;
    pstate.addBindingDescriptions({{0, sizeof(Vertex)}});
    pstate.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(Vertex, pos))},  // Position
    });

    VkPipelineRenderingCreateInfo prend_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    prend_info.colorAttachmentCount    = 1;
    prend_info.pColorAttachmentFormats = &m_colorFormat;
    prend_info.depthAttachmentFormat   = m_depthFormat;

    // Create the pipeline for "Image"
    {
      nvvk::GraphicsPipelineGenerator pgen(m_device, m_dset->getPipeLayout(), prend_info, pstate);
      pgen.addShader(m_vmodule, VK_SHADER_STAGE_VERTEX_BIT);
      pgen.addShader(m_fmodule, VK_SHADER_STAGE_FRAGMENT_BIT);
      m_pipelineImg = pgen.createPipeline();
      m_dutil->setObjectName(m_pipelineImg, "Image");
    }

    // Create the pipeline for "Buffer-A"
    {
      nvvk::GraphicsPipelineGenerator pgen(m_device, m_dset->getPipeLayout(), prend_info, pstate);
      pgen.addShader(m_vmodule, VK_SHADER_STAGE_VERTEX_BIT);
      pgen.addShader(m_fmoduleA, VK_SHADER_STAGE_FRAGMENT_BIT);
      m_pipelineBufA = pgen.createPipeline();
      m_dutil->setObjectName(m_pipelineBufA, "BufferA");
    }
  }

  // Render with one of the pipeline.
  // Buffer-A will write back to itself while Image will take Buffer-A as input
  void renderToBuffer(VkCommandBuffer cmd, VkPipeline pipeline, GbufItems inImage, GbufItems outImage)
  {
    // Always render to unused Color buffer
    nvvk::createRenderingInfo r_info({{0, 0}, m_viewSize}, {m_gBuffers->getColorImageView(eColor)},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_LOAD,
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    vkCmdBeginRendering(cmd, &r_info);
    m_app->setViewport(cmd);

    // Writing descriptor
    VkDescriptorImageInfo             in_desc  = m_gBuffers->getDescriptorImageInfo(inImage);
    VkDescriptorImageInfo             out_desc = m_gBuffers->getDescriptorImageInfo(outImage);
    std::vector<VkWriteDescriptorSet> writes;
    writes.push_back(m_dset->makeWrite(0, 0, &in_desc));
    writes.push_back(m_dset->makeWrite(0, 1, &out_desc));
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dset->getPipeLayout(), 0,
                              static_cast<uint32_t>(writes.size()), writes.data());

    // Pusing the "Input Uniforms"
    vkCmdPushConstants(cmd, m_dset->getPipeLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(InputUniforms), &m_inputUniform);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    VkDeviceSize offsets{0};
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
    vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);

    vkCmdEndRendering(cmd);
  }

  void createGbuffers(VkExtent2D size)
  {
    m_viewSize = size;

    // Creating the 4 buffers (see BufItems)
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), m_viewSize,
                                                   std::vector{m_colorFormat, m_rgba32Format, m_rgba32Format, m_rgba32Format},
                                                   m_depthFormat);
  }

  void createGeometryBuffers()
  {
    // -1-1 -- 1-1
    //   |      |
    // -1 1 -- 1 1
    const std::vector<Vertex>   vertices = {{{-1.0F, -1.0F}}, {{1.0F, -1.0F}}, {{1.0F, 1.0F}}, {{-1.0F, 1.0F}}};
    const std::vector<uint16_t> indices  = {0, 3, 2, 0, 2, 1};

    {
      auto* cmd  = m_app->createTempCmdBuffer();
      m_vertices = m_alloc->createBuffer(cmd, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      m_indices  = m_alloc->createBuffer(cmd, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      m_dutil->DBG_NAME(m_vertices.buffer);
      m_dutil->DBG_NAME(m_indices.buffer);
    }
  }


  void setCompilerOptions()
  {
    m_glslC->resetOptions();
    m_glslC->options()->SetTargetSpirv(shaderc_spirv_version::shaderc_spirv_version_1_2);
    m_glslC->options()->SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    m_glslC->options()->SetGenerateDebugInfo();
    m_glslC->options()->SetOptimizationLevel(shaderc_optimization_level_performance);
  }

  std::string compileShaders()
  {
    std::string error_msg;

    // Compile default shaders
    setCompilerOptions();
    const auto vert_result = m_glslC->compileFile("raster.vert", shaderc_shader_kind::shaderc_vertex_shader);
    m_glslC->options()->AddMacroDefinition("INCLUDE_FILE", "0");
    const auto frag_result = m_glslC->compileFile("raster.frag", shaderc_shader_kind::shaderc_fragment_shader);
    m_glslC->options()->AddMacroDefinition("INCLUDE_FILE", "1");
    const auto frag_result_a = m_glslC->compileFile("raster.frag", shaderc_shader_kind::shaderc_fragment_shader);

    if(vert_result.GetNumErrors() == 0 && frag_result.GetNumErrors() == 0 && frag_result_a.GetNumErrors() == 0)
    {
      // Deleting resources, but not immediately as they are still in used
      nvvkhl::Application::submitResourceFree([vmod = m_vmodule, fmod_i = m_fmodule, fmod_a = m_fmoduleA,
                                               device = m_device, gp = m_pipelineImg, gp_a = m_pipelineBufA]() {
        vkDestroyShaderModule(device, vmod, nullptr);
        vkDestroyShaderModule(device, fmod_i, nullptr);
        vkDestroyShaderModule(device, fmod_a, nullptr);
        vkDestroyPipeline(device, gp, nullptr);
        vkDestroyPipeline(device, gp_a, nullptr);
      });

      m_vmodule  = m_glslC->createModule(m_device, vert_result);
      m_fmodule  = m_glslC->createModule(m_device, frag_result);
      m_fmoduleA = m_glslC->createModule(m_device, frag_result_a);
      m_dutil->setObjectName(m_vmodule, "Vertex");
      m_dutil->setObjectName(m_fmodule, "Image");
      m_dutil->setObjectName(m_fmoduleA, "BufferA");
    }
    else
    {
      error_msg += vert_result.GetErrorMessage();
      error_msg += frag_result.GetErrorMessage();
      error_msg += frag_result_a.GetErrorMessage();
    }

    return error_msg;
  }

  void updateUniforms()
  {
    // Grab Data
    nvmath::vec2f mouse_pos = ImGui::GetMousePos();         // Current mouse pos in window
    nvmath::vec2f corner    = ImGui::GetCursorScreenPos();  // Corner of the viewport
    nvmath::vec2f size      = ImGui::GetContentRegionAvail();

    // Set uniforms
    m_inputUniform.iResolution           = nvmath::vec3f(size, 0);
    m_inputUniform.iChannelResolution[0] = nvmath::vec3f(size, 0);

    if(!m_pause)
    {
      m_inputUniform.iFrame          = m_frame;
      m_inputUniform.iFrameRate      = static_cast<int>(ImGui::GetIO().Framerate);
      m_inputUniform.iTimeDelta      = ImGui::GetIO().DeltaTime;
      m_inputUniform.iTime           = m_time;
      m_inputUniform.iChannelTime[0] = m_time;
      m_time += ImGui::GetIO().DeltaTime;
      m_frame++;
    }
    if(ImGui::GetIO().MouseDown[0])
    {
      nvmath::vec2f mpos(mouse_pos - corner);
      mpos.y                = size.y - mpos.y;  // Inverting mouse position
      m_inputUniform.iMouse = nvmath::vec4f(mpos, 1, 0);
    }
    else
    {
      m_inputUniform.iMouse.z = 0;  // Just setting the button down
    }
  }


  void detroyResources()
  {
    vkDestroyPipeline(m_device, m_pipelineImg, nullptr);
    vkDestroyPipeline(m_device, m_pipelineBufA, nullptr);

    m_alloc->destroy(m_vertices);
    m_alloc->destroy(m_indices);
    m_vertices = {};
    m_indices  = {};
    m_gBuffers.reset();
    m_dset->deinit();

    vkDestroyShaderModule(m_device, m_vmodule, nullptr);
    vkDestroyShaderModule(m_device, m_fmodule, nullptr);
    vkDestroyShaderModule(m_device, m_fmoduleA, nullptr);
  }

  //--------------------------------------------------------------------------------------------------
  nvvkhl::Application* m_app{nullptr};

  std::unique_ptr<nvvkhl::GBuffer>              m_gBuffers;
  std::unique_ptr<nvvk::DebugUtil>              m_dutil;
  std::unique_ptr<nvvkhl::AllocVma>             m_alloc;
  std::unique_ptr<nvvkhl::GlslCompiler>         m_glslC;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;  // Descriptor set

  VkExtent2D        m_viewSize{0, 0};
  VkFormat          m_colorFormat  = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of fragment
  VkFormat          m_rgba32Format = VK_FORMAT_R32G32B32A32_SFLOAT;  // Color format of the images
  VkFormat          m_depthFormat  = VK_FORMAT_UNDEFINED;            // Depth format of the depth buffer
  VkPipeline        m_pipelineImg  = VK_NULL_HANDLE;                 // The graphic pipeline to render
  VkPipeline        m_pipelineBufA = VK_NULL_HANDLE;                 // The graphic pipeline to render
  nvvk::Buffer      m_vertices;                                      // Buffer of the vertices
  nvvk::Buffer      m_indices;                                       // Buffer of the indices
  VkClearColorValue m_clearColor{{0.1F, 0.4F, 0.1F, 1.0F}};          // Clear color
  VkDevice          m_device = VK_NULL_HANDLE;                       // Convenient

  int           m_frame{0};
  float         m_time{0};
  bool          m_pause{false};
  InputUniforms m_inputUniform;

  VkShaderModule m_vmodule  = VK_NULL_HANDLE;
  VkShaderModule m_fmodule  = VK_NULL_HANDLE;
  VkShaderModule m_fmoduleA = VK_NULL_HANDLE;
};

}  // namespace nvvkhl

auto main(int argc, char** argv) -> int
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name             = PROJECT_NAME " Example";
  spec.vSync            = true;
  spec.vkSetup.apiMajor = 1;
  spec.vkSetup.apiMinor = 3;
  spec.vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::TinyShaderToy>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());

  app->run();
  app.reset();

  return test->errorCode();
}
