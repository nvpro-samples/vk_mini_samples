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
Note: This is primarily to show how to compile shaders on the fly.

This sample replicate in a simple form, the execution of shaders like 
 the ones found on https://www.shadertoy.com/. shows how shaders can be loaded and reloaded from disk.
 - Many uniforms can be accessed: iResolution, iTimes, iFrame, iChannelTime, iMouse, ... 
 - The BufferA can be persisted between frames, same as with ShaderToy and its result
   is stored in iChannel0

*/


#define USE_SLANG 1
#define SHADER_LANGUAGE_STR (USE_SLANG ? "Slang" : "GLSL")

#define VMA_IMPLEMENTATION
#define IMGUI_DEFINE_MATH_OPERATORS

#include <glm/glm.hpp>

// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
// clang-format on


#include <array>
#include <filesystem>
namespace fs = std::filesystem;

#include <vulkan/vulkan_core.h>

#include "common/utils.hpp"
#include "nvapp/application.hpp"
#include "nvapp/elem_default_menu.hpp"
#include "nvslang/slang.hpp"
#include "nvutils/logger.hpp"
#include "nvutils/parameter_parser.hpp"
#include "nvutils/timers.hpp"
#include "nvvk/barriers.hpp"
#include "nvvk/check_error.hpp"
#include "nvvk/context.hpp"
#include "nvvk/debug_util.hpp"
#include "nvvk/default_structs.hpp"
#include "nvvk/descriptors.hpp"
#include "nvvk/gbuffers.hpp"
#include "nvvk/graphics_pipeline.hpp"
#include "nvvk/resource_allocator.hpp"
#include "nvvk/sampler_pool.hpp"
#include "nvvk/staging.hpp"
#include "nvvkglsl/glsl.hpp"
#include <imgui.h>
#include <shaderc/shaderc.hpp>


// ShaderToy inputs
struct InputUniforms
{
  glm::vec3 iResolution = {0.F, 0.F, 0.F};
  float     iTime       = {0};
  glm::vec4 iMouse      = {0.F, 0.F, 0.F, 0.F};
  float     iTimeDelta  = {0};
  int       iFrame      = {0};
  int       iFrameRate  = {1};
  int       _pad0;
  float     iChannelTime[1] = {0.F};
  int       _pad1;
  int       _pad2;
  int       _pad3;
  glm::vec3 iChannelResolution[1] = {{0.F, 0.F, 0.F}};
};

// Simple utility that checks if a file has changed on disk.
// Calling hasChanged() will reset the writeTime.
class FileCheck
{
public:
  FileCheck(fs::path _path)
      : pathToFile(_path)
  {
    lastWriteTime = fs::last_write_time(pathToFile);
  }

  bool hasChanged()
  {
    if(lastWriteTime != fs::last_write_time(pathToFile))
    {
      lastWriteTime = fs::last_write_time(pathToFile);
      return true;
    }
    return false;
  }

private:
  fs::file_time_type lastWriteTime;
  fs::path           pathToFile;
};


class TinyShaderToy : public nvapp::IAppElement
{
  enum GbufItems
  {
    eImage,
    eBufA0,
    eBufA1
  };

  struct ResourceGroup
  {
    VkShaderModule vertexModule;
    VkShaderModule fragModule;
    VkShaderModule fragAModule;
    VkDevice       device;
    VkPipeline     pipeline;
    VkPipeline     pipelineA;
  };

public:
  TinyShaderToy()           = default;
  ~TinyShaderToy() override = default;


  void onAttach(nvapp::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();
    // Create the Vulkan allocator (VMA)
    m_alloc.init({
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = app->getPhysicalDevice(),
        .device           = app->getDevice(),
        .instance         = app->getInstance(),
        .vulkanApiVersion = VK_API_VERSION_1_4,
    });  // Allocator


    m_fileBufferA = std::make_unique<FileCheck>(getFilePath("buffer_a.glsl"));
    m_fileImage   = std::make_unique<FileCheck>(getFilePath("image.glsl"));

    // The texture sampler to use
    m_samplerPool.init(m_device);
    VkSampler linearSampler{};
    NVVK_CHECK(m_samplerPool.acquireSampler(linearSampler));
    NVVK_DBG_NAME(linearSampler);

    // Initialization of the G-Buffers we want use
    m_gBuffers.init({.allocator      = &m_alloc,
                     .colorFormats   = {m_rgba32Format, m_rgba32Format, m_rgba32Format},
                     .imageSampler   = linearSampler,
                     .descriptorPool = m_app->getTextureDescriptorPool()});

    // Setting up the Slang compiler
    m_slangCompiler.addSearchPaths(nvsamples::getShaderDirs());
    m_slangCompiler.defaultTarget();
    m_slangCompiler.defaultOptions();
    m_slangCompiler.addOption({slang::CompilerOptionName::DebugInformation, {slang::CompilerOptionValueKind::Int, 1}});
    m_slangCompiler.addOption({slang::CompilerOptionName::Optimization, {slang::CompilerOptionValueKind::Int, 0}});

    // Setting up the GLSL compiler
    m_glslCompiler.addSearchPaths(nvsamples::getShaderDirs());
    m_glslCompiler.defaultTarget();
    m_glslCompiler.defaultOptions();


    const std::string err_msg = compileShaders();
    if(!err_msg.empty())
    {
      LOGE("%s\n", err_msg.c_str());
      exit(1);
    }

    createPipelineLayout();
    createPipelines();
    createGeometryBuffers();
  }

  void onDetach() override
  {
    vkDeviceWaitIdle(m_device);
    destroyResources();
  }

  void onResize(VkCommandBuffer cmd, const VkExtent2D& size) { NVVK_CHECK(m_gBuffers.update(cmd, size)); }

  void onUIRender() override
  {

    static std::string error_msg;
    {  // Setting panel
      bool reloadShaders = false;

      ImGui::Begin("Settings");
      if(ImGui::RadioButton("GLSL", !m_useSlang))
      {
        m_useSlang    = false;
        reloadShaders = true;
      }
      if(ImGui::RadioButton("Slang", m_useSlang))
      {
        m_useSlang    = true;
        reloadShaders = true;
      }

      ImGui::Text("Edit the fragment shader, then click:");

      ImGui::Text("Open");
      ImGui::SameLine();
      if(ImGui::Button("Image"))
      {
        openFile("image.glsl");
      }
      ImGui::SameLine();
      if(ImGui::Button("Buffer A"))
      {
        openFile("buffer_a.glsl");
      }

      reloadShaders |= ImGui::Button("Reload Shaders");
      reloadShaders |= m_fileBufferA->hasChanged();
      reloadShaders |= m_fileImage->hasChanged();
      if(reloadShaders)
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
        ImGui::TextWrapped("%s", error_msg.c_str());
        ImGui::Separator();
      }

      if(ImGui::Button(m_pause ? "Play" : "Pause"))
      {
        m_pause = !m_pause;
      }
      ImGui::SameLine();
      if(ImGui::Button("Reset"))
      {
        m_frame        = 0;
        m_time         = 0;
        m_inputUniform = InputUniforms{};
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
      ImGui::Image((ImTextureID)m_gBuffers.getDescriptorSet(eImage), ImGui::GetContentRegionAvail());
      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

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
    const VkImageMemoryBarrier2 image_memory_barrier = nvvk::makeImageMemoryBarrier({
        .image     = m_gBuffers.getColorImage(out_image),
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
    });

    const VkDependencyInfo depInfo{.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                   .imageMemoryBarrierCount = 1,
                                   .pImageMemoryBarriers    = &image_memory_barrier};

    vkCmdPipelineBarrier2(cmd, &depInfo);


    renderToBuffer(cmd, m_pipelineImg, out_image, eImage);

    double_buffer = ((double_buffer == 1) ? 0 : 1);
  }


private:
  struct Vertex
  {
    glm::vec2 pos;
  };

  void createPipelineLayout()
  {
    m_bindings.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);

    NVVK_CHECK(m_bindings.createDescriptorSetLayout(m_device, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                                    &m_descriptorSetLayout));
    NVVK_DBG_NAME(m_descriptorSetLayout);

    const VkPushConstantRange pushConstants = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(InputUniforms)};
    NVVK_CHECK(nvvk::createPipelineLayout(m_device, &m_pipelineLayout, {m_descriptorSetLayout}, {pushConstants}));
    NVVK_DBG_NAME(m_pipelineLayout);
  }

  void createPipelines()
  {
    nvvk::GraphicsPipelineState pstate;

    pstate.vertexBindings = {
        {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, .stride = sizeof(Vertex), .divisor = 1}};
    pstate.vertexAttributes = {{.sType  = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                .format = VK_FORMAT_R32G32_SFLOAT,
                                .offset = offsetof(Vertex, pos)}};

    VkPipelineRenderingCreateInfo renderingInfo{.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
                                                .colorAttachmentCount    = 1,
                                                .pColorAttachmentFormats = &m_rgba32Format};

    // Create the pipeline for "Image"
    {
      nvvk::GraphicsPipelineCreator creator;
      creator.pipelineInfo.layout = m_pipelineLayout;
      creator.colorFormats        = {m_rgba32Format};
      if(m_useSlang)
      {
        creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", m_slagModule);
        creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentMain", m_slagModule);
      }
      else
      {
        creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_glslVertModule);
        creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_glslFragModule);
      }
      creator.createGraphicsPipeline(m_device, nullptr, pstate, &m_pipelineImg);
      NVVK_DBG_NAME(m_pipelineImg);
    }

    // Create the pipeline for "Buffer-A"
    {
      nvvk::GraphicsPipelineCreator creator;
      creator.pipelineInfo.layout = m_pipelineLayout;
      creator.colorFormats        = {m_rgba32Format};
      if(m_useSlang)
      {
        creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "vertexMain", m_slagModule);
        creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "fragmentBuffer", m_slagModule);
      }
      else
      {
        creator.addShader(VK_SHADER_STAGE_VERTEX_BIT, "main", m_glslVertModule);
        creator.addShader(VK_SHADER_STAGE_FRAGMENT_BIT, "main", m_glslFragAModule);
      }

      creator.createGraphicsPipeline(m_device, nullptr, pstate, &m_pipelineBufA);
      NVVK_DBG_NAME(m_pipelineBufA);
    }
  }

  // Render with one of the pipeline.
  // Buffer-A will write back to itself while Image will take Buffer-A as input
  void renderToBuffer(VkCommandBuffer cmd, VkPipeline pipeline, GbufItems inImage, GbufItems outImage)
  {
    NVVK_DBG_SCOPE(cmd);  // <-- Helps to debug in NSight

    // Rendering to GBuffer: attachment information
    VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
    colorAttachment.imageView                 = m_gBuffers.getColorImageView();
    colorAttachment.clearValue                = {.color = m_clearColor};
    colorAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_LOAD;
    VkRenderingInfo renderingInfo             = DEFAULT_VkRenderingInfo;
    renderingInfo.renderArea                  = DEFAULT_VkRect2D(m_gBuffers.getSize());
    renderingInfo.colorAttachmentCount        = 1;
    renderingInfo.pColorAttachments           = &colorAttachment;


    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
    vkCmdBeginRendering(cmd, &renderingInfo);
    {
      nvvk::GraphicsPipelineState::cmdSetViewportAndScissor(cmd, m_gBuffers.getSize());

      // Writing descriptor
      nvvk::WriteSetContainer writeContainer;
      writeContainer.append(m_bindings.getWriteSet(0), m_gBuffers.getDescriptorImageInfo(inImage));
      vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0,
                                static_cast<uint32_t>(writeContainer.size()), writeContainer.data());

      // Pushing the "Input Uniforms"
      vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(InputUniforms), &m_inputUniform);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

      const VkDeviceSize offsets{0};
      vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
      vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT16);
      vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);
    }
    vkCmdEndRendering(cmd);
    nvvk::cmdImageMemoryBarrier(cmd, {m_gBuffers.getColorImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL});
  }


  void createGeometryBuffers()
  {
    // -1-1 -- 1-1
    //   |      |
    // -1 1 -- 1 1
    const std::vector<Vertex>   vertices = {{{-1.0F, -1.0F}}, {{1.0F, -1.0F}}, {{1.0F, 1.0F}}, {{-1.0F, 1.0F}}};
    const std::vector<uint16_t> indices  = {0, 3, 2, 0, 2, 1};

    {
      VkCommandBuffer       cmd = m_app->createTempCmdBuffer();
      nvvk::StagingUploader uploader;
      uploader.init(&m_alloc);

      NVVK_CHECK(m_alloc.createBuffer(m_vertices, std::span(vertices).size_bytes(), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT));
      NVVK_CHECK(m_alloc.createBuffer(m_indices, std::span(indices).size_bytes(), VK_BUFFER_USAGE_INDEX_BUFFER_BIT));
      NVVK_CHECK(uploader.appendBuffer(m_vertices, 0, std::span(vertices)));
      NVVK_CHECK(uploader.appendBuffer(m_indices, 0, std::span(indices)));
      NVVK_DBG_NAME(m_vertices.buffer);
      NVVK_DBG_NAME(m_indices.buffer);

      uploader.cmdUploadAppended(cmd);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      uploader.deinit();
    }
  }


  std::string compileShaders()
  {
    vkQueueWaitIdle(m_app->getQueue(0).queue);
    if(m_useSlang)
      return compileSlangShaders();
    else
      return compileGlslShaders();
  }

  std::string compileGlslShaders()
  {
    nvutils::ScopedTimer st(__FUNCTION__);  // Prints the time for running this / compiling shaders

    std::string error_msg;

    // Compile default shaders

    shaderc::SpvCompilationResult vert_result;
    shaderc::SpvCompilationResult frag_result;
    shaderc::SpvCompilationResult frag_result_a;
    compileGlslShader("shader_toy.vert.glsl", shaderc_shader_kind::shaderc_vertex_shader, false, vert_result);
    compileGlslShader("shader_toy.frag.glsl", shaderc_shader_kind::shaderc_fragment_shader, true, frag_result);
    compileGlslShader("shader_toy.frag.glsl", shaderc_shader_kind::shaderc_fragment_shader, false, frag_result_a);

    if(vert_result.GetNumErrors() == 0 && frag_result.GetNumErrors() == 0 && frag_result_a.GetNumErrors() == 0)
    {
      // Deleting resources, but not immediately as they are still in used
      m_app->submitResourceFree([vertModule = m_glslVertModule, fragModule = m_glslFragModule, fragAModule = m_glslFragAModule,
                                 device = m_device, pipelineImg = m_pipelineImg, pipelineBufA = m_pipelineBufA]() {
        vkDestroyShaderModule(device, vertModule, nullptr);
        vkDestroyShaderModule(device, fragModule, nullptr);
        vkDestroyShaderModule(device, fragAModule, nullptr);
        vkDestroyPipeline(device, pipelineImg, nullptr);
        vkDestroyPipeline(device, pipelineBufA, nullptr);
      });


      VkShaderModuleCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      // Vertex shader
      createInfo.codeSize = m_glslCompiler.getSpirvSize(vert_result);
      createInfo.pCode    = m_glslCompiler.getSpirv(vert_result);
      NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &m_glslVertModule));
      NVVK_DBG_NAME(m_glslVertModule);
      // Fragment shaders (main)
      createInfo.codeSize = m_glslCompiler.getSpirvSize(frag_result);
      createInfo.pCode    = m_glslCompiler.getSpirv(frag_result);
      NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &m_glslFragModule));
      NVVK_DBG_NAME(m_glslFragModule);
      // Fragment shaders (buffer A)
      createInfo.codeSize = m_glslCompiler.getSpirvSize(frag_result_a);
      createInfo.pCode    = m_glslCompiler.getSpirv(frag_result_a);
      NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &m_glslFragAModule));
      NVVK_DBG_NAME(m_glslFragAModule);
    }
    else
    {
      error_msg += vert_result.GetErrorMessage();
      error_msg += frag_result.GetErrorMessage();
      error_msg += frag_result_a.GetErrorMessage();
    }


    return error_msg;
  }

  void compileGlslShader(const std::string& filename, shaderc_shader_kind shaderKind, bool macroImage, shaderc::SpvCompilationResult& result)
  {
    nvutils::ScopedTimer st(__FUNCTION__);
    // setCompilerOptions();
    if(macroImage)
      m_glslCompiler.options().AddMacroDefinition("INCLUDE_IMAGE");
    result = m_glslCompiler.compileFile(filename, shaderKind);
  }

  std::string compileSlangShaders()
  {
    nvutils::ScopedTimer st(__FUNCTION__);  // Prints the time for running this / compiling shaders
    std::string          error_msg;

    if(m_slangCompiler.compileFile("shader_toy.slang"))
    {
      // Deleting resources, but not immediately as they are still in used
      m_app->submitResourceFree(
          [slagModule = m_slagModule, device = m_device, pipelineImg = m_pipelineImg, pipelineBufA = m_pipelineBufA]() {
            vkDestroyShaderModule(device, slagModule, nullptr);
            vkDestroyPipeline(device, pipelineImg, nullptr);
            vkDestroyPipeline(device, pipelineBufA, nullptr);
          });


      const VkShaderModuleCreateInfo createInfo{
          .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
          .codeSize = m_slangCompiler.getSpirvSize(),
          .pCode    = m_slangCompiler.getSpirv(),
      };
      NVVK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &m_slagModule));
      NVVK_DBG_NAME(m_slagModule);
    }
    else
    {
      error_msg += "Error compiling Slang\n";
      error_msg += m_slangCompiler.getLastDiagnosticMessage();
      return error_msg;
    }


    return error_msg;
  }


  void updateUniforms()
  {
    // Grab Data
    const glm::vec2 mouse_pos = ImGui::GetMousePos();         // Current mouse pos in window
    const glm::vec2 corner    = ImGui::GetCursorScreenPos();  // Corner of the viewport
    const glm::vec2 size      = ImGui::GetContentRegionAvail();

    // Set uniforms
    m_inputUniform.iResolution           = glm::vec3(size, 0);
    m_inputUniform.iChannelResolution[0] = glm::vec3(size, 0);

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
      glm::vec2 mpos(mouse_pos - corner);
      mpos.y                = size.y - mpos.y;  // Inverting mouse position
      m_inputUniform.iMouse = glm::vec4(mpos, 1, 0);
    }
    else
    {
      m_inputUniform.iMouse.z = 0;  // Just setting the button down
    }
  }

  // Find the full path of the shader file
  std::string getFilePath(const std::string& file)
  {
    std::string directoryPath;
    for(const auto& dir : nvsamples::getShaderDirs())
    {
      fs::path p = fs::path(dir) / fs::path(file);
      if(fs::exists(p))
      {
        directoryPath = p.string();
        break;
      }
    }
    return directoryPath;
  }

  // Opening a file
  void openFile(const std::string& file)
  {
    std::string filePath = getFilePath(file);

#ifdef _WIN32  // For Windows
    std::string command = "start " + std::string(filePath);
    system(command.c_str());
#elif defined __linux__  // For Linux
    std::string command = "xdg-open " + std::string(filePath);
    system(command.c_str());
#endif
  }

  void destroyResources()
  {
    vkDestroyPipeline(m_device, m_pipelineImg, nullptr);
    vkDestroyPipeline(m_device, m_pipelineBufA, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_descriptorSetLayout, nullptr);


    vkDestroyShaderModule(m_device, m_slagModule, nullptr);
    vkDestroyShaderModule(m_device, m_glslVertModule, nullptr);
    vkDestroyShaderModule(m_device, m_glslFragModule, nullptr);
    vkDestroyShaderModule(m_device, m_glslFragAModule, nullptr);

    m_alloc.destroyBuffer(m_vertices);
    m_alloc.destroyBuffer(m_indices);
    m_vertices = {};
    m_indices  = {};
    m_gBuffers.deinit();
    m_samplerPool.deinit();
    m_alloc.deinit();
  }

  void onLastHeadlessFrame() override
  {
    m_app->saveImageToFile(m_gBuffers.getColorImage(), m_gBuffers.getSize(),
                           nvutils::getExecutablePath().replace_extension(".jpg").string());
  }

  //--------------------------------------------------------------------------------------------------
  nvapp::Application*     m_app{};
  nvvk::GBuffer           m_gBuffers{};
  nvvk::ResourceAllocator m_alloc{};
  nvvkglsl::GlslCompiler  m_glslCompiler{};
  nvslang::SlangCompiler  m_slangCompiler{};
  nvvk::SamplerPool       m_samplerPool;

  nvvk::DescriptorBindings    m_bindings;
  nvvk::GraphicsPipelineState m_graphicState;
  VkDescriptorSetLayout       m_descriptorSetLayout{};
  VkPipeline                  m_pipeline{};
  VkPipelineLayout            m_pipelineLayout{};

  std::unique_ptr<FileCheck> m_fileBufferA{};
  std::unique_ptr<FileCheck> m_fileImage{};


  VkFormat          m_rgba32Format = VK_FORMAT_R32G32B32A32_SFLOAT;  // Color format of the images
  VkPipeline        m_pipelineImg  = VK_NULL_HANDLE;                 // The graphic pipeline to render
  VkPipeline        m_pipelineBufA = VK_NULL_HANDLE;                 // The graphic pipeline to render
  nvvk::Buffer      m_vertices;                                      // Buffer of the vertices
  nvvk::Buffer      m_indices;                                       // Buffer of the indices
  VkClearColorValue m_clearColor{{0.1F, 0.4F, 0.1F, 1.0F}};          // Clear color
  VkDevice          m_device = VK_NULL_HANDLE;                       // Convenient

  // Settings
  int           m_frame{0};
  float         m_time{0};
  bool          m_pause{false};
  bool          m_useSlang{true};
  InputUniforms m_inputUniform;

  VkShaderModule m_slagModule{};
  VkShaderModule m_glslVertModule{};
  VkShaderModule m_glslFragModule{};
  VkShaderModule m_glslFragAModule{};
};

int main(int argc, char** argv)
{
  nvapp::ApplicationCreateInfo appInfo;
  nvvk::Context                vkContext;  // The Vulkan context

  nvutils::ParameterParser   cli(nvutils::getExecutablePath().stem().string());
  nvutils::ParameterRegistry reg;
  reg.add({"headless", "Run in headless mode"}, &appInfo.headless, true);
  reg.add({"frames", "Number of frames to render in headless mode"}, &appInfo.headlessFrameCount, true);
  cli.add(reg);
  cli.parse(argc, argv);


  nvvk::ContextInitInfo vkSetup{
      .instanceExtensions = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME},
      .deviceExtensions   = {{VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME}},
  };
  if(!appInfo.headless)
  {
    nvvk::addSurfaceExtensions(vkSetup.instanceExtensions);
    vkSetup.deviceExtensions.emplace_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }

  // Vulkan context creation
  if(vkContext.init(vkSetup) != VK_SUCCESS)
  {
    LOGE("Error in Vulkan context creation\n");
    return 1;
  }


  appInfo.name           = fmt::format("{} ({})", TARGET_NAME, SHADER_LANGUAGE_STR);
  appInfo.vSync          = true;
  appInfo.instance       = vkContext.getInstance();
  appInfo.device         = vkContext.getDevice();
  appInfo.physicalDevice = vkContext.getPhysicalDevice();
  appInfo.queues         = vkContext.getQueueInfos();

  // Create the application
  nvapp::Application app;
  app.init(appInfo);

  app.addElement(std::make_shared<nvapp::ElementDefaultMenu>());
  app.addElement(std::make_shared<TinyShaderToy>());

  app.run();

  app.deinit();
  vkContext.deinit();

  return 0;
}
