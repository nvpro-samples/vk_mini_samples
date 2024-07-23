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


/*
Note: This is primarily to show how to compile shaders on the fly.

This sample replicate in a simple form, the execution of shaders like 
 the ones found on https://www.shadertoy.com/. shows how shaders can be loaded and reloaded from disk.
 - Many uniforms can be accessed: iResolution, iTimes, iFrame, iChannelTime, iMouse, ... 
 - The BufferA can be persisted between frames, same as with ShaderToy and its result
   is stored in iChannel0

*/

#include <glm/glm.hpp>

// clang-format off
#define IM_VEC2_CLASS_EXTRA ImVec2(const glm::vec2& f) {x = f.x; y = f.y;} operator glm::vec2() const { return glm::vec2(x, y); }
// clang-format on

#include <array>
#include <filesystem>
namespace fs = std::filesystem;

#include <vulkan/vulkan_core.h>

#include <imgui.h>
#include <shaderc/shaderc.hpp>

#define VMA_IMPLEMENTATION
#define IMGUI_DEFINE_MATH_OPERATORS
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_benchmark_parameters.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/glsl_compiler.hpp"
#include "nvvkhl/pipeline_container.hpp"


#if USE_SLANG
#include "slang_compiler.hpp"
#endif

#include "nvvk/shaders_vk.hpp"

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


class TinyShaderToy : public nvvkhl::IAppElement
{
  enum GbufItems
  {
    eImage,
    eBufA0,
    eBufA1
  };


public:
  TinyShaderToy()           = default;
  ~TinyShaderToy() override = default;

  std::vector<std::string> getShaderDirs()
  {
    return {
        NVPSystem::exePath() + std::string(PROJECT_RELDIRECTORY) + std::string("shaders"),  //
        NVPSystem::exePath() + std::string(PROJECT_NAME) + std::string("/shaders"),         //
        NVPSystem::exePath()                                                                //
    };
  }


  void onAttach(nvvkhl::Application* app) override
  {
    m_app    = app;
    m_device = m_app->getDevice();
    m_dutil  = std::make_unique<nvvk::DebugUtil>(m_device);  // Debug utility
    m_alloc  = std::make_unique<nvvkhl::AllocVma>(VmaAllocatorCreateInfo{
         .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
         .physicalDevice = app->getPhysicalDevice(),
         .device         = app->getDevice(),
         .instance       = app->getInstance(),
    });  // Allocator
    m_dset   = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    m_fileBufferA = std::make_unique<FileCheck>(getFilePath("buffer_a.glsl"));
    m_fileImage   = std::make_unique<FileCheck>(getFilePath("image.glsl"));

#if USE_GLSL
    // shaderc compiler
    m_glslC = std::make_unique<nvvkhl::GlslCompiler>();
    // Add search paths
    for(const auto& path : getShaderDirs())
      m_glslC->addInclude(path);
#elif USE_SLANG
    m_slangC = std::make_unique<SlangCompiler>();
#endif

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

  void onResize(uint32_t width, uint32_t height) override { createGbuffers({width, height}); }

  void onUIRender() override
  {
    if(!m_gBuffers)
      return;

    static std::string error_msg;
    {  // Setting panel
      ImGui::Begin("Settings");
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

      bool reloadShaders = ImGui::Button("Reload Shaders");
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
      ImGui::Image(m_gBuffers->getDescriptorSet(eImage), ImGui::GetContentRegionAvail());
      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    if(!m_gBuffers)
      return;

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
    const VkImageMemoryBarrier image_memory_barrier =
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
    glm::vec2 pos;
  };

  void createPipelineLayout()
  {
    const VkPushConstantRange push_constants = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(InputUniforms)};

    m_dset->addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
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

    VkPipelineRenderingCreateInfo prend_info{.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
                                             .colorAttachmentCount    = 1,
                                             .pColorAttachmentFormats = &m_rgba32Format};

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
    const nvvk::DebugUtil::ScopedCmdLabel sdbg = m_dutil->DBG_SCOPE(cmd);

    // Render to `outImage` buffer, depth is unused
    nvvk::createRenderingInfo r_info({{0, 0}, m_gBuffers->getSize()}, {m_gBuffers->getColorImageView(outImage)},
                                     m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_CLEAR);
    r_info.pStencilAttachment = nullptr;

    vkCmdBeginRendering(cmd, &r_info);
    m_app->setViewport(cmd);

    // Writing descriptor
    const VkDescriptorImageInfo       in_desc = m_gBuffers->getDescriptorImageInfo(inImage);
    std::vector<VkWriteDescriptorSet> writes;
    writes.push_back(m_dset->makeWrite(0, 0, &in_desc));
    vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dset->getPipeLayout(), 0,
                              static_cast<uint32_t>(writes.size()), writes.data());

    // Pushing the "Input Uniforms"
    vkCmdPushConstants(cmd, m_dset->getPipeLayout(), VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(InputUniforms), &m_inputUniform);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    const VkDeviceSize offsets{0};
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
    vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);

    vkCmdEndRendering(cmd);
  }

  void createGbuffers(VkExtent2D size)
  {
    // Creating the 3 buffers (see BufItems)
    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), size,
                                                   std::vector{m_rgba32Format, m_rgba32Format, m_rgba32Format});
  }

  void createGeometryBuffers()
  {
    // -1-1 -- 1-1
    //   |      |
    // -1 1 -- 1 1
    const std::vector<Vertex>   vertices = {{{-1.0F, -1.0F}}, {{1.0F, -1.0F}}, {{1.0F, 1.0F}}, {{-1.0F, 1.0F}}};
    const std::vector<uint16_t> indices  = {0, 3, 2, 0, 2, 1};

    {
      VkCommandBuffer cmd = m_app->createTempCmdBuffer();
      m_vertices          = m_alloc->createBuffer(cmd, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      m_indices           = m_alloc->createBuffer(cmd, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
      m_app->submitAndWaitTempCmdBuffer(cmd);
      m_dutil->DBG_NAME(m_vertices.buffer);
      m_dutil->DBG_NAME(m_indices.buffer);
    }
  }


  void setCompilerOptions()
  {
    m_glslC->resetOptions();
    m_glslC->options()->SetTargetSpirv(shaderc_spirv_version::shaderc_spirv_version_1_3);
    m_glslC->options()->SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
    m_glslC->options()->SetGenerateDebugInfo();
    m_glslC->options()->SetOptimizationLevel(shaderc_optimization_level_zero);
  }

  std::string compileShaders()
  {
#if USE_GLSL
    return compileGlslShaders();
#elif USE_SLANG
    return compileSlangShaders();
#endif
  }

  std::string compileGlslShaders()
  {
    nvh::ScopedTimer st(__FUNCTION__);  // Prints the time for running this / compiling shaders
    std::string      error_msg;

    // Compile default shaders

    shaderc::SpvCompilationResult vert_result;
    shaderc::SpvCompilationResult frag_result;
    shaderc::SpvCompilationResult frag_result_a;
    compileGlslShader("raster.vert.glsl", shaderc_shader_kind::shaderc_vertex_shader, false, vert_result);
    compileGlslShader("raster.frag.glsl", shaderc_shader_kind::shaderc_fragment_shader, true, frag_result);
    compileGlslShader("raster.frag.glsl", shaderc_shader_kind::shaderc_fragment_shader, false, frag_result_a);

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

  void compileGlslShader(const std::string& filename, shaderc_shader_kind shaderKind, bool macroImage, shaderc::SpvCompilationResult& result)
  {
    nvh::ScopedTimer st(__FUNCTION__);
    setCompilerOptions();
    if(macroImage)
      m_glslC->options()->AddMacroDefinition("INCLUDE_IMAGE");
    result = m_glslC->compileFile(filename, shaderKind);
  }

  std::string compileSlangShaders()
  {
    nvh::ScopedTimer st(__FUNCTION__);  // Prints the time for running this / compiling shaders
    std::string      error_msg;

#if USE_SLANG
    // Spirv code for the shaders
    std::vector<uint32_t> vertexSpirvCode;
    std::vector<uint32_t> fragmentSpirvCode;
    std::vector<uint32_t> fragmentASpirvCode;

    // The shader file
    std::vector<std::string> searchPaths    = getShaderDirs();
    std::string              rasterFilename = nvh::findFile("raster.slang", searchPaths, true);

    // Always create a new session before compiling all files
    m_slangC->newSession();
    {
      nvh::ScopedTimer st("Vertex");

      auto compiler = m_slangC->createCompileRequest(rasterFilename, "vertexMain", SLANG_STAGE_VERTEX);
      {
        nvh::ScopedTimer st("SlangCompile");
        if(SLANG_FAILED(compiler->compile()))
          return compiler->getDiagnosticOutput();
      }
      m_slangC->getSpirvCode(compiler, vertexSpirvCode);
    }
    {
      nvh::ScopedTimer st("Frag Main");

      auto compiler = m_slangC->createCompileRequest(rasterFilename, "fragmentMain", SLANG_STAGE_FRAGMENT);
      compiler->addPreprocessorDefine("INCLUDE_IMAGE", "1");
      {
        nvh::ScopedTimer st("SlangCompile");
        if(SLANG_FAILED(compiler->compile()))
          return compiler->getDiagnosticOutput();
      }
      m_slangC->getSpirvCode(compiler, fragmentSpirvCode);
    }
    {
      nvh::ScopedTimer st("Frag Buffer-A");

      auto compiler = m_slangC->createCompileRequest(rasterFilename, "fragmentMain", SLANG_STAGE_FRAGMENT);
      compiler->addPreprocessorDefine("INCLUDE_BUFFER_A", "1");
      {
        nvh::ScopedTimer st("SlangCompile");
        if(SLANG_FAILED(compiler->compile()))
          return compiler->getDiagnosticOutput();
      }
      m_slangC->getSpirvCode(compiler, fragmentASpirvCode);
    }

    // Deleting resources, but not immediately as they are still in used
    nvvkhl::Application::submitResourceFree([vmod = m_vmodule, fmod_i = m_fmodule, fmod_a = m_fmoduleA,
                                             device = m_device, gp = m_pipelineImg, gp_a = m_pipelineBufA]() {
      vkDestroyShaderModule(device, vmod, nullptr);
      vkDestroyShaderModule(device, fmod_i, nullptr);
      vkDestroyShaderModule(device, fmod_a, nullptr);
      vkDestroyPipeline(device, gp, nullptr);
      vkDestroyPipeline(device, gp_a, nullptr);
    });

    {
      nvh::ScopedTimer st("Create Vulkan Shader Modules");
      m_vmodule  = nvvk::createShaderModule(m_device, static_cast<const uint32_t*>(vertexSpirvCode.data()),
                                            vertexSpirvCode.size() * sizeof(uint32_t));
      m_fmodule  = nvvk::createShaderModule(m_device, static_cast<const uint32_t*>(fragmentSpirvCode.data()),
                                            fragmentSpirvCode.size() * sizeof(uint32_t));
      m_fmoduleA = nvvk::createShaderModule(m_device, static_cast<const uint32_t*>(fragmentASpirvCode.data()),
                                            fragmentASpirvCode.size() * sizeof(uint32_t));
      m_dutil->setObjectName(m_vmodule, "Vertex");
      m_dutil->setObjectName(m_fmodule, "Image");
      m_dutil->setObjectName(m_fmoduleA, "BufferA");
    }

#endif
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
    for(const auto& dir : getShaderDirs())
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
  std::unique_ptr<FileCheck>                    m_fileBufferA;
  std::unique_ptr<FileCheck>                    m_fileImage;

#if USE_SLANG
  std::unique_ptr<SlangCompiler> m_slangC;
#endif


  VkFormat          m_rgba32Format = VK_FORMAT_R32G32B32A32_SFLOAT;  // Color format of the images
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

int main(int argc, char** argv)
{
  nvvk::ContextCreateInfo vkSetup;
  vkSetup.setVersion(1, 3);
  vkSetup.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  vkSetup.addDeviceExtension(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
  nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);

  nvvk::Context vkContext;
  vkContext.init(vkSetup);

  nvvkhl::ApplicationCreateInfo spec;
  spec.name           = fmt::format("{} ({})", PROJECT_NAME, SHADER_LANGUAGE_STR);
  spec.vSync          = true;
  spec.instance       = vkContext.m_instance;
  spec.device         = vkContext.m_device;
  spec.physicalDevice = vkContext.m_physicalDevice;
  spec.queues         = {vkContext.m_queueGCT, vkContext.m_queueC, vkContext.m_queueT};

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  auto test = std::make_shared<nvvkhl::ElementBenchmarkParameters>(argc, argv);
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());
  app->addElement(std::make_shared<TinyShaderToy>());

  app->run();
  app.reset();
  vkContext.deinit();

  return test->errorCode();
}
