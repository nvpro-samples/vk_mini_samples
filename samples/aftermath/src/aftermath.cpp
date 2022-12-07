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
 This sample shows how to integrate Nsight Aftermath.

 Note: if USE_NSIGHT_AFTERMATH is not defined, this means the CMake haven't 
       found the SDK. 
       - Download the NSight Aftermath SDK
       - Put the include/ and lib/ under aftermath/aftermath_sdk
       - Delete CMake cache and rerun Cmake

*/


#include <array>
#include <filesystem>
#include <vulkan/vulkan_core.h>
#include <imgui.h>

#define VMA_IMPLEMENTATION
#include "nvh/primitives.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/specialization.hpp"
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"

#include "shaders/device_host.h"


#include "_autogen/raster.frag.h"
#include "_autogen/raster.vert.h"


#include "GLFW/glfw3.h"

#ifdef USE_NSIGHT_AFTERMATH
#include "NsightAftermathGpuCrashTracker.h"
// Global marker map
::GpuCrashTracker::MarkerMap       g_marker_map;
std::unique_ptr<::GpuCrashTracker> g_aftermath_tracker;

// Display errors
#ifdef _WIN32
#define ERR_EXIT(err_msg, err_class)                                                                                   \
  do                                                                                                                   \
  {                                                                                                                    \
    MessageBox(nullptr, err_msg, err_class, MB_OK);                                                                    \
    exit(1);                                                                                                           \
  } while(0)
#else
#define ERR_EXIT(err_msg, err_class)                                                                                   \
  do                                                                                                                   \
  {                                                                                                                    \
    printf("%s\n", err_msg);                                                                                           \
    fflush(stdout);                                                                                                    \
    exit(1);                                                                                                           \
  } while(0)
#endif

#endif  // USE_NSIGHT_AFTERMATH


#ifdef USE_NSIGHT_AFTERMATH

// Override the default checkResult from nvvk/error_vk.cpp
bool nvvk::checkResult(VkResult result, const char* /*file*/, int32_t /*line*/)
{
  if(result == VK_SUCCESS)
  {
    return false;
  }
  if(result == VK_ERROR_DEVICE_LOST)
  {
    // Device lost notification is asynchronous to the NVIDIA display
    // driver's GPU crash handling. Give the Nsight Aftermath GPU crash dump
    // thread some time to do its work before terminating the process.
    auto tdr_termination_timeout = std::chrono::seconds(5);
    auto t_start                 = std::chrono::steady_clock::now();
    auto t_elapsed               = std::chrono::milliseconds::zero();

    GFSDK_Aftermath_CrashDump_Status status = GFSDK_Aftermath_CrashDump_Status_Unknown;
    AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetCrashDumpStatus(&status));

    while(status != GFSDK_Aftermath_CrashDump_Status_CollectingDataFailed
          && status != GFSDK_Aftermath_CrashDump_Status_Finished && t_elapsed < tdr_termination_timeout)
    {
      // Sleep 50ms and poll the status again until timeout or Aftermath finished processing the crash dump.
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      AFTERMATH_CHECK_ERROR(GFSDK_Aftermath_GetCrashDumpStatus(&status));

      auto t_end = std::chrono::steady_clock::now();
      t_elapsed  = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    }

    if(status != GFSDK_Aftermath_CrashDump_Status_Finished)
    {
      std::stringstream err_msg;
      err_msg << "Unexpected crash dump status: " << status;
      ERR_EXIT(err_msg.str().c_str(), "Aftermath Error");
    }

    std::stringstream err_msg;
    err_msg << "Aftermath file dumped under:\n\n";
    err_msg << std::filesystem::current_path().string();

    // Terminate on failure
#ifdef _WIN32
    err_msg << "\n\n\nSave path to clipboard?";
    int ret = MessageBox(nullptr, err_msg.str().c_str(), "Nsight Aftermath", MB_YESNO | MB_ICONEXCLAMATION);
    if(ret == IDYES)
    {
      ImGui::SetClipboardText(std::filesystem::current_path().string().c_str());
    }
#else
    printf("%s\n", err_msg.str().c_str());
#endif

    exit(1);
  }
  return false;
}
#endif

#ifdef USE_NSIGHT_AFTERMATH
// A helper that prepends the frame number to a string
static auto createMarkerStringForFrame(const char* marker_string, int frame_number) -> std::string
{
  std::stringstream ss;
  ss << "Frame " << frame_number << ": " << marker_string;
  return ss.str();
};

// A helper for setting a checkpoint marker
static void setCheckpointMarker(VkCommandBuffer cmd, const std::string& marker_data, int frame_number)
{
  // App is responsible for handling marker memory, and for resolving the memory at crash dump generation time.
  // The actual "const void* pCheckpointMarker" passed to setCheckpointNV in this case can be any uniquely identifying value that the app can resolve to the marker data later.
  // For this sample, we will use this approach to generating a unique marker value:
  // We keep a ringbuffer with a marker history of the last c_markerFrameHistory frames (currently 4).
  unsigned int marker_map_index         = frame_number % ::GpuCrashTracker::c_markerFrameHistory;
  auto&        current_frame_marker_map = g_marker_map[marker_map_index];
  // Take the index into the ringbuffer, multiply by 10000, and add the total number of markers logged so far in the current frame, +1 to avoid a value of zero.
  size_t marker_id = static_cast<uint64_t>(marker_map_index) * 10000 + current_frame_marker_map.size() + 1;
  // This value is the unique identifier we will pass to Aftermath and internally associate with the marker data in the map.
  current_frame_marker_map[marker_id] = marker_data;
  vkCmdSetCheckpointNV(cmd, (const void*)marker_id);
  // For example, if we are on frame 625, markerMapIndex = 625 % 4 = 1...
  // The first marker for the frame will have markerID = 1 * 10000 + 0 + 1 = 10001.
  // The 15th marker for the frame will have markerID = 1 * 10000 + 14 + 1 = 10015.
  // On the next frame, 626, markerMapIndex = 626 % 4 = 2.
  // The first marker for this frame will have markerID = 2 * 10000 + 0 + 1 = 20001.
  // The 15th marker for the frame will have markerID = 2 * 10000 + 14 + 1 = 20015.
  // So with this scheme, we can safely have up to 10000 markers per frame, and can guarantee a unique markerID for each one.
  // There are many ways to generate and track markers and unique marker identifiers!
};
#endif


class AftermathSample : public nvvkhl::IAppElement
{
  enum TdrReason
  {
    eNone,
    eOutOfBoundsVertexBufferOffset,
  };


public:
  AftermathSample()           = default;
  ~AftermathSample() override = default;

  void onAttach(nvvkhl::Application* app) override
  {
    m_app         = app;
    m_device      = m_app->getDevice();
    m_dutil       = std::make_unique<nvvk::DebugUtil>(m_device);                    // Debug utility
    m_alloc       = std::make_unique<nvvkhl::AllocVma>(m_app->getContext().get());  // Allocator
    m_depthFormat = nvvk::findDepthFormat(m_app->getPhysicalDevice());              // Not all depth are supported
    m_dset        = std::make_unique<nvvk::DescriptorSetContainer>(m_device);

    createPipeline();
    createVkResources();
    updateDescriptorSet();
  }

  void onDetach() override { detroyResources(); }

  void onResize(uint32_t width, uint32_t height) override { createGbuffers({width, height}); }

  void onUIRender() override
  {
    {  // Setting panel
      ImGui::Begin("Settings");

      if(ImGui::Button("1. Crash"))
      {
        m_currentPipe = 1;
      }
      ImGui::SameLine();
      ImGui::Text("Infinite loop in vertex shader");

      if(ImGui::Button("2. Crash"))
      {
        m_currentPipe = 2;
      }
      ImGui::SameLine();
      ImGui::Text("Infinite loop in fragment shader");

      if(ImGui::Button("3. Bug"))
      {
        m_tdrReason = TdrReason::eOutOfBoundsVertexBufferOffset;
      }
      ImGui::SameLine();
      ImGui::Text("Out of bound vertex buffer");

      if(ImGui::Button("4. Bug"))
      {
        m_alloc->destroy(m_vertices);
      }
      ImGui::SameLine();
      ImGui::Text("Delete vertex buffer");

      if(ImGui::Button("5. Bug"))
      {
        wrongDescriptorSet();
      }
      ImGui::SameLine();
      ImGui::Text("Wrong buffer address");

      if(ImGui::Button("6. Crash"))
      {
        m_currentPipe = 3;
      }
      ImGui::SameLine();
      ImGui::Text("Out of bound buffer");

      if(ImGui::Button("7. Crash"))
      {
        m_currentPipe = 4;
      }
      ImGui::SameLine();
      ImGui::Text("Bad texture access");

      ImGui::End();
    }

    {  // Display the G-Buffer image
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0F, 0.0F));
      ImGui::Begin("Viewport");
      ImGui::Image(m_gBuffers->getDescriptorSet(), ImGui::GetContentRegionAvail());
      ImGui::End();
      ImGui::PopStyleVar();
    }
  }

  void onRender(VkCommandBuffer cmd) override
  {
    auto _sdbg = m_dutil->DBG_SCOPE(cmd);
#if defined(USE_NSIGHT_AFTERMATH)
    m_frameNumber++;

    // clear the marker map for the current frame before writing any markers
    g_marker_map[m_frameNumber % ::GpuCrashTracker::c_markerFrameHistory].clear();

    // Insert a device diagnostic checkpoint into the command stream
    setCheckpointMarker(cmd, createMarkerStringForFrame("Draw Rect", m_frameNumber), m_frameNumber);
#endif

    nvvk::createRenderingInfo r_info({{0, 0}, m_viewSize}, {m_gBuffers->getColorImageView()}, m_gBuffers->getDepthImageView(),
                                     VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_LOAD_OP_CLEAR, m_clearColor);
    r_info.pStencilAttachment = nullptr;

    float         view_aspect_ratio = static_cast<float>(m_viewSize.width) / static_cast<float>(m_viewSize.height);
    const auto&   clip              = CameraManip.getClipPlanes();
    nvmath::mat4f matv              = CameraManip.getMatrix();
    nvmath::mat4f matp              = nvmath::perspectiveVK(CameraManip.getFov(), view_aspect_ratio, clip.x, clip.y);

    FrameInfo finfo{};
    finfo.time[0] = static_cast<float>(ImGui::GetTime());
    finfo.time[1] = 0;
    finfo.mpv     = matp * matv;

    finfo.resolution = nvmath::vec2f(m_viewSize.width, m_viewSize.height);
    finfo.badOffset  = std::rand();  // 0xDEADBEEF;
    vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(FrameInfo), &finfo);

    vkCmdBeginRendering(cmd, &r_info);
    m_app->setViewport(cmd);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipe.layout, 0, 1, m_dset->getSets(), 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipe.plines[m_currentPipe]);
    VkDeviceSize offsets{0};
    if(m_tdrReason == TdrReason::eOutOfBoundsVertexBufferOffset)
    {
      offsets = std::rand();
    }
    vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertices.buffer, &offsets);
    vkCmdBindIndexBuffer(cmd, m_indices.buffer, 0, VK_INDEX_TYPE_UINT32);
    auto index_count = static_cast<uint32_t>(m_meshes[0].indices.size());
    vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

    vkCmdEndRendering(cmd);
  }

private:
  void createPipeline()
  {
    auto& d = m_dset;
    d->addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_FRAGMENT_BIT);
    d->addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_FRAGMENT_BIT);
    d->addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_ALL | VK_SHADER_STAGE_FRAGMENT_BIT);
    d->initLayout();
    d->initPool(2);  // two frames - allow to change on the fly

    VkPipelineLayoutCreateInfo create_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    create_info.setLayoutCount = 1;
    create_info.pSetLayouts    = &d->getLayout();
    NVVK_CHECK(vkCreatePipelineLayout(m_device, &create_info, nullptr, &m_pipe.layout));

    VkPipelineRenderingCreateInfo prend_info{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    prend_info.colorAttachmentCount    = 1;
    prend_info.pColorAttachmentFormats = &m_colorFormat;
    prend_info.depthAttachmentFormat   = m_depthFormat;

    nvvk::GraphicsPipelineState pstate;
    pstate.addBindingDescriptions({{0, sizeof(nvh::PrimitiveVertex)}});
    pstate.addAttributeDescriptions({
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, p))},  // Position
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, n))},  // Normal
        {2, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(nvh::PrimitiveVertex, t))},     // Normal

    });

    auto vert_shd = std::vector<uint32_t>{std::begin(raster_vert), std::end(raster_vert)};
    auto frag_shd = std::vector<uint32_t>{std::begin(raster_frag), std::end(raster_frag)};

    // Shader sources, pre-compiled to Spir-V (see Makefile)
    nvvk::GraphicsPipelineGenerator pgen(m_device, m_pipe.layout, prend_info, pstate);
    pgen.addShader(vert_shd, VK_SHADER_STAGE_VERTEX_BIT);
    pgen.addShader(frag_shd, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_pipe.plines.push_back(pgen.createPipeline());
    m_dutil->DBG_NAME(m_pipe.plines[0]);
    pgen.clearShaders();

#ifdef USE_NSIGHT_AFTERMATH
    g_aftermath_tracker->addShaderBinary(vert_shd);
    g_aftermath_tracker->addShaderBinary(frag_shd);
#endif  // USE_NSIGHT_AFTERMATH

    // Create many specializations (shader with constant values)
    // 1- Loop in vertex, 2- Loop in Fragment, 3- Over buffer
    for(int i = 1; i <= 10; i++)
    {
      nvvk::Specialization specialization;
      specialization.add(0, i);
      pgen.addShader(std::vector<uint32_t>{std::begin(raster_vert), std::end(raster_vert)}, VK_SHADER_STAGE_VERTEX_BIT).pSpecializationInfo =
          specialization.getSpecialization();
      pgen.addShader(std::vector<uint32_t>{std::begin(raster_frag), std::end(raster_frag)}, VK_SHADER_STAGE_FRAGMENT_BIT)
          .pSpecializationInfo = specialization.getSpecialization();
      m_pipe.plines.push_back(pgen.createPipeline());
      m_dutil->setObjectName(m_pipe.plines.back(), "Crash " + std::to_string(i));
      pgen.clearShaders();
    }
  }
  void updateDescriptorSet()
  {
    auto& d = m_dset;
    // Writing to descriptors
    VkDescriptorBufferInfo dbi_unif{m_bFrameInfo.buffer, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo dbi_val{m_bValues.buffer, 0, VK_WHOLE_SIZE};

    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(d->makeWrite(0, 0, &dbi_unif));
    writes.emplace_back(d->makeWrite(0, 1, &dbi_val));
    writes.emplace_back(d->makeWrite(0, 2, &m_texture.descriptor));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void wrongDescriptorSet()
  {
    auto& d = m_dset;
    // Writing to descriptors
    VkDescriptorBufferInfo dbi_unif{nullptr, 0, VK_WHOLE_SIZE};
    dbi_unif.buffer = VkBuffer(0xDEADBEEFDEADBEEF);
    std::vector<VkWriteDescriptorSet> writes;
    writes.emplace_back(d->makeWrite(0, 1, &dbi_unif));
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
  }

  void createGbuffers(VkExtent2D size)
  {
    m_viewSize = size;
    VkImageFormatProperties prop;
    NVVK_CHECK(vkGetPhysicalDeviceImageFormatProperties(
        m_app->getPhysicalDevice(), m_colorFormat, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT, 0, &prop));

    m_gBuffers = std::make_unique<nvvkhl::GBuffer>(m_device, m_alloc.get(), m_viewSize, m_colorFormat, m_depthFormat);
  }

  void createVkResources()
  {
    m_meshes.emplace_back(nvh::sphere(0.5F, 20, 20));

    {
      auto* cmd = m_app->createTempCmdBuffer();

      // Create buffer of the mesh
      m_vertices = m_alloc->createBuffer(cmd, m_meshes[0].vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
      m_indices  = m_alloc->createBuffer(cmd, m_meshes[0].indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
      m_dutil->DBG_NAME(m_vertices.buffer);
      m_dutil->DBG_NAME(m_indices.buffer);

      // Frame information
      m_bFrameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      m_dutil->DBG_NAME(m_bFrameInfo.buffer);

      // Dummy buffer of values
      std::vector<float> values = {0.5F};
      m_bValues = m_alloc->createBuffer(cmd, values, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      m_dutil->DBG_NAME(m_bValues.buffer);

      // Create dummy texture
      {
        VkImageCreateInfo    create_info = nvvk::makeImage2DCreateInfo({1, 1}, VK_FORMAT_R8G8B8A8_UNORM);
        VkSamplerCreateInfo  sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        std::vector<uint8_t> image_data = {255, 0, 255, 255};
        m_texture = m_alloc->createTexture(cmd, image_data.size() * sizeof(uint8_t), image_data.data(), create_info, sampler_info);
        m_dutil->DBG_NAME(m_texture.image);
        m_dutil->DBG_NAME(m_texture.descriptor.sampler);
      }

      m_app->submitAndWaitTempCmdBuffer(cmd);
    }

    // Camera position
    CameraManip.setLookat({0, 0, 1}, {0, 0, 0}, {0, 1, 0});
  }

  void detroyResources()
  {
    m_alloc->destroy(m_bFrameInfo);
    m_alloc->destroy(m_bValues);
    m_alloc->destroy(m_vertices);
    m_alloc->destroy(m_indices);
    m_alloc->destroy(m_texture);
    m_pipe.destroy(m_device);
    m_dset->deinit();
    m_vertices = {};
    m_indices  = {};
    m_gBuffers.reset();
  }

  nvvkhl::Application* m_app{nullptr};

  std::unique_ptr<nvvkhl::GBuffer>              m_gBuffers;
  std::unique_ptr<nvvk::DebugUtil>              m_dutil;
  std::unique_ptr<nvvkhl::AllocVma>             m_alloc;
  std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;  // Descriptor set
  nvvk::Buffer                                  m_bFrameInfo;
  nvvk::Buffer                                  m_bValues;

  TdrReason m_tdrReason{eNone};


  VkExtent2D                m_viewSize{0, 0};
  VkFormat                  m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;  // Color format of the image
  VkFormat                  m_depthFormat = VK_FORMAT_UNDEFINED;       // Depth format of the depth buffer
  nvvkhl::PipelineContainer m_pipe;                                    // Multiple pipelines
  nvvk::Buffer              m_vertices;                                // Buffer of the vertices
  nvvk::Buffer              m_indices;                                 // Buffer of the indices
  VkClearColorValue         m_clearColor{{0.0F, 0.0F, 0.0F, 1.0F}};    // Clear color
  VkDevice                  m_device = VK_NULL_HANDLE;                 // Convenient
  int                       m_frameNumber{0};
  int                       m_currentPipe{0};

  std::vector<nvh::PrimitiveMesh> m_meshes;
  nvvk::Texture                   m_texture;
};

int main(int argc, char** argv)
{
  nvvkhl::ApplicationCreateInfo spec;
  spec.name             = PROJECT_NAME " Example";
  spec.vSync            = true;
  spec.vkSetup.apiMajor = 1;
  spec.vkSetup.apiMinor = 3;
  spec.vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);

#ifdef USE_NSIGHT_AFTERMATH
  // Enable NV_device_diagnostic_checkpoints extension to be able to use Aftermath event markers.
  spec.vkSetup.addDeviceExtension(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
  // Enable NV_device_diagnostics_config extension to configure Aftermath features.
  VkDeviceDiagnosticsConfigCreateInfoNV aftermath_info{VK_STRUCTURE_TYPE_DEVICE_DIAGNOSTICS_CONFIG_CREATE_INFO_NV};
  aftermath_info.flags = VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_SHADER_DEBUG_INFO_BIT_NV
                         | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_RESOURCE_TRACKING_BIT_NV
                         | VK_DEVICE_DIAGNOSTICS_CONFIG_ENABLE_AUTOMATIC_CHECKPOINTS_BIT_NV;
  spec.vkSetup.addDeviceExtension(VK_NV_DEVICE_DIAGNOSTICS_CONFIG_EXTENSION_NAME, false, &aftermath_info);
#endif

  // #Aftermath - Initialization
#ifdef USE_NSIGHT_AFTERMATH
  g_aftermath_tracker = std::make_unique<::GpuCrashTracker>(g_marker_map);
  g_aftermath_tracker->initialize();
#endif

  // Create the application
  auto app = std::make_unique<nvvkhl::Application>(spec);

  // Create the test framework
  auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);

  // Add all application elements
  app->addElement(test);
  app->addElement(std::make_shared<nvvkhl::ElementCamera>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultMenu>());
  app->addElement(std::make_shared<nvvkhl::ElementDefaultWindowTitle>());
  app->addElement(std::make_shared<AftermathSample>());


  app->run();
  app.reset();
#ifdef USE_NSIGHT_AFTERMATH
  g_aftermath_tracker.reset();
#endif

  return test->errorCode();
}
