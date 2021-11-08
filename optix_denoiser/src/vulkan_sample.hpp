/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nvvk/appbase_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#include "nvh/gltfscene.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/raypicker_vk.hpp"

#include "shaders/host_device.h"

#include "hdr_env.hpp"
#include "denoiser.hpp"

constexpr int FRAME_IN_FLIGHT = 3;


//--------------------------------------------------------------------------------------------------
// Simple rasterizer and raytracer of glTF scenes
//
class VulkanSample : public nvvk::AppBaseVk
{
public:
  // Keep the handles of Vulkan, initialize resource manager
  void create(const nvvk::AppBaseVkCreateInfo& info) override;
  // Load the scene, create resources and pipelines
  void createScene(const std::string& filename);
  // Load Hdr and set image
  void createHdr(const std::string& hdrFilename);
  // Display info in the title bar
  void titleBar();
  // Rendering the user interface
  void renderUI();
  // Update/push the per frame buffer
  void updateUniformBuffer(VkCommandBuffer cmdBuf);
  // Run the post pipeline (tone mapper)
  void drawPost(VkCommandBuffer cmdBuf);
  // Raytrace the scene
  void raytrace(VkCommandBuffer cmdBuf);
  // Rasterize the scene
  void rasterize(VkCommandBuffer cmdBuff);
  // Free all resources allocated in create
  void freeResources();
  // Destroy everything
  void destroy() override;

  const VkClearColorValue& clearColor() { return m_clearColor; }
  const VkRenderPass&      offscreenRenderPass() { return m_offscreenRenderPass; }
  const VkFramebuffer&     offscreenFramebuffer() { return m_offscreenFramebuffer; }

  // Overriding to create 2x more command buffer per frame
  void createSwapchain(const VkSurfaceKHR& surface,
                       uint32_t            width,
                       uint32_t            height,
                       VkFormat            colorFormat = VK_FORMAT_B8G8R8A8_UNORM,
                       VkFormat            depthFormat = VK_FORMAT_UNDEFINED,
                       bool                vsync       = false) override;


  void submitWithTLSemaphore(const VkCommandBuffer& cmdBuf);
  void submitFrame(const VkCommandBuffer& cmdBuf);
  void createGbuffers();
  void denoise();
  void setImageToDisplay();
  bool needToDenoise();
  void copyImagesToCuda(const VkCommandBuffer& cmdBuf);
  void copyCudaImagesToVulkan(const VkCommandBuffer& cmdBuf);

  enum class RenderMode  // All rendering modes
  {
    eRaster,
    eRayTracer
  } m_renderMode{RenderMode::eRayTracer};

private:
  void loadScene(const std::string& filename);
  void createVertexBuffer(VkCommandBuffer cmdBuf);
  void createMaterialBuffer(VkCommandBuffer cmdBuf);
  void createInstanceInfoBuffer(VkCommandBuffer cmdBuf);
  void createUniformBuffer();
  void createGraphicPipeline();
  void createTextureImages(VkCommandBuffer cmdBuf, tinygltf::Model& gltfModel);
  void createOffscreenRender();
  void createPostPipeline();
  void updatePostDescriptorSet(const VkDescriptorImageInfo& descriptor);
  auto primitiveToGeometry(const nvh::GltfPrimMesh& prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress);
  void initRayTracing();
  void createBottomLevelAS();
  void createTopLevelAS();
  void createRtPipeline();
  void updateRtDescriptorSet();
  bool updateFrame();
  void resetFrame();
  void screenPicking();
  void onResize(int /*w*/, int /*h*/) override;
  void onMouseButton(int button, int action, int mods) override;
  void onKeyboard(int key, int scancode, int action, int mods) override;
  void onFileDrop(const char* filename) override;

  struct PipelineContainer  // All it needs to create a pipeline
  {
    VkPipeline            pipeline{VK_NULL_HANDLE};
    VkPipelineLayout      pipelineLayout{VK_NULL_HANDLE};
    VkDescriptorSet       dstSet{VK_NULL_HANDLE};
    VkDescriptorSetLayout dstLayout{VK_NULL_HANDLE};
    VkDescriptorPool      dstPool{VK_NULL_HANDLE};
  };

  enum Pipelines
  {
    eGraphic,   // Rasterize: vert/frag
    ePost,      // Post: passthrough / post
    eRaytrace,  // Pathtrace
    eNb
  };
  std::array<PipelineContainer, Pipelines::eNb> m_pContainer;  // All pipelines

  std::array<VkDescriptorPool, FRAME_IN_FLIGHT> m_postDstPool;
  std::array<VkDescriptorSet, FRAME_IN_FLIGHT>  m_postDstSet;
  uint32_t                                      m_postFrame{0};


  VkCommandBuffer m_recordedCmdBuffer{VK_NULL_HANDLE};  // Used by raster to replay rendering commands

  nvvk::ResourceAllocatorDma m_alloc;   // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil            m_debug;   // Utility to name objects
  nvvk::SBTWrapper           m_sbt;     // Shading binding table wrapper
  nvvk::RayPickerKHR         m_picker;  // Send ray at mouse coordinates

  nvh::GltfScene               m_gltfScene;  // Loaded scene
  std::array<Light, NB_LIGHTS> m_lights = {{
      {{0.0f, 4.42f, 0.0f},  // 0 - position
       5.f,                  // 0 - intensity
       {1.0f, 1.0f, 1.0f},   // 0 - color
       0},                   // 0 - type
      {{3.9f, -2.5f, 4.6f},  // 1 - position
       15.f,                 // 1 - intensity
       {1.0f, 1.0f, 1.0f},   // 1 - color
       0}                    // 1 - type
  }};
  VkClearColorValue            m_clearColor{0.5f, 0.5f, 0.5f, 1.f};

  // Resources
  std::vector<nvvk::Buffer>  m_vertices;        // One buffer per primitive (Vertex)
  std::vector<nvvk::Buffer>  m_indices;         // One buffer per primitive (uint32_t)
  nvvk::Buffer               m_materialBuffer;  // Array of ShadingMaterial
  nvvk::Buffer               m_primInfo;        // Array of PrimMeshInfo
  nvvk::Buffer               m_instInfoBuffer;  // Array of InstanceInfo
  nvvk::Buffer               m_sceneDesc;       // SceneDescription
  nvvk::Buffer               m_frameInfo;       // Device-Host of FrameInfo
  std::vector<nvvk::Texture> m_textures;        // Vector of all textures of the scene

  // Information pushed at each draw call
  RasterPushConstant m_pcRaster{};
  RtxPushConstant    m_pcRay{
      -1,    // frame
      10.f,  // magic-scene number
      7,     // max ray recursion
      1,     // max samples per pixel
  };

  // Post/offscreen pipeline
  VkRenderPass  m_offscreenRenderPass{VK_NULL_HANDLE};
  VkFramebuffer m_offscreenFramebuffer{VK_NULL_HANDLE};
  nvvk::Texture m_offscreenColor;
  nvvk::Texture m_offscreenDepth;
  Tonemapper    m_tonemapper{
      1,  // exposure
      1,  // brightness
      1,  // contrast
      1,  // saturation
      0,  // vignette
  };

  nvvk::Texture m_gAlbedo;
  nvvk::Texture m_gNormal;


  // Ray tracing pipeline
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::RaytracingBuilderKHR                      m_rtBuilder;
  HdrEnv                                          m_hdrEnv;


  //////////////////////////////////////////////////////////////////////////
  DenoiserOptix m_denoiser;
  nvvk::Texture m_gDenoised;

  // Timeline semaphores
  uint64_t m_fenceValue{0};
  bool     m_denoiseApply{true};
  bool     m_denoiseFirstFrame{false};
  int      m_denoiseEveryNFrames{100};
  int      m_maxFrames{2000};
};
