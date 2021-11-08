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
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/memallocator_dma_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"

// #VKRay
#include "nvh/gltfscene.hpp"
#include "nvvk/raytraceKHR_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"


#include "shaders/host_device.h"
#include "nvvk/raypicker_vk.hpp"


//--------------------------------------------------------------------------------------------------
// Simple rasterizer of glTF objects
// - Each OBJ loaded are stored in an `ObjModel` and referenced by a `ObjInstance`
// - It is possible to have many `ObjInstance` referencing the same `ObjModel`
// - Rendering is done in an offscreen framebuffer
// - The image of the framebuffer is displayed in post-process in a full-screen quad
//
class VulkanSample : public nvvk::AppBaseVk
{
public:
  void create(const nvvk::AppBaseVkCreateInfo& info) override;
  void createScene(const std::string& filename);
  void titleBar();
  void renderUI();
  void updateUniformBuffer(const VkCommandBuffer& cmdBuf, const VkExtent2D& size);
  void drawPost(VkCommandBuffer cmdBuf);
  void raytrace(const VkCommandBuffer& cmdBuf);
  void rasterize(const VkCommandBuffer& cmdBuff);
  void destroyResources();
  void fourViews(VkCommandBuffer cmdBuf);

  const VkClearColorValue& clearColor() { return m_clearColor; }
  const VkRenderPass&      offscreenRenderPass() { return m_offscreenRenderPass; }
  const VkFramebuffer&     offscreenFramebuffer() { return m_offscreenFramebuffer; }

  enum RenderMode
  {
    eRaster,
    eRayTracer
  } m_renderMode{eRayTracer};

private:
  void loadScene(const std::string& filename);
  void createVertexBuffer(VkCommandBuffer cmdBuf);
  void createMaterialBuffer(VkCommandBuffer cmdBuf);
  void createInstanceInfoBuffer(VkCommandBuffer cmdBuf);
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void updateDescriptorSet();
  void createUniformBuffer();
  void createTextureImages(const VkCommandBuffer& cmdBuf, tinygltf::Model& gltfModel);
  void onResize(int /*w*/, int /*h*/) override;
  void createOffscreenRender();
  void createPostPipeline();
  void createPostDescriptor();
  void updatePostDescriptorSet();
  auto primitiveToGeometry(const nvh::GltfPrimMesh& prim, VkDeviceAddress vertexAddress, VkDeviceAddress indexAddress);
  void initRayTracing();
  void createBottomLevelAS();
  void createTopLevelAS();
  void createRtDescriptorSet();
  void updateRtDescriptorSet();
  void createRtPipeline();
  void updateFrame();
  void resetFrame();
  void screenPicking();
  void onMouseButton(int button, int action, int mods) override;
  void onKeyboard(int key, int scancode, int action, int mods) override;


  VkCommandBuffer m_recordedCmdBuffer{VK_NULL_HANDLE};  // Used by raster to replay rendering commands

  nvvk::ResourceAllocatorDma m_alloc;       // Allocator for buffer, images, acceleration structures
  nvvk::DebugUtil            m_debug;       // Utility to name objects
  nvvk::SBTWrapper           m_sbtWrapper;  // Shading binding table wrapper
  nvvk::RayPickerKHR         m_picker;

  nvh::GltfScene    m_gltfScene;  // Loaded scene
  Light             m_light[1];   // Global Light
  VkClearColorValue m_clearColor{0.4f, 0.4f, 0.4f, 1.f};

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
  RasterPushConstant m_pushConstant{};
  RtxPushConstant    m_rtxPushConstant{};

  // Graphic pipeline
  VkPipelineLayout            m_pipelineLayout{VK_NULL_HANDLE};
  VkPipeline                  m_graphicsPipeline{VK_NULL_HANDLE};
  nvvk::DescriptorSetBindings m_descSetLayoutBind;
  VkDescriptorPool            m_descPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout       m_descSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet             m_descSet{VK_NULL_HANDLE};

  // Post/offscreen pipeline
  nvvk::DescriptorSetBindings m_postDescSetLayoutBind;
  VkDescriptorPool            m_postDescPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout       m_postDescSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet             m_postDescSet{VK_NULL_HANDLE};
  VkPipeline                  m_postPipeline{VK_NULL_HANDLE};
  VkPipelineLayout            m_postPipelineLayout{VK_NULL_HANDLE};
  VkRenderPass                m_offscreenRenderPass{VK_NULL_HANDLE};
  VkFramebuffer               m_offscreenFramebuffer{VK_NULL_HANDLE};
  nvvk::Texture               m_offscreenColor;
  nvvk::Texture               m_offscreenDepth;
  VkFormat                    m_offscreenColorFormat{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkFormat                    m_offscreenDepthFormat{VK_FORMAT_X8_D24_UNORM_PACK32};

  // Ray tracing pipeline
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};
  nvvk::RaytracingBuilderKHR                      m_rtBuilder;
  nvvk::DescriptorSetBindings                     m_rtDescSetLayoutBind;
  VkDescriptorPool                                m_rtDescPool{VK_NULL_HANDLE};
  VkDescriptorSetLayout                           m_rtDescSetLayout{VK_NULL_HANDLE};
  VkDescriptorSet                                 m_rtDescSet{VK_NULL_HANDLE};
  std::vector<VkRayTracingShaderGroupCreateInfoKHR> m_rtShaderGroups;
  VkPipelineLayout                                  m_rtPipelineLayout{VK_NULL_HANDLE};
  VkPipeline                                        m_rtPipeline{VK_NULL_HANDLE};


  VkPhysicalDeviceInheritedViewportScissorFeaturesNV m_inheritedViewport{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INHERITED_VIEWPORT_SCISSOR_FEATURES_NV};
  VkViewport    m_viewportDepth{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};  // Only care about minDepth and maxDepth
  nvmath::vec2f m_viewCenter{0.3f, 0.3f};
};
