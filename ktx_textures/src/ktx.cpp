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

//#define TINYGLTF_IMPLEMENTATION
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "fileformats/nv_ktx.h"

#include "ktx.hpp"

#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/buffers_vk.hpp"


// For printing VkFormat names
#include "vulkan/vulkan.hpp"
#include "stb_image.h"

//--------------------------------------------------------------------------------------------------
// Override: Adding profiler
//--------------------------------------------------------------------------------------------------
void KtxSample::create(const nvvk::AppBaseVkCreateInfo& info)
{
  VulkanSample::create(info);
  m_profiler.init(info.device, info.physicalDevice, info.queueIndices[0]);
  m_profiler.setLabelUsage(true);  // depends on VK_EXT_debug_utils

  m_clearColor = {0.5f, 0.5f, 0.5f, 1.0f};
}

//--------------------------------------------------------------------------------------------------
// Override: removing Image Loader
//
void KtxSample::loadScene(const std::string& filename)
{
  nvh::Stopwatch sw;
  using vkBU = VkBufferUsageFlagBits;
  tinygltf::Model    tmodel;
  tinygltf::TinyGLTF tcontext;
  std::string        warn, error;

  LOGI("- Loading file:\n\t %s\n", filename.c_str());
  // #KTX
  tcontext.RemoveImageLoader();
  if(!tcontext.LoadASCIIFromFile(&tmodel, &error, &warn, filename))
  {
    LOGW(warn.c_str());
    LOGE(error.c_str());
    assert(!"Error while loading scene");
  }

  m_gltfScene.importMaterials(tmodel);
  m_gltfScene.importDrawableNodes(tmodel, nvh::GltfAttributes::Normal | nvh::GltfAttributes::Texcoord_0 | nvh::GltfAttributes::Tangent);


  CameraManip.setClipPlanes(nvmath::vec2f(0.001f * m_gltfScene.m_dimensions.radius, 10.0f * m_gltfScene.m_dimensions.radius));

  // Create the buffers, copy vertices, indices and materials
  nvvk::CommandPool cmdPool(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = cmdPool.createCommandBuffer();

  createMaterialBuffer(cmdBuf);
  createInstanceInfoBuffer(cmdBuf);
  createVertexBuffer(cmdBuf);
  createTextureImages(cmdBuf, tmodel, filename);

  // Buffer references
  SceneDescription sceneDesc{};
  sceneDesc.materialAddress = nvvk::getBufferDeviceAddress(m_device, m_materialBuffer.buffer);
  sceneDesc.primInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_primInfo.buffer);
  sceneDesc.instInfoAddress = nvvk::getBufferDeviceAddress(m_device, m_instInfoBuffer.buffer);
  m_sceneDesc               = m_alloc.createBuffer(cmdBuf, sizeof(SceneDescription), &sceneDesc,
                                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
  NAME2_VK(m_sceneDesc.buffer, "Scene Description");

  cmdPool.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
  LOGI("  --> %7.2fms\n", sw.elapsed());
}

//--------------------------------------------------------------------------------------------------
// Override: using member m_hostUBO
//
void KtxSample::updateUniformBuffer(VkCommandBuffer cmdBuf)
{
  LABEL_SCOPE_VK(cmdBuf);
  CameraManip.updateAnim();

  // Prepare new UBO contents on host.
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);
  auto&       clip        = CameraManip.getClipPlanes();

  m_hostUBO.view       = CameraManip.getMatrix();
  m_hostUBO.proj       = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, clip.x, clip.y);
  m_hostUBO.viewInv    = nvmath::invert(m_hostUBO.view);
  m_hostUBO.projInv    = nvmath::invert(m_hostUBO.proj);
  m_hostUBO.light[0]   = m_lights[0];
  m_hostUBO.light[1]   = m_lights[1];
  m_hostUBO.clearColor = m_clearColor.float32;

  // Schedule the host-to-device upload. (hostUBO is copied into the cmd buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_frameInfo.buffer, 0, sizeof(FrameInfo), &m_hostUBO);
}


//--------------------------------------------------------------------------------------------------
// Override: calling loadCreateImage
//
void KtxSample::createTextureImages(VkCommandBuffer cmdBuf, tinygltf::Model& gltfModel, const std::string& filename)
{
  namespace fs = std::filesystem;
  auto basedir = fs::path(filename).parent_path();

  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.maxLod     = FLT_MAX;


  // Make dummy image(1,1), needed as we cannot have an empty array
  auto addDefaultImage = [this, cmdBuf](const std::array<uint8_t, 4> color) {
    VkImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(VkExtent2D{1, 1});
    nvvk::Image       image           = m_alloc.createImage(cmdBuf, 4, color.data(), imageCreateInfo);
    m_images.emplace_back(image, imageCreateInfo);
    m_debug.setObjectName(m_images.back().first.image, "dummy");
  };

  // Make dummy texture/image(1,1), needed as we cannot have an empty array
  auto addDefaultTexture = [&]() {
    if(m_images.empty())
      addDefaultImage({255, 255, 255, 255});

    std::pair<nvvk::Image, VkImageCreateInfo>& image  = m_images[0];
    VkImageViewCreateInfo                      ivInfo = nvvk::makeImageViewCreateInfo(image.first.image, image.second);
    m_textures.emplace_back(m_alloc.createTexture(image.first, ivInfo, samplerCreateInfo));
  };

  if(gltfModel.textures.empty())
  {
    addDefaultTexture();
    return;
  }

  // We assume images are Uniform
  m_hostUBO.isSrgb = false;

  // First - create the images
  m_images.reserve(gltfModel.images.size());
  for(auto& image : gltfModel.images)
  {
    // #KTX
    if(loadCreateImage(cmdBuf, basedir, image) == false)
    {
      addDefaultImage({255, 0, 255, 255});  // Image not present or incorrectly loaded (image.empty)
      continue;
    }
  }

  // Creating the textures using the above images
  m_textures.reserve(gltfModel.textures.size());
  for(size_t i = 0; i < gltfModel.textures.size(); i++)
  {
    auto& txt         = gltfModel.textures[i];
    int   sourceImage = txt.source;
    if(txt.extensions.find("KHR_texture_basisu") != txt.extensions.end())
    {
      const auto& ext = txt.extensions.find("KHR_texture_basisu")->second;
      if(ext.Has("source"))
      {
        sourceImage = ext.Get("source").Get<int>();
      }
    }
    if(sourceImage >= gltfModel.images.size() || sourceImage < 0)
    {
      addDefaultTexture();  // Incorrect source image
      continue;
    }

    std::pair<nvvk::Image, VkImageCreateInfo>& image  = m_images[sourceImage];
    VkImageViewCreateInfo                      ivInfo = nvvk::makeImageViewCreateInfo(image.first.image, image.second);
    m_textures.emplace_back(m_alloc.createTexture(image.first, ivInfo, samplerCreateInfo));
  }
}


//--------------------------------------------------------------------------------------------------
// Override: profiler
//
void KtxSample::destroy()
{
  freeResources();
  m_alloc.deinit();
  m_profiler.deinit();
  AppBaseVk::destroy();
}


//--------------------------------------------------------------------------------------------------
// Override: profiler
//
void KtxSample::raytrace(VkCommandBuffer cmdBuf)
{
  auto sec = m_profiler.timeRecurring("raytrace", cmdBuf);

  VulkanSample::raytrace(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Rendering UI
//
void KtxSample::renderUI()
{
  if(showGui() == false)
    return;

  bool changed{false};

  ImGuiH::Panel::Begin();
  float widgetWidth = std::min(std::max(ImGui::GetWindowWidth() - 150.0f, 100.0f), 300.0f);
  ImGui::PushItemWidth(widgetWidth);

  if(ImGui::CollapsingHeader("Render Mode", ImGuiTreeNodeFlags_DefaultOpen))
  {
    changed |= ImGui::RadioButton("Raster", (int*)&m_renderMode, (int)RenderMode::eRaster);
    ImGui::SameLine();
    changed |= ImGui::RadioButton("Ray Tracing", (int*)&m_renderMode, (int)RenderMode::eRayTracer);
    if(m_renderMode == RenderMode::eRayTracer && ImGui::TreeNode("Ray Tracing"))
    {
      changed |= ImGui::SliderFloat("Max Luminance", &m_pcRay.maxLuminance, 0.01f, 20.f);
      changed |= ImGui::SliderInt("Depth", (int*)&m_pcRay.maxDepth, 1, 15);
      changed |= ImGui::SliderInt("Samples", (int*)&m_pcRay.maxSamples, 1, 20);
      ImGui::TreePop();
    }
  }

  if(ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen))
  {
    ImGuiH::CameraWidget();
  }

  if(ImGui::CollapsingHeader("Environment", ImGuiTreeNodeFlags_DefaultOpen))
  {
    changed |= ImGui::ColorEdit3("Color", &m_clearColor.float32[0], ImGuiColorEditFlags_Float);
    if(ImGui::TreeNode("Tonemapper"))
    {
      ImGui::SliderFloat("Exposure", &m_tonemapper.exposure, 0.001f, 5.0f);
      ImGui::SliderFloat("Brightness", &m_tonemapper.brightness, 0.0f, 2.0f);
      ImGui::SliderFloat("Contrast", &m_tonemapper.contrast, 0.0f, 2.0f);
      ImGui::SliderFloat("Saturation", &m_tonemapper.saturation, 0.0f, 2.0f);
      ImGui::SliderFloat("Vignette", &m_tonemapper.vignette, 0.0f, 1.0f);
      ImGui::TreePop();
    }

    uint32_t i = 0;
    for(auto& light : m_lights)
    {
      if(ImGui::TreeNode((void*)(intptr_t)i, "Light %d", i))
      {
        changed |= ImGui::RadioButton("Point", &light.type, 0);
        ImGui::SameLine();
        changed |= ImGui::RadioButton("Infinite", &light.type, 1);
        changed |= ImGui::DragFloat3("Position", &light.position.x, 0.1f);
        changed |= ImGui::DragFloat("Intensity", &light.intensity, 1.f, 0.0f, 1000.f, nullptr,
                                    ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
        changed |= ImGui::ColorEdit3("Color", reinterpret_cast<float*>(&light.color));
        ImGui::TreePop();
      }
      i++;
    }
  }
  // #KTX
  ImGui::Separator();
  changed |= ImGui::Checkbox("Is sRGB", (bool*)&m_hostUBO.isSrgb);
  ImGuiH::tooltip("Images are in sRGB domain and converted to linear by the hardware.", true);

  ImGui::PopItemWidth();
  ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

  nvh::Profiler::TimerInfo info;
  m_profiler.getTimerInfo("raytrace", info);
  ImGui::Text("GPU %.3f ", float(info.gpu.average / 1000.0f));


  ImGui::TextDisabled("(M) Toggle Render Mode");
  ImGui::TextDisabled("(F) Frame All");
  ImGui::TextDisabled("(R) Restart rendering");
  ImGui::TextDisabled("(SPACE) Set Eye position");
  ImGui::TextDisabled("(F10) Toggle Pane");
  ImGui::TextDisabled("(ESC) Quit");
  ImGuiH::Panel::End();

  if(changed)
    resetFrame();
}


// #KTX
//--------------------------------------------------------------------------------------------------
// Loading and creating images
// - KTX2 : load and create mipmaps, always used the unorm as the conversion is done in the shader
// - JPG, PNG: Using stbi_load
//
bool KtxSample::loadCreateImage(const VkCommandBuffer& cmdBuf, const std::filesystem::path& basedir, tinygltf::Image& gltfImage)
{
  namespace fs = std::filesystem;

  fs::path    uri       = fs::path(gltfImage.uri);
  std::string extension = uri.extension().string();
  std::string imgName   = uri.filename().string();
  std::string img_uri   = fs::path(basedir / uri).string();


  if(extension == ".ktx2" || extension == ".ktx")
  {
    nv_ktx::KTXImage      ktximage;
    nv_ktx::ReadSettings  ktxReadSettings;
    nv_ktx::ErrorWithText maybe_error = ktximage.readFromFile(img_uri.c_str(), ktxReadSettings);
    if(maybe_error.has_value())
    {
      LOGE("KTX Error: %s\n", maybe_error->c_str());
      return false;
    }

    // Converting SRGB to UNORM: the gamma correction will be done in the shader.
    VkFormat format = ktximage.format;
    if(ktximage.is_srgb)
    {
      m_hostUBO.isSrgb = true;  // If one image is found to be sRGB, we assume that all images are properly defined

      // If this was not the fact, we would need to convert all images to UNORM and do the conversion in the shader
      // format = static_cast<VkFormat>(static_cast<int>(format) - 1);  // SRGB always follow UNORM
    }

    // Unsupported formats on NVIDIA hardware, this would need to be uncompressed and compressed to another format.
    if((format >= VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK_EXT && format <= VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK_EXT)
       || (format >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && format <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK))
    {
      vk::Format f = static_cast<vk::Format>(format);
      LOGE("Format unsupported: %s\n", vk::to_string(f).c_str());
      return false;
    }

    auto              imgSize         = VkExtent2D{ktximage.mip_0_width, ktximage.mip_0_height};
    VkImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);

    // Creating image level 0
    std::vector<char>& data        = ktximage.subresource();
    VkDeviceSize       bufferSize  = data.size();
    nvvk::Image        resultImage = m_alloc.createImage(cmdBuf, bufferSize, data.data(), imageCreateInfo);

    // Create all mip-levels
    nvvk::cmdBarrierImageLayout(cmdBuf, resultImage.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    auto staging = m_alloc.getStaging();
    for(uint32_t mip = 1; mip < ktximage.num_mips; mip++)
    {
      imageCreateInfo.extent.width  = std::max(1u, ktximage.mip_0_width >> mip);
      imageCreateInfo.extent.height = std::max(1u, ktximage.mip_0_height >> mip);


      VkOffset3D               offset{};
      VkImageSubresourceLayers subresource{};
      subresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      subresource.layerCount = 1;
      subresource.mipLevel   = mip;

      std::vector<char>& mipresource = ktximage.subresource(mip, 0, 0);
      VkDeviceSize       bufferSize  = mipresource.size();
      if(imageCreateInfo.extent.width > 0 && imageCreateInfo.extent.height > 0)
      {
        staging->cmdToImage(cmdBuf, resultImage.image, offset, imageCreateInfo.extent, subresource, bufferSize,
                            mipresource.data());
      }
    }
    nvvk::cmdBarrierImageLayout(cmdBuf, resultImage.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    m_images.emplace_back(resultImage, imageCreateInfo);
    m_debug.setObjectName(resultImage.image, imgName);
  }
  else if(extension == ".jpg" || extension == ".jpeg" || extension == ".png")
  {
    int      w = 0, h = 0, comp = 0, req_comp = 4;
    stbi_uc* data = stbi_load(img_uri.c_str(), &w, &h, &comp, req_comp);
    if(!data || w < 1 || h < 1)
      return false;

    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(w) * h * req_comp;

    VkFormat format  = VK_FORMAT_R8G8B8A8_UNORM;
    auto     imgSize = VkExtent2D{(uint32_t)w, (uint32_t)h};

    VkImageCreateInfo imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);
    nvvk::Image       resultImage     = m_alloc.createImage(cmdBuf, bufferSize, data, imageCreateInfo);
    nvvk::cmdGenerateMipmaps(cmdBuf, resultImage.image, format, imgSize, imageCreateInfo.mipLevels);
    m_images.emplace_back(resultImage, imageCreateInfo);
    m_debug.setObjectName(resultImage.image, imgName);

    stbi_image_free(data);
  }
  else
  {
    LOGE("Extensions not supported: %s /n", extension.c_str());
    return false;
  }

  return true;
}
