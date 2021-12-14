/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#define GRID_SIZE 16  // Grid size used by compute shaders

// clang-format off
#ifdef __cplusplus // GLSL Type
using vec2 = nvmath::vec2f;
using vec3 = nvmath::vec3f;
using vec4 = nvmath::vec4f;
using mat4 = nvmath::mat4f;
using uint = uint32_t;
#endif

#ifdef __cplusplus // Descriptor binding helper for C++ and GLSL
 #define START_BINDING(a) enum a {
 #define END_BINDING() }
#else
 #define START_BINDING(a)  const uint
 #define END_BINDING() 
#endif

#define NB_LIGHTS 2

START_BINDING(SceneBindings)
  eFrameInfo = 0,
  eSceneDesc = 1,
  eTextures  = 2
END_BINDING();

START_BINDING(RtxBindings)
  eTlas     = 0,
  eOutImage = 1
END_BINDING();

START_BINDING(PostBindings)
  ePostImage = 0
END_BINDING();
// clang-format on

struct GltfShadeMaterial
{
  vec4 pbrBaseColorFactor;
  vec3 emissiveFactor;
  int  pbrBaseColorTexture;

  int   normalTexture;
  float normalTextureScale;
  int   shadingModel;
  float pbrRoughnessFactor;

  float pbrMetallicFactor;
  int   pbrMetallicRoughnessTexture;
  int   khrSpecularGlossinessTexture;
  int   khrDiffuseTexture;

  vec4  khrDiffuseFactor;
  vec3  khrSpecularFactor;
  float khrGlossinessFactor;

  int   emissiveTexture;
  int   alphaMode;
  float alphaCutoff;
};

struct PrimMeshInfo
{
  uint64_t vertexAddress;
  uint64_t indexAddress;
  int      materialIndex;
};

struct SceneDescription
{
  uint64_t materialAddress;
  uint64_t instInfoAddress;
  uint64_t primInfoAddress;
};

struct Light
{
  vec3  position;
  float intensity;
  vec3  color;
  int   type;
};

// Tonemapper used in post.frag
struct Tonemapper
{
  float exposure;
  float brightness;
  float contrast;
  float saturation;
  float vignette;
};

struct FrameInfo
{
  mat4  view;
  mat4  proj;
  mat4  viewInv;
  mat4  projInv;
  vec4  clearColor;
  Light light[NB_LIGHTS];
  int   isSrgb;  // Image are sRGB already (no conversion)
};

struct InstanceInfo
{
  mat4 objMatrix;
  mat4 objMatrixIT;
};

struct RasterPushConstant
{
  int materialId;
  int instanceId;
};

struct RtxPushConstant
{
  int   frame;
  float maxLuminance;
  uint  maxDepth;
  uint  maxSamples;
};

struct Vertex
{
  vec4 position;  // w == texcood.u
  vec4 normal;    // w == texcood.v
  vec4 tangent;
};

#endif  // HOST_DEVICE_H
