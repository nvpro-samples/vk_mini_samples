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

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#ifdef __cplusplus
using mat4 = nvmath::mat4f;
using vec4 = nvmath::vec4f;
using vec3 = nvmath::vec3f;
using vec2 = nvmath::vec2f;
#elif defined(__hlsl) || defined(__slang)
#define mat4 float4x4
#define vec4 float4
#define vec3 float3
#define vec2 float2
#endif  // __cplusplus


struct PushConstant
{
  float metallic;
  float roughness;
  float intensity;
  int   maxDepth;
};


struct FrameInfo
{
  mat4 proj;
  mat4 view;
  mat4 projInv;
  mat4 viewInv;
  vec3 camPos;
};

struct PrimMeshInfo
{
  uint64_t vertexAddress;
};

struct InstanceInfo
{
  int materialID;
};

struct SceneDescription
{
  uint64_t materialAddress;
  uint64_t instInfoAddress;
};


#endif  // HOST_DEVICE_H
