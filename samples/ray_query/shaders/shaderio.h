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

#ifndef HOST_DEVICE_H
#define HOST_DEVICE_H

#define WORKGROUP_SIZE 16

#include "nvshaders/slang_types.h"

NAMESPACE_SHADERIO_BEGIN()

struct Light
{
  float3 position;
  float  intensity;
  float  radius;  // on XZ plane
  float  pad0;    // alignment of 8
};


struct PushConstant
{
  int   maxDepth;
  int   frame;
  float fireflyClampThreshold;
  int   maxSamples;
  Light light;
};


struct CameraInfo
{
  float4x4 projInv;
  float4x4 viewInv;
};


struct Material
{
  float3 albedo;
  float  roughness;
  float  metallic;
  float  transmission;
  float  _pad1;
  float  _pad2;
};

// From primitive
struct Vertex
{
  float3 position;
  float3 normal;
  float2 t;
};

struct PrimMeshInfo
{
  uint64_t vertexAddress;
  uint64_t indexAddress;
};

struct InstanceInfo
{
  float4x4 transform;
  int      materialID;
  int      _pad0;
  int      _pad1;
  int      _pad2;
};

struct SceneInfo
{
  uint64_t materialAddress;
  uint64_t instInfoAddress;
  uint64_t primInfoAddress;
  Light    light;
};

NAMESPACE_SHADERIO_END()

#endif  // HOST_DEVICE_H
