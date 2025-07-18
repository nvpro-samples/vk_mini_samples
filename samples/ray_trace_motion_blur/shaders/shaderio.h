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

#include "nvshaders/slang_types.h"

struct PushConstant
{
  float4 clearColor;
  float3 lightPosition;
  float  lightIntensity;
  int    frame;
  int    numSamples;
};


struct FrameInfo
{
  float4x4 proj;
  float4x4 view;
  float4x4 projInv;
  float4x4 viewInv;
  float3   camPos;
};


struct Vertex
{
  float3 position;
  float3 normal;
  float2 texCoord;
};


struct InstanceInfo
{
  float4x4 transform;
};

#endif  // HOST_DEVICE_H
