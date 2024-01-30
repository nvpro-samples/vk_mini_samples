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

#include "common/shaders/glsl_type.hlsli"
#include "device_host.h"


struct VSIn
{
  [[vk::location(0)]] float2 pos : POSITION;
  [[vk::location(1)]] float2 uv : TEXCOORD;
};

struct VSOut
{
  float2 uv : TEXCOORD;
  float4 sv_position : SV_Position;
};

// Output of the fragment shader
struct PSOut
{
  float4 color : SV_TARGET;
};

struct PushConstant
{
  float4x4 transfo;
  float2 scale;
};

// Descriptor Set
[[vk::push_constant]] ConstantBuffer<PushConstant> pushConst;
[[vk::binding(0)]] [[vk::combinedImageSampler]] Texture2D g_Texture;
[[vk::binding(0)]] [[vk::combinedImageSampler]] SamplerState g_Sampler;


// Vertex  Shader
[shader("vertex")]
VSOut vertexMain(VSIn input)
{
  VSOut output;

  float4 pos = mul(pushConst.transfo, float4(pushConst.scale * input.pos, 0.0, 1.0));
  output.sv_position = pos;
  output.uv = input.uv;
   
  return output;
}


// Fragment Shader
[shader("pixel")]
PSOut fragmentMain(VSOut input)
{
  PSOut output;
  float2 uv = input.uv; 
 
  float3 color = g_Texture.Sample(g_Sampler, uv).xyz;
  
  output.color = float4(color, 1);
  
  return output;
}
