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

#include "nvshaders/simple_shading.h.slang"
#include "shaderio_msaa.h"

// Per-vertex attributes to be assembled from bound vertex buffers.
struct VSin
{
  float3 position : POSITION;
  float3 normal : NORMAl;
};

// Output of the vertex shader, and input to the fragment shader.
struct PSin
{
  float3 position : POSITION;
  float3 normal : NORMAL;
};

// Output of the vertex shader
struct VSout
{
  PSin   stage;
  float4 sv_position : SV_Position;
};

// Output of the fragment shader
struct PSout
{
  float4 color : SV_Target;
};


[[vk::push_constant]]
ConstantBuffer<PushConstant> pushConst;
[[vk::binding(MsaaBinding::eFrameInfo)]]
ConstantBuffer<FrameInfo> frameInfo;

// Vertex  Shader
[shader("vertex")]
VSout vertexMain(VSin input)
{
  float4 pos = mul(float4(input.position.xyz, 1.0), pushConst.transfo);

  // Transform normal using the model matrix
  float3 worldNormal = mul(input.normal, (float3x3)pushConst.transfo);

  VSout output;
  output.sv_position    = mul(mul(pos, frameInfo.view), frameInfo.proj);
  output.stage.normal   = worldNormal;
  output.stage.position = pos.xyz;

  return output;
}


// Fragment Shader
[shader("pixel")]
PSout fragmentMain(PSin stage)
{
  // Ensure normal is normalized
  float3 normal = normalize(stage.normal);

  // View vector from camera to fragment
  float3 V = normalize(frameInfo.camPos - stage.position);

  // Calculate shading using the simple shading model
  float3 color = simpleShading(V, V, normal, pushConst.color.xyz, 16.0);

  PSout output;
  output.color = float4(color, pushConst.color.w);

  return output;
}
