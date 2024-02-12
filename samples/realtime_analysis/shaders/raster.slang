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


#include "common/shaders/glsl_type.slang"
#include "common/shaders/functions.hlsli"
#include "device_host.h"

void updateBuiltins(float4 fragCoord)
{
  gl_FragCoord = fragCoord;
}

#define INSPECTOR_MODE_FRAGMENT
#define INSPECTOR_DESCRIPTOR_SET 0
#define INSPECTOR_INSPECTION_DATA_BINDING eFragInspectorData
#define INSPECTOR_METADATA_BINDING eFragInspectorMeta
#include "common/shaders/dh_inspector.hlsli"

[[vk::push_constant]] ConstantBuffer<PushConstant> pushConst;
[[vk::binding(eFrameInfo)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(eParticles)]] RWStructuredBuffer<Particle> particles;

#define M_PI_OVER_2 1.570796

// Per-vertex attributes to be assembled from bound vertex buffers.
struct VSin
{
  [[vk::location(0)]] float3 position : POSITION;
  [[vk::location(1)]] float2 uv : TEXCOORD;
};

// Output of the vertex shader, and input to the fragment shader.
struct PSin
{
  nointerpolation uint id : DRAW_ID;
  float2 uv : TEXCOORD;
};

// Output of the vertex shader
struct VSout
{
  PSin stage;
  float4 sv_position : SV_Position;
};

// Output of the fragment shader
struct PSout
{
  float4 color : SV_Target;
};


// Vertex  Shader
[shader("vertex")]
VSout vertexMain(VSin input)
{
  uint particleID = InstanceIndex();
  float2 particle = input.position.xy * frameInfo.radius + (particles[particleID].position * frameInfo.scale * 2);
  
  VSout output;
  output.sv_position = mul(frameInfo.proj, float4(particle, 0.0, 1.0));
  output.stage.uv = input.uv;
  output.stage.id = particleID;

  return output;
}


// Fragment Shader
[shader("pixel")]
PSout fragmentMain(PSin stage, float4 fragCoord : SV_Position)
{
  updateBuiltins(fragCoord);
  
  uint particleID = stage.id;
  
  float2 centerOffset = (stage.uv - 0.5) * 2.0;
  float sqrDstFromDistance = dot(centerOffset, centerOffset);
  float circleAlpha = cos(sqrDstFromDistance * sqrDstFromDistance * M_PI_OVER_2);
  float delta = fwidth(sqrt(sqrDstFromDistance));
  circleAlpha = min(circleAlpha, 1 - smoothstep(1 - delta, 1 + delta, sqrDstFromDistance));

  float maxVelocity = 0.5 / frameInfo.scale;
  float velocity = length(particles[particleID].velocity);
  float3 col = clamp(temperature(velocity / maxVelocity), 0, 1);

  // #INSPECTOR
  inspect32BitValue(0, asuint(velocity));
  
  PSout output;
  output.color = float4(col.xyz, circleAlpha);
  
  return output;
}
