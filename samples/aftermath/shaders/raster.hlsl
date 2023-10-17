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

#include "device_host.h"
#include "functions.hlsli"
#include "constants.hlsli"

// Per-vertex attributes to be assembled from bound vertex buffers.
struct VSin
{
  [[vk::location(0)]] float3 position : POSITION;
  [[vk::location(1)]] float3 normal : NORMAl;
  [[vk::location(2)]] float2 uv : TEXCOORD0;
};

// Output of the vertex shader, and input to the fragment shader.
struct PSin
{
  float4 color : COLOR0;
  float2 uv : TEXCOORD0;
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

[[vk::constant_id(0)]] const int CRASH_TEST = 0;
[[vk::binding(0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(1)]] RWStructuredBuffer<float> values;
[[vk::binding(2)]] [[vk::combinedImageSampler]] Texture2DArray inTexture;
[[vk::binding(2)]] [[vk::combinedImageSampler]] SamplerState inSampler;


// Vertex  Shader
[shader("vertex")]
VSout vertexMain(VSin input)
{
  VSout output;
  output.sv_position = mul(frameInfo.mpv, float4(input.position, 1));
  output.stage.color.xyz = 0.5 + 0.5 * input.normal;
  output.stage.uv = input.uv;

  output.stage.color.a = 1.0;
  if (CRASH_TEST == 1)
  {
    while (output.stage.color.a > 0.0)
    {
      output.stage.color.a -= input.normal.x;
      output.stage.color.a = abs(output.stage.color.a);
    }
    output.stage.color.a *= 0.1 * output.stage.color.a;
  }

  return output;
}


// Fragment Shader
[shader("pixel")]
PSout fragmentMain(PSin stage, float4 fragCoord : SV_Position)
{
  float iTime = frameInfo.time[0];

  // Ripple color
  float3 color = float3(stage.uv, 0.5 + 0.5 * sin(iTime * 6.5));
  float r = length(float2(0.5, 0.5) - stage.uv);
  float z = 1.0 + 0.5 * sin((r + iTime * 0.05) / 0.005);
  color *= z;

  if (CRASH_TEST == 2)
  {
    // Entering an infinite loop
    color = min(color, float3(1.0, 1.0, 1.0));
    while (color.x <= 10.0)
    {
      color.x *= sin(color.x);
    }
  }
  
  if (CRASH_TEST == 3)
  {
    // This is not good, but might not crash
    values[frameInfo.badOffset] = 0.5 + 0.5 * abs(sin(fragCoord.y + iTime) + sin(fragCoord.x + iTime * 0.5));
    color *= values[frameInfo.badOffset];

    // Bad access
    //BadStruct_ tbuf = BadStruct_(frameInfo.bufferAddr);
    //BadStruct buff = tbuf.s[frameInfo.badOffset + 100000000]; // <------ Make it crash !!!
    // color += buff.bad_value;
  }

  if (CRASH_TEST == 4)
  {
    color += inTexture.Sample(inSampler, float3(stage.uv, frameInfo.badOffset + 100000000)).xyz;
    color *= IntegerToColor(frameInfo.badOffset);
  }

  
  PSout output;
  output.color = float4(color, 1.0);

  return output;
}
