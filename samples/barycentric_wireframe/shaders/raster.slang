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

#version 460

#include "common/shaders/constants.hlsli"
#include "common/shaders/functions.hlsli"

#include "device_host.h"

// Per-vertex attributes to be assembled from bound vertex buffers.
struct VSin
{
  [[vk::location(0)]] float3 position : POSITION;
  [[vk::location(1)]] float3 normal : NORMAl;
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
  PSin stage;
  float4 sv_position : SV_Position;
};

// Output of the fragment shader
struct PSout
{
  float4 color : SV_Target;
};

[[vk::push_constant]] ConstantBuffer<PushConstant> pushConst;
[[vk::binding(0, 0)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(1, 0)]] ConstantBuffer<WireframeSettings> settings;

// Return the width [0..1] for which the line should be displayed or not
float getLineWidth(in float3 deltas, in float thickness, in float smoothing, in float3 barys)
{
  barys = smoothstep(deltas * (thickness), deltas * (thickness + smoothing), barys);
  float minBary = min(barys.x, min(barys.y, barys.z));
  return 1.0 - minBary;
}

// Position along the edge [0..1]
float edgePosition(float3 baryWeights)
{
  return max(baryWeights.z, max(baryWeights.y, baryWeights.x));
}

// Return 0 or 1 if edgePos should be diplayed or not
float stipple(in float stippleRepeats, in float stippleLength, in float edgePos)
{
  float offset = 1.0 / stippleRepeats;
  offset *= 0.5 * stippleLength;
  float pattern = frac((edgePos + offset) * stippleRepeats);
  return 1.0 - step(stippleLength, pattern);
}

// Vary the thickness along the edge
float edgeThickness(in float2 thicknessVar, in float edgePos)
{
  return lerp(thicknessVar.x, thicknessVar.y, (1.0 - sin(edgePos * M_PI)));
}


// Vertex  Shader
[shader("vertex")]
VSout vertexMain(VSin input)
{
  float4 pos = mul(pushConst.transfo, float4(input.position.xyz, 1.0));

  VSout output;
  output.sv_position = mul(frameInfo.proj, mul(frameInfo.view, float4(pos)));
  output.stage.normal = input.normal;
  output.stage.position = pos.xyz;

  return output;
}


// Fragment Shader
[shader("pixel")]
PSout fragmentMain(PSin stage, bool isFrontFacing : SV_IsFrontFace, float3 baryWeights : SV_Barycentrics)
{
  float3 V = normalize(frameInfo.camPos - stage.position);
  float3 color = simpleShading(V, V, stage.normal, pushConst.color.xyz);

  // For a one liner simple wireframe, this can be done for grey wireframe on top of the geometry
  // color = mix(color, float3(0.8), getLineWidth(fwidthFine(gl_BaryCoordEXT), 0.5, 0.5, gl_BaryCoordEXT));

  // Wireframe Settings
  float thickness = settings.thickness * 0.5; // Thickness for both side of the edge, must be divided by 2
  float3 wireColor = settings.color; // Color of the wireframe
  float smoothing = settings.thickness * settings.smoothing; // Could be thickness
  bool enableStipple = (settings.enableStipple == 1);

  // Uniform position on the edge [0, 1]
  float edgePos = edgePosition(baryWeights);

  if (!isFrontFacing)
  {
    enableStipple = true; // Forcing backface to always stipple the line
    wireColor = settings.backFaceColor;
  }

  // [optional] Vary the thickness along the edge
  thickness *= edgeThickness(settings.thicknessVar, edgePos);

  // fwidth � return the sum of the absolute value of derivatives in x and y
  //          which makes the width in screen space
  float3 deltas = (settings.screenSpace == 1) ? fwidth(baryWeights) : float3(1, 1, 1);

  // Get the wireframe line width
  float lineWidth = getLineWidth(deltas, thickness, smoothing, baryWeights);

  // [optional]
  if (enableStipple)
  {
    float stippleFact = stipple(settings.stippleRepeats, settings.stippleLength, edgePos);
    lineWidth *= stippleFact; // 0 or 1
  }

  // To see through, we discard faces and blend with the background
  if (settings.onlyWire == 1)
  {
    color = pushConst.clearColor.xyz;
    if (lineWidth < 0.1)
      discard;
  }

  // Final color
  color = lerp(color, wireColor, lineWidth);
  
  
  
  PSout output;
  output.color = float4(color, pushConst.color.w);
  
  return output;
}
