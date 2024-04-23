/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common/shaders/glsl_type.hlsli"
#include "device_host.h"
#define WORKGROUP_SIZE 16

[[vk::binding(0)]] RWTexture2D<float4> outImage;
[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;


//https://iquilezles.org/articles/palettes/
float3 palette(float t)
{
  float3 a = float3(0.5, 0.5, 0.5);
  float3 b = float3(0.5, 0.5, 0.5);
  float3 c = float3(1.0, 1.0, 0.5);
  float3 d = float3(0.0, 0.15, 0.2);

  return a + b * cos(6.28318f * (c * t + d));
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  float2 fragCoord   = threadIdx.xy;
  uint2 iResolution;
  outImage.GetDimensions(iResolution.x, iResolution.y); //DispatchRaysDimensions();
  
  if(fragCoord.x >= iResolution.x || fragCoord.y >= iResolution.y)
    return;
  float iTime = pushConst.time;

  float2 uv         = (fragCoord * 2.0F - iResolution) / iResolution.y;
  float2 uv0        = uv;
  float3 finalColor = float3(0.0,0.0,0.0);

  for(float i = 0.0; i < pushConst.iter; i++)
  {
    uv = frac(uv * pushConst.zoom) - 0.5f;

    float d = length(uv) * exp(-length(uv0));

    float3 col = palette(length(uv0) + i  + iTime * .6);

    d = sin(d * 8 + iTime) / 2;
    d = abs(d);
    d = pow(0.01 / d, 1.2);

    finalColor += col * d;
  }

  outImage[int2(fragCoord)] = float4(finalColor, 1.0F);
}
