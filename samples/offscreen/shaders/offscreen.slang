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

#include "shaderio.h"

struct PSin
{
  float2 uv : TEXCOORD;
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


// Vertex  Shader
[shader("vertex")]
VSout vertexMain(uint vid: SV_VertexID)  // gl_VertexIndex
{
  VSout  output;
  float2 uv          = float2((vid << 1) & 2, vid & 2);
  output.stage.uv    = uv;
  output.sv_position = float4(uv * 2.0f - 1.0f, 0.0f, 1.0f);

  return output;
}


// Fragment Shader
[shader("pixel")]
PSout fragmentMain(PSin input)
{
  PSout  output;
  float2 uv = input.uv;

  uv.x *= pushConst.aspectRatio;

  // Time varying pixel color
  float3 col = 0.5 + 0.5 * cos(pushConst.iTime + uv.xyx + float3(0, 2, 4));

  for(int i = 0; i < 2; ++i)
  {
    uv -= 0.5;
    uv *= 5.;

    float  radius = .5 + (sin(pushConst.iTime * .3) * .15) - cos(pushConst.iTime) * float(i) * .1;
    float2 fuv    = frac(uv) - .5;
    float  x      = smoothstep(radius * .9, radius * 1.1, length(fuv));
    float  y      = 1.0 - smoothstep(radius * .3, radius * .7, length(fuv));

    float val = x + y;
    col -= float3(val, val, val);
  }

  output.color = float4(col, 1);

  return output;
}
