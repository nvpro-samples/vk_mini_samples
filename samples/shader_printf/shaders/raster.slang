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

// Per-vertex attributes to be assembled from bound vertex buffers.
struct VSin
{
  float2 position : POSITION;
  float3 color : COLOR;
};

// Output of the vertex shader, and input to the fragment shader.
struct PSin
{
  float3 color : COLOR;
};

// Output of the vertex shader
struct VSout
{
  PSin stage;
  float4 sv_position : SV_Position;
};


[[vk::push_constant]] ConstantBuffer<PushConstant> pushConst;


// Vertex  Shader
[shader("vertex")]
VSout vertexMain(VSin input)
{
  VSout output;
  output.sv_position = float4(input.position.xy, 0.0, 1.0);
  output.stage.color = input.color;

  return output;
}

// Fragment Shader
[shader("pixel")]
float4 fragmentMain(PSin stage, float4 sv_fragCoord : SV_Position) : SV_Target
{
  // Check if the mouse is over the fragment, if yes, print values
  int2 fragCoord = int2(floor(sv_fragCoord.xy));
  if(all(fragCoord == int2(pushConst.mouseCoord)))
  {
    printf("\n[%d, %d] Color: %f, %f, %f\n", fragCoord.x, fragCoord.y, stage.color.x, stage.color.y, stage.color.z);
  }

  // Return interpolated color
  return float4(stage.color, 1);
}
