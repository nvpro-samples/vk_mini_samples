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


// Incoming to vertex shader
struct vertexInfo
{
  [[vk::location(0)]] float2 position : POSITION;
  [[vk::location(1)]] float3 color : COLOR;
};

// Vertex shader to pixel shader
struct v2p
{
  float3 color : COLOR;
  float4 sv_position : SV_Position; // Specify the position of a vertex in screen space (after projection).
};

// Vertex shader
[shader("vertex")]
v2p vertexMain(vertexInfo input)
{
  v2p output;
  output.color = input.color; // Pass through, will be interpolated
  output.sv_position = float4(input.position, 0.0, 1.0);

  return output;
}

// Pixel shader
[shader("pixel")]
float4 fragmentMain(v2p input) : SV_Target
{
  return float4(input.color, 1.0);
}
