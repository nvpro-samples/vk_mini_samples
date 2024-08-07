/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

struct PushConstant
{
    float3 color;
    float rotate;
    float2 translate;
    int2 _pad;
};

[[vk::push_constant]] ConstantBuffer<PushConstant> pushConstants;

struct VSInput
{
    float3 position : POSITION;
};

struct PSInput
{
    float4 position : SV_POSITION;
    float4 color : COLOR;
};


// Vertex shader
[shader("vertex")]
PSInput vertexMain(VSInput input)
{
    PSInput output;

    float2 pos = input.position.xy;
    pos.y *= -1.0; // Flip the y coordinate

    // Translate and rotate the vertex position
    float angle = pushConstants.rotate;
    float2x2 rot = float2x2(cos(angle), -sin(angle), sin(angle), cos(angle));
    pos = mul(rot, pos);
    pos += pushConstants.translate;

    output.position = float4(pos, input.position.z, 1.0);
    output.color = float4(pushConstants.color, 1.0);

    return output;
}

// Pixel shader
[shader("pixel")]
float4 fragmentMain(PSInput input) : SV_Target
{
  return input.color;
}
