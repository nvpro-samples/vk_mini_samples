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


struct VertexInput
{
    float2 pos : POSITION;
    float2 uv : TEXCOORD;
};

struct FragmentInput
{
    float2 uv : TEXCOORD;
    float4 sv_position : SV_Position;
};

struct PushConstant
{
    float4x4 transfo;
    float2   scale;
};

// Descriptor Set
// clang-format off
[[vk::push_constant]]   ConstantBuffer<PushConstant>    pushConst;
[[vk::binding(0)]]      Sampler2D                       mainTexture;
// clang-format on


// Vertex  Shader
[shader("vertex")]
FragmentInput vertexMain(VertexInput input)
{
    return {
        input.uv,                                                             // uv : TEXCOORD
        mul(float4(pushConst.scale * input.pos, 0.0, 1.0), pushConst.transfo) // sv_position : SV_Position
    };
}


// Fragment Shader
[shader("pixel")]
float4 fragmentMain(FragmentInput input)
    : SV_TARGET
{
    return mainTexture.Sample(input.uv);
}
