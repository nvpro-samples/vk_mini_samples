/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
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
  // Flat: same sampler index for whole primitive (single quad).
  nointerpolation uint samplerIdx : SAMPLERIDX;
  float4               sv_position : SV_Position;
};

struct PushConstant
{
  float4x4 transfo;
  float2   scale;
  uint     samplerIdx;
};

[[vk::push_constant]]
ConstantBuffer<PushConstant> pushConst;

// Bindless via SPV_EXT_descriptor_heap: [require(spvDescriptorHeapEXT)] makes Slang emit
// ResourceHeapEXT/SamplerHeapEXT builtins (accessed via SPV_KHR_untyped_pointers) instead of
// a typed `__slang_resource_heap` OpVariable with DescriptorSet/Binding decorations. The
// latter would otherwise trip VUID-VkShaderCreateInfoEXT-flags-11292 when creating the
// shader object with VK_SHADER_CREATE_DESCRIPTOR_HEAP_BIT_EXT.
[shader("vertex")]
FragmentInput vertexMain(VertexInput input)
{
  FragmentInput o;
  o.uv          = input.uv;
  o.samplerIdx  = pushConst.samplerIdx;
  o.sv_position = mul(float4(pushConst.scale * input.pos, 0.0, 1.0), pushConst.transfo);
  return o;
}


[shader("pixel")]
float4 fragmentMain(FragmentInput input)
    : SV_TARGET
{
  const uint texID = 0;  // Always use the first texture in the heap for now. The sampler index is passed via push constants.
  Sampler2D sampled = Sampler2D.Handle(uint2(texID, input.samplerIdx));
  return sampled.Sample(input.uv);
}
