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

#version 450

#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_descriptor_heap : enable

layout(location = 0) in vec2 inFragUv;
layout(location = 1) flat in uint inSamplerIdx;
layout(location = 0) out vec4 outColor;

// Bindless: explicit heap indices (see vk_mini_samples image_viewer.cpp — heap slot 0 = texture, 0/1 = samplers).
layout(descriptor_heap) uniform texture2D heapTextures[];
layout(descriptor_heap) uniform sampler heapSamplers[];

void main()
{
  vec3 color =
      texture(sampler2D(heapTextures[nonuniformEXT(0)], heapSamplers[nonuniformEXT(inSamplerIdx)]), inFragUv).xyz;
  outColor = vec4(color, 1.0);
}
