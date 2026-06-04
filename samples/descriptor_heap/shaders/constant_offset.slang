/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "instanced_common.slang.h"

// The shader stays in set/binding form. Mapping in createShaders() maps these
// unsized arrays to the descriptor heaps with HEAP_WITH_CONSTANT_OFFSET.
[[vk::binding(0, 0)]]
Texture2D heapTextures[];
[[vk::binding(1, 0)]]
SamplerState heapSamplers[];

[shader("vertex")]
VSout vertexMain(VSin input, int instanceIndex : SV_InstanceID)
{
  return instancedVertex(input, instanceIndex);
}

[shader("pixel")]
PSout fragmentMain(VSout stage)
{
  uint   texIdx   = stage.baseFaceTexIdx + stage.faceIdx;
  float4 texColor = heapTextures[NonUniformResourceIndex(texIdx)].Sample(heapSamplers[0], stage.uv);

  // Replace border pixels with the single-draw border color.
  float borderWidth = 1.0 / 48.0;
  if(stage.uv.x < borderWidth || stage.uv.x > 1.0 - borderWidth || stage.uv.y < borderWidth || stage.uv.y > 1.0 - borderWidth)
  {
    texColor.rgb = unpackColor(push.instanced.borderColor);
  }

  float3 N     = normalize(stage.normal);
  float  NdotL = max(dot(N, normalize(push.frame.lightDir)), 0.0);

  PSout output;
  output.color = float4(texColor.rgb * (0.3 + 0.7 * NdotL), 1.0);
  return output;
}
