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

#ifdef __cplusplus
using mat4 = glm::mat4;
using vec4 = glm::vec4;
using vec3 = glm::vec3;
#elif defined(__hlsl) || defined(__slang)
#define mat4 float4x4
#define vec4 float4
#define vec3 float3
#else
#define static
#endif  // __cplusplus

static const int BKtxFrameInfo = 0;
static const int BKtxTex       = 1;

struct PushConstant
{
  mat4 transfo;
  vec4 color;
};

struct FrameInfo
{
  mat4 proj;
  mat4 view;
  vec3 camPos;
#if defined(__hlsl)
  //Sampler2D texture;
#endif
};
