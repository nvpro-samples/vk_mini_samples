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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#ifdef __cplusplus
using mat4 = nvmath::mat4f;
using vec2 = nvmath::vec2f;
#elif defined(__hlsl)
#define mat4 float4x4
#define vec2 float2
#endif  // __cplusplus


struct FrameInfo
{
  mat4     mpv;
  int      badOffset;
  int      _pad;
  vec2     resolution;
  float    time[2];
  int      _pad2;
  uint64_t bufferAddr;
};
