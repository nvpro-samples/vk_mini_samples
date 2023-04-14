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

#ifndef PAYLOAD_H
#define PAYLOAD_H

#ifdef __cplusplus
using vec3 = nvmath::vec3f;
#endif  // __cplusplus

#define MISS_DEPTH 1000

struct HitPayload
{
  vec3 color;
  vec3 weight;
  int  depth;
};

HitPayload initPayload()
{
  HitPayload p;
  p.color  = vec3(0, 0, 0);
  p.depth  = 0;
  p.weight = vec3(1);
  return p;
}

#endif  // PAYLOAD_H
