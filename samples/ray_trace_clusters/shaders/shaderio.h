/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RAYTRACE_CLUSTERS_SHADERIO_H
#define RAYTRACE_CLUSTERS_SHADERIO_H

#include "nvshaders/slang_types.h"

// Descriptor set bindings (classic push-descriptor set, shared by Slang and GLSL)
#define B_tlas 0
#define B_outImage 1
#define B_frameInfo 2

NAMESPACE_SHADERIO_BEGIN()

// Per-frame camera transforms (uniform buffer)
struct FrameInfo
{
  float4x4 projInv;  // inverse projection: clip -> view
  float4x4 viewInv;  // inverse view:       view -> world
};

// Small constants pushed every frame
struct PushConstant
{
  float3 lightDir;        // world-space direction toward the light
  int    colorByCluster;  // 1: color each surface by its cluster ID, 0: plain shading
};

NAMESPACE_SHADERIO_END()

#endif  // RAYTRACE_CLUSTERS_SHADERIO_H
