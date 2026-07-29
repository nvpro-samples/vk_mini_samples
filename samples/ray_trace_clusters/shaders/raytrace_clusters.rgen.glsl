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

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "shaderio.h"

#define INFINITE 1e32

layout(location = 0) rayPayloadEXT vec3 payloadColor;

layout(set = 0, binding = B_tlas) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = B_outImage, rgba8) uniform image2D outImage;
layout(set = 0, binding = B_frameInfo) uniform FrameInfo_
{
  FrameInfo frameInfo;
};

layout(push_constant) uniform PushConstant_
{
  PushConstant pc;
};

void main()
{
  const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + 0.5;
  const vec2 inUV        = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
  const vec2 d           = inUV * 2.0 - 1.0;

  const vec4 origin    = frameInfo.viewInv * vec4(0.0, 0.0, 0.0, 1.0);
  const vec4 target    = frameInfo.projInv * vec4(d.x, d.y, 0.01, 1.0);
  const vec4 direction = frameInfo.viewInv * vec4(normalize(target.xyz), 0.0);

  payloadColor = vec3(0.0);
  traceRayEXT(topLevelAS, gl_RayFlagsNoneEXT, 0xFF, 0, 0, 0, origin.xyz, 0.001, direction.xyz, INFINITE, 0);

  imageStore(outImage, ivec2(gl_LaunchIDEXT.xy), vec4(payloadColor, 1.0));
}
