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

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_scalar_block_layout : require

#include "device_host.h"
#include "payload.h"
#include "dh_bindings.h"
#include "nvvkhl/shaders/random.h"
#include "nvvkhl/shaders/constants.h"

// clang-format off
layout(location = 0) rayPayloadEXT HitPayload payload;

layout(set = 0, binding = B_tlas) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = B_outImage, rgba32f) uniform image2D image;
layout(set = 0, binding = B_frameInfo, scalar) uniform FrameInfo_ { FrameInfo frameInfo; };
// clang-format on

layout(push_constant) uniform RtxPushConstant_
{
  PushConstant pc;
};


void main()
{
  payload = initPayload();

  const vec2 inUV   = vec2(gl_LaunchIDEXT.xy) / vec2(gl_LaunchSizeEXT.xy);
  const vec2 d      = inUV * 2.0 - 1.0;
  const vec4 target = normalize(frameInfo.projInv * vec4(d.x, d.y, 1.0, 1.0));

  const vec3 rayOrigin    = vec3(frameInfo.viewInv * vec4(0.0, 0.0, 0.0, 1.0));
  const vec3 rayDirection = vec3(frameInfo.viewInv * vec4(target.xyz, 0.0));

  const uint  rayFlags = gl_RayFlagsCullBackFacingTrianglesEXT;
  const float tMin     = 0.001;
  const float tMax     = INFINITE;

  traceRayEXT(topLevelAS,    // acceleration structure
              rayFlags,      // rayFlags
              0xFF,          // cullMask
              0,             // sbtRecordOffset
              0,             // sbtRecordStride
              0,             // missIndex
              rayOrigin,     // ray origin
              tMin,          // ray min range
              rayDirection,  // ray direction
              tMax,          // ray max range
              0              // payload (location = 0)
  );

  imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(payload.color, 1.F));
}
