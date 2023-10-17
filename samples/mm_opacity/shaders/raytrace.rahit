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


#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "device_host.h"
#include "payload.h"


layout(location = 0) rayPayloadInEXT HitPayload payload;
layout(push_constant) uniform RtxPushConstant_
{
  PushConstant pc;
};


void main()
{
  // Find where the ray hit
  vec3 pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

  // Cut out the plane if outside the radius
  if((pc.useAnyhit == 1) && (length(pos) > pc.radius))
  {
    ignoreIntersectionEXT;
    return;
  }

  // To show that AnyHitr shader was invoked, we are tinted the color
  payload.color  = vec3(1.0F, 0.0F, 0.0F);
  payload.weight = vec3(0.5F);
}
