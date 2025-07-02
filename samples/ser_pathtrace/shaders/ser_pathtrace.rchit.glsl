/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "shaderio.h"
#include "dh_bindings.h"
#include "payload.h"

// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;
hitAttributeEXT vec3 objAttribs; //layout(location = 0) hitObjectAttributeNV vec3 objAttribs;

layout(set = 0, binding = B_instances, scalar) buffer InstanceInfo_ { InstanceInfo i[]; } instanceInfo;
layout(set = 0, binding = B_vertex, scalar) buffer Vertex_ { Vertex v[]; } vertices[];
layout(set = 0, binding = B_index, scalar) buffer Index_ { uvec3 i[]; } indices[];
// clang-format on


#include "gethit.h"

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void main()
{
  // Retrieve the Primitive mesh buffer information
  uint   meshID        = gl_InstanceCustomIndexEXT;
  uint   triID         = gl_PrimitiveID;
  mat4x3 objectToWorld = gl_ObjectToWorldEXT;
  mat4x3 worldToObject = gl_WorldToObjectEXT;

  HitState hit = getHitState(meshID, triID, objectToWorld, worldToObject, objAttribs);

  payload.hitT          = gl_HitTEXT;
  payload.pos           = hit.pos;
  payload.nrm           = hit.nrm;
  payload.instanceIndex = gl_InstanceID;
}
