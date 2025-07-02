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
#extension GL_EXT_ray_tracing_position_fetch : require  // #FETCH


#include "shaderio.h"
#include "dh_bindings.h"
#include "payload.h"
#include "nvshaders/constants.h.slang"
#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/constants.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/ray_utils.h.slang"
#include "nvshaders/sky_functions.h.slang"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(set = 0, binding = B_tlas ) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = B_frameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = B_sceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(set = 0, binding = B_skyParam) uniform SkyInfo_ { SkySimpleParameters skyInfo; };
layout(set = 0, binding = B_materials, scalar) buffer Materials_ { vec4 m[]; } materials;
layout(set = 0, binding = B_instances, scalar) buffer InstanceInfo_ { InstanceInfo i[]; } instanceInfo;

layout(push_constant) uniform RtxPushConstant_ { PushConstant pc; };
// clang-format on


//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  vec3 pos;
  vec3 nrm;
  vec3 geonrm;
};

//-----------------------------------------------------------------------
// Return hit position and normal in world space
HitState getHitState()
{
  HitState hit;

  // Barycentric coordinate on the triangle
  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Position
  const vec3 pos0     = gl_HitTriangleVertexPositionsEXT[0];  // #FETCH
  const vec3 pos1     = gl_HitTriangleVertexPositionsEXT[1];
  const vec3 pos2     = gl_HitTriangleVertexPositionsEXT[2];
  const vec3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos             = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

  // Normal
  const vec3 geoNormal      = normalize(cross(pos1 - pos0, pos2 - pos0));
  vec3       worldGeoNormal = normalize(vec3(geoNormal * gl_WorldToObjectEXT));
  hit.geonrm                = worldGeoNormal;
  hit.nrm                   = worldGeoNormal;  // #FETCH

  return hit;
}

//-----------------------------------------------------------------------
// Return TRUE if there is no occluder, meaning that the light is visible from P toward L
bool shadowRay(vec3 P, vec3 L)
{
  const uint rayFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
  HitPayload savedP = payload;
  traceRayEXT(topLevelAS, rayFlags, 0xFF, 0, 0, 0, P, 0.0001, L, 100.0, 0);
  bool visible = (payload.depth == MISS_DEPTH);
  payload      = savedP;
  return visible;
}

vec3 ggxEvaluate(vec3 V, vec3 L, PbrMaterial mat)
{
  BsdfEvaluateData data;
  data.k1 = V;
  data.k2 = L;

  bsdfEvaluateSimple(data, mat);

  return data.bsdf_glossy + data.bsdf_diffuse;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void main()
{
  // We hit our max depth
  if(payload.depth >= pc.maxDepth)
  {
    return;
  }

  vec3 P = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
  vec3 D = normalize(gl_WorldRayDirectionEXT);
  vec3 V = -D;

  // Vector to the light
  vec3 L       = normalize(skyInfo.sunDirection);
  bool visible = shadowRay(P, L);

  // Retrieve the Instance buffer information
  InstanceInfo iInfo = instanceInfo.i[gl_InstanceID];

  HitState hit = getHitState();

  vec3 albedo = materials.m[iInfo.materialID].xyz;

  PbrMaterial mat = defaultPbrMaterial(albedo, pc.metallic, pc.roughness, hit.nrm, hit.geonrm);

  // Color at hit point
  vec3 color = ggxEvaluate(V, L, mat);

  // Under shader, dimm the contribution
  if(!visible)
    color *= 0.3F;

  payload.color += color * payload.weight * pc.intensity;

  // Reflection
  vec3 refl_dir = reflect(D, hit.nrm);

  payload.depth += 1;
  payload.weight *= pc.metallic;  // more or less reflective

  traceRayEXT(topLevelAS, gl_RayFlagsCullBackFacingTrianglesEXT, 0xFF, 0, 0, 0, P, 0.001, refl_dir, 100.0, 0);
}
