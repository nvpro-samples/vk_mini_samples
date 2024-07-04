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
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "device_host.h"
#include "dh_bindings.h"
#include "payload.h"
#include "nvvkhl/shaders/constants.h"
#include "nvvkhl/shaders/ggx.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/bsdf_functions.h"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;

//layout(buffer_reference, scalar) readonly buffer Vertices  { Vertex v[]; };
//layout(buffer_reference, scalar) readonly buffer Indices   { uvec3 i[]; };
// layout(buffer_reference, scalar) readonly buffer PrimMeshInfos { PrimMeshInfo i[]; };
// layout(buffer_reference, scalar) readonly buffer InstanceInfos { InstanceInfo i[]; };
// layout(buffer_reference, scalar) readonly buffer Materials { vec4 m[]; };

layout(set = 0, binding = B_tlas ) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = B_frameInfo, scalar) uniform FrameInfo_ { FrameInfo frameInfo; };
// layout(set = 0, binding = B_sceneDesc, scalar) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(set = 0, binding = B_skyParam,  scalar) uniform SkyInfo_ { ProceduralSkyShaderParameters skyInfo; };
layout(set = 0, binding = B_materials, scalar) buffer Materials_ { vec4 m[]; } materials;
layout(set = 0, binding = B_instances, scalar) buffer InstanceInfo_ { InstanceInfo i[]; } instanceInfo;
layout(set = 0, binding = B_vertex, scalar) buffer Vertex_ { Vertex v[]; } vertices[];
layout(set = 0, binding = B_index, scalar) buffer Index_ { uvec3 i[]; } indices[];

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
HitState getHitState(int meshID)
{
  HitState hit;

  // Vextex and indices of the primitive
  //Vertices vertices = Vertices(pinfo.vertexAddress);
  //Indices  indices  = Indices(pinfo.indexAddress);

  // Barycentric coordinate on the triangle
  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = indices[meshID].i[gl_PrimitiveID];

  // All vertex attributes of the triangle.
  Vertex v0 = vertices[meshID].v[triangleIndex.x];
  Vertex v1 = vertices[meshID].v[triangleIndex.y];
  Vertex v2 = vertices[meshID].v[triangleIndex.z];

  // Position
  const vec3 pos0     = v0.position.xyz;
  const vec3 pos1     = v1.position.xyz;
  const vec3 pos2     = v2.position.xyz;
  const vec3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos             = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

  // Normal
  const vec3 nrm0           = v0.normal.xyz;
  const vec3 nrm1           = v1.normal.xyz;
  const vec3 nrm2           = v2.normal.xyz;
  const vec3 normal         = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  vec3       worldNormal    = normalize(vec3(normal * gl_WorldToObjectEXT));
  const vec3 geoNormal      = normalize(cross(pos1 - pos0, pos2 - pos0));
  vec3       worldGeoNormal = normalize(vec3(geoNormal * gl_WorldToObjectEXT));
  hit.geonrm                = worldGeoNormal;
  hit.nrm                   = worldNormal;

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
#define gl_HitKindFrontFacingMicroTriangleNV 222
#define gl_HitKindBackFacingMicroTriangleNV 223
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
  vec3 L       = normalize(skyInfo.directionToLight);
  bool visible = shadowRay(P, L);

  // Retrieve the Primitive mesh buffer information
  //PrimMeshInfos pInfo_ = PrimMeshInfos(sceneDesc.primInfoAddress);
  //PrimMeshInfo  pinfo  = pInfo_.i[gl_InstanceCustomIndexEXT];

  // Retrieve the Instance buffer information
  // InstanceInfos iInfo_ = InstanceInfos(sceneDesc.instInfoAddress);
  // InstanceInfo iInfo = iInfo_.i[gl_InstanceID];
  InstanceInfo iInfo = instanceInfo.i[gl_InstanceID];

  HitState hit = getHitState(gl_InstanceCustomIndexEXT);

  //Materials materials = Materials(sceneDesc.materialAddress);
  vec3 albedo = materials.m[iInfo.materialID].xyz;

  // Color at hit point
  PbrMaterial mat   = defaultPbrMaterial(albedo, pc.metallic, pc.roughness, hit.nrm, hit.geonrm);

  vec3 color = ggxEvaluate(V, L, mat);

  // Under shader, dimm the contribution
  if(!visible)
    color *= 0.3F;

  payload.color += color * payload.weight * pc.intensity;

  if(pc.numBaseTriangles > 0)
  {
    const float thickness  = 0.002F * pc.numBaseTriangles;
    const float smoothness = 0.002F;
    const vec3 bary = fract(vec3(1.0F - attribs.x - attribs.y, attribs.x, attribs.y) * pc.numBaseTriangles + (thickness * 0.5F));
    float      minBary    = min(bary.x, min(bary.y, bary.z));
    const vec3 wire_color = vec3(0.3F, 0.3F, 0.3F);
    minBary               = smoothstep(thickness, thickness + smoothness, minBary);
    payload.color         = mix(wire_color, payload.color, minBary);
  }

  // Reflection
  vec3 refl_dir = reflect(D, hit.nrm);

  payload.depth += 1;
  payload.weight *= pc.metallic;  // more or less reflective

  traceRayEXT(topLevelAS, gl_RayFlagsCullBackFacingTrianglesEXT, 0xFF, 0, 0, 0, P, 0.0001, refl_dir, 100.0, 0);
}
