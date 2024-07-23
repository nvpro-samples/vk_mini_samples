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
#extension GL_NV_displacement_micromap : require


#include "device_host.h"
#include "dh_bindings.h"
#include "payload.h"
#include "nvvkhl/shaders/constants.h"
#include "nvvkhl/shaders/ggx.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/bsdf_structs.h"
#include "nvvkhl/shaders/bsdf_functions.h"
#include "nvvkhl/shaders/pbr_mat_struct.h"
#include "nvvkhl/shaders/func.h"

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
layout(set = 0, binding = B_skyParam,  scalar) uniform SkyInfo_ { SimpleSkyParameters skyInfo; };
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

// Utility for temperature and landscapeColor:
// Smoothly transforms the range [low, high] to [0, 1], with 0 derivative at
// low, high, and (low + high) * 0.5.
float fade(float low, float high, float value)
{
  float mid   = (low + high) * 0.5;
  float range = (high - low) * 0.5;
  float x     = 1.0 - clamp(abs(mid - value) / range, 0.0, 1.0);
  return smoothstep(0.0, 1.0, x);
}

// Return a landscape color based on height [0-1]
vec3 landscapeColor(float height)
{
  const vec3 water = vec3(0.0, 0.0, 0.5);
  const vec3 sand  = vec3(0.8, 0.7, 0.4);
  const vec3 green = vec3(0.1, 0.4, 0.1);
  const vec3 rock  = vec3(0.4, 0.4, 0.4);
  const vec3 snow  = vec3(1.0, 1.0, 1.0);


  vec3 color = (fade(-0.25, 0.25, height) * water   //
                + fade(0.0, 0.5, height) * sand     //
                + fade(0.25, 0.75, height) * green  //
                + fade(0.5, 1.0, height) * rock     //
                + smoothstep(0.75, 1.0, height) * snow);
  return color;
}

vec2 baseToMicro(vec2 barycentrics[3], vec2 p)
{
  vec2  ap   = p - barycentrics[0];
  vec2  ab   = barycentrics[1] - barycentrics[0];
  vec2  ac   = barycentrics[2] - barycentrics[0];
  float rdet = 1.f / (ab.x * ac.y - ab.y * ac.x);
  return vec2(ap.x * ac.y - ap.y * ac.x, ap.y * ab.x - ap.x * ab.y) * rdet;
}

vec3 wireframe(in vec3 color, float width, vec3 bary)
{
  const vec3 wireColor = vec3(0.3F, 0.3F, 0.3F);
  float      minBary   = min(bary.x, min(bary.y, bary.z));
  return mix(wireColor, color, smoothstep(width, width + 0.002F, minBary));
}

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

  // Retrieve the Primitive mesh buffer information
  //PrimMeshInfos pInfo_ = PrimMeshInfos(sceneDesc.primInfoAddress);
  //PrimMeshInfo  pinfo  = pInfo_.i[gl_InstanceCustomIndexEXT];
  InstanceInfo iInfo = instanceInfo.i[gl_InstanceID];

  uint hitKind = gl_HitKindEXT;
  if(hitKind == gl_HitKindFrontFacingMicroTriangleNV || hitKind == gl_HitKindBackFacingMicroTriangleNV)
  {
    payload.color = landscapeColor(P.y * 2.0F);

    // Add wireframe
    if(pc.numBaseTriangles > 0)
    {
      // Micro-triangles
      const vec2 microBary2 = baseToMicro(gl_HitMicroTriangleVertexBarycentricsNV, attribs);
      const vec3 microBary  = vec3(1.0F - microBary2.x - microBary2.y, microBary2.xy);
      payload.color         = wireframe(payload.color, 0.001F * pc.numBaseTriangles, microBary);

      // Base triangles
      const vec3 baseBary = vec3(1.0 - attribs.x - attribs.y, attribs.xy);
      payload.color       = wireframe(payload.color, 0.002F, baseBary);
    }
    return;
  }

  HitState hit = getHitState(gl_InstanceCustomIndexEXT);

  //Materials materials = Materials(sceneDesc.materialAddress);
  vec3 albedo = materials.m[iInfo.materialID].xyz;

  // Vector to the light
  vec3 L       = normalize(skyInfo.directionToLight);
  bool visible = shadowRay(P, L);

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

  traceRayEXT(topLevelAS, gl_RayFlagsCullBackFacingTrianglesEXT, 0xFF, 0, 0, 0, P, 0.0001, refl_dir, 100.0, 0);
}
