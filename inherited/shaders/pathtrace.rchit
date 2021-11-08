/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "host_device.h"

#include "raycommon.glsl"
#include "sampling.glsl"


hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT hitPayload prd;

layout(buffer_reference, scalar) readonly buffer Vertices  { Vertex v[]; };
layout(buffer_reference, scalar) readonly buffer Indices   { uvec3 i[]; };
layout(buffer_reference, scalar) readonly buffer PrimMeshInfos { PrimMeshInfo i[]; };
layout(buffer_reference, scalar) readonly buffer Materials { ShadingMaterial m[]; };

layout(set = 0, binding = eTlas ) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 1, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(set = 1, binding = eTextures)  uniform sampler2D texturesMap[]; // all textures
// clang-format on


void main()
{
  // Retrieve the Primitive mesh buffer information
  PrimMeshInfos pInfo_ = PrimMeshInfos(sceneDesc.primInfoAddress);
  PrimMeshInfo  pinfo  = pInfo_.i[gl_InstanceCustomIndexEXT];

  // Vextex and indices of the primitive
  Vertices vertices = Vertices(pinfo.vertexAddress);
  Indices  indices  = Indices(pinfo.indexAddress);

  // Material used
  uint      matIndex  = max(0, pinfo.materialIndex);  // material of primitive mesh
  Materials materials = Materials(sceneDesc.materialAddress);

  // barycentric coordinate on the triangle
  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = indices.i[gl_PrimitiveID];

  // All vertex attributes of the triangle.
  Vertex v0 = vertices.v[triangleIndex.x];
  Vertex v1 = vertices.v[triangleIndex.y];
  Vertex v2 = vertices.v[triangleIndex.z];

  // Hit position
  const vec3 pos0           = v0.position.xyz;
  const vec3 pos1           = v1.position.xyz;
  const vec3 pos2           = v2.position.xyz;
  const vec3 position       = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  const vec3 world_position = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

  // Normal
  const vec3 nrm0         = v0.normal.xyz;
  const vec3 nrm1         = v1.normal.xyz;
  const vec3 nrm2         = v2.normal.xyz;
  vec3       normal       = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  const vec3 world_normal = normalize(vec3(normal * gl_WorldToObjectEXT));
  const vec3 geom_normal  = normalize(cross(pos1 - pos0, pos2 - pos0));

  // TexCoord
  const vec2 uv0       = vec2(v0.position.w, v0.normal.w);
  const vec2 uv1       = vec2(v1.position.w, v1.normal.w);
  const vec2 uv2       = vec2(v2.position.w, v2.normal.w);
  const vec2 texcoord0 = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

  // https://en.wikipedia.org/wiki/Path_tracing
  // Material of the object
  ShadingMaterial mat       = materials.m[matIndex];
  vec3            emittance = mat.emissiveFactor;


  // Pick a random direction from here and keep going.
  vec3 tangent, bitangent;
  createCoordinateSystem(world_normal, tangent, bitangent);
  vec3 rayOrigin    = world_position;
  vec3 rayDirection = cosineSamplingHemisphere(prd.seed, tangent, bitangent, world_normal);

  // Probability of the newRay (cosine distributed)
  const float pdf = 1 / M_PI;

  // Compute the BRDF for this ray (assuming Lambertian reflection)
  vec3 albedo = mat.pbrBaseColorFactor.xyz;
  if(mat.pbrBaseColorTexture > -1)
  {
    uint txtId = mat.pbrBaseColorTexture;
    albedo *= texture(texturesMap[nonuniformEXT(txtId)], texcoord0).xyz;
  }
  vec3 BRDF = albedo / M_PI;

  // Next-event estimation
  Light light = frameInfo.light[0];

  // Vector toward light
  vec3  L              = light.position;
  float lightIntensity = light.intensity;
  if(light.type == 0 /*point*/)
  {
    L -= world_position;
    float d = length(L);
    lightIntensity /= (d * d);
  }
  L = normalize(L);

  float rayLength = (light.type != 0) ? 1e37f : length(rayOrigin - light.position);

  vec3 lightContrib = BRDF * max(dot(world_normal, L), 0.0) * lightIntensity * light.color;

  uint currentDepth = prd.depth;
  prd.depth         = 0;
  prd.rayOrigin     = rayOrigin;
  prd.rayDirection  = rayDirection;
  prd.weight        = BRDF / pdf;


  traceRayEXT(topLevelAS,  // acceleration structure
              // Stop at the first intersection, don't invoke the closest hit shader
              gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
              0xFF,          // cullMask
              0,             // sbtRecordOffset
              0,             // sbtRecordStride
              0,             // missIndex
              rayOrigin,     // ray origin
              0.001,         // ray min range
              rayDirection,  // ray direction
              rayLength,     // ray max range
              0              // payload (location = 0)
  );
  if(prd.depth != 0)
    emittance += lightContrib;

  prd.hitValue = emittance;
  prd.depth    = currentDepth;
}
