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
#include "ray_common.glsl"
#include "common/shaders/sampling.glsl"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 1) rayPayloadInEXT GbufferPayload payloadGbuf;

layout(buffer_reference, scalar) readonly buffer Vertices  { Vertex v[]; };
layout(buffer_reference, scalar) readonly buffer Indices   { uvec3 i[]; };
layout(buffer_reference, scalar) readonly buffer PrimMeshInfos { PrimMeshInfo i[]; };
layout(buffer_reference, scalar) readonly buffer Materials { GltfShadeMaterial m[]; };

layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 1, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(set = 1, binding = eTextures)  uniform sampler2D texturesMap[]; // all textures
// clang-format on


#include "common/shaders/pbr_gltf.glsl"
#include "compress.glsl"

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  vec3 nrm;
  vec2 uv;
  vec3 tangent;
  vec3 bitangent;
};


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
HitState GetHitState(PrimMeshInfo pinfo)
{
  HitState hit;

  // Vextex and indices of the primitive
  Vertices vertices = Vertices(pinfo.vertexAddress);
  Indices  indices  = Indices(pinfo.indexAddress);

  // Barycentric coordinate on the triangle
  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = indices.i[gl_PrimitiveID];

  // All vertex attributes of the triangle.
  Vertex v0 = vertices.v[triangleIndex.x];
  Vertex v1 = vertices.v[triangleIndex.y];
  Vertex v2 = vertices.v[triangleIndex.z];

  // Position
  //  const vec3 pos0     = v0.position.xyz;
  //  const vec3 pos1     = v1.position.xyz;
  //  const vec3 pos2     = v2.position.xyz;
  //  const vec3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  //  hit.pos             = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

  // Normal
  const vec3 nrm0        = v0.normal.xyz;
  const vec3 nrm1        = v1.normal.xyz;
  const vec3 nrm2        = v2.normal.xyz;
  const vec3 normal      = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  const vec3 worldNormal = normalize(vec3(normal * gl_WorldToObjectEXT));
  //  const vec3 geomNormal  = normalize(cross(pos1 - pos0, pos2 - pos0));
  hit.nrm = dot(worldNormal, gl_WorldRayDirectionEXT) <= 0.0 ? worldNormal : -worldNormal;  // Front-face

  // TexCoord
  const vec2 uv0 = vec2(v0.position.w, v0.normal.w);
  const vec2 uv1 = vec2(v1.position.w, v1.normal.w);
  const vec2 uv2 = vec2(v2.position.w, v2.normal.w);
  hit.uv         = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

  // Tangent - Bitangent
  const vec4 tng0    = vec4(v0.tangent);
  const vec4 tng1    = vec4(v1.tangent);
  const vec4 tng2    = vec4(v2.tangent);
  vec3       tangent = normalize(tng0.xyz * barycentrics.x + tng1.xyz * barycentrics.y + tng2.xyz * barycentrics.z);
  vec3       world_tangent  = normalize(vec3(tangent * gl_WorldToObjectEXT));
  vec3       world_binormal = cross(worldNormal, world_tangent) * tng0.w;
  hit.tangent               = world_tangent;
  hit.bitangent             = world_binormal;

  return hit;
}


//-----------------------------------------------------------------------
// Returning the G-Buffer informations
//-----------------------------------------------------------------------
void main()
{
  // Retrieve the Primitive mesh buffer information
  PrimMeshInfos pInfo_ = PrimMeshInfos(sceneDesc.primInfoAddress);
  PrimMeshInfo  pinfo  = pInfo_.i[gl_InstanceCustomIndexEXT];

  HitState hit = GetHitState(pinfo);

  // Scene materials
  uint      matIndex  = max(0, pinfo.materialIndex);  // material of primitive mesh
  Materials materials = Materials(sceneDesc.materialAddress);

  // Material of the object and evaluated material (includes textures)
  GltfShadeMaterial mat     = materials.m[matIndex];
  MaterialEval      matEval = evaluateMaterial(mat, hit.nrm, hit.tangent, hit.bitangent, hit.uv);

  payloadGbuf.packAlbedo = packUnorm4x8(matEval.albedo);
  payloadGbuf.packNormal = compress_unit_vec(matEval.normal);
}
