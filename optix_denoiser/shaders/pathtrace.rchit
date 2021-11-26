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
#include "compress.glsl"

hitAttributeEXT vec2 attribs;

// clang-format off
layout(location = 0) rayPayloadInEXT HitPayload payload;

layout(buffer_reference, scalar) readonly buffer Vertices  { Vertex v[]; };
layout(buffer_reference, scalar) readonly buffer Indices   { uvec3 i[]; };
layout(buffer_reference, scalar) readonly buffer PrimMeshInfos { PrimMeshInfo i[]; };
layout(buffer_reference, scalar) readonly buffer Materials { GltfShadeMaterial m[]; };

layout(set = 0, binding = eTlas ) uniform accelerationStructureEXT topLevelAS;

layout(set = 1, binding = eFrameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 1, binding = eSceneDesc) readonly buffer SceneDesc_ { SceneDescription sceneDesc; };
layout(set = 1, binding = eTextures)  uniform sampler2D texturesMap[]; // all textures

layout(set = 2, binding = eImpSamples,  scalar)	buffer _EnvAccel { EnvAccel envSamplingData[]; };
layout(set = 2, binding = eHdr) uniform sampler2D hdrTexture;
// clang-format on


#include "pbr_gltf.glsl"
#include "hdr_env_sampling.glsl"

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  vec3 pos;
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
  const vec3 pos0     = v0.position.xyz;
  const vec3 pos1     = v1.position.xyz;
  const vec3 pos2     = v2.position.xyz;
  const vec3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos             = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

  // Normal
  const vec3 nrm0        = v0.normal.xyz;
  const vec3 nrm1        = v1.normal.xyz;
  const vec3 nrm2        = v2.normal.xyz;
  const vec3 normal      = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  const vec3 worldNormal = normalize(vec3(normal * gl_WorldToObjectEXT));
  const vec3 geomNormal  = normalize(cross(pos1 - pos0, pos2 - pos0));
  hit.nrm                = worldNormal;

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
// Use for light/env contribution
struct VisibilityContribution
{
  vec3  radiance;   // Radiance at the point if light is visible
  vec3  lightDir;   // Direction to the light, to shoot shadow ray
  float lightDist;  // Distance to the light (1e32 for infinite or sky)
  bool  visible;    // true if in front of the face and should shoot shadow ray
};


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
VisibilityContribution DirectLight(MaterialEval matEval, HitState hit)
{
  VisibilityContribution contrib;
  contrib.radiance = vec3(0);
  contrib.visible  = false;
  vec3  lightDir;
  float lightPdf = 1 / float(NB_LIGHTS);
  vec3  lightContrib;
  float lightDist = 1e32;
  bool  isLight   = false;

  if(NB_LIGHTS != 0 && rand(payload.seed) <= 0.5)
  {
    // randomly select one of the lights
    int   light_index = int(min(rand(payload.seed) * NB_LIGHTS, NB_LIGHTS));
    Light light       = frameInfo.light[light_index];

    lightContrib = lightContribution(light, hit.pos, hit.nrm, lightDir);
    lightDist    = (light.type != 0) ? 1e37f : length(hit.pos - light.position);
    isLight      = true;
  }
  else
  {
    vec3 randVal     = vec3(rand(payload.seed), rand(payload.seed), rand(payload.seed));
    vec4 radiancePdf = environmentSample(hdrTexture, randVal, lightDir);
    lightContrib     = radiancePdf.xyz * frameInfo.clearColor.xyz;
    lightPdf         = radiancePdf.w;
  }

  float dotNL = dot(lightDir, hit.nrm);
  if(dotNL > 0.0)
  {
    float pdf       = 0;
    vec3  brdf      = pbrEval(matEval, -gl_WorldRayDirectionEXT, lightDir, pdf);
    float misWeight = isLight ? 1.0 : max(0.0, powerHeuristic(lightPdf, pdf));
    vec3  radiance  = misWeight * brdf * dotNL * lightContrib / lightPdf;

    contrib.visible   = true;
    contrib.lightDir  = lightDir;
    contrib.lightDist = lightDist;
    contrib.radiance  = radiance;
  }

  return contrib;
}

void StopRay()
{
  payload.hitT = INFINITE;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 shading(in MaterialEval matEval, in HitState hit)
{
  vec3 radiance = matEval.emissive;  // Emissive material

  if(radiance.x > 0 || radiance.y > 0 || radiance.z > 0)
  {  // Stop on emmissive material
    StopRay();
    return radiance;
  }

  // Sampling for the next ray
  vec3  rayDirection;
  float pdf  = 0;
  vec3  brdf = pbrSample(matEval, -gl_WorldRayDirectionEXT, rayDirection, pdf, payload.seed);

  if(pdf > 0.0)
  {
    payload.weight = brdf * abs(dot(hit.nrm, rayDirection)) / pdf;
  }
  else
  {
    StopRay();
    return radiance;
  }

  // Next ray
  payload.rayDirection = rayDirection;
  payload.rayOrigin    = offsetRay(hit.pos, dot(rayDirection, hit.nrm) > 0 ? hit.nrm : -hit.nrm);

  // Light and environment contribution at hit position
  VisibilityContribution vcontrib = DirectLight(matEval, hit);

  if(vcontrib.visible)
  {
    // Shadow ray - stop at the first intersection, don't invoke the closest hit shader (fails for transparent objects)
    uint rayflag = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
    payload.hitT = 0;
    traceRayEXT(topLevelAS, rayflag, 0xFF, 0, 0, 0, payload.rayOrigin, 0.001, vcontrib.lightDir, vcontrib.lightDist, 0);
    // If hitting nothing, add light contribution
    if(payload.hitT == INFINITE)
      radiance += vcontrib.radiance;
    payload.hitT = gl_HitTEXT;
  }
  return radiance;
}

//-----------------------------------------------------------------------
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

  payload.hitT    = gl_HitTEXT;
  payload.contrib = shading(matEval, hit);

  // -- Debug --
  //  payload.contrib = hit.nrm * .5 + .5;
  //  payload.contrib = matEval.albedo.xyz;
  //  payload.contrib = matEval.tangent * .5 + .5;
  //  StopRay();
}
