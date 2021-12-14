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


#include "common/shaders/pbr_gltf.glsl"
#include "common/shaders/hdr_env_sampling.glsl"
#include "common/shaders/get_hit.glsl"

// --vvvv-- Adding HDR sampling --vvvv-- 

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
  { // <------ Adding HDR sampling
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
    vec3  radiance  = misWeight * brdf * lightContrib / lightPdf;

    contrib.visible   = true;
    contrib.lightDir  = lightDir;
    contrib.lightDist = lightDist;
    contrib.radiance  = radiance;
  }

  return contrib;
}


#include "common/shaders/shading.glsl"

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
  //payload.contrib = shading(matEval, hit);
    ShadingResult result;
  shading(matEval, hit, result);

  payload.weight       = result.weight;
  payload.contrib      = result.radiance;
  payload.rayOrigin    = result.rayOrigin;
  payload.rayDirection = result.rayDirection;

  // -- Debug --
  //  payload.contrib = hit.nrm * .5 + .5;
  //  payload.contrib = matEval.albedo.xyz;
  //  payload.contrib = matEval.tangent * .5 + .5;
  //  StopRay();
}
