/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_NV_shader_sm_builtins : require     // Debug - gl_WarpIDNV, gl_SMIDNV
#extension GL_ARB_gpu_shader_int64 : enable       // Debug - heatmap value
#extension GL_EXT_shader_realtime_clock : enable  // Debug - heatmap timing

#extension GL_NV_shader_invocation_reorder : enable
#extension GL_EXT_ray_query : require


#include "device_host.h"
#include "dh_bindings.h"
#include "payload.h"
#include "temperature.glsl"
#include "nvvkhl/shaders/random.h"
#include "nvvkhl/shaders/constants.h"
#include "nvvkhl/shaders/dh_sky.h"
#include "nvvkhl/shaders/ggx.h"
#include "nvvkhl/shaders/ray_util.h"
#include "nvvkhl/shaders/pbr_mat_struct.h"
#include "nvvkhl/shaders/bsdf_structs.h"
#include "nvvkhl/shaders/bsdf_functions.h"


// clang-format off
layout(location = 0) rayPayloadEXT HitPayload payload;


layout(set = 0, binding = B_tlas) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = B_outImage, rgba32f) uniform image2D image;
layout(set = 0, binding = B_outHeatmap, rgba32f) uniform image2D heatmap;
layout(set = 0, binding = B_frameInfo) uniform FrameInfo_ { FrameInfo frameInfo; };
layout(set = 0, binding = B_skyParam) uniform SkyInfo_ { ProceduralSkyShaderParameters skyInfo; };
layout(set = 0, binding = B_heatStats) buffer HeatStats_ { HeatStats heatStats; };
layout(set = 0, binding = B_materials, scalar) buffer Materials_ { vec4 m[]; } materials;
layout(set = 0, binding = B_instances, scalar) buffer InstanceInfo_ { InstanceInfo i[]; } instanceInfo;

layout(set = 0, binding = B_vertex, scalar) buffer Vertex_ { Vertex v[]; } vertices[];
layout(set = 0, binding = B_index, scalar) buffer Index_ { uvec3 i[]; } indices[];

layout(location = 0) hitObjectAttributeNV vec3 objAttribs;

layout(constant_id = 0) const int USE_SER = 1;

// clang-format on


#include "gethit.h"

layout(push_constant) uniform RtxPushConstant_
{
  PushConstant pc;
};

struct Ray
{
  vec3 origin;
  vec3 direction;
};


//-----------------------------------------------------------------------
// Shoot a ray an return the information of the closest hit, in the
// PtPayload structure (PRD)
//
void traceRay(Ray r)
{
  payload.hitT  = 0.0F;
  uint rayFlags = gl_RayFlagsCullBackFacingTrianglesEXT;
  if(USE_SER == 1)
  {
    hitObjectNV hObj;
    hitObjectRecordEmptyNV(hObj);  //Initialize to an empty hit object
    hitObjectTraceRayNV(hObj, topLevelAS, rayFlags, 0xFF, 0, 0, 0, r.origin, 0.0, r.direction, INFINITE, 0);
    reorderThreadNV(hObj);
    hitObjectGetAttributesNV(hObj, 0);

    payload.hitT = hitObjectGetRayTMaxNV(hObj);

    if(hitObjectIsHitNV(hObj))
    {
      payload.instanceIndex = hitObjectGetInstanceIdNV(hObj);

      int    meshID        = hitObjectGetInstanceCustomIndexNV(hObj);
      int    triID         = hitObjectGetPrimitiveIndexNV(hObj);
      mat4x3 objectToWorld = hitObjectGetObjectToWorldNV(hObj);
      mat4x3 worldToObject = hitObjectGetWorldToObjectNV(hObj);

      HitState hit = getHitState(meshID, triID, objectToWorld, worldToObject, objAttribs);
      payload.pos  = hit.pos;
      payload.nrm  = hit.nrm;
    }

    // Skipping call to Chit and Miss shaders, the information can be retrieved from hitObjectGet functions
    // hitObjectExecuteShaderNV(hObj, 0);
  }
  else
  {
    traceRayEXT(topLevelAS, rayFlags, 0xFF, 0, 0, 0, r.origin, 0.0, r.direction, INFINITE, 0);
  }
}

//-----------------------------------------------------------------------
// Shadow ray - return true if a ray hits anything
//
bool traceShadow(Ray r, float maxDist)
{
  payload.hitT = 0.0F;
  uint rayFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsCullBackFacingTrianglesEXT | gl_RayFlagsSkipClosestHitShaderEXT;
  bool isHit;
  if(USE_SER == 1)
  {
    hitObjectNV hObj;
    hitObjectRecordEmptyNV(hObj);
    hitObjectTraceRayNV(hObj, topLevelAS, rayFlags, 0xFF, 0, 0, 0, r.origin, 0.0, r.direction, maxDist, 0);
    isHit = hitObjectIsHitNV(hObj);
  }
  else
  {
    traceRayEXT(topLevelAS, rayFlags, 0xFF, 0, 0, 0, r.origin, 0.0, r.direction, maxDist, 0);
    isHit = payload.hitT < INFINITE;
  }
  return isHit;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 pathTrace(Ray r, inout uint seed)
{
  vec3 radiance   = vec3(0.0F);
  vec3 throughput = vec3(1.0F);

  for(int depth = 0; depth < pc.maxDepth; depth++)
  {
    traceRay(r);

    // Hitting the environment, then exit
    if(payload.hitT == INFINITE)
    {
      vec3 sky_color = proceduralSky(skyInfo, r.direction, 0.0F);
      return radiance + (sky_color * throughput);
    }

    // Retrieve the Instance buffer information
    InstanceInfo iInfo = instanceInfo.i[payload.instanceIndex];

    float pdf = 0.0F;
    vec3  V   = -r.direction;
    vec3  L   = normalize(skyInfo.directionToLight);

    // Setting up the material
    vec3 baseColor = materials.m[iInfo.materialID].xyz;
    PbrMaterial pbrMat    = defaultPbrMaterial(baseColor, pc.metallic, pc.roughness, payload.nrm, payload.nrm);

    // Add dummy divergence
    // This sample shader is too simple to show the benefits of sorting
    // rays, with this we create some artificial workload depending on the
    // material ID

    vec3 dummy     = payload.nrm;
    uint dummyLoop = (iInfo.materialID * 128) & (1024 - 1);
    for(uint i = 0; i < dummyLoop; i++)
    {
      dummy = sin(dummy);
    }

    pbrMat.baseColor.xyz += dummy * 0.01;

    vec3 contrib = vec3(0);

    // Evaluation of direct light (sun)
    bool nextEventValid = (dot(L, payload.nrm) > 0.0f);
    if(nextEventValid)
    {
      BsdfEvaluateData evalData;
      evalData.k1 = -r.direction;
      evalData.k2 = normalize(skyInfo.directionToLight);
      evalData.xi = vec3(rand(seed), rand(seed), rand(seed));
      bsdfEvaluate(evalData, pbrMat);

      const vec3 w = pc.intensity.xxx;
      contrib += w * evalData.bsdf_diffuse;
      contrib += w * evalData.bsdf_glossy;
      contrib *= throughput;
    }


    // Sample BSDF
    {
      BsdfSampleData sampleData;
      sampleData.k1 = -r.direction;  // outgoing direction
      sampleData.xi = vec3(rand(seed), rand(seed), rand(seed));

      bsdfSample(sampleData, pbrMat);
      if(sampleData.event_type == BSDF_EVENT_ABSORB)
      {
        break;  // Need to add the contribution ?
      }

      throughput *= sampleData.bsdf_over_pdf;
      r.origin    = offsetRay(payload.pos, payload.nrm);
      r.direction = sampleData.k2;
    }


    // Russian-Roulette (minimizing live state)
    float rrPcont = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001F, 0.95F);
    if(rand(seed) >= rrPcont)
      break;                // paths with low throughput that won't contribute
    throughput /= rrPcont;  // boost the energy of the non-terminated paths

    // We are adding the contribution to the radiance only if the ray is not occluded by an object.
    if(nextEventValid)
    {
      Ray  shadowRay = Ray(r.origin, L);
      bool inShadow  = traceShadow(shadowRay, INFINITE);
      if(!inShadow)
      {
        radiance += contrib;
      }
    }
  }

  return radiance;
}


//-----------------------------------------------------------------------
// Sampling the pixel
//-----------------------------------------------------------------------
vec3 samplePixel(inout uint seed)
{
  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  vec2 subpixel_jitter = pc.frame == 0 ? vec2(0.5f, 0.5f) : vec2(rand(seed), rand(seed));

  const vec2 pixelCenter = vec2(gl_LaunchIDEXT.xy) + subpixel_jitter;
  const vec2 inUV        = pixelCenter / vec2(gl_LaunchSizeEXT.xy);
  const vec2 d           = inUV * 2.0 - 1.0;

  const vec4 origin    = frameInfo.viewInv * vec4(0.0F, 0.0F, 0.0F, 1.0F);
  const vec4 target    = frameInfo.projInv * vec4(d.x, d.y, 0.01F, 1.0F);
  const vec4 direction = frameInfo.viewInv * vec4(normalize(target.xyz), 0.0F);

  Ray ray = Ray(origin.xyz, direction.xyz);

  vec3 radiance = pathTrace(ray, seed);

  // Removing fireflies
  float lum = dot(radiance, vec3(0.212671F, 0.715160F, 0.072169F));
  if(lum > pc.fireflyClampThreshold)
  {
    radiance *= pc.fireflyClampThreshold / lum;
  }

  return radiance;
}


void main()
{
  uint64_t start = clockRealtimeEXT();  // Debug - Heatmap

  // Initialize the random number
  uint seed = xxhash32(uvec3(gl_LaunchIDEXT.xy, pc.frame));

  // Sampling n times the pixel
  vec3 pixel_color = vec3(0.0F, 0.0F, 0.0F);
  for(int s = 0; s < pc.maxSamples; s++)
  {
    pixel_color += samplePixel(seed);
  }
  pixel_color /= pc.maxSamples;

  bool first_frame = (pc.frame == 0);

  // Debug - Heatmap
  //if(pc.heatmap == 1)
  {
    uint64_t end      = clockRealtimeEXT();
    uint     duration = uint(end - start);

    // log maximum of current frame
    uint statsIndex = uint(pc.frame) & 1;
    atomicMax(heatStats.maxDuration[statsIndex], duration);
    // use previous frame's maximum
    uint maxDuration = heatStats.maxDuration[statsIndex ^ 1];

    // lower ceiling to see some more red ;)
    float high = float(maxDuration) * 0.50;

    float val        = clamp(float(duration) / high, 0.0F, 1.0F);
    vec3  heat_color = temperature(val);

    // Wrap & SM visualization
    // heatcolor = temperature(float(gl_SMIDNV) / float(gl_SMCountNV - 1)) * float(gl_WarpIDNV) / float(gl_WarpsPerSMNV - 1);

    imageStore(heatmap, ivec2(gl_LaunchIDEXT.xy), vec4(heat_color, 1.0F));
  }

  // Saving result
  if(first_frame)
  {  // First frame, replace the value in the buffer
    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(pixel_color, 1.0F));
  }
  else
  {  // Do accumulation over time
    float a         = 1.0F / float(pc.frame + 1);
    vec3  old_color = imageLoad(image, ivec2(gl_LaunchIDEXT.xy)).xyz;
    imageStore(image, ivec2(gl_LaunchIDEXT.xy), vec4(mix(old_color, pixel_color, a), 1.0F));
  }
}
