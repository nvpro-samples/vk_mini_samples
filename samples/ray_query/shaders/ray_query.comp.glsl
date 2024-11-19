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
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_buffer_reference2 : require

#extension GL_NV_shader_invocation_reorder : enable


const int GROUP_SIZE = 16;
layout(local_size_x = GROUP_SIZE, local_size_y = GROUP_SIZE) in;

#include "device_host.h"
#include "dh_bindings.h"
#include "nvvkhl/shaders/random.h"
#include "nvvkhl/shaders/constants.h"
#include "nvvkhl/shaders/ggx.h"
#include "nvvkhl/shaders/ray_util.h"
#include "nvvkhl/shaders/pbr_mat_struct.h"
#include "nvvkhl/shaders/bsdf_structs.h"
#include "nvvkhl/shaders/bsdf_functions.h"

// clang-format off
layout(buffer_reference, scalar) readonly buffer Materials { Material m[]; };
layout(buffer_reference, scalar) readonly buffer InstanceInfos { InstanceInfo i[]; };
layout(buffer_reference, scalar) readonly buffer PrimMeshInfos { PrimMeshInfo i[]; };
layout(buffer_reference, scalar) readonly buffer Vertices { Vertex v[]; };
layout(buffer_reference, scalar) readonly buffer Indices { uvec3 i[]; };

layout(set = 0, binding = B_tlas) uniform accelerationStructureEXT topLevelAS;
layout(set = 0, binding = B_outImage, rgba32f) uniform image2D image;
layout(set = 0, binding = B_cameraInfo, scalar) uniform cameraInfo_ { CameraInfo cameraInfo; };
layout(set = 0, binding = B_sceneDesc, scalar) uniform SceneDesc_ { SceneInfo sceneDesc; };

// clang-format on

layout(push_constant, scalar) uniform RtxPushConstant_
{
  PushConstant pushConst;
};

struct Ray
{
  vec3 origin;
  vec3 direction;
};

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  vec3 pos;
  vec3 nrm;
  vec3 geonrm;
};

struct HitPayload
{
  float hitT;
  int   instanceIndex;
  vec3  pos;
  vec3  nrm;
  vec3  geonrm;
};

//-----------------------------------------------------------------------
// Return hit position and normal in world space
HitState getHitState(PrimMeshInfo pinfo, vec2 barycentricCoords, mat4x3 worldToObject, mat4x3 objectToWorld, int primitiveID, vec3 worldRayDirection)
{
  HitState hit;

  // Vextex and indices of the primitive
  Vertices vertices = Vertices(pinfo.vertexAddress);
  Indices  indices  = Indices(pinfo.indexAddress);

  // Barycentric coordinate on the triangle
  const vec3 barycentrics = vec3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = indices.i[primitiveID];

  // All vertex attributes of the triangle.
  Vertex v0 = vertices.v[triangleIndex.x];
  Vertex v1 = vertices.v[triangleIndex.y];
  Vertex v2 = vertices.v[triangleIndex.z];

  // Position
  const vec3 pos0     = v0.position.xyz;
  const vec3 pos1     = v1.position.xyz;
  const vec3 pos2     = v2.position.xyz;
  const vec3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos             = vec3(objectToWorld * vec4(position, 1.0));

  // Normal
  const vec3 nrm0           = v0.normal.xyz;
  const vec3 nrm1           = v1.normal.xyz;
  const vec3 nrm2           = v2.normal.xyz;
  const vec3 normal         = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  vec3       worldNormal    = normalize(vec3(normal * worldToObject));
  const vec3 geoNormal      = normalize(cross(pos1 - pos0, pos2 - pos0));
  vec3       worldGeoNormal = normalize(vec3(geoNormal * worldToObject));
  hit.geonrm                = worldGeoNormal;
  hit.nrm                   = worldNormal;

  // ** Adjusting normal **

  // Flip if back facing
  if(dot(hit.geonrm, -worldRayDirection) < 0)
    hit.geonrm = -hit.geonrm;

  // Make Normal and GeoNormal on the same side
  if(dot(hit.geonrm, hit.nrm) < 0)
  {
    hit.nrm = -hit.nrm;
  }

  // For low tessalated, avoid internal reflection
  vec3 r = reflect(normalize(worldRayDirection), hit.nrm);
  if(dot(r, hit.geonrm) < 0)
    hit.nrm = hit.geonrm;

  return hit;
}


//-----------------------------------------------------------------------
// Shoot a ray an return the information of the closest hit, in the
// PtPayload structure (PRD)
//
void traceRay(in rayQueryEXT rayQuery, Ray ray, inout HitPayload payload)
{
  payload.hitT  = 0.0F;
  uint rayFlags = gl_RayFlagsNoneEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
  rayQueryInitializeEXT(rayQuery, topLevelAS, rayFlags, 0xFF, ray.origin, 0.0, ray.direction, INFINITE);

  while(rayQueryProceedEXT(rayQuery))
  {
    if(rayQueryGetIntersectionTypeEXT(rayQuery, false) == gl_RayQueryCandidateIntersectionTriangleEXT)
    {
      rayQueryConfirmIntersectionEXT(rayQuery);
    }
  }

  if(rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT)
  {
    vec2   barycentricCoords = rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    int    meshID            = rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    int    primitiveID       = rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    mat4x3 worldToObject     = rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
    mat4x3 objectToWorld     = rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
    float  hitT              = rayQueryGetIntersectionTEXT(rayQuery, true);
    int    instanceID        = rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);
    vec3   worldRayDirection = rayQueryGetWorldRayDirectionEXT(rayQuery);

    // Retrieve the Primitive mesh buffer information
    PrimMeshInfos pInfo_ = PrimMeshInfos(sceneDesc.primInfoAddress);
    PrimMeshInfo  pinfo  = pInfo_.i[meshID];

    HitState hit = getHitState(pinfo, barycentricCoords, worldToObject, objectToWorld, primitiveID, worldRayDirection);

    payload.hitT          = hitT;
    payload.pos           = hit.pos;
    payload.nrm           = hit.nrm;
    payload.geonrm        = hit.geonrm;
    payload.instanceIndex = instanceID;
  }
  else
  {
    payload.hitT = INFINITE;
  }
}

//-----------------------------------------------------------------------
// Shadow ray - return true if a ray hits anything
//
bool traceShadow(in rayQueryEXT rayQuery, Ray ray, float maxDist)
{
  uint rayFlags = gl_RayFlagsNoneEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
  rayQueryInitializeEXT(rayQuery, topLevelAS, rayFlags, 0xFF, ray.origin, 0.0, ray.direction, maxDist);

  while(rayQueryProceedEXT(rayQuery))
  {  // Force opaque, therefore, no intersection confirmation needed
    rayQueryConfirmIntersectionEXT(rayQuery);
  }

  return (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);  // Is Hit ?
}

vec3 getRandomPosition(vec3 position, float radius, vec2 randomValues)
{
  float angle    = randomValues.x * 2.0 * 3.14159;
  float distance = sqrt(randomValues.y) * radius;

  vec2 offset      = vec2(cos(angle), sin(angle)) * distance;
  vec3 newPosition = vec3(offset.x, 0, offset.y);

  return position + newPosition;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
vec3 pathTrace(Ray ray, inout uint seed)
{
  vec3 radiance   = vec3(0.0F);
  vec3 throughput = vec3(1.0F);
  bool isInside   = false;

  Materials   materials = Materials(sceneDesc.materialAddress);
  rayQueryEXT rayQuery;
  HitPayload  payload;

  for(int depth = 0; depth < pushConst.maxDepth; depth++)
  {
    traceRay(rayQuery, ray, payload);

    // Hitting the environment, then exit
    if(payload.hitT == INFINITE)
    {
      vec3 sky_color = vec3(0.1, 0.1, 0.15);  // Light blue grey
      return radiance + (sky_color * throughput);
    }

    // Retrieve the Instance buffer information
    InstanceInfos iInfo_ = InstanceInfos(sceneDesc.instInfoAddress);
    InstanceInfo  iInfo  = iInfo_.i[payload.instanceIndex];


    vec3  lightPos = getRandomPosition(pushConst.light.position, pushConst.light.radius, vec2(rand(seed), rand(seed)));
    float distanceToLight = length(lightPos - payload.pos);

    float pdf = 0.0F;
    vec3  V   = -ray.direction;
    vec3  L   = normalize(lightPos - payload.pos);

    // Setting up the material
    Material    mat     = materials.m[iInfo.materialID];
    PbrMaterial pbrMat  = defaultPbrMaterial(mat.albedo, mat.metallic, mat.roughness, payload.nrm, payload.geonrm);
    pbrMat.transmission = mat.transmission;
    pbrMat.isThinWalled = false;

    float matIor = 1.1;

    if(isInside)
    {
      pbrMat.ior1 = matIor;
      pbrMat.ior2 = 1.0;
    }
    else
    {
      pbrMat.ior1 = 1.0;
      pbrMat.ior2 = matIor;
    }

    vec3 contrib = vec3(0);

    // Evaluation of direct light (sun)
    bool nextEventValid = (dot(L, payload.nrm) > 0.0f);
    if(nextEventValid)
    {
      BsdfEvaluateData evalData;
      evalData.k1 = -ray.direction;
      evalData.k2 = L;
      evalData.xi = vec3(rand(seed), rand(seed), rand(seed));
      bsdfEvaluate(evalData, pbrMat);

      if(evalData.pdf > 0.0)
      {
        const vec3 w = pushConst.light.intensity.xxx * 1.0 / (distanceToLight * distanceToLight);
        contrib += w * evalData.bsdf_diffuse;
        contrib += w * evalData.bsdf_glossy;
        contrib *= throughput;
      }
    }

    // Sample BSDF
    {
      BsdfSampleData sampleData;
      sampleData.k1 = -ray.direction;  // outgoing direction
      sampleData.xi = vec3(rand(seed), rand(seed), rand(seed));

      bsdfSample(sampleData, pbrMat);
      if(sampleData.event_type == BSDF_EVENT_ABSORB)
      {
        break;  // Need to add the contribution ?
      }

      if(sampleData.event_type == BSDF_EVENT_TRANSMISSION)
        isInside = !isInside;

      throughput *= sampleData.bsdf_over_pdf;
      vec3 offsetDir = dot(sampleData.k2, payload.geonrm) < 0.0 ? -payload.geonrm : payload.geonrm;
      ray.origin     = offsetRay(payload.pos, offsetDir);
      ray.direction  = sampleData.k2;
    }

    // Russian-Roulette (minimizing live state)
    float rrPcont = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001F, 0.95F);
    if(rand(seed) >= rrPcont)
      break;                // paths with low throughput that won't contribute
    throughput /= rrPcont;  // boost the energy of the non-terminated paths

    // We are adding the contribution to the radiance only if the ray is not occluded by an object.
    if(nextEventValid)
    {
      Ray  shadowRay = Ray(ray.origin, L);
      bool inShadow  = traceShadow(rayQuery, shadowRay, distanceToLight);
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
vec3 samplePixel(inout uint seed, uvec2 launchID, ivec2 launchSize)
{
  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  vec2 subpixel_jitter = pushConst.frame == 0 ? vec2(0.5f, 0.5f) : vec2(rand(seed), rand(seed));
  vec2 clipCoords      = (vec2(launchID) + subpixel_jitter) / vec2(launchSize) * 2.0 - 1.0;
  vec4 viewCoords      = cameraInfo.projInv * vec4(clipCoords, -1.0, 1.0);
  viewCoords /= viewCoords.w;

  const vec3 origin    = vec3(cameraInfo.viewInv[3].xyz);
  const vec3 direction = normalize((cameraInfo.viewInv * viewCoords).xyz - origin);

  Ray ray = Ray(origin.xyz, direction.xyz);

  vec3 radiance = pathTrace(ray, seed);

  // Removing fireflies
  float lum = dot(radiance, vec3(0.212671F, 0.715160F, 0.072169F));
  if(lum > pushConst.fireflyClampThreshold)
  {
    radiance *= pushConst.fireflyClampThreshold / lum;
  }

  return radiance;
}


void main()
{
  ivec2 LaunchSize = imageSize(image);
  uvec2 LaunchID   = gl_GlobalInvocationID.xy;

  // Check if not outside boundaries
  if(LaunchID.x >= LaunchSize.x || LaunchID.y >= LaunchSize.y)
    return;


  // Initialize the random number
  uint seed = xxhash32(uvec3(LaunchID.xy, pushConst.frame));

  // Sampling n times the pixel
  vec3 pixel_color = vec3(0.0F, 0.0F, 0.0F);
  for(int s = 0; s < pushConst.maxSamples; s++)
  {
    pixel_color += samplePixel(seed, LaunchID, LaunchSize);
  }
  pixel_color /= pushConst.maxSamples;

  bool first_frame = (pushConst.frame <= 1);
  // Saving result
  if(first_frame)
  {  // First frame, replace the value in the buffer
    imageStore(image, ivec2(LaunchID.xy), vec4(pixel_color, 1.0F));
  }
  else
  {  // Do accumulation over time
    float a         = 1.0F / float(pushConst.frame + 1);
    vec3  old_color = imageLoad(image, ivec2(LaunchID.xy)).xyz;
    imageStore(image, ivec2(LaunchID.xy), vec4(mix(old_color, pixel_color, a), 1.0F));
  }
}
