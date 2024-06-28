/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#rayquery

#version 460

#include "common/shaders/constants.hlsli"
#include "common/shaders/ggx.hlsli"
#include "common/shaders/random.hlsli"
#include "common/shaders/sky.hlsli"
#include "device_host.h"
#include "dh_bindings.h"

#define WORKGROUP_SIZE 16

struct PrimMeshInfoSlang
{
    Vertex* vertexAddress;
    uint3* indexAddress;
};
    
struct SceneDescriptionSlang
{
    Material* materialAddress;
    InstanceInfo* instInfoAddress;
    PrimMeshInfoSlang* primInfoAddress;
    Light light;
};

// Bindings
[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;
[[vk::binding(B_tlas)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(B_outImage)]] RWTexture2D<float4> outImage;
[[vk::binding(B_cameraInfo)]] ConstantBuffer<CameraInfo> cameraInfo;
[[vk::binding(B_sceneDesc)]] ConstantBuffer<SceneDescriptionSlang> sceneDesc;


//-----------------------------------------------------------------------
// Payload 
// See: https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#example
struct HitPayload
{
  float hitT;
  int instanceIndex;
  float3 pos;
  float3 nrm;
  float3 geonrm;
};

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  float3 pos;
  float3 nrm;
  float3 geonrm;
};

//-----------------------------------------------------------------------
// Return hit position, normal and geometric normal in world space
HitState getHitState(float2 barycentricCoords, float3x4 worldToObject3x4, float3x4 objectToWorld3x4, int meshID, int primitiveID, float3 worldRayDirection)
{
  HitState hit;

  // Barycentric coordinate on the triangle
  const float3 barycentrics = float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);
 
  PrimMeshInfoSlang pimInfo = sceneDesc.primInfoAddress[meshID];
  uint3 triangleIndex = pimInfo.indexAddress[primitiveID];
  
  // Vertex and indices of the primitive
  Vertex v0 = pimInfo.vertexAddress[triangleIndex.x];
  Vertex v1 = pimInfo.vertexAddress[triangleIndex.y];
  Vertex v2 = pimInfo.vertexAddress[triangleIndex.z];

  // Position
  const float3 pos0 = v0.position.xyz;
  const float3 pos1 = v1.position.xyz;
  const float3 pos2 = v2.position.xyz;
  const float3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos = float3(mul(objectToWorld3x4, float4(position, 1.0)));

  // Normal
  const float3 nrm0 = v0.normal.xyz;
  const float3 nrm1 = v1.normal.xyz;
  const float3 nrm2 = v2.normal.xyz;
  const float3 normal = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  float3 worldNormal = normalize(mul(normal, worldToObject3x4).xyz);
  const float3 geoNormal = normalize(cross(pos1 - pos0, pos2 - pos0));
  float3 worldGeoNormal = normalize(mul(geoNormal, worldToObject3x4).xyz);
  hit.geonrm = worldGeoNormal;
  hit.nrm = worldNormal;

  // ** Adjusting normal **

  // Flip if back facing
  if (dot(hit.geonrm, -worldRayDirection) < 0)
      hit.geonrm = -hit.geonrm;

  // Make Normal and GeoNormal on the same side
  if (dot(hit.geonrm, hit.nrm) < 0)
  {
      hit.nrm = -hit.nrm;
  }

  // For low tessalated, avoid internal reflection
  vec3 r = reflect(normalize(worldRayDirection), hit.nrm);
  if (dot(r, hit.geonrm) < 0)
      hit.nrm = hit.geonrm;
  
  return hit;
}


//-----------------------------------------------------------------------
// Shoot a ray an return the information of the closest hit, in the
// PtPayload structure (PRD)
//
void traceRay(RayDesc ray, inout HitPayload payload)
{
  RayQuery < RAY_FLAG_NONE > q;
  q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);
  while(q.Proceed())
  {
    if(q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
      q.CommitNonOpaqueTriangleHit(); // forcing to be opaque
  }
  
  if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
  {
    float2 barycentricCoords = q.CommittedTriangleBarycentrics();
    int meshID = q.CommittedInstanceID(); // rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    int primitiveID = q.CommittedPrimitiveIndex(); // rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    float3x4 worldToObject = q.CommittedWorldToObject3x4();
    float3x4 objectToWorld = q.CommittedObjectToWorld3x4();
    float hitT = q.CommittedRayT();
    int instanceIndex = q.CommittedInstanceIndex(); //rayQueryGetIntersectionInstanceIdEXT(rayQuery, true);

    HitState hit = getHitState(barycentricCoords, worldToObject, objectToWorld, meshID, primitiveID, ray.Direction);

    payload.hitT = hitT;
    payload.pos = hit.pos;
    payload.nrm = hit.nrm;
    payload.geonrm = hit.geonrm;
    payload.instanceIndex = instanceIndex;
  }
  else
  {
    payload.hitT = INFINITE;
  }
}

//-----------------------------------------------------------------------
// Shadow ray - return true if a ray hits anything
//
bool traceShadow(RayDesc ray)
{
  RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
  q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);
  while (q.Proceed())
  {
      if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
          q.CommitNonOpaqueTriangleHit(); // forcing to be opaque
  }
  return (q.CommittedStatus() != COMMITTED_NOTHING);
}

float3 getRandomPosition(float3 position, float radius, float2 randomValues)
{
  float angle = randomValues.x * 2.0 * 3.14159;
  float distance = sqrt(randomValues.y) * radius;

  float2 offset = float2(cos(angle), sin(angle)) * distance;
  float3 newPosition = float3(offset.x, 0, offset.y);

  return position + newPosition;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float3 pathTrace(RayDesc ray, inout uint seed)
{
  float3 radiance = float3(0.0F, 0.0F, 0.0F);
  float3 throughput = float3(1.0F, 1.0F, 1.0F);
  bool isInside = false;

  HitPayload  payload;
  for(int depth = 0; depth < pushConst.maxDepth; depth++)
  {
    traceRay(ray, payload);

    // Hitting the environment, then exit
    if(payload.hitT == INFINITE)
    {
      float3 sky_color = float3(0.1, 0.1, 0.15); // Light blue grey
      return radiance + (sky_color * throughput);
    }

    // Retrieve the Instance buffer information
    InstanceInfo instInfo = sceneDesc.instInfoAddress[payload.instanceIndex];

    float3 lightPos = getRandomPosition(pushConst.light.position, pushConst.light.radius, float2(rand(seed), rand(seed)));
    float distanceToLight = length(lightPos - payload.pos);
        
    float pdf = 0.0F;
    float3 V = -ray.Direction;
    float3 L = normalize(lightPos - payload.pos);
   
    // Retrieve the material color
    Material mat = sceneDesc.materialAddress[instInfo.materialID];
    

    // Setting up the material
    PbrMaterial pbrMat;
    pbrMat.albedo = float4(mat.albedo, 1);
    pbrMat.roughness = mat.roughness;
    pbrMat.metallic =  mat.metallic;
    pbrMat.normal = payload.nrm;
    pbrMat.emissive = float3(0.0F, 0.0F, 0.0F);
    pbrMat.f0 = lerp(float3(0.04F, 0.04F, 0.04F), pbrMat.albedo.xyz, pbrMat.metallic);
    pbrMat.f90 = vec3(1, 1, 1);
    pbrMat.specularWeight = 1.0;                  // product of specularFactor and specularTexture.a
    pbrMat.transmissionFactor = mat.transmission; // KHR_materials_transmission
    pbrMat.thicknessFactor = 0;

    float matIor = 1.1;

    if (isInside)
        pbrMat.eta = matIor / 1.0;
    else
        pbrMat.eta = 1.0 / matIor;
    float3 contrib = float3(0, 0, 0);

    // Evaluation of direct light (sun)
    bool nextEventValid = (dot(L, payload.nrm) > 0.0f);
    if(nextEventValid)
    {
      BsdfEvaluateData evalData;
      evalData.k1 = -ray.Direction;
      evalData.k2 = L;
      bsdfEvaluate(evalData, pbrMat);

      const float3 w = sceneDesc.light.intensity.xxx * 1.0/(distanceToLight*distanceToLight);
      contrib += w * evalData.bsdf_diffuse;
      contrib += w * evalData.bsdf_glossy;
      contrib *= throughput;
    }


    // Sample BSDF
    {
      BsdfSampleData sampleData;
      sampleData.k1 = -ray.Direction; // outgoing direction
      sampleData.xi = float4(rand(seed), rand(seed), rand(seed), rand(seed));

      bsdfSample(sampleData, pbrMat);
      if(sampleData.event_type == BSDF_EVENT_ABSORB)
      {
        break; // Need to add the contribution ?
      }

      if (sampleData.event_type == BSDF_EVENT_TRANSMISSION)
          isInside = !isInside;

      throughput *= sampleData.bsdf_over_pdf;
      vec3 offsetDir = dot(sampleData.k2, payload.geonrm) < 0.0 ? -payload.geonrm : payload.geonrm;
      ray.Origin = offsetRay(payload.pos, offsetDir);
      ray.Direction = sampleData.k2;
    }


    // Russian-Roulette (minimizing live state)
    float rrPcont = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001F, 0.95F);
    if(rand(seed) >= rrPcont)
      break; // paths with low throughput that won't contribute
    throughput /= rrPcont; // boost the energy of the non-terminated paths

    // We are adding the contribution to the radiance only if the ray is not occluded by an object.
    if(nextEventValid)
    {
      RayDesc shadowRay;
      shadowRay.Origin = ray.Origin;
      shadowRay.Direction = L;
      shadowRay.TMin = 0.01;
      shadowRay.TMax = distanceToLight;
      bool inShadow = traceShadow(shadowRay);
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
float3 samplePixel(inout uint seed, float2 launchID, float2 launchSize)
{
  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  const float2 subpixel_jitter = pushConst.frame == 0 ? float2(0.5f, 0.5f) : float2(rand(seed), rand(seed));
  const float2 clipCoords = (launchID + subpixel_jitter) / launchSize * 2.0 - 1.0;
  float4 viewCoords = mul(cameraInfo.projInv, float4(clipCoords, -1.0, 1.0)); 
  viewCoords /= viewCoords.w; 

  RayDesc ray;
  ray.Origin = float3(cameraInfo.viewInv[0].w, cameraInfo.viewInv[1].w, cameraInfo.viewInv[2].w);
  ray.Direction = normalize(mul(cameraInfo.viewInv, viewCoords).xyz - ray.Origin);
  ray.TMin = 0.001;
  ray.TMax = INFINITE;

  float3 radiance = pathTrace(ray, seed);

  // Removing fireflies
  float lum = dot(radiance, float3(0.212671F, 0.715160F, 0.072169F));
  if(lum > pushConst.fireflyClampThreshold)
  {
    radiance *= pushConst.fireflyClampThreshold / lum;
  }

  return radiance;
}


//-----------------------------------------------------------------------
// RAY GENERATION
//-----------------------------------------------------------------------
[shader("compute")]
[numthreads(WORKGROUP_SIZE, WORKGROUP_SIZE, 1)]
void computeMain(uint3 threadIdx : SV_DispatchThreadID)
{
  float2 launchID = (float2)threadIdx.xy;
  
  uint2 imgSize;
  outImage.GetDimensions(imgSize.x, imgSize.y); //DispatchRaysDimensions();
  float2 launchSize = imgSize;

  if(launchID.x >= launchSize.x || launchID.y >= launchSize.y)
    return;
  
  // Initialize the random number
  uint seed = xxhash32(uint3(uint2(launchID.xy), pushConst.frame));

  // Sampling n times the pixel
  float3 pixel_color = float3(0.0F, 0.0F, 0.0F);
  for(int s = 0; s < pushConst.maxSamples; s++)
  {
    pixel_color += samplePixel(seed, launchID, launchSize);
  }
  pixel_color /= pushConst.maxSamples;

  bool first_frame = (pushConst.frame == 0);

  // Saving result
  if(first_frame)
  { // First frame, replace the value in the buffer
    outImage[int2(launchID)] = float4(pixel_color, 1.0);
  }
  else
  { // Do accumulation over time
    float a = 1.0F / float(pushConst.frame + 1);
    float3 old_color = outImage[int2(launchID)].xyz;
    outImage[int2(launchID)] = float4(lerp(old_color, pixel_color, a), 1.0F);
  }
}

