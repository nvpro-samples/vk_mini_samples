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

#version 460

#include "common/shaders/constants.hlsli"
#include "common/shaders/ggx.hlsli"
#include "common/shaders/random.hlsli"
#include "common/shaders/sky.hlsli"
#include "device_host.h"
#include "dh_bindings.h"

#define MISS_DEPTH 1000

// Bindings
[[vk::push_constant]] ConstantBuffer<PushConstant> pushConst;
[[vk::binding(B_tlas)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(B_outImage)]] RWTexture2D<float4> outImage;
[[vk::binding(B_frameInfo)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(B_materials)]] StructuredBuffer<float4> materials;
[[vk::binding(B_instances)]] StructuredBuffer<InstanceInfo> instanceInfo;
[[vk::binding(B_vertex)]] StructuredBuffer<Vertex> vertices[];
[[vk::binding(B_index)]] StructuredBuffer<uint3> indices[];
[[vk::binding(B_triMat)]] StructuredBuffer<int> triMat[];

//-----------------------------------------------------------------------
// Payload 
// See: https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#example
struct HitPayload
{
  float3 color;
  int depth;
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
HitState getHitState(int meshID, int primitiveID, float3 barycentrics)
{
  HitState hit;

  // Getting the 3 indices of the triangle (local)
  uint3 triangleIndex = indices[NonUniformResourceIndex(meshID)][primitiveID];

  // All vertex attributes of the triangle.
  Vertex v0 = vertices[NonUniformResourceIndex(meshID)][triangleIndex.x];
  Vertex v1 = vertices[NonUniformResourceIndex(meshID)][triangleIndex.y];
  Vertex v2 = vertices[NonUniformResourceIndex(meshID)][triangleIndex.z];

  // Position
  const float3 pos0 = v0.position.xyz;
  const float3 pos1 = v1.position.xyz;
  const float3 pos2 = v2.position.xyz;
  const float3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos = float3(mul(ObjectToWorld3x4(), float4(position, 1.0)));

  // Normal
  const float3 nrm0 = v0.normal.xyz;
  const float3 nrm1 = v1.normal.xyz;
  const float3 nrm2 = v2.normal.xyz;
  const float3 normal = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  float3 worldNormal = normalize(mul(normal, WorldToObject3x4()).xyz);
  const float3 geoNormal = normalize(cross(pos1 - pos0, pos2 - pos0));
  float3 worldGeoNormal = normalize(mul(geoNormal, WorldToObject3x4()).xyz);
  hit.geonrm = worldGeoNormal;
  hit.nrm = worldNormal;

  return hit;
}

//-----------------------------------------------------------------------
// Return TRUE if there is no occluder, 
// meaning that the light is visible from P toward L
bool shadowRay(float3 P, float3 L)
{
  // https://learn.microsoft.com/en-us/windows/win32/direct3d12/ray_flag
  const uint rayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE;
  
  RayDesc ray;
  ray.Origin = P;
  ray.Direction = L;
  ray.TMin = 0.001;
  ray.TMax = INFINITE;
  
  HitPayload payload;
  payload.depth = 0;
  payload.color = float3(0, 0, 0);

  TraceRay(topLevelAS, rayFlags, 0xFF, 0, 0, 0, ray, payload);
  bool visible = (payload.depth == MISS_DEPTH);

  float3 color = payload.color;

  return visible;
}

// http://filmicworlds.com/blog/filmic-tonemapping-operators/
float3 tonemapFilmic(float3 color)
{
  float3 temp = max(float3(0.0F), color - float3(0.004F));
  float3 result = (temp * (float3(6.2F) * temp + float3(0.5F))) / (temp * (float3(6.2F) * temp + float3(1.7F)) + float3(0.06F));
  return result;
}


//-----------------------------------------------------------------------
// RAY GENERATION
//-----------------------------------------------------------------------
[shader("raygeneration")]
void rgenMain()
{
  float2 launchID = (float2)DispatchRaysIndex().xy;
  float2 launchSize = (float2)DispatchRaysDimensions().xy;

  const float2 pixelCenter = launchID;
  const float2 inUV = pixelCenter / launchSize;
  const float2 d = inUV * 2.0 - 1.0;
  const float4 target = mul(frameInfo.projInv, float4(d.x, d.y, 0.01, 1.0));
  const uint rayFlags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;

  RayDesc ray;
  ray.Origin = mul(frameInfo.viewInv, float4(0.0, 0.0, 0.0, 1.0)).xyz;
  ray.Direction = mul(frameInfo.viewInv, float4(normalize(target.xyz), 0.0)).xyz;
  ray.TMin = 0.001;
  ray.TMax = INFINITE;

  // Initial state
  HitPayload payload;
  payload.color = float3(0, 0, 0);
  payload.depth = 0;

  // Initialize the random number
  uint seed = xxhash32(uint3(DispatchRaysIndex().xy, 0));

  float time;
  float3 result = float3(0, 0, 0);
  for (int i = 0; i < pushConst.numSamples; i++)
  {
    time = rand(seed);
    TraceMotionRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, time, payload);
    result += payload.color;
  }

  float3 color = tonemapFilmic(result / pushConst.numSamples);

  outImage[int2(launchID)] = float4(color, 1.0);
}


//-----------------------------------------------------------------------
// CLOSEST HIT
//-----------------------------------------------------------------------
[shader("closesthit")]
void rchitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
  float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
  uint instanceID = InstanceIndex();
  uint meshID = InstanceID();
  uint primitiveID = PrimitiveIndex();

  // Rays
  float3 P = WorldRayOrigin() + RayTCurrent() * WorldRayDirection();
  float3 D = normalize(WorldRayDirection()); // Incoming direction
  float3 V = -D; // Vector to origin of ray: view direction

  float3 L; // Vector to the light
  float lightIntensity = pushConst.lightIntensity;
  float lightDistance = 100000.0;
  {
      float3 lDir = pushConst.lightPosition - P;
      lightDistance = length(lDir);
      lightIntensity = lightIntensity / (lightDistance * lightDistance);
      L = normalize(lDir);
  }
  bool visible = shadowRay(P, L);

  // Material info
  int matID = triMat[meshID][primitiveID];
  float3 albedo = materials[matID].xyz;

  // Hit information: position, normal, geo normal
  HitState hitState = getHitState(meshID, primitiveID, barycentrics);

  // Color at hit point
  vec3 color;
  {
      // Lambertian
      float dotNL = max(dot(hitState.nrm, L), 0.0);
      color = albedo * dotNL;
  }

  // Fake: lower the contribution if the hit wasn't visible
  if(!visible)
    color *= 0.3F;

  // Add contribution
  payload.color = color * lightIntensity;
  
}


//-----------------------------------------------------------------------
// MISS 
//-----------------------------------------------------------------------
[shader("miss")]
void rmissMain(inout HitPayload payload)
{
  payload.color += pushConst.clearColor.xyz;
  payload.depth = MISS_DEPTH; // Stop
}

