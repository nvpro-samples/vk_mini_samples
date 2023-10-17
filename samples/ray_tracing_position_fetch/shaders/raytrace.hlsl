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


#include "device_host.h"
#include "dh_bindings.h"

#include "sky.hlsli"
#include "ggx.hlsli"
#include "constants.hlsli"

#define MISS_DEPTH 1000

// Bindings
[[vk::push_constant]] ConstantBuffer<PushConstant> pushConst;
[[vk::binding(B_tlas)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(B_outImage)]] RWTexture2D<float4> outImage;
[[vk::binding(B_frameInfo)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(B_skyParam)]] ConstantBuffer<ProceduralSkyShaderParameters> skyInfo;
[[vk::binding(B_materials)]] StructuredBuffer<float4> materials;
[[vk::binding(B_instances)]] StructuredBuffer<InstanceInfo> instanceInfo;


//-----------------------------------------------------------------------
// Payload 
// See: https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#example
struct [raypayload] HitPayload
{
  float3 color : write(caller, closesthit, miss) : read(caller, closesthit, miss);
float weight : write(caller, closesthit) : read(caller, closesthit, miss);
int depth : write(caller, miss, closesthit) : read(caller, closesthit);
};


//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  float3 pos;
  float3 nrm;
  float3 geonrm;
};

// Adding capability and extension for KHR_ray_tracing_position_fetch
// * http://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/main/extensions/KHR/SPV_KHR_ray_tracing_position_fetch.html
// * https://github.com/KhronosGroup/SPIRV-Headers/blob/main/include/spirv/unified1/spirv.json

// Adding access to the vertex positions stored in the acceleration structure. 
#define BuiltIn 11
#define RayTracingPositionFetchKHR 5336
#define HitTriangleVertexPositionsKHR 5391

static float3 gl_HitTriangleVertexPositions[3];

[[vk::ext_extension("SPV_KHR_ray_tracing_position_fetch")]]
[[vk::ext_capability(RayTracingPositionFetchKHR)]] 

//-----------------------------------------------------------------------
// Return hit position, normal and geometric normal in world space
HitState getHitState(int meshID, int primitiveID, float3 barycentrics, float3 hitTriangleVertexPositions[3])
{
  HitState hit;
  
  // Position
  const float3 pos0 = hitTriangleVertexPositions[0]; // #FETCH
  const float3 pos1 = hitTriangleVertexPositions[1];
  const float3 pos2 = hitTriangleVertexPositions[2];
  const float3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos = float3(mul(ObjectToWorld3x4(), float4(position, 1.0)));

  // Normal
  const float3 geoNormal = normalize(cross(pos1 - pos0, pos2 - pos0));
  float3 worldGeoNormal = normalize(mul(geoNormal, WorldToObject3x4()).xyz);
  hit.geonrm = worldGeoNormal;
  hit.nrm = worldGeoNormal; // #FETCH

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
  // -Wpayload-access-perf
  payload.color = float3(0, 0, 0);
  payload.weight = 1;

  TraceRay(topLevelAS, rayFlags, 0xFF, 0, 0, 0, ray, payload);
  bool visible = (payload.depth == MISS_DEPTH);

  // -Wpayload-access-perf
  float3 color = payload.color;
  float weight = payload.weight;

  return visible;
}

//-----------------------------------------------------------------------
// RAY GENERATION
//-----------------------------------------------------------------------
[shader("raygeneration")]
void rgenMain()
{
  float2 launchID = (float2)DispatchRaysIndex();
  float2 launchSize = (float2)DispatchRaysDimensions();

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
  payload.weight = 1;
  payload.depth = 0;

  TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, payload);
  float3 color = payload.color;

  // -Wpayload-access-perf
  float weight = payload.weight;
  int depth = payload.depth;

  outImage[int2(launchID)] = float4(color, 1.0);
}

//-----------------------------------------------------------------------
// CLOSEST HIT
//-----------------------------------------------------------------------
struct BuiltInHitTriangleVertexPositions
{
  [[vk::ext_decorate(BuiltIn, HitTriangleVertexPositionsKHR)]] 
  float3 HitTriangleVertexPositions[3];
};

struct MyAttributes
{
  [[vk::ext_decorate(BuiltIn, HitTriangleVertexPositionsKHR)]] float3 HitTriangleVertexPositions[3];
  BuiltInTriangleIntersectionAttributes intersect;
};

[shader("closesthit")]
void rchitMain(inout HitPayload payload, in MyAttributes attr)
{

  gl_HitTriangleVertexPositions[0] = attr.HitTriangleVertexPositions[0];
  gl_HitTriangleVertexPositions[1] = attr.HitTriangleVertexPositions[1];
  gl_HitTriangleVertexPositions[2] = attr.HitTriangleVertexPositions[2];
  
  //BuiltInTriangleIntersectionAttributes attr;
  //initializeArr();
  float2 bary = attr.intersect.barycentrics;
  float3 barycentrics = float3(1 - bary.x - bary.y, bary.x, bary.y);
  uint instanceID = InstanceIndex();
  uint meshID = InstanceID();
  uint primitiveID = PrimitiveIndex();

  // Rays
  float3 D = normalize(WorldRayDirection()); // Incoming direction
  float3 V = -D; // Vector to origin of ray: view direction
  float3 L = normalize(skyInfo.directionToLight); // Vector to the light

  // Material info
  InstanceInfo iInfo = instanceInfo[instanceID];
  float3 albedo = materials[iInfo.materialID].xyz;
  
  // Hit information: position, normal, geo normal
  HitState hitState = getHitState(meshID, primitiveID, barycentrics, gl_HitTriangleVertexPositions);

  //payload.color = albedo;
  //payload.color = frac(hitState.pos);
  //payload.color = hitState.nrm;
  payload.color = frac(gl_HitTriangleVertexPositions[0]);
  return;
  
  // Send a ray toward light and return true if there was no hit between P and L
  bool visible = shadowRay(hitState.pos, L);

  // Color at hit point
  float3 color = ggxEvaluate(V, hitState.nrm, L, albedo, pushConst.metallic, pushConst.roughness);

  // Fake: lower the contribution if the hit wasn't visible
  if(!visible)
    color *= 0.3F;

  // Add contribution
  payload.color += color * pushConst.intensity * payload.weight;

  // Recursive bounce
  payload.depth += 1;
  payload.weight *= pushConst.metallic; // more or less reflective

  // Reflection
  float3 refl_dir = reflect(D, hitState.nrm);

  // We hit our max depth
  if(payload.depth >= pushConst.maxDepth)
    return;

  // Trace bouncing ray
  RayDesc ray;
  ray.Origin = hitState.pos;
  ray.Direction = refl_dir;
  ray.TMin = 0.001;
  ray.TMax = INFINITE;
  TraceRay(topLevelAS, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 0, 0, ray, payload);

  // -Wpayload-access-perf
  color = payload.color;
  float weight = payload.weight;
  int depth = payload.depth;
}


//-----------------------------------------------------------------------
// MISS 
//-----------------------------------------------------------------------
[shader("miss")]
void rmissMain(inout HitPayload payload)
{
  float3 sky_color = proceduralSky(skyInfo, WorldRayDirection(), 0);
  payload.color += sky_color * payload.weight;
  payload.depth = MISS_DEPTH; // Stop
}

