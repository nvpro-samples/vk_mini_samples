/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "dh_bindings.h"
#include "shaderio.h"


#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/constants.h.slang"
#include "nvshaders/sky_functions.h.slang"

#define MISS_DEPTH 1000

// clang-format off
// Bindings
[[vk::push_constant]]         ConstantBuffer<PushConstant> pushConst;
[[vk::binding(B_tlas)]]       RaytracingAccelerationStructure topLevelAS;
[[vk::binding(B_outImage)]]   RWTexture2D<float4> outImage;
[[vk::binding(B_frameInfo)]]  ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(B_skyParam)]]   ConstantBuffer<SkySimpleParameters> skyInfo;
[[vk::binding(B_materials)]]  StructuredBuffer<float4> materials;
[[vk::binding(B_instances)]]  StructuredBuffer<InstanceInfo> instanceInfo;
[[vk::binding(B_vertex)]]     StructuredBuffer<Vertex> vertices[];
[[vk::binding(B_index)]]      StructuredBuffer<uint3> indices[];
// clang-format on

//-----------------------------------------------------------------------
// Payload
// See: https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#example
struct HitPayload
{
  float3 color;
  float  weight;
  int    depth;
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
  const float3 pos0     = v0.position.xyz;
  const float3 pos1     = v1.position.xyz;
  const float3 pos2     = v2.position.xyz;
  const float3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos               = float3(mul(float4(position, 1.0), ObjectToWorld4x3()));

  // Normal
  const float3 nrm0           = v0.normal.xyz;
  const float3 nrm1           = v1.normal.xyz;
  const float3 nrm2           = v2.normal.xyz;
  const float3 normal         = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  float3       worldNormal    = normalize(mul(WorldToObject4x3(), normal).xyz);
  const float3 geoNormal      = normalize(cross(pos1 - pos0, pos2 - pos0));
  float3       worldGeoNormal = normalize(mul(WorldToObject4x3(), geoNormal).xyz);
  hit.geonrm                  = worldGeoNormal;
  hit.nrm                     = worldNormal;

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
  ray.Origin    = P;
  ray.Direction = L;
  ray.TMin      = 0.001;
  ray.TMax      = INFINITE;

  HitPayload payload;
  payload.depth = 0;
  // -Wpayload-access-perf
  payload.color  = float3(0, 0, 0);
  payload.weight = 1;

  TraceRay(topLevelAS, rayFlags, 0xFF, 0, 0, 0, ray, payload);
  bool visible = (payload.depth == MISS_DEPTH);

  // -Wpayload-access-perf
  float3 color  = payload.color;
  float  weight = payload.weight;

  return visible;
}

float3 wireFrame(float3 color, float3 barycentrics)
{
  const float  thickness  = 0.002F * pushConst.numBaseTriangles;
  const float  smoothness = 0.002F;
  const float3 bary       = frac(barycentrics * pushConst.numBaseTriangles + (thickness * 0.5F));
  float        minBary    = min(bary.x, min(bary.y, bary.z));
  const float3 wire_color = float3(0.3F, 0.3F, 0.3F);
  minBary                 = smoothstep(thickness, thickness + smoothness, minBary);
  return lerp(wire_color, color, minBary);
}

float3 ggxEvaluate(float3 V, float3 L, PbrMaterial mat)
{
  BsdfEvaluateData data;
  data.k1 = V;
  data.k2 = L;

  bsdfEvaluateSimple(data, mat);

  return data.bsdf_glossy + data.bsdf_diffuse;
}


//-----------------------------------------------------------------------
// RAY GENERATION
//-----------------------------------------------------------------------
[shader("raygeneration")]
void rgenMain()
{
  float2 launchID   = (float2)DispatchRaysIndex().xy;
  float2 launchSize = (float2)DispatchRaysDimensions().xy;

  const float2 pixelCenter = launchID;
  const float2 inUV        = pixelCenter / launchSize;
  const float2 d           = inUV * 2.0 - 1.0;
  const float4 target      = mul(float4(d.x, d.y, 0.01, 1.0), frameInfo.projInv);
  const uint   rayFlags    = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;

  RayDesc ray;
  ray.Origin    = mul(float4(0.0, 0.0, 0.0, 1.0), frameInfo.viewInv).xyz;
  ray.Direction = mul(float4(normalize(target.xyz), 0.0), frameInfo.viewInv).xyz;
  ray.TMin      = 0.001;
  ray.TMax      = INFINITE;

  // Initial state
  HitPayload payload;
  payload.color  = float3(0, 0, 0);
  payload.weight = 1;
  payload.depth  = 0;

  TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, payload);
  float3 color = payload.color;

  // -Wpayload-access-perf
  float weight = payload.weight;
  int   depth  = payload.depth;

  outImage[int2(launchID)] = float4(color, 1.0);
}


//-----------------------------------------------------------------------
// CLOSEST HIT
//-----------------------------------------------------------------------
[shader("closesthit")]
void rchitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
  float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
  uint   instanceID   = InstanceIndex();
  uint   meshID       = InstanceID();
  uint   primitiveID  = PrimitiveIndex();

  // Rays
  float3 D = normalize(WorldRayDirection());   // Incoming direction
  float3 V = -D;                               // Vector to origin of ray: view direction
  float3 L = normalize(skyInfo.sunDirection);  // Vector to the light

  // Material info
  InstanceInfo iInfo  = instanceInfo[instanceID];
  float3       albedo = materials[iInfo.materialID].xyz;

  // Hit information: position, normal, geo normal
  HitState hitState = getHitState(meshID, primitiveID, barycentrics);

  // Send a ray toward light and return true if there was no hit between P and L
  bool visible = shadowRay(hitState.pos, L);

  // Color at hit point
  PbrMaterial pbrMat = defaultPbrMaterial(albedo, pushConst.metallic, pushConst.roughness, hitState.nrm, hitState.geonrm);
  float3 color = ggxEvaluate(V, L, pbrMat);

  if(pushConst.numBaseTriangles > 0)
  {
    color = wireFrame(color, barycentrics);
  }

  // Fake: lower the contribution if the hit wasn't visible
  if(!visible)
    color *= 0.3F;

  // Add contribution
  payload.color += color * pushConst.intensity * payload.weight;

  // Recursive bounce
  payload.depth += 1;
  payload.weight *= pushConst.metallic;  // more or less reflective

  // Reflection
  float3 refl_dir = reflect(D, hitState.nrm);

  // We hit our max depth
  if(payload.depth >= pushConst.maxDepth)
    return;

  // Trace bouncing ray
  RayDesc ray;
  ray.Origin    = hitState.pos;
  ray.Direction = refl_dir;
  ray.TMin      = 0.001;
  ray.TMax      = INFINITE;
  TraceRay(topLevelAS, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 0, 0, ray, payload);

  // -Wpayload-access-perf
  color        = payload.color;
  float weight = payload.weight;
  int   depth  = payload.depth;
}


//-----------------------------------------------------------------------
// MISS
//-----------------------------------------------------------------------
[shader("miss")]
void rmissMain(inout HitPayload payload)
{
  float3 sky_color = evalSimpleSky(skyInfo, WorldRayDirection());
  payload.color += sky_color * payload.weight;
  payload.depth = MISS_DEPTH;  // Stop
}

//-----------------------------------------------------------------------
// ANY HIT
//-----------------------------------------------------------------------
[shader("anyhit")]
void rahitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
  // Find where the ray hit
  float3 pos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();

  // Cut out the plane if outside the radius
  if((pushConst.useAnyhit == 1) && (length(pos) > pushConst.radius))
  {
    IgnoreHit();
    return;
  }

  // To show that AnyHit shader was invoked, we are tinted the color
  payload.color  = float3(1.0F, 0.0F, 0.0F);
  payload.weight = 0.5F;
}


