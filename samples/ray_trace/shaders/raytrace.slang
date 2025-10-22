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

#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/pbr_material_eval.h.slang"
#include "nvshaders/sky_functions.h.slang"
#include "nvshaders/tonemap_functions.h.slang"
#include "nvshaders/ray_utils.h.slang"

#include "shaderio_rt_gltf.h"

#define MISS_DEPTH 1000
#define INFINITE 1e32

// Bindings
// clang-format off
[[vk::push_constant]]                       ConstantBuffer<RtGltfPushConstant> pushConst;
[[vk::binding(BindingIndex::eTlas)]]        RaytracingAccelerationStructure topLevelAS;
[[vk::binding(BindingIndex::eOutImage)]]    RWTexture2D<float4> outImage;
[[vk::binding(BindingIndex::eSceneDesc)]]   ConstantBuffer<RtGltfSceneInfo> sceneInfo;
// clang-format on

// Raytracing Payload
struct HitPayload
{
  float3 color;
  float  weight;
  int    depth;
};

// Hit state information
struct HitState
{
  float3 pos;
  float3 nrm;
  float3 geonrm;
};

int3 getTriangleIndices(uint64_t dataBufferAddress, const Ptr<TriangleMesh> mesh, int primitiveID)
{
  if(mesh->indices.byteStride == sizeof(int16_t))
  {
    int16_t3* indices = (int16_t3*)(dataBufferAddress + mesh->indices.offset);
    return indices[primitiveID];
  }
  else if(mesh->indices.byteStride == sizeof(int32_t))
  {
    int3* indices = (int3*)(dataBufferAddress + mesh->indices.offset);
    return indices[primitiveID];
  }

  return int3(-1);  // Error case
}

float3 getAttribFloat3(uint64_t dataBufferAddress, uint64_t offset, int vertexID)
{
  float3* attribs = (float3*)(dataBufferAddress + offset);
  return attribs[vertexID];
}

float3 getVertexPosition(uint64_t dataBufferAddress, const Ptr<TriangleMesh> mesh, int vertexID)
{
  return getAttribFloat3(dataBufferAddress, mesh->positions.offset, vertexID);
}

float3 getVertexNormal(uint64_t dataBufferAddress, const Ptr<TriangleMesh> mesh, int vertexID)
{
  return getAttribFloat3(dataBufferAddress, mesh->normals.offset, vertexID);
}

float3 getVertexNormalInterpolated(uint64_t dataBufferAddress, const Ptr<TriangleMesh> mesh, int3 vertexID, float3 baryWeights)
{
  float3* normals = (float3*)(dataBufferAddress + mesh->normals.offset);
  float3  n0      = normals[vertexID.x];
  float3  n1      = normals[vertexID.y];
  float3  n2      = normals[vertexID.z];
  // Object space normal
  return normalize(n0 * baryWeights.x + n1 * baryWeights.y + n2 * baryWeights.z);
}

//-----------------------------------------------------------------------
// Return hit position, normal and geometric normal in world space
HitState getHitState(RtMeshInfo meshInfo, int primitiveID, float3 barycentrics)
{
  HitState hit = {};

  uint3 triangleIndex = getTriangleIndices(meshInfo.baseBufferAddress, meshInfo.mesh, primitiveID);

  //// Position
  float3       pos0     = getVertexPosition(meshInfo.baseBufferAddress, meshInfo.mesh, triangleIndex.x);
  float3       pos1     = getVertexPosition(meshInfo.baseBufferAddress, meshInfo.mesh, triangleIndex.y);
  float3       pos2     = getVertexPosition(meshInfo.baseBufferAddress, meshInfo.mesh, triangleIndex.z);
  const float3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos               = float3(mul(float4(position, 1.0), ObjectToWorld4x3()));

  //// Normal
  float3 normal = getVertexNormalInterpolated(meshInfo.baseBufferAddress, meshInfo.mesh, triangleIndex, barycentrics);
  float3 worldNormal          = normalize(mul(WorldToObject4x3(), normal).xyz);
  const float3 geoNormal      = normalize(cross(pos1 - pos0, pos2 - pos0));
  float3       worldGeoNormal = normalize(mul(WorldToObject4x3(), geoNormal).xyz);
  hit.geonrm                  = worldGeoNormal;
  hit.nrm                     = worldNormal;

  return hit;
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
// Return TRUE if there is no occluder,
// meaning that the light is visible from P toward L
bool shadowRay(float3 P, float3 L)
{
  // https://learn.microsoft.com/en-us/windows/win32/direct3d12/ray_flag
  const uint rayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_FORCE_OPAQUE;

  RayDesc ray;
  ray.Origin    = offsetRay(P, L);
  ray.Direction = L;
  ray.TMin      = 0.00;
  ray.TMax      = INFINITE;

  HitPayload payload;
  payload.depth  = 0;
  payload.color  = float3(0, 0, 0);
  payload.weight = 1;

  TraceRay(topLevelAS, rayFlags, 0xFF, 0, 0, 0, ray, payload);
  bool visible = (payload.depth == MISS_DEPTH);

  return visible;
}

//-----------------------------------------------------------------------
// RAY GENERATION
//-----------------------------------------------------------------------
[shader("raygeneration")]
void rgenMain()
{
  float2 launchID   = (float2)DispatchRaysIndex().xy;
  float2 launchSize = (float2)DispatchRaysDimensions().xy;

  const uint   rayFlags   = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
  const float2 clipCoords = launchID / launchSize * 2.0 - 1.0;
  const float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);

  RayDesc ray;
  ray.Origin    = mul(float4(0.0, 0.0, 0.0, 1.0), sceneInfo.viewInvMatrix).xyz;
  ray.Direction = mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz;

  ray.TMin = 0.001;
  ray.TMax = INFINITE;

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

  // Tonemap
  // if (sceneInfo.tonemapper.isActive == 1)
  //     color = applyTonemap(sceneInfo.tonemapper, color, launchID / launchSize);

  outImage[int2(launchID)] = float4(color, 1.0);
}


//-----------------------------------------------------------------------
// CLOSEST HIT
//-----------------------------------------------------------------------
[shader("closesthit")]
void rchitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
  float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

  uint instanceID  = InstanceIndex();
  uint meshID      = InstanceID();
  uint primitiveID = PrimitiveIndex();

  RtInstanceInfo instInfo = sceneInfo.instances[instanceID];
  RtMeshInfo     meshInfo = sceneInfo.meshes[instInfo.meshIndex];

  // Rays
  float3 D = normalize(WorldRayDirection());    // Incoming direction
  float3 V = -D;                                // Vector to origin of ray: view direction
  float3 L = sceneInfo.skyParams.sunDirection;  //(1, 1, 1); // normalize(skyInfo.directionToLight);
                                                //// Vector to the light

  // Material info
  // InstanceInfo iInfo = instanceInfo[instanceID];
  float3 albedo = instInfo.color;

  // Hit information: position, normal, geo normal
  HitState hitState = getHitState(meshInfo, primitiveID, barycentrics);

  // Send a ray toward light and return true if there was no hit between P and L
  bool visible = shadowRay(hitState.pos, L);

  // Color at hit point
  PbrMaterial mat = defaultPbrMaterial(albedo, pushConst.metallic, pushConst.roughness, hitState.nrm, hitState.geonrm);
  float3      color = ggxEvaluate(V, L, mat);


  // Fake: lower the contribution if the hit wasn't visible
  if(!visible)
    color *= 0.3F;

  // Add contribution
  payload.color += color * /*pushConst.intensity **/ payload.weight;

  // Recursive bounce
  payload.depth += 1;
  payload.weight *= pushConst.metallic;  // more or less reflective

  // We hit our max depth
  if(payload.depth >= pushConst.maxDepth)
    return;

  // Reflection
  float3 refl_dir = reflect(D, hitState.nrm);

  // Trace bouncing ray
  RayDesc ray;
  ray.Origin    = hitState.pos;
  ray.Direction = refl_dir;
  ray.TMin      = 0.001;
  ray.TMax      = INFINITE;
  TraceRay(topLevelAS, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, 0, 0, 0, ray, payload);
}


//-----------------------------------------------------------------------
// MISS
//-----------------------------------------------------------------------
[shader("miss")]
void rmissMain(inout HitPayload payload)
{
  float3 skyColor = evalPhysicalSky(sceneInfo.skyParams, WorldRayDirection());

  payload.color += skyColor * payload.weight;
  payload.depth = MISS_DEPTH;  // Stop
}

