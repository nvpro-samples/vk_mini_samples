/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */


#include "common/shaders/glsl_type.hlsli"
#include "device_host.h"
#include "dh_bindings.h"

#include "common/shaders/sky.hlsli"
#include "common/shaders/ggx.hlsli"
#include "common/shaders/constants.hlsli"
#include "common/shaders/functions.hlsli"
#include "common/shaders/random.hlsli"

// Bindings
[[vk::constant_id(0)]] const int USE_SER = 0;
[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;
[[vk::binding(B_tlas)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(B_outImage)]] RWTexture2D<float4> outImage;
[[vk::binding(B_outHeatmap)]] RWTexture2D<float4> outHeatmap;
[[vk::binding(B_frameInfo)]] ConstantBuffer<FrameInfo> frameInfo;
[[vk::binding(B_skyParam)]] ConstantBuffer<SimpleSkyParameters> skyInfo;
[[vk::binding(B_heatStats)]] RWStructuredBuffer<HeatStats> heatStats;
[[vk::binding(B_materials)]] StructuredBuffer<float4> materials;
[[vk::binding(B_instances)]] StructuredBuffer<InstanceInfo> instanceInfo;
[[vk::binding(B_vertex)]] StructuredBuffer<Vertex> vertices[];
[[vk::binding(B_index)]] StructuredBuffer<uint3> indices[];

//-----------------------------------------------------------------------
// Payload 
// See: https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#example
struct [raypayload] HitPayload
{
  float hitT : write(closesthit, miss) : read(caller);
int instanceIndex : write(closesthit) : read(caller);
float3 pos : write(closesthit) : read(caller);
float3 nrm : write(closesthit) : read(caller);
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
//
// Shader Execution Reordering (SER)
//
// SPV_NV_shader_invocation_reorder
// https://htmlpreview.github.io/?https://github.com/KhronosGroup/SPIRV-Registry/blob/master/extensions/NV/SPV_NV_shader_invocation_reorder.html
#define ShaderInvocationReorderNV 5383
#define HitObjectAttributeNV 5385
#define RayPayloadKHR 5338


#define OpHitObjectRecordHitMotionNV 5249
#define OpHitObjectRecordHitWithIndexMotionNV 5250
#define OpHitObjectRecordMissMotionNV 5251
#define OpHitObjectGetWorldToObjectNV 5252
#define OpHitObjectGetObjectToWorldNV 5253
#define OpHitObjectGetObjectRayDirectionNV 5254
#define OpHitObjectGetObjectRayOriginNV 5255
#define OpHitObjectTraceRayMotionNV 5256
#define OpHitObjectGetShaderRecordBufferHandleNV 5257
#define OpHitObjectGetShaderBindingTableRecordIndexNV 5258
#define OpHitObjectRecordEmptyNV 5259
#define OpHitObjectTraceRayNV 5260
#define OpHitObjectRecordHitNV 5261
#define OpHitObjectRecordHitWithIndexNV 5262
#define OpHitObjectRecordMissNV 5263
#define OpHitObjectExecuteShaderNV 5264
#define OpHitObjectGetCurrentTimeNV 5265
#define OpHitObjectGetAttributesNV 5266
#define OpHitObjectGetHitKindNV 5267
#define OpHitObjectGetPrimitiveIndexNV 5268
#define OpHitObjectGetGeometryIndexNV 5269
#define OpHitObjectGetInstanceIdNV 5270
#define OpHitObjectGetInstanceCustomIndexNV 5271
#define OpHitObjectGetWorldRayDirectionNV 5272
#define OpHitObjectGetWorldRayOriginNV 5273
#define OpHitObjectGetRayTMaxNV 5274
#define OpHitObjectGetRayTMinNV 5275
#define OpHitObjectIsEmptyNV 5276
#define OpHitObjectIsHitNV 5277
#define OpHitObjectIsMissNV 5278
#define OpReorderThreadWithHitObjectNV 5279
#define OpReorderThreadWithHintNV 5280
#define OpTypeHitObjectNV 5281

// Definition of the Storage classes
[[vk::ext_storage_class(RayPayloadKHR)]] static HitPayload payload;
[[vk::ext_storage_class(HitObjectAttributeNV)]] static float3 objAttribs;

// Adding capabilities
[[vk::ext_capability(ShaderInvocationReorderNV)]]
[[vk::ext_extension("SPV_NV_shader_invocation_reorder")]]

// Pointer to the HitObject: used in parameter declarations
#define HitObjectNV vk::ext_type<HitObjectAttributeNV>
#define RefHitObjectNV [[vk::ext_reference]] HitObjectNV

// Shader Execution Reorder :  Function Declaration

[[vk::ext_type_def(HitObjectAttributeNV, OpTypeHitObjectNV)]]
void createHitObjectNV();

[[vk::ext_type_def(1, RayPayloadKHR)]]
void createRayPayloadKHR();

[[vk::ext_instruction(OpHitObjectRecordEmptyNV)]]
void hitObjectRecordEmptyNV(RefHitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectTraceRayNV)]]
void hitObjectTraceRayNV(
    RefHitObjectNV hitObject,
    RaytracingAccelerationStructure as,
    uint RayFlags,
    uint CullMask,
    uint SBTOffset,
    uint SBTStride,
    uint MissIndex,
    float3 RayOrigin,
    float RayTmin,
    float3 RayDirection,
    float RayTMax,
    [[vk::ext_reference]] [[vk::ext_storage_class(RayPayloadKHR)]] HitPayload payload
  );

[[vk::ext_instruction(OpReorderThreadWithHintNV)]]
void reorderThreadWithHintNV(int Hint, int Bits);

[[vk::ext_instruction(OpReorderThreadWithHitObjectNV)]]
void reorderThreadWithHitObjectNV(RefHitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectExecuteShaderNV)]]
void hitObjectExecuteShaderNV(
  RefHitObjectNV hitObject, 
  [[vk::ext_reference]] [[vk::ext_storage_class(RayPayloadKHR)]] HitPayload payload);

[[vk::ext_instruction(OpHitObjectIsHitNV)]]
bool hitObjectIsHitNV(RefHitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectGetRayTMaxNV)]]
float hitObjectGetRayTMaxNV(RefHitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectGetWorldToObjectNV)]]
float4x3 hitObjectGetWorldToObjectNV(RefHitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectGetObjectToWorldNV)]]
float4x3 hitObjectGetObjectToWorldNV(RefHitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectGetInstanceIdNV)]]
int hitObjectGetInstanceIdNV(RefHitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectGetInstanceCustomIndexNV)]]
int hitObjectGetInstanceCustomIndexNV(RefHitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectGetPrimitiveIndexNV)]]
int hitObjectGetPrimitiveIndexNV(RefHitObjectNV hitObject);

[[vk::ext_instruction(OpHitObjectGetAttributesNV)]]
void hitObjectGetAttributesNV(
    RefHitObjectNV hitObject,
    [[vk::ext_reference]] [[vk::ext_storage_class(HitObjectAttributeNV)]] float3 attribs
  );

//
// End -- Shader Execution Reordering (SER)
//


//-----------------------------------------------------------------------
// Return hit position, normal and geometric normal in world space
HitState getHitState(uint meshID, uint triID, float4x3 objectToWorld, float4x3 worldToObject, float3 barycentrics)
{
  HitState hit;
  
  uint3 triangleIndex = indices[meshID][triID];

  // Vertex and indices of the primitive
  Vertex v0 = vertices[meshID][triangleIndex.x];
  Vertex v1 = vertices[meshID][triangleIndex.y];
  Vertex v2 = vertices[meshID][triangleIndex.z];
   
  // Position
  const float3 pos0 = v0.position.xyz;
  const float3 pos1 = v1.position.xyz;
  const float3 pos2 = v2.position.xyz;
  const float3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos = float3(mul(float4(position, 1.0), objectToWorld));

  // Normal
  const float3 nrm0 = v0.normal.xyz;
  const float3 nrm1 = v1.normal.xyz;
  const float3 nrm2 = v2.normal.xyz;
  const float3 normal = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  float3 worldNormal = normalize(mul(worldToObject, normal).xyz);
  const float3 geoNormal = normalize(cross(pos1 - pos0, pos2 - pos0));
  float3 worldGeoNormal = normalize(mul(worldToObject, geoNormal).xyz);
  hit.geonrm = worldGeoNormal;
  hit.nrm = worldNormal;

  return hit;
}

//-----------------------------------------------------------------------
// Shoot a ray an return the information of the closest hit, in the
// PtPayload structure (PRD)
//
void traceRay(RayDesc r, inout HitPayload payload)
{
  payload.hitT = 0.0F;
  uint rayFlags = RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
  if(USE_SER == 1)
  {
    HitObjectNV hObj;
    hitObjectRecordEmptyNV(hObj); //Initialize to an empty hit object
    hitObjectTraceRayNV(hObj, topLevelAS, rayFlags, 0xFF, 0, 0, 0, r.Origin, 0.0, r.Direction, INFINITE, payload);
    reorderThreadWithHitObjectNV(hObj);
    
    payload.hitT = hitObjectGetRayTMaxNV(hObj);

    hitObjectGetAttributesNV(hObj, objAttribs);
        
    if(hitObjectIsHitNV(hObj))
    {
      payload.instanceIndex = hitObjectGetInstanceIdNV(hObj);

      int meshID = hitObjectGetInstanceCustomIndexNV(hObj);
      int triID = hitObjectGetPrimitiveIndexNV(hObj);
      float4x3 objectToWorld = hitObjectGetObjectToWorldNV(hObj);
      float4x3 worldToObject = hitObjectGetWorldToObjectNV(hObj);

      float3 barycentrics = float3(1 - objAttribs.x - objAttribs.y, objAttribs.x, objAttribs.y);
      HitState hit = getHitState(meshID, triID, objectToWorld, worldToObject, barycentrics);
      payload.pos = hit.pos;
      payload.nrm = hit.nrm;
    }
        
    // No need to execute the closest hit shader, as all necessary shading information has already been retrieved above.
    // hitObjectExecuteShaderNV(hObj, payload);
  }
  else
  {
    TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, r, payload);
  }
}

//-----------------------------------------------------------------------
// Shadow ray - return true if a ray hits anything
//
bool traceShadow(RayDesc r, inout HitPayload payload)
{
  payload.hitT = 0.0F;
  uint rayFlags = RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_CULL_BACK_FACING_TRIANGLES | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER;
  bool isHit;
  if(USE_SER == 1)
  {
    HitObjectNV hObj;
    hitObjectRecordEmptyNV(hObj); //Initialize to an empty hit object
    hitObjectTraceRayNV(hObj, topLevelAS, rayFlags, 0xFF, 0, 0, 0, r.Origin, 0.0, r.Direction, INFINITE, payload);
    isHit = hitObjectIsHitNV(hObj);
  }
  else
  {
    TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, r, payload);
    isHit = payload.hitT < INFINITE;
  }
  return isHit;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float3 pathTrace(RayDesc r, inout uint seed, inout HitPayload payload)
{
  float3 radiance = float3(0.0F, 0.0F, 0.0F);
  float3 throughput = float3(1.0F, 1.0F, 1.0F);

  //Materials materials = Materials(sceneDesc.materialAddress);

  for(int depth = 0; depth < pushConst.maxDepth; depth++)
  {
    traceRay(r, payload);

    // Hitting the environment, then exit
    if(payload.hitT == INFINITE)
    {
      float3 sky_color = evalSimpleSky(skyInfo, r.Direction);
      return radiance + (sky_color * throughput);
    }
    
    // Retrieve the Instance buffer information
    uint matID = instanceInfo[payload.instanceIndex].materialID;
  
    // Retrieve the material color
    float4 albedo = materials[matID];
   
    float pdf = 0.0F;
    float3 V = -r.Direction;
    float3 L = normalize(skyInfo.directionToLight);

    // Setting up the material
    PbrMaterial pbrMat = defaultPbrMaterial(albedo.xyz,pushConst.metallic, pushConst.roughness,payload.nrm,payload.nrm);


    // Add dummy divergence
    // This sample shader is too simple to show the benefits of sorting
    // rays, with this we create some artificial workload depending on the
    // material ID
    float3 dummy = payload.nrm;
    uint dummyLoop = (matID * 128) & (1024 - 1);
    for(uint i = 0; i < dummyLoop; i++)
    {
      dummy = sin(dummy);
    }

    pbrMat.baseColor.xyz += dummy * 0.01;

    float3 contrib = float3(0, 0, 0);

    // Evaluation of direct light (sun)
    bool nextEventValid = (dot(L, payload.nrm) > 0.0F);
    if(nextEventValid)
    {
      BsdfEvaluateData evalData;
      evalData.k1 = -r.Direction;
      evalData.k2 = normalize(skyInfo.directionToLight);
      evalData.xi = float3(rand(seed), rand(seed), rand(seed));
      bsdfEvaluate(evalData, pbrMat);

      const float3 w = pushConst.intensity.xxx;
      contrib += w * evalData.bsdf_diffuse;
      contrib += w * evalData.bsdf_glossy;
      contrib *= throughput;
    }


    // Sample BSDF
    {
      BsdfSampleData sampleData;
      sampleData.k1 = -r.Direction; // outgoing direction
      sampleData.xi = float3(rand(seed), rand(seed), rand(seed));

      bsdfSample(sampleData, pbrMat);
      if(sampleData.event_type == BSDF_EVENT_ABSORB)
      {
        break; // Need to add the contribution ?
      }

      throughput *= sampleData.bsdf_over_pdf;
      r.Origin = offsetRay(payload.pos, payload.nrm);
      r.Direction = sampleData.k2;
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
      shadowRay.Origin = r.Origin;
      shadowRay.Direction = L;
      shadowRay.TMin = 0.01;
      shadowRay.TMax = INFINITE;
      bool inShadow = traceShadow(shadowRay, payload);
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
float3 samplePixel(inout uint seed, inout HitPayload payload)
{
  float2 launchID = (float2)DispatchRaysIndex();
  float2 launchSize = (float2)DispatchRaysDimensions();

  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  const float2 subpixel_jitter = pushConst.frame == 0 ? float2(0.5f, 0.5f) : float2(rand(seed), rand(seed));
  const float2 pixelCenter = launchID + subpixel_jitter;
  const float2 inUV = pixelCenter / launchSize;
  const float2 d = inUV * 2.0 - 1.0;
  const float4 target = mul(frameInfo.projInv, float4(d.x, d.y, 0.01, 1.0));

  RayDesc ray;
  ray.Origin = mul(frameInfo.viewInv, float4(0.0, 0.0, 0.0, 1.0)).xyz;
  ray.Direction = mul(frameInfo.viewInv, float4(normalize(target.xyz), 0.0)).xyz;
  ray.TMin = 0.001;
  ray.TMax = INFINITE;

  // Initial state
  payload.hitT = 0;
  payload.instanceIndex = 0;
  payload.pos = float3(0, 0, 0);
  payload.nrm = float3(0, 0, 0);

  float3 radiance = pathTrace(ray, seed, payload);

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
[shader("raygeneration")]
void rgenMain()
{
  createHitObjectNV();

  float2 launchID = (float2)DispatchRaysIndex();
  float2 launchSize = (float2)DispatchRaysDimensions();

  uint64_t start = vk::ReadClock(vk::DeviceScope); // Debug - Heatmap

  // Initialize the random number
  uint seed = xxhash32(uint3(launchID.xy, pushConst.frame));

  // Sampling n times the pixel
  float3 pixel_color = float3(0.0F, 0.0F, 0.0F);
  for(int s = 0; s < pushConst.maxSamples; s++)
  {
    pixel_color += samplePixel(seed, payload);
  }
  pixel_color /= pushConst.maxSamples;

  bool first_frame = (pushConst.frame == 0);

  // Debug - Heatmap
  {
    uint64_t end = vk::ReadClock(vk::DeviceScope);
    uint duration = uint(end - start);

    // log maximum of current frame
    uint statsIndex = uint(pushConst.frame) & 1;
    InterlockedMax(heatStats[0].maxDuration[statsIndex], duration, heatStats[0].maxDuration[statsIndex]);
    // use previous frame's maximum
    uint maxDuration = heatStats[0].maxDuration[statsIndex ^ 1];

    // lower ceiling to see some more red ;)
    float high = float(maxDuration) * 0.50;

    float val = clamp(float(duration) / high, 0.0F, 1.0F);
    float3 heat_color = temperature(val);

    // Wrap & SM visualization
    // heat_color = temperature(float(gl_SMIDNV) / float(gl_SMCountNV - 1)) * float(gl_WarpIDNV) / float(gl_WarpsPerSMNV - 1);
    
    outHeatmap[int2(launchID)] = float4(heat_color, 1.0);
  }

  // Saving result
  if(first_frame)
  { // First frame, replace the value in the buffer
    outImage[int2(launchID)] = float4(pixel_color, 1.0);
  }
  else
  { // Do accumulation over time
    float a = 1.0F / float(pushConst.frame + 1);
    float3 old_color = outImage[int2(launchID)].xyz; //imageLoad(image, ivec2(gl_LaunchIDEXT.xy)).xyz;
    outImage[int2(launchID)] = float4(lerp(old_color, pixel_color, a), 1.0F);
  }
}


//-----------------------------------------------------------------------
// CLOSEST HIT
//-----------------------------------------------------------------------
[shader("closesthit")]
void rchitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
  float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);

  uint meshID = InstanceID();
  uint triID = PrimitiveIndex();
  float4x3 objectToWorld = ObjectToWorld4x3();
  float4x3 worldToObject = WorldToObject4x3();
  
  HitState hit = getHitState(meshID, triID, objectToWorld, worldToObject, barycentrics);

  payload.hitT = RayTCurrent();
  payload.pos = hit.pos;
  payload.nrm = hit.nrm;
  payload.instanceIndex = InstanceIndex();
}


//-----------------------------------------------------------------------
// MISS 
//-----------------------------------------------------------------------
[shader("miss")]
void rmissMain(inout HitPayload payload)
{
  payload.hitT = INFINITE;
}

