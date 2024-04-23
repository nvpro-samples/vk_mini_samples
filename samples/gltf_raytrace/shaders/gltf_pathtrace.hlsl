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

//#version 460

#include "common/shaders/constants.hlsli"
#include "common/shaders/dh_comp.hlsli"
#include "common/shaders/dh_lighting.hlsli"
#include "common/shaders/dh_scn_desc.hlsli"
#include "common/shaders/ggx.hlsli"
#include "common/shaders/light_contrib.hlsli"
#include "common/shaders/pbr_mat_eval.hlsli"
#include "common/shaders/random.hlsli"
#include "common/shaders/sky.hlsli"

#include "device_host.h"
#include "dh_bindings.h"

static const float DIRAC = -1.0; // Used for direct light sampling

struct PrimMeshInfoHlsl
{
  uint64_t vertexAddress; // Array of Vertex  (Vertex)
  uint64_t indexAddress; // Array of Index (uint3)
  int materialIndex; // Material used by Mesh
};
    
struct SceneDescriptionHlsl
{
  uint64_t materialAddress; // GltfShadeMaterial
  uint64_t instInfoAddress; // InstanceInfo
  uint64_t primInfoAddress; // PrimMeshInfoHlsl
  Light light;
};

struct Ray
{
  float3 origin;
  float3 direction;
};

// Bindings
[[vk::push_constant]]  ConstantBuffer<PushConstant> pushConst;
[[vk::binding(B_tlas)]] RaytracingAccelerationStructure topLevelAS;
[[vk::binding(B_outImage)]] RWTexture2D<float4> outImage;
[[vk::binding(B_cameraInfo)]] ConstantBuffer<CameraInfo> cameraInfo;
[[vk::binding(B_sceneDesc)]] StructuredBuffer<SceneDescriptionHlsl> sceneDesc;
[[vk::binding(B_skyParam)]] ConstantBuffer<ProceduralSkyShaderParameters> skyInfo;
[[vk::binding(B_textures)]]  Texture2D   texturesMap[] : register(t4);
[[vk::binding(B_textures)]] SamplerState samplers;

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  float3 pos;
  float3 nrm;
  float3 geonrm;
  float2 uv;
  float3 tangent;
  float3 bitangent;
};

// Payload for the path tracer
struct HitPayload
{
  float hitT;
  int meshID;
  HitState hitState;
};

template<typename  T>
T mixBary(T a, T b, T c, float3 bary)
{
  return a * bary.x + b * bary.y + c * bary.z;
}

// This helper is used to load data from a buffer
// The startAddress is the address of the buffer
// The offset is the starting position in the buffer
//
// The loadValue function will load the data from the buffer
// NOTE: loadValue will increment the offset, therefore all the loadValue should be called in the same order as the data is stored in the buffer
struct LoaderHelper
{
  void init(uint64_t startAddress, uint64_t offset=0) { m_address = startAddress; m_offset = offset;}

  template<typename T>
  void loadValue(inout T value)
  {
    value = vk::RawBufferLoad<T>(m_address + m_offset);
    m_offset += sizeof(T);
  }

  uint64_t m_address;
  uint64_t m_offset;
};

PrimMeshInfoHlsl getPrimMeshInfo(uint64_t primAddress, uint64_t offset)
{
  PrimMeshInfoHlsl pInfo;
  {
    LoaderHelper loader;
    loader.init(primAddress, offset);
    
    loader.loadValue<uint64_t>(pInfo.vertexAddress);
    loader.loadValue<uint64_t>(pInfo.indexAddress);
    loader.loadValue<int>(pInfo.materialIndex);
  }
  return pInfo;
}


// Return the Vertex structure, from a buffer address and an offset
Vertex getVertex(uint64_t vertAddress, uint64_t offset)
{
  LoaderHelper loader;
  loader.init(vertAddress, offset);
  Vertex v;
  loader.loadValue<float4>(v.position);
  loader.loadValue<float4>(v.normal);
  loader.loadValue<float4>(v.tangent);

  return v;
}

// Return the material structure, from a buffer address and an offset
GltfShadeMaterial getMaterial(uint64_t materialAddress, uint64_t offset)
{
  LoaderHelper loader;
  loader.init(materialAddress, offset);
  
  GltfShadeMaterial m;
  loader.loadValue<float4>(m.pbrBaseColorFactor);
  loader.loadValue<float3>(m.emissiveFactor);
  loader.loadValue<int>(m.pbrBaseColorTexture);

  loader.loadValue<int>(m.normalTexture);
  loader.loadValue<float>(m.normalTextureScale);
  loader.loadValue<int>(m._pad0);
  loader.loadValue<float>(m.pbrRoughnessFactor);

  loader.loadValue<float>(m.pbrMetallicFactor);
  loader.loadValue<int>(m.pbrMetallicRoughnessTexture);

  loader.loadValue<int>(m.emissiveTexture);
  loader.loadValue<int>(m.alphaMode);
  loader.loadValue<float>(m.alphaCutoff);

  // KHR_materials_transmission
  loader.loadValue<float>(m.transmissionFactor);
  loader.loadValue<int>(m.transmissionTexture);
  
  // KHR_materials_ior
  loader.loadValue<float>(m.ior);
  
  // KHR_materials_volume
  loader.loadValue<float3>(m.attenuationColor);
  loader.loadValue<float>(m.thicknessFactor);
  loader.loadValue<int>(m.thicknessTexture);
  loader.loadValue<bool>(m.thinWalled);
  loader.loadValue<float>(m.attenuationDistance);

  // KHR_materials_clearcoat
  loader.loadValue<float>(m.clearcoatFactor);
  loader.loadValue<float>(m.clearcoatRoughness);
  loader.loadValue<int>(m.clearcoatTexture);
  loader.loadValue<int>(m.clearcoatRoughnessTexture);
  loader.loadValue<int>(m.clearcoatNormalTexture);
  
   // KHR_materials_specular
  loader.loadValue<float>(m.specularFactor);
  loader.loadValue<int>(m.specularTexture);
  loader.loadValue<float3>(m.specularColorFactor);
  loader.loadValue<int>(m.specularColorTexture);
  
  // KHR_texture_transform
  loader.loadValue<float3x3>(m.uvTransform);
  return m;
}


//-----------------------------------------------------------------------
// Return hit information: position, normal, geonormal, uv, tangent, bitangent
HitState getHitState(int meshID, float2 barycentricCoords, float3x4 worldToObject, float3x4 objectToWorld, int primitiveID, float3 worldRayDirection)
{
  HitState hit;

  // Vextex and indices of the primitive
  uint64_t primOffset = sizeof(PrimMeshInfoHlsl) * meshID;
  PrimMeshInfoHlsl pInfo  = getPrimMeshInfo(sceneDesc[0].primInfoAddress, primOffset);

  // Barycentric coordinate on the triangle
  const float3 barycentrics = float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

  // Getting the 3 indices of the triangle (local)
  uint64_t indexOffset = sizeof(uint3) * primitiveID;
  uint3 triangleIndex = vk::RawBufferLoad<uint3> (pInfo.indexAddress + indexOffset);
  
  // Vertex and indices of the primitive
  Vertex v0 = getVertex(pInfo.vertexAddress, sizeof(Vertex) * triangleIndex.x);
  Vertex v1 = getVertex(pInfo.vertexAddress, sizeof(Vertex) * triangleIndex.y);
  Vertex v2 = getVertex(pInfo.vertexAddress, sizeof(Vertex) * triangleIndex.z);
  

  // Position
  const float3 pos0 = v0.position.xyz;
  const float3 pos1 = v1.position.xyz;
  const float3 pos2 = v2.position.xyz;
  const float3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos = float3(mul(objectToWorld, float4(position, 1.0)));

  // Normal
  const float3 nrm0 = v0.normal.xyz;
  const float3 nrm1 = v1.normal.xyz;
  const float3 nrm2 = v2.normal.xyz;
  const float3 normal = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  float3 worldNormal = normalize(float3(mul(normal, worldToObject).xyz));
  const float3 geoNormal = normalize(cross(pos1 - pos0, pos2 - pos0));
  float3 worldGeoNormal = normalize(float3(mul(geoNormal, worldToObject).xyz));
  hit.geonrm = worldGeoNormal;
  hit.nrm = worldNormal;

  // TexCoord
  const float2 uv0 = float2(v0.position.w, v0.normal.w);
  const float2 uv1 = float2(v1.position.w, v1.normal.w);
  const float2 uv2 = float2(v2.position.w, v2.normal.w);
  hit.uv = mixBary(uv0, uv1, uv2, barycentrics);

  // Tangent - Bitangent
  const float4 tng0 = float4(v0.tangent);
  const float4 tng1 = float4(v1.tangent);
  const float4 tng2 = float4(v2.tangent);
  hit.tangent = normalize(mixBary(tng0.xyz, tng1.xyz, tng2.xyz, barycentrics));
  hit.tangent = float3(mul(objectToWorld, float4(hit.tangent, 0.0)));
  hit.tangent = normalize(hit.tangent - hit.nrm * dot(hit.nrm, hit.tangent));
  hit.bitangent = cross(hit.nrm, hit.tangent) * tng0.w;

  // Adjusting normal
  const float3 V = (-worldRayDirection);
  if(dot(hit.geonrm, V) < 0)  // Flip if back facing
    hit.geonrm = -hit.geonrm;

  // If backface
  if(dot(hit.geonrm, hit.nrm) < 0)  // Make Normal and GeoNormal on the same side
  {
    hit.nrm = -hit.nrm;
    hit.tangent = -hit.tangent;
    hit.bitangent = -hit.bitangent;
  }

  // handle low tessellated meshes with smooth normals
  float3 k2 = reflect(-V, hit.nrm);
  if(dot(hit.geonrm, k2) < 0.0F)
    hit.nrm = hit.geonrm;

  // For low tessalated, avoid internal reflection
  float3 r = reflect(normalize(worldRayDirection), hit.nrm);
  if(dot(r, hit.geonrm) < 0)
    hit.nrm = hit.geonrm;

  return hit;
}

struct DirectLight
{
  float3 direction; // Direction to the light
  float3 radianceOverPdf; // Radiance over pdf
  float distance; // Distance to the light
  float pdf; // Probability of sampling this light
};


//-----------------------------------------------------------------------
// This should sample any lights in the scene, but we only have the sun
void sampleLights(in float3 pos, float3 normal, in float3 worldRayDirection, inout uint seed, out DirectLight directLight)
{
  float3 radiance = float3(0, 0, 0);
  float lightPdf = 1.0;

  // Light contribution
  Light sun;
  sun.type = eLightTypeDirectional;
  sun.angularSizeOrInvRange = skyInfo.angularSizeOfLight;
  sun.direction = -skyInfo.directionToLight;
  sun.color = skyInfo.lightColor;
  sun.intensity = 1;
  float2 rand_val = float2(rand(seed), rand(seed));
  LightContrib lightContrib = singleLightContribution(sun, pos, normal, -worldRayDirection, rand_val);

  // Return the light contribution
  directLight.direction = normalize(-lightContrib.incidentVector);
  directLight.radianceOverPdf = lightContrib.intensity / lightPdf;
  directLight.distance = INFINITE;
  directLight.pdf = DIRAC;
}

//----------------------------------------------------------
// Testing if the hit is opaque or alpha-transparent
// Return true if it is opaque
bool hitTest(in RayQuery<RAY_FLAG_NONE> rayQuery, inout uint seed)
{
  int meshID = rayQuery.CandidateInstanceID(); // rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, false);
  int primitiveID = rayQuery.CandidatePrimitiveIndex(); // rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, false);

  // Retrieve the Primitive mesh buffer information
  uint64_t primOffset = sizeof(PrimMeshInfoHlsl) * meshID;
  PrimMeshInfoHlsl pInfo  = getPrimMeshInfo(sceneDesc[0].primInfoAddress, primOffset);
  
  // Find the material of the primitive
  const uint matIndex = max(0, pInfo.materialIndex); // material of primitive mesh
  uint64_t matOffset = sizeof(GltfShadeMaterial) * matIndex;
  GltfShadeMaterial material = getMaterial(sceneDesc[0].materialAddress, matOffset);
  if(material.alphaMode == ALPHA_OPAQUE)
    return true;

  float baseColorAlpha = material.pbrBaseColorFactor.a;
  if(material.pbrBaseColorTexture > -1)
  {
    uint64_t indexOffset = sizeof(uint3) * primitiveID;
    uint3 triangleIndex = vk::RawBufferLoad<uint3>(pInfo.indexAddress + indexOffset);
    
    // Vertex and indices of the primitive
    Vertex v0 = getVertex(pInfo.vertexAddress, sizeof(Vertex) * triangleIndex.x);
    Vertex v1 = getVertex(pInfo.vertexAddress, sizeof(Vertex) * triangleIndex.y);
    Vertex v2 = getVertex(pInfo.vertexAddress, sizeof(Vertex) * triangleIndex.z);
    

    // Get the texture coordinate
    float2 bary = rayQuery.CandidateTriangleBarycentrics(); // rayQueryGetIntersectionBarycentricsEXT(rayQuery, false);
    const float3 barycentrics = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    const float2 uv0 = float2(v0.position.w, v0.normal.w);
    const float2 uv1 = float2(v1.position.w, v1.normal.w);
    const float2 uv2 = float2(v2.position.w, v2.normal.w);
    float2 texcoord0 = mixBary(uv0, uv1, uv2, barycentrics);

    // Uv Transform
    //texcoord0 = (float4(texcoord0.xy, 1, 1) * material.uvTransform).xy;

    baseColorAlpha *= texturesMap[material.pbrBaseColorTexture].SampleLevel(samplers, texcoord0.xy, 0).a;
  }

  float opacity;
  if(material.alphaMode == ALPHA_MASK)
  {
    opacity = baseColorAlpha > material.alphaCutoff ? 1.0 : 0.0;
  }
  else
  {
    opacity = baseColorAlpha;
  }

  // do alpha blending the stochastically way
  if(rand(seed) > opacity)
    return false;

  return true;
}

//-----------------------------------------------------------------------
// Shoot a ray an return the information of the closest hit, in the
// PtPayload structure (PRD)
//
void traceRay(Ray ray, inout HitPayload payload, inout uint seed)
{
  payload.hitT = INFINITE; // Default when not hitting anything

  RayDesc r = {ray.origin, 0, ray.direction, INFINITE};
  RayQuery<RAY_FLAG_NONE> q;
  q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, r);

  // uint rayFlags = gl_RayFlagsNoneEXT;
  // rayQueryInitializeEXT(rayQuery, topLevelAS, rayFlags, 0xFF, ray.origin, 0.0, ray.direction, INFINITE);
  while(q.Proceed())//rayQueryProceedEXT(rayQuery))
  {
    if(hitTest(q, seed))
    {
      q.CommitNonOpaqueTriangleHit(); // rayQueryConfirmIntersectionEXT(rayQuery);
    }
  }

  //if(rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT)
  if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
  {
    float2 barycentricCoords = q.CommittedTriangleBarycentrics(); // rayQueryGetIntersectionBarycentricsEXT(rayQuery, true);
    int meshID = q.CommittedInstanceID(); // rayQueryGetIntersectionInstanceCustomIndexEXT(rayQuery, true);
    int primitiveID = q.CommittedPrimitiveIndex(); // rayQueryGetIntersectionPrimitiveIndexEXT(rayQuery, true);
    float3x4 worldToObject = q.CommittedWorldToObject3x4(); // rayQueryGetIntersectionWorldToObjectEXT(rayQuery, true);
    float3x4 objectToWorld = q.CommittedObjectToWorld3x4(); // rayQueryGetIntersectionObjectToWorldEXT(rayQuery, true);
    float hitT = q.CommittedRayT(); // rayQueryGetIntersectionTEXT(rayQuery, true);
    float3 worldRayDirection = r.Direction; // rayQueryGetWorldRayDirectionEXT(rayQuery);

    HitState hit = getHitState(meshID, barycentricCoords, worldToObject, objectToWorld, primitiveID, worldRayDirection);

    payload.hitT = hitT;
    payload.meshID = meshID;
    payload.hitState = hit;
  }
}

//-----------------------------------------------------------------------
// Shadow ray - return true if a ray hits anything
//
bool traceShadow(Ray ray, float maxDist, inout uint seed)
{
  RayDesc r = {ray.origin, 0, ray.direction, maxDist};
  RayQuery<RAY_FLAG_NONE> q;
  q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, r);

  while(q.Proceed())
  {
    if(hitTest(q, seed))
    {
      q.CommitNonOpaqueTriangleHit(); // rayQueryConfirmIntersectionEXT(rayQuery);
    }
  }

  //return (rayQueryGetIntersectionTypeEXT(rayQuery, true) != gl_RayQueryCommittedIntersectionNoneEXT);  // Is Hit ?
  return q.CommittedStatus() != COMMITTED_NOTHING;
}

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
float3 pathTrace(Ray ray, inout uint seed)
{
  float3 radiance = float3(0.0F,0.0F,0.0F);
  float3 throughput = float3(1.0F,1.0F,1.0F);
  bool isInside = false;


  HitPayload payload;

  for(int depth = 0; depth < pushConst.maxDepth; depth++)
  {
    traceRay(ray, payload, seed);

    // Hitting the environment, then exit
    if(payload.hitT == INFINITE)
    {
      float3 sky_color = proceduralSky(skyInfo, ray.direction, 0);
      return radiance + (sky_color * throughput);
    }

    // Getting the hit information (primitive/mesh that was hit)
    HitState hit = payload.hitState;

    // Setting up the material
    uint64_t primOffset = sizeof(PrimMeshInfoHlsl) * payload.meshID;
    PrimMeshInfoHlsl pInfo  = getPrimMeshInfo(sceneDesc[0].primInfoAddress, primOffset);

    int materialIndex = max(0, pInfo.materialIndex); // Material ID of hit mesh
    uint64_t matOffset = sizeof(GltfShadeMaterial) * materialIndex;
    GltfShadeMaterial material = getMaterial(sceneDesc[0].materialAddress, matOffset);
    

    PbrMaterial pbrMat = evaluateMaterial(material, texturesMap, samplers, hit.nrm, hit.tangent, hit.bitangent, hit.uv, isInside);

    // Adding emissive
    radiance += pbrMat.emissive * throughput;

    // Apply volume attenuation
    bool thin_walled = pbrMat.thicknessFactor == 0;
    if(isInside && !thin_walled)
    {
      const float3 abs_coeff = absorptionCoefficient(pbrMat);
      throughput.x *= abs_coeff.x > 0.0 ? exp(-abs_coeff.x * payload.hitT) : 1.0;
      throughput.y *= abs_coeff.y > 0.0 ? exp(-abs_coeff.y * payload.hitT) : 1.0;
      throughput.z *= abs_coeff.z > 0.0 ? exp(-abs_coeff.z * payload.hitT) : 1.0;
    }
    float3 contribution = float3(0, 0, 0); // Direct lighting contribution

    // Light contribution; can be environment or punctual lights
    DirectLight directLight;
    sampleLights(hit.pos, pbrMat.normal, ray.direction, seed, directLight);

    // Evaluation of direct light (sun)
    const bool nextEventValid = ((dot(directLight.direction, hit.geonrm) > 0.0F) != isInside) && directLight.pdf != 0.0F;
    if(nextEventValid)
    {
      BsdfEvaluateData evalData;
      evalData.k1 = -ray.direction;
      evalData.k2 = directLight.direction;
      bsdfEvaluate(evalData, pbrMat);

      if(evalData.pdf > 0.0)
      {
        const float mis_weight = (directLight.pdf == DIRAC) ? 1.0F : directLight.pdf / (directLight.pdf + evalData.pdf);

        // sample weight
        const float3 w = throughput * directLight.radianceOverPdf * mis_weight;
        contribution += w * evalData.bsdf_diffuse;
        contribution += w * evalData.bsdf_glossy;
      }
    }
    
    // Sample BSDF
    {
      BsdfSampleData sampleData;
      sampleData.k1 = -ray.direction; // outgoing direction
      sampleData.xi = float4(rand(seed), rand(seed), rand(seed), rand(seed));
      bsdfSample(sampleData, pbrMat);

      throughput *= sampleData.bsdf_over_pdf;
      ray.direction = sampleData.k2;

      if(sampleData.event_type == BSDF_EVENT_ABSORB)
      {
        break; // Need to add the contribution ?
      }
      else
      {
        // Continue path
        bool isSpecular = (sampleData.event_type & BSDF_EVENT_SPECULAR) != 0;
        bool isTransmission = (sampleData.event_type & BSDF_EVENT_TRANSMISSION) != 0;

        float3 offsetDir = dot(ray.direction, hit.geonrm) > 0 ? hit.geonrm : -hit.geonrm;
        ray.origin = offsetRay(hit.pos, offsetDir);

        if(isTransmission)
        {
          isInside = !isInside;
        }
      }
    }

    // We are adding the contribution to the radiance only if the ray is not occluded by an object.
    if(nextEventValid)
    {
      Ray shadowRay = {ray.origin, directLight.direction};
      bool inShadow = traceShadow(shadowRay, directLight.distance, seed);
      if(!inShadow)
      {
        radiance += contribution;
      }
    }

    // Russian-Roulette (minimizing live state)
    float rrPcont = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001F, 0.95F);
    if(rand(seed) >= rrPcont)
      break; // paths with low throughput that won't contribute
    throughput /= rrPcont; // boost the energy of the non-terminated paths
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

  Ray ray;
  ray.origin = float3(cameraInfo.viewInv[0].w, cameraInfo.viewInv[1].w, cameraInfo.viewInv[2].w);
  ray.direction = normalize(mul(cameraInfo.viewInv, viewCoords).xyz - ray.origin);


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

