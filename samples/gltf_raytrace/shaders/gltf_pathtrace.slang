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

#include "nvshaders/functions.h.slang"
#include "nvshaders/random.h.slang"
#include "nvshaders/sky_functions.h.slang"

#include "nvshaders/bsdf_functions.h.slang"
#include "nvshaders/pbr_material_eval.h.slang"
#include "nvshaders/ray_utils.h.slang"
#include "nvshaders/gltf_scene_io.h.slang"
#include "nvshaders/gltf_vertex_access.h.slang"

#include "nvshaders/hdr_env_sampling.h.slang"
#include "nvshaders/hdr_io.h.slang"

#include "shaderio.h"

struct Ray
{
  float3 origin;
  float3 direction;
};

// Bindings
// clang-format off
[[vk::push_constant]]           ConstantBuffer<PushConstant> pushConst;
[[vk::binding(B_textures, 0)]]  Sampler2D allTextures[];
[[vk::binding(B_tlas, 1)]]      RaytracingAccelerationStructure topLevelAS;
[[vk::binding(B_outImage, 1)]]  RWTexture2D<float4> outImage;

// HDR Environment
[[vk::binding(EnvBindings::eHdr, 2)]]           Sampler2D hdrIBL;
[[vk::binding(EnvBindings::eImpSamples, 2)]]    StructuredBuffer<EnvAccel> envSamplingData;

// clang-format on

//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  float3 pos;
  float3 nrm;
  float4 color;
  float3 geonrm;
  float2 uv[2];
  float3 tangent;
  float3 bitangent;
};

// Payload for the path tracer
struct HitPayload
{
  float    hitT;
  int      rnodeID;
  int      rprimID;
  HitState hitState;
};

static bool doDebug = false;

// Direct light structure
struct DirectLight
{
  float3 direction;        // Direction to the light
  float3 radianceOverPdf;  // Radiance over pdf
  float  distance;         // Distance to the light
  float  pdf;              // Probability of sampling this light
};

//-----------------------------------------------------------------------
// This should sample any lights in the scene, but we only have the sun
void sampleLights(in float3 pos, float3 normal, in float3 worldRayDirection, inout uint seed, out DirectLight directLight)
{
  directLight.distance = INFINITE;

  if(pushConst.environmentType == 0)
  {
    float2            randVal   = float2(rand(seed), rand(seed));
    SkySamplingResult skySample = samplePhysicalSky(*pushConst.skyParams, randVal);

    directLight.direction       = skySample.direction;
    directLight.pdf             = skySample.pdf;
    directLight.radianceOverPdf = skySample.radiance / skySample.pdf;
  }
  else
  {
    float3 randVal      = float3(rand(seed), rand(seed), rand(seed));
    float4 radiance_pdf = environmentSample(hdrIBL, envSamplingData, randVal, directLight.direction);

    float  envPdf   = radiance_pdf.w;
    float3 radiance = radiance_pdf.xyz / envPdf;

    directLight.pdf             = envPdf;
    directLight.radianceOverPdf = radiance;
  }
}


//-----------------------------------------------------------------------
// Return hit information: position, normal, geonormal, uv, tangent, bitangent
HitState getHitState(GltfRenderPrimitive renderPrim,  //
                     float2              barycentricCoords,
                     float4x3            worldToObject,
                     float4x3            objectToWorld,
                     int                 primitiveID,
                     float3              worldRayDirection)
{
  HitState hit;

  // Barycentric coordinate on the triangle
  const float3 barycentrics =
      float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

  // Getting the 3 indices of the triangle (local)
  uint3 triangleIndex = getTriangleIndices(renderPrim, primitiveID);

  // Position
  const float3 pos0     = getVertexPosition(renderPrim, triangleIndex.x);
  const float3 pos1     = getVertexPosition(renderPrim, triangleIndex.y);
  const float3 pos2     = getVertexPosition(renderPrim, triangleIndex.z);
  const float3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos               = float3(mul(float4(position, 1.0), objectToWorld));

  // Normal
  const float3 geoNormal      = normalize(cross(pos1 - pos0, pos2 - pos0));
  float3       worldGeoNormal = normalize(float3(mul(worldToObject, geoNormal).xyz));
  float3       normal         = geoNormal;
  float3       worldNormal    = worldGeoNormal;
  if(hasVertexNormal(renderPrim))
  {
    normal      = getInterpolatedVertexNormal(renderPrim, triangleIndex, barycentrics);
    worldNormal = normalize(float3(mul(worldToObject, normal).xyz));
  }
  hit.geonrm = worldGeoNormal;
  hit.nrm    = worldNormal;

  // TexCoord
  hit.uv[0] = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);
  hit.uv[1] = getInterpolatedVertexTexCoord1(renderPrim, triangleIndex, barycentrics);

  // Color
  hit.color = getInterpolatedVertexColor(renderPrim, triangleIndex, barycentrics);

  // Tangent - Bitangent
  float4 tng[3];
  if(hasVertexTangent(renderPrim))
  {
    tng[0] = getVertexTangent(renderPrim, triangleIndex.x);
    tng[1] = getVertexTangent(renderPrim, triangleIndex.y);
    tng[2] = getVertexTangent(renderPrim, triangleIndex.z);
  }
  else
  {
    float4 t = makeFastTangent(normal);
    tng[0]   = t;
    tng[1]   = t;
    tng[2]   = t;
  }
  hit.tangent   = normalize(mixBary(tng[0].xyz, tng[1].xyz, tng[2].xyz, barycentrics));
  hit.tangent   = float3(mul(float4(hit.tangent, 0.0), objectToWorld));
  hit.tangent   = normalize(hit.tangent - hit.nrm * dot(hit.nrm, hit.tangent));
  hit.bitangent = cross(hit.nrm, hit.tangent) * tng[0].w;

  // Adjusting normal
  const float3 V = (-worldRayDirection);
  if(dot(hit.geonrm, V) < 0)  // Flip if back facing
  {
    hit.geonrm = -hit.geonrm;
  }

  // If backface
  if(dot(hit.geonrm, hit.nrm) < 0)  // Make Normal and GeoNormal on the same side
  {
    hit.nrm       = -hit.nrm;
    hit.tangent   = -hit.tangent;
    hit.bitangent = -hit.bitangent;
  }

  // handle low tessellated meshes with smooth normals
  float3 k2 = reflect(-V, hit.nrm);
  if(dot(hit.geonrm, k2) < 0.0f)
  {
    hit.nrm = hit.geonrm;
  }

  // For low tessalated, avoid internal reflection
  float3 r = reflect(normalize(worldRayDirection), hit.nrm);
  if(dot(r, hit.geonrm) < 0)
  {
    hit.nrm = hit.geonrm;
  }

  return hit;
}

//----------------------------------------------------------
// Testing if the hit is opaque or alpha-transparent
// Return true if it is opaque
bool hitTest(in RayQuery rayQuery, inout uint seed)
{
  int rnodeID = rayQuery.CandidateInstanceIndex();
  int rprimID = rayQuery.CandidateInstanceID();
  int triID   = rayQuery.CandidatePrimitiveIndex();

  // Retrieve the Primitive mesh buffer information
  GltfRenderNode      renderNode = pushConst.gltfScene->renderNodes[rnodeID];
  GltfRenderPrimitive renderPrim = pushConst.gltfScene->renderPrimitives[rprimID];

  // Find the material of the primitive
  const uint         matIndex  = max(0, renderNode.materialID);      // material of primitive mesh
  GltfShadeMaterial* materials = pushConst.gltfScene->materials;     // Buffer of materials
  GltfTextureInfo*   texInfos  = pushConst.gltfScene->textureInfos;  // Buffer of textures
  GltfShadeMaterial  material  = materials[matIndex];
  if(material.alphaMode == AlphaMode::eAlphaModeOpaque)
    return true;

  float baseColorAlpha = material.pbrBaseColorFactor.a;
  if(isTexturePresent(material.pbrBaseColorTexture))
  {
    // Getting the 3 indices of the triangle (local)
    uint3 triangleIndex = getTriangleIndices(renderPrim, triID);  // renderPrim.

    // Get the texture coordinate
    const float2 bary         = rayQuery.CandidateTriangleBarycentrics();
    const float3 barycentrics = float3(1.0 - bary.x - bary.y, bary.x, bary.y);
    float2       texcoord0    = getInterpolatedVertexTexCoord0(renderPrim, triangleIndex, barycentrics);

    // Uv Transform
    GltfTextureInfo tinfo    = texInfos[material.pbrBaseColorTexture];
    float2          texCoord = mul(float3(texcoord0.xy, 1), tinfo.uvTransform).xy;

    baseColorAlpha *= allTextures[tinfo.index].SampleLevel(texCoord, 0).a;
  }

  float opacity;
  if(material.alphaMode == AlphaMode::eAlphaModeMask)
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
  payload.hitT = INFINITE;  // Default when not hitting anything

  RayDesc  r = { ray.origin, 0, ray.direction, INFINITE };
  RayQuery q;
  q.TraceRayInline(topLevelAS, RAY_FLAG_CULL_BACK_FACING_TRIANGLES, 0xFF, r);

  while(q.Proceed())
  {
    if(hitTest(q, seed))
    {
      q.CommitNonOpaqueTriangleHit();
    }
  }

  if(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
  {
    float2 barycentricCoords = q.CommittedTriangleBarycentrics();

    int rnodeID = q.CommittedInstanceIndex();
    int rprimID = q.CommittedInstanceID();
    int triID   = q.CommittedPrimitiveIndex();

    float4x3 worldToObject = q.CommittedWorldToObject4x3();
    float4x3 objectToWorld = q.CommittedObjectToWorld4x3();

    float  hitT              = q.CommittedRayT();
    float3 worldRayDirection = r.Direction;

    // Retrieve the Primitive mesh buffer information
    GltfRenderPrimitive* meshes     = pushConst.gltfScene->renderPrimitives;
    GltfRenderPrimitive  renderPrim = meshes[rprimID];

    HitState hit = getHitState(renderPrim, barycentricCoords, worldToObject, objectToWorld, triID, worldRayDirection);

    payload.hitT     = hitT;
    payload.rprimID  = rprimID;
    payload.rnodeID  = rnodeID;
    payload.hitState = hit;
  }
}

//-----------------------------------------------------------------------
// Shadow ray - return true if a ray hits anything
//
bool traceShadow(Ray ray, float maxDist, inout uint seed)
{
  RayDesc  r = { ray.origin, 0, ray.direction, maxDist };
  RayQuery q;
  q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, r);

  while(q.Proceed())
  {
    if(hitTest(q, seed))
    {
      q.CommitNonOpaqueTriangleHit();  // rayQueryConfirmIntersectionEXT(rayQuery);
    }
  }

  return q.CommittedStatus() != COMMITTED_NOTHING;
}

//-----------------------------------------------------------------------
// Path tracing
//
// This function:
// 1. Traces rays through a scene, bouncing them off surfaces according to their material properties (like reflection, transmission, etc.) up to a maximum depth
// 2. At each intersection, it calculates direct lighting contribution from light sources (currently only sky/sun) and samples the BSDF to determine the next ray direction
// 3. Accumulates radiance along the path while applying Russian Roulette for optimization, handling both surface and volumetric effects, and returns the final color contribution for that ray path
//
float3 pathTrace(Ray ray, inout uint seed)
{
  float3 radiance   = float3(0.0F, 0.0F, 0.0F);
  float3 throughput = float3(1.0F, 1.0F, 1.0F);
  bool   isInside   = false;

  // Setting up the material
  GltfShadeMaterial*   materials       = pushConst.gltfScene->materials;         // Buffer of materials
  GltfTextureInfo*     texInfos        = pushConst.gltfScene->textureInfos;      // Buffer of textures
  GltfRenderNode*      renderNodes     = pushConst.gltfScene->renderNodes;       // Buffer of render nodes
  GltfRenderPrimitive* renderPrimitive = pushConst.gltfScene->renderPrimitives;  // Buffer of meshes
  HitPayload           payload;

  // Path tracing loop, until the ray hits the environment or the maximum depth is reached or the ray is absorbed
  for(int depth = 0; depth < pushConst.maxDepth; depth++)
  {
    // Trace the ray through the scene
    traceRay(ray, payload, seed);

    // Hitting the environment, then exit
    if(payload.hitT == INFINITE)
    {
      if(pushConst.environmentType == 0)
      {
        float3 sky_color = evalPhysicalSky(*pushConst.skyParams, ray.direction);
        return radiance + (sky_color * throughput);
      }
      else
      {
        float2 uv       = getSphericalUv(ray.direction);
        float4 hdrColor = hdrIBL.SampleLevel(uv, 0);
        return radiance + (hdrColor.xyz * throughput);
      }
    }

    // Getting the hit information (primitive/mesh that was hit)
    HitState hit = payload.hitState;

    // Setting up the material
    GltfRenderPrimitive renderPrim    = renderPrimitive[payload.rprimID];  // Primitive information
    GltfRenderNode      renderNode    = renderNodes[payload.rnodeID];      // Node information
    int                 materialIndex = max(0, renderNode.materialID);     // Material ID of hit mesh
    GltfShadeMaterial   material      = materials[materialIndex];          // Material of the hit object

    material.pbrBaseColorFactor *= hit.color;  // Modulate the base color with the vertex color

    // Evaluate the material at the hit point
    MeshState   mesh   = MeshState(hit.nrm, hit.tangent, hit.bitangent, hit.geonrm, hit.uv, isInside);
    PbrMaterial pbrMat = evaluateMaterial(material, mesh, allTextures, texInfos);

    // Adding emissive
    radiance += pbrMat.emissive * throughput;

    // Apply volume attenuation
    bool thin_walled = pbrMat.isThinWalled;
    if(isInside && !thin_walled)
    {
      const float3 abs_coeff = absorptionCoefficient(pbrMat);
      throughput.x *= abs_coeff.x > 0.0 ? exp(-abs_coeff.x * payload.hitT) : 1.0;
      throughput.y *= abs_coeff.y > 0.0 ? exp(-abs_coeff.y * payload.hitT) : 1.0;
      throughput.z *= abs_coeff.z > 0.0 ? exp(-abs_coeff.z * payload.hitT) : 1.0;
    }

    float3 contribution = float3(0);  // Direct lighting contribution

    // Light contribution; can be environment or punctual lights
    DirectLight directLight;
    sampleLights(hit.pos, pbrMat.N, ray.direction, seed, directLight);

    // Evaluation of direct light (sun)
    const bool nextEventValid = ((dot(directLight.direction, hit.geonrm) > 0.0F) != isInside) && directLight.pdf != 0.0F;
    if(nextEventValid)
    {
      // Evaluate the BSDF at the hit point
      BsdfEvaluateData evalData;
      evalData.k1 = -ray.direction;
      evalData.k2 = directLight.direction;
      evalData.xi = float3(rand(seed), rand(seed), rand(seed));
      bsdfEvaluate(evalData, pbrMat);

      // If the PDF is greater than 0, then we can sample the BSDF
      if(evalData.pdf > 0.0)
      {
        // Weight for combining light and BSDF sampling strategies (Multiple Importance Sampling)
        const float mis_weight = (directLight.pdf == DIRAC) ? 1.0F : directLight.pdf / (directLight.pdf + evalData.pdf);

        // sample weight
        const float3 w = throughput * directLight.radianceOverPdf * mis_weight;
        contribution += w * evalData.bsdf_diffuse;
        contribution += w * evalData.bsdf_glossy;
      }
    }

    {
      // Sample the BSDF
      BsdfSampleData sampleData;
      sampleData.k1 = -ray.direction;                              // outgoing direction
      sampleData.xi = float3(rand(seed), rand(seed), rand(seed));  // random number
      bsdfSample(sampleData, pbrMat);

      // Update the throughput
      throughput *= sampleData.bsdf_over_pdf;
      ray.direction = sampleData.k2;  // new direction

      // If the ray is absorbed, then break
      if(sampleData.event_type == BSDF_EVENT_ABSORB)
      {
        break;  // Need to add the contribution ?
      }
      else
      {
        // Continue path
        bool isSpecular     = (sampleData.event_type & BSDF_EVENT_IMPULSE) != 0;
        bool isTransmission = (sampleData.event_type & BSDF_EVENT_TRANSMISSION) != 0;

        float3 offsetDir = dot(ray.direction, hit.geonrm) > 0 ? hit.geonrm : -hit.geonrm;
        ray.origin       = offsetRay(hit.pos, offsetDir);

        if(isTransmission)
        {
          isInside = !isInside;
        }
      }
    }

    // We are adding the contribution to the radiance only if the ray is not occluded by an object.
    if(nextEventValid)
    {
      // Trace the shadow ray
      Ray  shadowRay = Ray(ray.origin, directLight.direction);
      bool inShadow  = traceShadow(shadowRay, directLight.distance, seed);
      if(!inShadow)
      {
        radiance += contribution;
      }
    }

    // Russian-Roulette (minimizing live state)
    float rrPcont = min(max(throughput.x, max(throughput.y, throughput.z)) + 0.001F, 0.95F);
    if(rand(seed) >= rrPcont)
      break;                // paths with low throughput that won't contribute
    throughput /= rrPcont;  // boost the energy of the non-terminated paths
  }

  // Return the radiance
  return radiance;
}

//-----------------------------------------------------------------------
// Sampling the pixel
//-----------------------------------------------------------------------
float3 samplePixel(inout uint seed, float2 launchID, float2 launchSize)
{
  // Subpixel jitter: send the ray through a different position inside the pixel each time, to provide antialiasing.
  const float2 subpixel_jitter = pushConst.frame == 0 ? float2(0.5f, 0.5f) : float2(rand(seed), rand(seed));
  const float2 clipCoords      = (launchID + subpixel_jitter) / launchSize * 2.0 - 1.0;
  float4       viewCoords      = mul(float4(clipCoords, -1.0, 1.0), pushConst.cameraInfo->projInv);
  viewCoords /= viewCoords.w;

  Ray ray;
  ray.origin    = float3(pushConst.cameraInfo->viewInv[3].xyz);
  ray.direction = normalize(mul(viewCoords, pushConst.cameraInfo->viewInv).xyz - ray.origin);

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
void computeMain(uint3 threadIdx: SV_DispatchThreadID)
{
  float2 launchID = (float2)threadIdx.xy;
  uint2  launchSize;
  outImage.GetDimensions(launchSize.x, launchSize.y);

  if(launchID.x >= launchSize.x || launchID.y >= launchSize.y)
    return;

  if(launchID.x == pushConst.mouseCoord.x && launchID.y == pushConst.mouseCoord.y)
  {
    doDebug = true;
  }

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

  // float2 uv = launchID / launchSize;
  // pixel_color = float3(uv, 0);
  // uint width;
  // uint height;
  // allTextures[0].GetDimensions(width, height);
  // if (doDebug)
  //     printf("Width: %d, Height:%d\n", width, height);
  // pixel_color = allTextures[0][int2(threadIdx.xy)].xyz;

  // Saving result
  if(first_frame)
  {  // First frame, replace the value in the buffer
    outImage[int2(launchID)] = float4(pixel_color, 1.0);
  }
  else
  {  // Do accumulation over time
    float  a                 = 1.0F / float(pushConst.frame + 1);
    float3 old_color         = outImage[int2(launchID)].xyz;
    outImage[int2(launchID)] = float4(lerp(old_color, pixel_color, a), 1.0F);
  }
}

