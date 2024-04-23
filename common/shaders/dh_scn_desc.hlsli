
/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2022 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

/// @DOC_SKIP

#ifndef DH_SCN_DESC_H
#define DH_SCN_DESC_H 1

struct InstanceInfo
{
  float4x4 objectToWorld;
  float4x4 worldToObject;
  int materialID;
};

struct Vertex
{
  float4 position; // POS.xyz + UV.x
  float4 normal; // NRM.xyz + UV.y
  float4 tangent; // TNG.xyz + sign: 1, -1
};

//struct PrimMeshInfo
//{
//  uint64_t vertexAddress;
//  uint64_t indexAddress;
//  int      materialIndex;
//};

//struct SceneDescription
//{
//  uint64_t materialAddress;
//  uint64_t instInfoAddress;
//  uint64_t primInfoAddress;
//};

// shadingModel
#define MATERIAL_METALLICROUGHNESS 0
#define MATERIAL_SPECULARGLOSSINESS 1
// alphaMode
#define ALPHA_OPAQUE 0
#define ALPHA_MASK 1
#define ALPHA_BLEND 2

struct GltfShadeMaterial
{
  // Core
  float4 pbrBaseColorFactor;
  float3 emissiveFactor;
  int pbrBaseColorTexture;

  int   normalTexture;
  float normalTextureScale;
  int   _pad0;
  float pbrRoughnessFactor;

  float pbrMetallicFactor;
  int   pbrMetallicRoughnessTexture;

  int   emissiveTexture;
  int   alphaMode;
  float alphaCutoff;

  // KHR_materials_transmission
  float transmissionFactor;
  int transmissionTexture;
  // KHR_materials_ior
  float ior;
  // KHR_materials_volume
  float3 attenuationColor;
  float thicknessFactor;
  int thicknessTexture;
  bool thinWalled;
  float attenuationDistance;
  // KHR_materials_clearcoat
  float clearcoatFactor;
  float clearcoatRoughness;
  int clearcoatTexture;
  int clearcoatRoughnessTexture;
  int clearcoatNormalTexture;
  // KHR_materials_specular
  float specularFactor;
  int specularTexture;
  float3 specularColorFactor;
  int specularColorTexture;
  // KHR_texture_transform
  float3x3 uvTransform;
};

#endif
