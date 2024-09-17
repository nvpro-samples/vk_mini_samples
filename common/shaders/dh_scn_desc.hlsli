
/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/// @DOC_SKIP

#ifndef DH_SCN_DESC_H
#define DH_SCN_DESC_H 1

struct RenderNode
{
  float4x4 objectToWorld;
  float4x4 worldToObject;
  int materialID;
  int renderPrimID;
};

#ifdef __SLANG__

// This is all the information about a vertex buffer
struct VertexBuf
{
  float3* positionAddress;
  float3* normalAddress;
  uint* colorAddress;
  float4* tangentAddress;
  float2* texCoord0Address;
  float2* texCoord1Address;
};

// This is the GLTF Primitive structure, promoted to a Mesh.
struct RenderPrimitive
{
  uint3*  indexAddress;
  VertexBuf vertexBuffer;
};

// The scene description is a pointer to the material, render node and primitive
// The buffers are all arrays of the above structures
struct SceneDescription
{
  GltfShadeMaterial* materialAddress;
  RenderNode* renderNodeAddress;
  RenderPrimitive* renderPrimitiveAddress;
};

#elif __HLSL_VERSION > 1


// This is all the information about a vertex buffer
struct VertexBuf
{
  uint64_t positionAddress;
  uint64_t normalAddress;
  uint64_t colorAddress;
  uint64_t tangentAddress;
  uint64_t texCoord0Address;
  uint64_t texCoord1Address;
};

// This is the GLTF Primitive structure, promoted to a Mesh.
struct RenderPrimitive
{
  uint64_t  indexAddress;
  VertexBuf vertexBuffer;
};

// The scene description is a pointer to the material, render node and primitive
// The buffers are all arrays of the above structures
struct SceneDescription
{
  uint64_t materialAddress;
  uint64_t renderNodeAddress;
  uint64_t renderPrimitiveAddress;
};


#endif


// shadingModel
#define MATERIAL_METALLICROUGHNESS 0
#define MATERIAL_SPECULARGLOSSINESS 1
// alphaMode
#define ALPHA_OPAQUE 0
#define ALPHA_MASK 1
#define ALPHA_BLEND 2

struct GltfTextureInfo
{
  float3x3 uvTransform; // 36 bytes (3x3 matrix)
  int index; // 4 bytes
  int texCoord; // 4 bytes
}; // Total: 44 bytes


struct GltfShadeMaterial
{
  float4 pbrBaseColorFactor; // offset 0 - 16 bytes    - glTF Core
  float3 emissiveFactor; // offset 16 - 12 bytes
  float normalTextureScale; // offset 28 - 4 bytes
  float pbrRoughnessFactor; // offset 32 - 4 bytes
  float pbrMetallicFactor; // offset 36 - 4 bytes
  int alphaMode; // offset 40 - 4 bytes
  float alphaCutoff; // offset 44 - 4 bytes
  float transmissionFactor; // offset 48 - 4 bytes    - KHR_materials_transmission
  float ior; // offset 52 - 4 bytes    - KHR_materials_ior
  float3 attenuationColor; // offset 56 - 12 bytes   - KHR_materials_volume
  float thicknessFactor; // offset 68 - 4 bytes
  float attenuationDistance; // offset 72 - 4 bytes
  float clearcoatFactor; // offset 76 - 4 bytes    - KHR_materials_clearcoat
  float clearcoatRoughness; // offset 80 - 4 bytes
  float3 specularColorFactor; // offset 84 - 12 bytes   - KHR_materials_specular
  float specularFactor; // offset 96 - 4 bytes
  int unlit; // offset 100 - 4 bytes   - KHR_materials_unlit
  float iridescenceFactor; // offset 104 - 4 bytes   - KHR_materials_iridescence
  float iridescenceThicknessMaximum; // offset 108 - 4 bytes
  float iridescenceThicknessMinimum; // offset 112 - 4 bytes
  float iridescenceIor; // offset 116 - 4 bytes
  float anisotropyStrength; // offset 120 - 4 bytes   - KHR_materials_anisotropy
  float2 anisotropyRotation; // offset 124 - 8 bytes
  float sheenRoughnessFactor; // offset 132 - 4 bytes   - KHR_materials_sheen
  float3 sheenColorFactor; // offset 136 - 12 bytes

  // Texture infos (44 bytes each)
  GltfTextureInfo pbrBaseColorTexture; // offset 148 - 44 bytes
  GltfTextureInfo normalTexture; // offset 192 - 44 bytes
  GltfTextureInfo pbrMetallicRoughnessTexture; // offset 236 - 44 bytes
  GltfTextureInfo emissiveTexture; // offset 280 - 44 bytes
  GltfTextureInfo transmissionTexture; // offset 324 - 44 bytes
  GltfTextureInfo thicknessTexture; // offset 368 - 44 bytes
  GltfTextureInfo clearcoatTexture; // offset 412 - 44 bytes
  GltfTextureInfo clearcoatRoughnessTexture; // offset 456 - 44 bytes
  GltfTextureInfo clearcoatNormalTexture; // offset 500 - 44 bytes
  GltfTextureInfo specularTexture; // offset 544 - 44 bytes
  GltfTextureInfo specularColorTexture; // offset 588 - 44 bytes
  GltfTextureInfo iridescenceTexture; // offset 632 - 44 bytes
  GltfTextureInfo iridescenceThicknessTexture; // offset 676 - 44 bytes
  GltfTextureInfo anisotropyTexture; // offset 720 - 44 bytes
  GltfTextureInfo sheenColorTexture; // offset 764 - 44 bytes
  GltfTextureInfo sheenRoughnessTexture; // offset 808 - 44 bytes

  // 4 bytes of padding to align to 16-byte boundary
  int padding; // offset 852 - 4 bytes
}; // Total size: 856 bytes

GltfTextureInfo getDefaultTextureInfo()
{
  GltfTextureInfo t;
  t.uvTransform = float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
  t.index = -1;
  t.texCoord = 0;
  return t;
}

GltfShadeMaterial getDefaultGltfMaterial()
{
  GltfShadeMaterial m;
  m.pbrBaseColorFactor = float4(1, 1, 1, 1);
  m.emissiveFactor = float3(0, 0, 0);
  m.pbrBaseColorTexture = getDefaultTextureInfo();
  m.normalTexture = getDefaultTextureInfo();
  m.normalTextureScale = 1;
  m.pbrRoughnessFactor = 1;
  m.pbrMetallicFactor = 1;
  m.pbrMetallicRoughnessTexture = getDefaultTextureInfo();
  m.emissiveTexture = getDefaultTextureInfo();
  m.alphaMode = ALPHA_OPAQUE;
  m.alphaCutoff = 0.5;
  m.transmissionFactor = 0;
  m.transmissionTexture = getDefaultTextureInfo();
  m.ior = 1.5;
  m.attenuationColor = float3(1, 1, 1);
  m.thicknessFactor = 0;
  m.thicknessTexture = getDefaultTextureInfo();
  m.attenuationDistance = 0;
  m.clearcoatFactor = 0;
  m.clearcoatRoughness = 0;
  m.clearcoatTexture = getDefaultTextureInfo();
  m.clearcoatRoughnessTexture = getDefaultTextureInfo();
  m.clearcoatNormalTexture = getDefaultTextureInfo();
  m.specularFactor = 0;
  m.specularTexture = getDefaultTextureInfo();
  m.specularColorFactor = float3(1, 1, 1);
  m.specularColorTexture = getDefaultTextureInfo();
  
  return m;
};

#endif
