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

#ifndef PBR_MAT_STRUCT_H
#define PBR_MAT_STRUCT_H 1

#include "functions.hlsli"

struct PbrMaterial
{
  float3 baseColor; // base color
  float opacity; // 1 = opaque, 0 = fully transparent
  float2 roughness; // 0 = smooth, 1 = rough (anisotropic: x = U, y = V)
  float metallic; // 0 = dielectric, 1 = metallic
  float3 emissive; // emissive color

  float3 N; // shading normal
  float3 T; // shading normal
  float3 B; // shading normal
  float3 Ng; // geometric normal


  float ior1; // index of refraction : current medium (i.e. air)
  float ior2; // index of refraction : the other side (i.e. glass)

  float specular; // weight of the dielectric specular layer
  float3 specularColor; // color of the dielectric specular layer
  float transmission; // KHR_materials_transmission

  float3 attenuationColor; // KHR_materials_volume
  float attenuationDistance; //
  float thickness; // Replace for isThinWalled?

  float clearcoat; // KHR_materials_clearcoat
  float clearcoatRoughness; //
  float3 Nc; // clearcoat normal

  float iridescence;
  float iridescenceIor;
  float iridescenceThickness;

  float3 sheenColor;
  float sheenRoughness;
};

PbrMaterial defaultPbrMaterial()
{
  PbrMaterial mat;
  mat.baseColor = float3(1.0F,1.0F,1.0F);
  mat.opacity = 1.0F;
  mat.roughness = float2(1.0F,1.0F);
  mat.metallic = 1.0F;
  mat.emissive = float3(0.0F,0.0F,0.0F);

  mat.N = float3(0.0F,0.0F,1.0F);
  mat.Ng = float3(0.0F,0.0F,1.0F);
  mat.T = float3(1.0F,0.0F,0.0F);
  mat.B = float3(0.0F,1.0F,0.0F);

  mat.ior1 = 1.0F;
  mat.ior2 = 1.5F;

  mat.specular = 1.0F;
  mat.specularColor = float3(1.0F,1.0F,1.0F);
  mat.transmission = 0.0F;

  mat.attenuationColor = float3(1.0F,1.0F,1.0F);
  mat.attenuationDistance = 1.0F;
  mat.thickness = 0.0F;

  mat.clearcoat = 0.0F;
  mat.clearcoatRoughness = 0.01F;
  mat.Nc = float3(0.0F,0.0F,1.0F);

  mat.iridescence = 0.0F;
  mat.iridescenceIor = 1.5F;
  mat.iridescenceThickness = 0.1F;

  mat.sheenColor = float3(0.0F,0.0F,0.0F);
  mat.sheenRoughness = 0.0F;

  return mat;
}

PbrMaterial defaultPbrMaterial(float3 baseColor,float metallic,float roughness,float3 N,float3 Ng)
{
  PbrMaterial mat = defaultPbrMaterial();
  mat.baseColor = baseColor;
  mat.metallic = metallic;
  float r2 = roughness * roughness;
  mat.roughness = float2(r2, r2);
  mat.Ng = Ng;
  mat.N = N;
  orthonormalBasis(mat.N,mat.T,mat.B);

  return mat;
}
#endif
