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


#ifndef SKY_HLSLI
#define SKY_HLSLI 1

struct ProceduralSkyShaderParameters
{
  float3  directionToLight;
  float angularSizeOfLight;

  float3  lightColor;
  float glowSize;

  float3  skyColor;
  float glowIntensity;

  float3  horizonColor;
  float horizonSize;

  float3  groundColor;
  float glowSharpness;

  float3  directionUp;
  float pad1;
};

struct SkyPushConstant
{
  float4x4 mvp;
};

inline ProceduralSkyShaderParameters initSkyShaderParameters()
{
  ProceduralSkyShaderParameters p;
  p.directionToLight   = float3(0.0F, 0.707F, 0.707F);
  p.angularSizeOfLight = 0.059F;
  p.lightColor         = float3(1.0F, 1.0F, 1.0F);
  p.skyColor           = float3(0.17F, 0.37F, 0.65F);
  p.horizonColor       = float3(0.50F, 0.70F, 0.92F);
  p.groundColor        = float3(0.62F, 0.59F, 0.55F);
  p.directionUp        = float3(0.F, 1.F, 0.F);
  p.horizonSize        = 0.5F;    // +/- degrees
  p.glowSize           = 0.091F;  // degrees, starting from the edge of the light disk
  p.glowIntensity      = 0.9F;    // [0-1] relative to light intensity
  p.glowSharpness      = 4.F;     // [1-10] is the glow power exponent

  return p;
}

inline float3 proceduralSky(ProceduralSkyShaderParameters params, float3 direction, float angularSizeOfPixel)
{
  float elevation   = asin(clamp(dot(direction, params.directionUp), -1.0F, 1.0F));
  float top         = smoothstep(0.F, params.horizonSize, elevation);
  float bottom      = smoothstep(0.F, params.horizonSize, -elevation);
  float3  environment = lerp(lerp(params.horizonColor, params.groundColor, bottom), params.skyColor, top);

  float angle_to_light    = acos(clamp(dot(direction, params.directionToLight), 0.0F, 1.0F));
  float half_angular_size = params.angularSizeOfLight * 0.5F;
  float light_intensity =
      clamp(1.0F - smoothstep(half_angular_size - angularSizeOfPixel * 2.0F, half_angular_size + angularSizeOfPixel * 2.0F, angle_to_light),
            0.0F, 1.0F);
  light_intensity = pow(light_intensity, 4.0F);
  float glow_input =
      clamp(2.0F * (1.0F - smoothstep(half_angular_size - params.glowSize, half_angular_size + params.glowSize, angle_to_light)),
            0.0F, 1.0F);
  float glow_intensity = params.glowIntensity * pow(glow_input, params.glowSharpness);
  float3  light          = max(light_intensity, glow_intensity) * params.lightColor;

  return environment + light;
}

#endif  // SKY_HLSLI
