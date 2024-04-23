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
#ifndef DH_LIGHTING_H
#define DH_LIGHTING_H 1

static const int eLightTypeNone = 0;
static const int eLightTypeDirectional = 1;
static const int eLightTypeSpot = 2;
static const int eLightTypePoint = 3;

//-----------------------------------------------------------------------
// Use for light/env contribution
struct VisibilityContribution
{
  float3 radiance; // Radiance at the point if light is visible
  float3 lightDir; // Direction to the light, to shoot shadow ray
  float lightDist; // Distance to the light (1e32 for infinite or sky)
  bool visible; // true if in front of the face and should shoot shadow ray
};

struct LightContrib
{
  float3 incidentVector;
  float halfAngularSize;
  float3 intensity;
};

struct Light
{
  float3 direction;
  int type;

  float3 position;
  float radius;

  float3 color;
  float intensity; // illuminance (lm/m2) for directional lights, luminous intensity (lm/sr) for positional lights

  float angularSizeOrInvRange; // angular size for directional lights, 1/range for spot and point lights
  float innerAngle;
  float outerAngle;
  float outOfBoundsShadow;
};


#endif  // DH_LIGHTING_H
