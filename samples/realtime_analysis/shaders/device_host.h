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
 * SPDX-FileCopyrightText: Copyright (c) 2014-2023 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

// Shared between Host and Device

// Number of particles in the simulation
#define NUM_PARTICLES 15000

#define WORKGROUP_SIZE 128

#ifdef _glsl
#define static
#endif

static const int eFrameInfo         = 0;
static const int eParticles         = 1;
static const int eFragInspectorData = 2;
static const int eFragInspectorMeta = 3;

static const int eCompParticles    = 0;
static const int eCompSort         = 1;
static const int eCompSetting      = 2;
static const int eThreadInspection = 3;
static const int eThreadMetadata   = 4;

struct PushConstant
{
  vec4  color;
  uint  groupWidth;
  uint  groupHeight;
  uint  stepIndex;
};

struct FrameInfo
{
  mat4  proj;
  float scale;
  float radius;
};

struct Particle
{
  vec2 position;
  vec2 predictedPosition;
  vec2 velocity;
  vec2 density;
};

struct SpatialInfo
{
  uvec3 indices;
  uint  offsets;
};


struct ParticleSetting
{
  uint  numParticles;
  float gravity;
  float deltaTime;
  float collisionDamping;
  float smoothingRadius;
  float targetDensity;
  float pressureMultiplier;
  float nearPressureMultiplier;
  float viscosityStrength;
  float interactionInputStrength;
  float interactionInputRadius;

  float poly6ScalingFactor;
  float spikyPow3ScalingFactor;
  float spikyPow2ScalingFactor;
  float spikyPow3DerivativeScalingFactor;
  float spikyPow2DerivativeScalingFactor;

  // aligned 64
  float boundsMultiplier;
  int   _pad0;
  //
  vec2 boundsSize;
  vec2 interactionInputPoint;
  vec2 obstacleSize;
  vec2 obstacleCentre;
};
