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

// Code converted from: https://github.com/SebLague/Fluid-Sim

#version 450 // This is adding support fot GLSL 
// Includes
#include "device_host.h"

[[vk::push_constant]] ConstantBuffer<PushConstant> pushC;
[[vk::binding(eCompParticles)]] RWStructuredBuffer<Particle> particles;
[[vk::binding(eCompSort)]] RWStructuredBuffer<SpatialInfo> spatialInfo;
[[vk::binding(eCompSetting)]] ConstantBuffer<ParticleSetting> setting;

 #define INSPECTOR_MODE_COMPUTE 
 #define INSPECTOR_DESCRIPTOR_SET 0 
 #define INSPECTOR_INSPECTION_DATA_BINDING eThreadInspection
 #define INSPECTOR_METADATA_BINDING eThreadMetadata
 #include "common/shaders/dh_inspector.hlsli"

#include "fluid_maths_2D.glsl"

static const int2 offsets2D[9] =
{
  int2(-1, 1),
  int2(0, 1),
  int2(1, 1),
  int2(-1, 0),
  int2(0, 0),
  int2(1, 0),
  int2(-1, -1),
  int2(0, -1),
  int2(1, -1),
};

// Constants used for hashing
static const uint hashK1 = 15823;
static const uint hashK2 = 9737333;

// Convert floating point position into an integer cell coordinate
int2 getCell2D(float2 position, float radius)
{
  return int2(floor(position / radius));
}

// Hash cell coordinate to a single unsigned integer
uint hashCell2D(int2 cell)
{
  cell = int2(uint2(cell.x, cell.y));
  uint a = cell.x * hashK1;
  uint b = cell.y * hashK2;
  return (a + b);
}

uint keyFromHash(uint hash, uint tableSize)
{
  return hash % tableSize;
}

float densityKernel(float dst, float radius)
{
  return spikyKernelPow2(dst, radius);
}

float nearDensityKernel(float dst, float radius)
{
  return spikyKernelPow3(dst, radius);
}

float densityDerivative(float dst, float radius)
{
  return derivativeSpikyPow2(dst, radius);
}

float nearDensityDerivative(float dst, float radius)
{
  return derivativeSpikyPow3(dst, radius);
}

float viscosityKernel(float dst, float radius)
{
  return smoothingKernelPoly6(dst, setting.smoothingRadius);
}

float2 calculateDensity(float2 pos)
{
  int2 originCell = getCell2D(pos, setting.smoothingRadius);
  float sqrRadius = setting.smoothingRadius * setting.smoothingRadius;
  float density = 0;
  float nearDensity = 0;

  // Neighbour search
  for (int i = 0; i < 9; i++)
  {
    uint hash = hashCell2D(originCell + offsets2D[i]);
    uint key = keyFromHash(hash, setting.numParticles);
    uint currIndex = spatialInfo[key].offsets;

    while (currIndex < setting.numParticles)
    {
      uint3 indexData = spatialInfo[currIndex].indices;
      currIndex++;
      // Exit if no longer looking at correct bin
      if (indexData.z != key)
        break;
      // Skip if hash does not match
      if (indexData.y != hash)
        continue;

      uint neighbourIndex = indexData[0];
      float2 neighbourPos = particles[neighbourIndex].predictedPosition;
      float2 offsetToNeighbour = neighbourPos - pos;
      float sqrDstToNeighbour = dot(offsetToNeighbour, offsetToNeighbour);

      // Skip if not within radius
      if (sqrDstToNeighbour > sqrRadius)
        continue;

      // Calculate density and near density
      float dst = sqrt(sqrDstToNeighbour);
      density += densityKernel(dst, setting.smoothingRadius);
      nearDensity += nearDensityKernel(dst, setting.smoothingRadius);
    }
  }
  
  return float2(density, nearDensity);
}

float pressureFromDensity(float density)
{
  return (density - setting.targetDensity) * setting.pressureMultiplier;
}

float nearPressureFromDensity(float nearDensity)
{
  return setting.nearPressureMultiplier * nearDensity;
}

float2 externalForces(float2 pos, float2 velocity)
{
  // Gravity
  float2 gravityAccel = float2(0, setting.gravity);

  // Input interactions modify gravity
  if (setting.interactionInputStrength != 0)
  {
    float2 inputPointOffset = setting.interactionInputPoint - pos;
    float sqrDst = dot(inputPointOffset, inputPointOffset);
    if (sqrDst < setting.interactionInputRadius * setting.interactionInputRadius)
    {
      float dst = sqrt(sqrDst);
      float edgeT = (dst / setting.interactionInputRadius);
      float centreT = 1 - edgeT;
      float2 dirToCentre = inputPointOffset / dst;

      float gravityWeight = 1 - (centreT * clamp(setting.interactionInputStrength / 10, 0, 1));
      float2 accel = gravityAccel * gravityWeight + dirToCentre * centreT * setting.interactionInputStrength;
      accel -= velocity * centreT;
      return accel;
    }
  }

  return gravityAccel;
}

void handleCollisions(uint particleIndex)
{
  float2 pos = particles[particleIndex].position;
  float2 vel = particles[particleIndex].velocity;

  // Keep particle inside bounds
  const float2 halfSize = setting.boundsSize * 0.5;
  float2 edgeDst = halfSize - abs(pos);

  if (edgeDst.x <= 0)
  {
    pos.x = halfSize.x * sign(pos.x);
    vel.x *= -1 * setting.collisionDamping;
  }
  if (edgeDst.y <= 0)
  {
    pos.y = halfSize.y * sign(pos.y);
    vel.y *= -1 * setting.collisionDamping;
  }

  // Collide particle against the test obstacle
  const float2 obstacleHalfSize = setting.obstacleSize * 0.5;
  float2 obstacleEdgeDst = obstacleHalfSize - abs(pos - setting.obstacleCentre);

  if (obstacleEdgeDst.x >= 0 && obstacleEdgeDst.y >= 0)
  {
    if (obstacleEdgeDst.x < obstacleEdgeDst.y)
    {
      pos.x = obstacleHalfSize.x * sign(pos.x - setting.obstacleCentre.x) + setting.obstacleCentre.x;
      vel.x *= -1 * setting.collisionDamping;
    }
    else
    {
      pos.y = obstacleHalfSize.y * sign(pos.y - setting.obstacleCentre.y) + setting.obstacleCentre.y;
      vel.y *= -1 * setting.collisionDamping;
    }
  }

  // Update position and velocity
  particles[particleIndex].position = pos;
  particles[particleIndex].velocity = vel;
}

float2 calculatePressure(uint particleID)
{
  float density = particles[particleID].density.x;
  float densityNear = particles[particleID].density.y;
  float pressure = pressureFromDensity(density);
  float nearPressure = nearPressureFromDensity(densityNear);
  float2 pressureForce = float2(0);

  float2 pos = particles[particleID].predictedPosition;
  int2 originCell = getCell2D(pos, setting.smoothingRadius);
  float sqrRadius = setting.smoothingRadius * setting.smoothingRadius;

  uint numElements = 0;

  // Neighbour search
  for (int i = 0; i < 9; i++)
  {
    uint hash = hashCell2D(originCell + offsets2D[i]);
    uint key = keyFromHash(hash, setting.numParticles);
    uint currIndex = spatialInfo[key].offsets;

    while (currIndex < setting.numParticles)
    {
      uint3 indexData = spatialInfo[currIndex].indices;
      currIndex++;
      // Exit if no longer looking at correct bin
      if (indexData.z != key)
        break;
      // Skip if hash does not match
      if (indexData.y != hash)
        continue;

      uint neighbourIndex = indexData[0];
      // Skip if looking at self
      if (neighbourIndex == particleID)
        continue;

      float2 neighbourPos = particles[neighbourIndex].predictedPosition;
      float2 offsetToNeighbour = neighbourPos - pos;
      float sqrDstToNeighbour = dot(offsetToNeighbour, offsetToNeighbour);

      // Skip if not within radius
      if (sqrDstToNeighbour > sqrRadius)
        continue;

      // Calculate pressure force
      float dst = sqrt(sqrDstToNeighbour);
      float2 dirToNeighbour = dst > 0 ? offsetToNeighbour / dst : float2(0, 1);

      float neighbourDensity = particles[neighbourIndex].density.x;
      float neighbourNearDensity = particles[neighbourIndex].density.y;
      float neighbourPressure = pressureFromDensity(neighbourDensity);
      float neighbourNearPressure = nearPressureFromDensity(neighbourNearDensity);

      float sharedPressure = (pressure + neighbourPressure) * 0.5;
      float sharedNearPressure = (nearPressure + neighbourNearPressure) * 0.5;

      pressureForce += dirToNeighbour * densityDerivative(dst, setting.smoothingRadius) * sharedPressure / neighbourDensity;
      pressureForce += dirToNeighbour * nearDensityDerivative(dst, setting.smoothingRadius) * sharedNearPressure / neighbourNearDensity;
      ++numElements;
    }
  }
  
  inspect32BitValue(2, numElements);

  float2 acceleration = pressureForce / density;
  return acceleration;
}

float2 calculateViscosity(uint particleID)
{
  float2 pos = particles[particleID].predictedPosition;
  int2 originCell = getCell2D(pos, setting.smoothingRadius);
  float sqrRadius = setting.smoothingRadius * setting.smoothingRadius;
  float2 viscosityForce = float2(0);
  float2 velocity = particles[particleID].velocity;

  for (int i = 0; i < 9; i++)
  {
    uint hash = hashCell2D(originCell + offsets2D[i]);
    uint key = keyFromHash(hash, setting.numParticles);
    uint currIndex = spatialInfo[key].offsets;

    while (currIndex < setting.numParticles)
    {
      uint3 indexData = spatialInfo[currIndex].indices;
      currIndex++;
      // Exit if no longer looking at correct bin
      if (indexData.z != key)
        break;
      // Skip if hash does not match
      if (indexData.y != hash)
        continue;

      uint neighbourIndex = indexData[0];
      // Skip if looking at self
      if (neighbourIndex == particleID)
        continue;

      float2 neighbourPos = particles[neighbourIndex].predictedPosition;
      float2 offsetToNeighbour = neighbourPos - pos;
      float sqrDstToNeighbour = dot(offsetToNeighbour, offsetToNeighbour);

      // Skip if not within radius
      if (sqrDstToNeighbour > sqrRadius)
        continue;

      float dst = sqrt(sqrDstToNeighbour);
      float2 neighbourVelocity = particles[neighbourIndex].velocity;
      viscosityForce += (neighbourVelocity - velocity) * viscosityKernel(dst, setting.smoothingRadius);
    }
  }
  
  return viscosityForce;
}


[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void calculateDensity(uint3 threadIdx : SV_DispatchThreadID)
{
  uint particleID = threadIdx.x;
  if (particleID >= setting.numParticles)
    return;

  float2 pos = particles[particleID].predictedPosition;
  particles[particleID].density = calculateDensity(pos);
}


[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void calculatePressureForce(uint3 threadIdx : SV_DispatchThreadID)
{
  uint particleID = threadIdx.x;
  if (particleID >= setting.numParticles)
    return;

  float2 acceleration = calculatePressure(particleID);
  inspect32BitValue(0, asuint(acceleration.x));
  inspect32BitValue(1, asuint(acceleration.y));
  particles[particleID].velocity += acceleration * setting.deltaTime;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void calculateViscosity(uint3 threadIdx : SV_DispatchThreadID)
{
  uint particleID = threadIdx.x;
  if (particleID >= setting.numParticles)
    return;
  
  float2 viscosityForce = calculateViscosity(particleID);
  particles[particleID].velocity += viscosityForce * setting.viscosityStrength * setting.deltaTime;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void externalForces(uint3 threadIdx : SV_DispatchThreadID)
{
  uint particleID = threadIdx.x;
  if (particleID >= setting.numParticles)
    return;
  
  // External forces (gravity and input interaction)
  float2 forces = externalForces(particles[particleID].position, particles[particleID].velocity);
  particles[particleID].velocity += forces * setting.deltaTime;

  // Predict
  const float predictionFactor = setting.deltaTime / 2.0;
  particles[particleID].predictedPosition = particles[particleID].position + particles[particleID].velocity * predictionFactor;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void updatePositions(uint3 threadIdx : SV_DispatchThreadID)
{
  uint particleID = threadIdx.x;
  if (particleID >= setting.numParticles)
    return;
  
  float2 pos = particles[particleID].velocity * setting.deltaTime;
  if (isnan(pos.x) || isnan(pos.y))
  {
    pos = float2(0);
    particles[particleID].velocity = float2(0);
  }

  particles[particleID].position += pos;
  handleCollisions(particleID);
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void updateSpatialHash(uint3 threadIdx : SV_DispatchThreadID)
{
  uint particleID = threadIdx.x;
  if (particleID >= setting.numParticles)
    return;
  
  // Reset offsets
  spatialInfo[particleID].offsets = setting.numParticles;

  // Update index buffer
  int2 cell = getCell2D(particles[particleID].predictedPosition, setting.smoothingRadius);
  uint hash = hashCell2D(cell);
  uint key = keyFromHash(hash, setting.numParticles);
  spatialInfo[particleID].indices = uint3(particleID, hash, key);
}

///

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void bitonicSort(uint3 threadIdx : SV_DispatchThreadID)
{
  uint particleID = threadIdx.x;
  if (particleID >= setting.numParticles)
    return;
  uint i = particleID;

  uint hIndex = i & (pushC.groupWidth - 1);
  uint indexLeft = hIndex + (pushC.groupHeight + 1) * (i / pushC.groupWidth);
  uint rightStepSize = pushC.stepIndex == 0 ? pushC.groupHeight - 2 * hIndex : (pushC.groupHeight + 1) / 2;
  uint indexRight = indexLeft + rightStepSize;

  // Exit if out of bounds (for non-power of 2 input sizes)
  if (indexRight >= setting.numParticles)
    return;

  uint valueLeft = spatialInfo[indexLeft].indices.z;
  uint valueRight = spatialInfo[indexRight].indices.z;

  // Swap entries if value is descending
  if (valueLeft > valueRight)
  {
    uint3 temp = spatialInfo[indexLeft].indices;
    spatialInfo[indexLeft].indices = spatialInfo[indexRight].indices;
    spatialInfo[indexRight].indices = temp;
  }
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void bitonicSortOffset(uint3 threadIdx : SV_DispatchThreadID)
{
  uint particleID = threadIdx.x;

  if (particleID >= setting.numParticles - 1)
  {
    return;
  }

  uint i = particleID;
  uint null = setting.numParticles;

  uint key = spatialInfo[i].indices.z;
  uint keyPrev = (i == 0 ? null : spatialInfo[i - 1].indices.z);

  if (key != keyPrev)
  {
    spatialInfo[key].offsets = particleID;
  }
}
