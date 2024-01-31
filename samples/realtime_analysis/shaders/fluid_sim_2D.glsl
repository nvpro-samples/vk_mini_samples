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

// Includes
#include "./fluid_maths_2D.glsl"
#include "./spatial_hash.glsl"

#ifndef INSPECTOR_MODE_COMPUTE
void inspect32BitValue(uint a, uint b) {}
#endif


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

vec2 calculateDensity(vec2 pos)
{
  ivec2 originCell  = getCell2D(pos, setting.smoothingRadius);
  float sqrRadius   = setting.smoothingRadius * setting.smoothingRadius;
  float density     = 0;
  float nearDensity = 0;

  // Neighbour search
  for(int i = 0; i < 9; i++)
  {
    uint hash      = hashCell2D(originCell + offsets2D[i]);
    uint key       = keyFromHash(hash, setting.numParticles);
    uint currIndex = spatialInfo[key].offsets;

    while(currIndex < setting.numParticles)
    {
      uvec3 indexData = spatialInfo[currIndex].indices;
      currIndex++;
      // Exit if no longer looking at correct bin
      if(indexData.z != key)
        break;
      // Skip if hash does not match
      if(indexData.y != hash)
        continue;

      uint  neighbourIndex    = indexData[0];
      vec2  neighbourPos      = particles[neighbourIndex].predictedPosition;
      vec2  offsetToNeighbour = neighbourPos - pos;
      float sqrDstToNeighbour = dot(offsetToNeighbour, offsetToNeighbour);

      // Skip if not within radius
      if(sqrDstToNeighbour > sqrRadius)
        continue;

      // Calculate density and near density
      float dst = sqrt(sqrDstToNeighbour);
      density += densityKernel(dst, setting.smoothingRadius);
      nearDensity += nearDensityKernel(dst, setting.smoothingRadius);
    }
  }

  return vec2(density, nearDensity);
}

float pressureFromDensity(float density)
{
  return (density - setting.targetDensity) * setting.pressureMultiplier;
}

float nearPressureFromDensity(float nearDensity)
{
  return setting.nearPressureMultiplier * nearDensity;
}

vec2 externalForces(vec2 pos, vec2 velocity)
{
  // Gravity
  vec2 gravityAccel = vec2(0, setting.gravity);

  // Input interactions modify gravity
  if(setting.interactionInputStrength != 0)
  {
    vec2  inputPointOffset = setting.interactionInputPoint - pos;
    float sqrDst           = dot(inputPointOffset, inputPointOffset);
    if(sqrDst < setting.interactionInputRadius * setting.interactionInputRadius)
    {
      float dst         = sqrt(sqrDst);
      float edgeT       = (dst / setting.interactionInputRadius);
      float centreT     = 1 - edgeT;
      vec2  dirToCentre = inputPointOffset / dst;

      float gravityWeight = 1 - (centreT * clamp(setting.interactionInputStrength / 10, 0, 1));
      vec2  accel         = gravityAccel * gravityWeight + dirToCentre * centreT * setting.interactionInputStrength;
      accel -= velocity * centreT;
      return accel;
    }
  }

  return gravityAccel;
}

void handleCollisions(uint particleIndex)
{
  vec2 pos = particles[particleIndex].position;
  vec2 vel = particles[particleIndex].velocity;

  // Keep particle inside bounds
  const vec2 halfSize = setting.boundsSize * 0.5;
  vec2       edgeDst  = halfSize - abs(pos);

  if(edgeDst.x <= 0)
  {
    pos.x = halfSize.x * sign(pos.x);
    vel.x *= -1 * setting.collisionDamping;
  }
  if(edgeDst.y <= 0)
  {
    pos.y = halfSize.y * sign(pos.y);
    vel.y *= -1 * setting.collisionDamping;
  }

  // Collide particle against the test obstacle
  const vec2 obstacleHalfSize = setting.obstacleSize * 0.5;
  vec2       obstacleEdgeDst  = obstacleHalfSize - abs(pos - setting.obstacleCentre);

  if(obstacleEdgeDst.x >= 0 && obstacleEdgeDst.y >= 0)
  {
    if(obstacleEdgeDst.x < obstacleEdgeDst.y)
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

vec2 calculatePressure(uint particleID)
{
  float density       = particles[particleID].density.x;
  float densityNear   = particles[particleID].density.y;
  float pressure      = pressureFromDensity(density);
  float nearPressure  = nearPressureFromDensity(densityNear);
  vec2  pressureForce = vec2(0);

  vec2  pos        = particles[particleID].predictedPosition;
  ivec2 originCell = getCell2D(pos, setting.smoothingRadius);
  float sqrRadius  = setting.smoothingRadius * setting.smoothingRadius;

  uint numElem = 0;

  // Neighbour search
  for(int i = 0; i < 9; i++)
  {
    uint hash      = hashCell2D(originCell + offsets2D[i]);
    uint key       = keyFromHash(hash, setting.numParticles);
    uint currIndex = spatialInfo[key].offsets;

    while(currIndex < setting.numParticles)
    {
      uvec3 indexData = spatialInfo[currIndex].indices;
      currIndex++;
      // Exit if no longer looking at correct bin
      if(indexData.z != key)
        break;
      // Skip if hash does not match
      if(indexData.y != hash)
        continue;

      uint neighbourIndex = indexData[0];
      // Skip if looking at self
      if(neighbourIndex == particleID)
        continue;

      vec2  neighbourPos      = particles[neighbourIndex].predictedPosition;
      vec2  offsetToNeighbour = neighbourPos - pos;
      float sqrDstToNeighbour = dot(offsetToNeighbour, offsetToNeighbour);

      // Skip if not within radius
      if(sqrDstToNeighbour > sqrRadius)
        continue;

      // Calculate pressure force
      float dst            = sqrt(sqrDstToNeighbour);
      vec2  dirToNeighbour = dst > 0 ? offsetToNeighbour / dst : vec2(0, 1);

      float neighbourDensity      = particles[neighbourIndex].density.x;
      float neighbourNearDensity  = particles[neighbourIndex].density.y;
      float neighbourPressure     = pressureFromDensity(neighbourDensity);
      float neighbourNearPressure = nearPressureFromDensity(neighbourNearDensity);

      float sharedPressure     = (pressure + neighbourPressure) * 0.5;
      float sharedNearPressure = (nearPressure + neighbourNearPressure) * 0.5;

      pressureForce += dirToNeighbour * densityDerivative(dst, setting.smoothingRadius) * sharedPressure / neighbourDensity;
      pressureForce += dirToNeighbour * nearDensityDerivative(dst, setting.smoothingRadius) * sharedNearPressure / neighbourNearDensity;
      ++numElem;
    }
  }

  // Tell how many other particles where tested
  inspect32BitValue(2, numElem);

  vec2 acceleration = pressureForce / density;

  return acceleration;
}

vec2 calculateViscosity(uint particleID)
{

  vec2  pos            = particles[particleID].predictedPosition;
  ivec2 originCell     = getCell2D(pos, setting.smoothingRadius);
  float sqrRadius      = setting.smoothingRadius * setting.smoothingRadius;
  vec2  viscosityForce = vec2(0);
  vec2  velocity       = particles[particleID].velocity;
  uint  currIndex      = 0;


  for(int i = 0; i < 9; i++)
  {
    uint hash      = hashCell2D(originCell + offsets2D[i]);
    uint key       = keyFromHash(hash, setting.numParticles);
    uint currIndex = spatialInfo[key].offsets;

    while(currIndex < setting.numParticles)
    {
      uvec3 indexData = spatialInfo[currIndex].indices;
      currIndex++;
      // Exit if no longer looking at correct bin
      if(indexData.z != key)
        break;
      // Skip if hash does not match
      if(indexData.y != hash)
        continue;

      uint neighbourIndex = indexData[0];
      // Skip if looking at self
      if(neighbourIndex == particleID)
        continue;

      vec2  neighbourPos      = particles[neighbourIndex].predictedPosition;
      vec2  offsetToNeighbour = neighbourPos - pos;
      float sqrDstToNeighbour = dot(offsetToNeighbour, offsetToNeighbour);

      // Skip if not within radius
      if(sqrDstToNeighbour > sqrRadius)
        continue;

      float dst               = sqrt(sqrDstToNeighbour);
      vec2  neighbourVelocity = particles[neighbourIndex].velocity;
      viscosityForce += (neighbourVelocity - velocity) * viscosityKernel(dst, setting.smoothingRadius);
    }
  }

  return viscosityForce;
}