
#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_scalar_block_layout : enable

#include "device_host.h"
#include "layouts.h"
#include "fluid_sim_2D.h"

void main()
{
  uint particleID = gl_GlobalInvocationID.x;
  if(particleID >= setting.numParticles)
    return;

  // Reset offsets
  spatialInfo[particleID].offsets = setting.numParticles;

  // Update index buffer
  ivec2 cell                      = getCell2D(particles[particleID].predictedPosition, setting.smoothingRadius);
  uint  hash                      = hashCell2D(cell);
  uint  key                       = keyFromHash(hash, setting.numParticles);
  spatialInfo[particleID].indices = uvec3(particleID, hash, key);
}