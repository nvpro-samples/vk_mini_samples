
#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_scalar_block_layout : enable

#include "device_host.h"
#include "layouts.glsl"
#include "fluid_sim_2D.glsl"


void main()
{
  uint particleID = gl_GlobalInvocationID.x;
  if(particleID >= setting.numParticles)
    return;

  vec2 pos                      = particles[particleID].predictedPosition;
  particles[particleID].density = calculateDensity(pos);
}