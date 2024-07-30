
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

  vec2 pos = particles[particleID].velocity * setting.deltaTime;
  if(isnan(pos.x) || isnan(pos.y))
  {
    pos                            = vec2(0);
    particles[particleID].velocity = vec2(0);
  }

  particles[particleID].position += pos;
  handleCollisions(particleID);
}