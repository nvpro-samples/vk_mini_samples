
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

  // External forces (gravity and input interaction)
  vec2 forces = externalForces(particles[particleID].position, particles[particleID].velocity);
  particles[particleID].velocity += forces * setting.deltaTime;

  // Predict
  const float predictionFactor = setting.deltaTime / 2.0;
  particles[particleID].predictedPosition = particles[particleID].position + particles[particleID].velocity * predictionFactor;
}