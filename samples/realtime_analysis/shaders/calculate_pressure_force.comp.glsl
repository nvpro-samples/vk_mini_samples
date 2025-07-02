
#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_shader_subgroup_basic : enable

#include "device_host.h"
#include "layouts.h"

#define INSPECTOR_MODE_COMPUTE
#define INSPECTOR_DESCRIPTOR_SET 0
#define INSPECTOR_INSPECTION_DATA_BINDING eThreadInspection
#define INSPECTOR_METADATA_BINDING eThreadMetadata
#include "nvshaders/inspector_io.h"

#include "fluid_sim_2D.h"

void main()
{
  uint particleID = gl_GlobalInvocationID.x;
  if(particleID >= setting.numParticles)
    return;

  vec2 acceleration = calculatePressure(particleID);
  inspect32BitValue(0, floatBitsToUint(acceleration.x));
  inspect32BitValue(1, floatBitsToUint(acceleration.y));
  particles[particleID].velocity += acceleration * setting.deltaTime;
}