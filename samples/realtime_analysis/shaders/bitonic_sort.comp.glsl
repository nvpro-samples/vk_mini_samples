
#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_scalar_block_layout : enable

#include "device_host.h"
#include "layouts.glsl"

// Sort the given entries by their keys (smallest to largest)
// This is done using bitonic merge sort, and takes multiple iterations

void main()
{
  uint particleID = gl_GlobalInvocationID.x;

  uint i = particleID;

  uint hIndex        = i & (pushC.groupWidth - 1);
  uint indexLeft     = hIndex + (pushC.groupHeight + 1) * (i / pushC.groupWidth);
  uint rightStepSize = pushC.stepIndex == 0 ? pushC.groupHeight - 2 * hIndex : (pushC.groupHeight + 1) / 2;
  uint indexRight    = indexLeft + rightStepSize;

  // Exit if out of bounds (for non-power of 2 input sizes)
  if(indexRight >= setting.numParticles)
    return;

  uint valueLeft  = spatialInfo[indexLeft].indices.z;
  uint valueRight = spatialInfo[indexRight].indices.z;

  // Swap entries if value is descending
  if(valueLeft > valueRight)
  {
    uvec3 temp                      = spatialInfo[indexLeft].indices;
    spatialInfo[indexLeft].indices  = spatialInfo[indexRight].indices;
    spatialInfo[indexRight].indices = temp;
  }
}
