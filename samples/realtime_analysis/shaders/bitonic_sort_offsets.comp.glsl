
#version 450
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_scalar_block_layout : enable


#include "device_host.h"

#include "layouts.h"

// Calculate offsets into the sorted Entries buffer (used for spatial hashing).
// For example, given an Entries buffer sorted by key like so: {2, 2, 2, 3, 6, 6, 9, 9, 9, 9}
// The resulting Offsets calculated here should be:            {0, -, -, 3, 4, -, 6, -, -, -}
// (where '-' represents elements that won't be read/written)
//
// Usage example:
// Say we have a particular particle P, and we want to know which particles are in the same grid cell as it.
// First we would calculate the Key of P based on its position. Let's say in this example that Key = 9.
// Next we can look up Offsets[Key] to get: Offsets[9] = 6
// This tells us that SortedEntries[6] is the first particle that's in the same cell as P.
// We can then loop until we reach a particle with a different cell key in order to iterate over all the particles in the cell.
//
// NOTE: offsets buffer must filled with values equal to (or greater than) its length to ensure that this works correctly
void main()
{
  uint particleID = gl_GlobalInvocationID.x;
  if(particleID >= setting.numParticles - 1)
  {
    return;
  }

  uint i    = particleID;
  uint null = setting.numParticles;

  uint key     = spatialInfo[i].indices.z;
  uint keyPrev = (i == 0 ? null : spatialInfo[i - 1].indices.z);

  if(key != keyPrev)
  {
    spatialInfo[key].offsets = particleID;
  }
}