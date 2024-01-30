// Shared between Host and Device


#define USING_SPACIAL_INFO 1

// Number of particles in the simulation
#define NUM_PARTICLES 15000

#define WORKGROUP_SIZE 128

#define eFrameInfo 0
#define eParticles 1
#define eFragInspectorData 2
#define eFragInspectorMeta 3

#define eCompParticles 0
#define eCompSort 1
#define eThreadInspection 2
#define eThreadMetadata 3

struct PushConstant
{
  vec4 color;
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
  uint  groupWidth;
  uint  groupHeight;
  uint  stepIndex;
  float boundsMultiplier;

  //
  vec2 boundsSize;
  vec2 interactionInputPoint;
  vec2 obstacleSize;
  vec2 obstacleCentre;

  uvec3 numWorkGroups;
  int   _pad;
};
