// This function returns the contribution of one light, choose randomly and weighted accordently

#ifndef DIRECTLIGHT_GLSL
#define DIRECTLIGHT_GLSL

//-----------------------------------------------------------------------
// Use for light/env contribution
struct VisibilityContribution
{
  vec3  radiance;   // Radiance at the point if light is visible
  vec3  lightDir;   // Direction to the light, to shoot shadow ray
  float lightDist;  // Distance to the light (1e32 for infinite or sky)
  bool  visible;    // true if in front of the face and should shoot shadow ray
};


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
VisibilityContribution DirectLight(MaterialEval matEval, HitState hit)
{
  VisibilityContribution contrib;
  contrib.radiance = vec3(0);
  contrib.visible  = false;

  // randomly select one of the lights
  int   light_index = int(min(rand(payload.seed) * NB_LIGHTS, NB_LIGHTS));
  Light light       = frameInfo.light[light_index];

  vec3  lightDir;
  vec3  lightContrib = lightContribution(light, hit.pos, hit.nrm, lightDir);
  float lightDist    = (light.type != 0) ? 1e37f : length(hit.pos - light.position);
  float dotNL        = dot(lightDir, hit.nrm);

  if(dotNL > 0.0)
  {
    float lightPdf = 1.0f / float(NB_LIGHTS);

    float pdf      = 0;
    vec3  brdf     = pbrEval(matEval, -gl_WorldRayDirectionEXT, lightDir, pdf);
    vec3  radiance = brdf * lightContrib / lightPdf;

    contrib.visible   = true;
    contrib.lightDir  = lightDir;
    contrib.lightDist = lightDist;
    contrib.radiance  = radiance;
  }

  return contrib;
}

#endif
