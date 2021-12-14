// This function evaluates a material at hit position

#ifndef SHADING_GLSL
#define SHADING_GLSL

void StopRay()
{
  payload.hitT = INFINITE;
}

struct ShadingResult
{
  vec3 weight;
  vec3 radiance;
  vec3 rayOrigin;
  vec3 rayDirection;
};

//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
void shading(in MaterialEval matEval, in HitState hit, out ShadingResult result)
{
  result.radiance = matEval.emissive;  // Emissive material

  if(result.radiance.x > 0 || result.radiance.y > 0 || result.radiance.z > 0)
  {  // Stop on emmissive material
    StopRay();
    return;
  }

  // Sampling for the next ray
  vec3  rayDirection;
  float pdf     = 0;
  vec3  randVal = vec3(rand(payload.seed), rand(payload.seed), rand(payload.seed));
  vec3  brdf    = pbrSample(matEval, -gl_WorldRayDirectionEXT, rayDirection, pdf, randVal);

  if(dot(hit.nrm, rayDirection) > 0.0 && pdf > 0.0)
  {
    result.weight = brdf / pdf;
  }
  else
  {
    StopRay();
    return;
  }

  // Next ray
  result.rayDirection = rayDirection;
  result.rayOrigin    = offsetRay(hit.pos, dot(rayDirection, hit.nrm) > 0 ? hit.nrm : -hit.nrm);


  // Light and environment contribution at hit position
  VisibilityContribution vcontrib = DirectLight(matEval, hit);

  if(vcontrib.visible)
  {
    // Shadow ray - stop at the first intersection, don't invoke the closest hit shader (fails for transparent objects)
    uint rayflag = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT | gl_RayFlagsCullBackFacingTrianglesEXT;
    payload.hitT = 0;
    traceRayEXT(topLevelAS, rayflag, 0xFF, 0, 0, 0, result.rayOrigin, 0.001, vcontrib.lightDir, vcontrib.lightDist, 0);
    // If hitting nothing, add light contribution
    if(payload.hitT == INFINITE)
      result.radiance += vcontrib.radiance;
    payload.hitT = gl_HitTEXT;
  }
}

#endif  // SHADING_GLSL
