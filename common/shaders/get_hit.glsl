// This function returns the geometric information at hit point
// Note: depends on the buffer layout PrimMeshInfo

#ifndef GETHIT_GLSL
#define GETHIT_GLSL


//-----------------------------------------------------------------------
// Hit state information
struct HitState
{
  vec3 pos;
  vec3 nrm;
  vec3 geonrm;
  vec2 uv;
  vec3 tangent;
  vec3 bitangent;
};


//--------------------------------------------------------------
// Flipping Back-face
vec3 adjustShadingNormalToRayDir(inout vec3 N, inout vec3 G)
{
  const vec3 V = -gl_WorldRayDirectionEXT;

  if(dot(G, V) < 0)  // Flip if back facing
    G = -G;

  if(dot(G, N) < 0)  // Make Normal and GeoNormal on the same side
    N = -N;

  return N;
}


//-----------------------------------------------------------------------
//-----------------------------------------------------------------------
HitState GetHitState(PrimMeshInfo pinfo)
{
  HitState hit;

  // Vextex and indices of the primitive
  Vertices vertices = Vertices(pinfo.vertexAddress);
  Indices  indices  = Indices(pinfo.indexAddress);

  // Barycentric coordinate on the triangle
  const vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

  // Getting the 3 indices of the triangle (local)
  uvec3 triangleIndex = indices.i[gl_PrimitiveID];

  // All vertex attributes of the triangle.
  Vertex v0 = vertices.v[triangleIndex.x];
  Vertex v1 = vertices.v[triangleIndex.y];
  Vertex v2 = vertices.v[triangleIndex.z];

  // Position
  const vec3 pos0     = v0.position.xyz;
  const vec3 pos1     = v1.position.xyz;
  const vec3 pos2     = v2.position.xyz;
  const vec3 position = pos0 * barycentrics.x + pos1 * barycentrics.y + pos2 * barycentrics.z;
  hit.pos             = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

  // Normal
  const vec3 nrm0           = v0.normal.xyz;
  const vec3 nrm1           = v1.normal.xyz;
  const vec3 nrm2           = v2.normal.xyz;
  const vec3 normal         = normalize(nrm0 * barycentrics.x + nrm1 * barycentrics.y + nrm2 * barycentrics.z);
  vec3       worldNormal    = normalize(vec3(normal * gl_WorldToObjectEXT));
  const vec3 geoNormal      = normalize(cross(pos1 - pos0, pos2 - pos0));
  vec3       worldGeoNormal = normalize(vec3(geoNormal * gl_WorldToObjectEXT));
  adjustShadingNormalToRayDir(worldNormal, worldGeoNormal);
  hit.geonrm = worldGeoNormal;
  hit.nrm    = worldNormal;

  // TexCoord
  const vec2 uv0 = vec2(v0.position.w, v0.normal.w);
  const vec2 uv1 = vec2(v1.position.w, v1.normal.w);
  const vec2 uv2 = vec2(v2.position.w, v2.normal.w);
  hit.uv         = uv0 * barycentrics.x + uv1 * barycentrics.y + uv2 * barycentrics.z;

  // Tangent - Bitangent
  const vec4 tng0    = vec4(v0.tangent);
  const vec4 tng1    = vec4(v1.tangent);
  const vec4 tng2    = vec4(v2.tangent);
  vec3       tangent = normalize(tng0.xyz * barycentrics.x + tng1.xyz * barycentrics.y + tng2.xyz * barycentrics.z);
  vec3       world_tangent  = normalize(vec3(tangent * gl_WorldToObjectEXT));
  vec3       world_binormal = cross(worldNormal, world_tangent) * tng0.w;
  hit.tangent               = world_tangent;
  hit.bitangent             = world_binormal;

  return hit;
}


#endif
