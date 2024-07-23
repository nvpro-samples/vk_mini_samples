# Simple Polygons

![Barycentric Wireframe](docs/bary_wireframe.png)

This sample demonstrates single-pass wireframe rendering on geometry using [`gl_BaryCoordNV`](https://github.com/KhronosGroup/GLSL/blob/master/extensions/nv/GLSL_NV_fragment_shader_barycentric.txt).

## Technical Overview

The technique employs a distance field function utilizing the barycentric coordinates of the triangle to render lines along the edges.

### Barycentric Coordinates

Barycentric coordinates provide a method to describe a point's position relative to a triangle's vertices. For a triangle ABC, any point P can be uniquely represented by a set of three weights (u, v, w) where:

u + v + w = 1

- u: weight assigned to vertex A
- v: weight assigned to vertex B
- w: weight assigned to vertex C

These weights indicate the relative distances from P to the triangle's vertices.

![Barycentric Coordinate Visualization](docs/Barycentric.jpg)

In the above visualization, a weight of zero on the `x` axis corresponds to a position on the edge opposite vertex `A`.

### Implementation Details

The [`gl_BaryCoordNV`](https://github.com/KhronosGroup/GLSL/blob/master/extensions/nv/GLSL_NV_fragment_shader_barycentric.txt) built-in variable returns the weights for a given fragment. This information enables the calculation of the distance to the edge. If this value falls below a specified threshold, the edge can be highlighted.

## Fragment Shader Implementation

The following GLSL code snippet demonstrates the core logic for generating screen-space wireframe lines with adjustable width:

```glsl
float thickness = 1.0;
float smoothing = thickness * 0.5;
vec3 wireColor  = vec3(1,0,0);
vec3 deltas     = fwidth(gl_BaryCoordNV);
vec3 barys      = smoothstep(deltas * thickness, deltas * (thickness + smoothing), gl_BaryCoordNV);
float minBary   = min(barys.x, min(barys.y, barys.z));
float lineWidth = 1.0 - minBary;

// Final color computation
color = mix(color, wireColor, lineWidth);
```

### Key Components:

- `thickness`: Controls the width of the wireframe lines
- `smoothing`: Defines the anti-aliasing factor for the lines
- `fwidth()`: Calculates the partial derivatives of the barycentric coordinates
- `smoothstep()`: Provides smooth interpolation for line edges
- `mix()`: Blends between the base color and the wireframe color based on the calculated line width

This implementation offers an efficient method for rendering wireframes without the need for separate geometry or multiple rendering passes.