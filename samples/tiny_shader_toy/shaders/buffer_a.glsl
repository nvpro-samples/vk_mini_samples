// Instead of being displayed (as for buffer image), the result is stored in the special texture of same name.
// Evaluated at each frame
// Can be used for persistent or incremental effect by reading the iChannel0

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
  fragColor = vec4(0, 0, 0, 1);
}
