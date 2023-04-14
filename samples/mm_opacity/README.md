# Opacity Micro-Map

![](docs/opacity.png)

With Opacity Micro-map, it is possible to encode visibility to smaller portion of the triangle.

In the above picture, the blue triangles are flaged opaque, the red ones are flaged *transparent unknown* and the ones we don't see are flagged transparent. This allows to invoke the AnyHit shader only for the unknown state triangles.

## Sample

There are `4^subdivision` triangles for a specific subdivision level. For each triangle index, there is a function (`BirdCurveHelper::micro2bary`) that is returning the tripplet of barycentric coordinates. With those values, it is possible to project the base triangle information. In this sample, we find the world position of the micro-triangle and check if the triangle is outside , inside or crossing the radius.

The values stored are either a 2-states (1 bit) or 4-states (2-bits), telling if the triangle is opaque, transparent or unknown. We create a buffer holding all values for each triangle of the mesh. There is also a buffer of `VkMicromapTriangleEXT` telling where each triangle find its data in the value buffer, and finaly an index buffer, telling which `VkMicromapTriangleEXT` is used by the mesh triangle. For the latest, since we aren't reusing triangle data, the index is simply a continuoius array of `0, 1, 2, 3, 4, .. number of triangles`. With all those informations, we can build the `VkMicromapEXT`.

To the BLAS, we will attach the micromap information. This is done by filling the `VkAccelerationStructureTrianglesOpacityMicromapEXT` and attaching it to the `pNext` of `VkAccelerationStructureGeometryTrianglesDataKHR`.
