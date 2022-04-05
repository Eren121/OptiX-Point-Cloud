#ifndef POINT_H
#define POINT_H

#include <optix.h>
#include "helper_math.h"

#define SCALE 1.0f
        
struct Point
{
    Point() = default;
    Point(float x, float y, float z, float r)
        : pos(make_float3(x, y, z)), r(r)
    {}

    Point(float x, float y, float z)
        : pos(make_float3(x, y, z))
    {}
    
    float3 pos = make_float3(0.0f, 0.0f, 0.0f); // Position
    float3 nor = make_float3(0.0f, 1.0f, 0.0f); // Normale
    uchar3 col = make_uchar3(255, 255, 255);    // Couleur
    float  r   = 1.0f;                          // Rayon
    
    OptixAabb toAabb() const
    {
        OptixAabb aabb;
        aabb.minX = pos.x - r;
        aabb.maxX = pos.x + r;
        aabb.minY = pos.y - r;
        aabb.maxY = pos.y + r;
        aabb.minZ = pos.z - r;
        aabb.maxZ = pos.z + r;
        return aabb;
    }
};

#endif /* POINT_H */
