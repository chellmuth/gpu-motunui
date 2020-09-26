#pragma once

#include "moana/core/vec3.hpp"

namespace moana {

struct PerRayData {
    bool isHit;
    float t;
    float3 point;
    Vec3 normal;
    Vec3 woWorld;
    float3 baseColor;
    int materialID;
    int primitiveID;
    int textureIndex;
    float2 barycentrics;
    bool isInside;
};

}
