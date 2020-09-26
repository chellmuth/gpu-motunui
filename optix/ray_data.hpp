#pragma once

#include "moana/core/vec3.hpp"
#include "moana/cuda/bsdf.hpp"

namespace moana {

struct PerRayData {
    bool isHit;
    float t;
    float3 point;
    Vec3 normal;
    Vec3 woWorld;
    float3 baseColor;
    int materialID;
    BSDFType bsdfType;
    int primitiveID;
    int textureIndex;
    float2 barycentrics;
    bool isInside;
};

}
