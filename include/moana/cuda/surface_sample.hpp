#pragma once

#include "moana/core/vec3.hpp"
#include "moana/cuda/world_frame.hpp"

namespace moana {

struct SurfaceSample {
    Vec3 point;
    Vec3 normal;
    float areaPDF;

    __device__ float solidAnglePDF(const Vec3 &referencePoint) const
    {
        const Vec3 lightDirection = point - referencePoint;
        const Vec3 lightWo = -normalized(lightDirection);
        const float distance = lightDirection.length();

        const float distance2 = distance * distance;
        const float projectedArea = WorldFrame::absCosTheta(normal, lightWo);

        if (projectedArea == 0.f) { return 0.f; }
        return areaPDF * distance2 / projectedArea;
    }
};


}
