#pragma once

#include "moana/core/vec3.hpp"

namespace moana { namespace Sample {

__device__ inline Vec3 uniformHemisphere(float xi1, float xi2)
{
    const float z = xi1;
    const float r = sqrtf(fmaxf(0.f, 1.f - z * z));

    const float phi = 2 * M_PI * xi2;
    const float x = r * cos(phi);
    const float y = r * sin(phi);

    return Vec3(x, y, z);
}

__device__ inline float uniformHemispherePDF(const Vec3& vector)
{
    if (vector.z() < 0.f) { return 0.f; }

    return 1.f / (2.f * M_PI);
}

} }
