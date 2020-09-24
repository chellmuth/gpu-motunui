#pragma once

#include "moana/core/vec3.hpp"

namespace moana { namespace WorldFrame {
    __device__ inline float cosTheta(const Vec3 &normal, const Vec3 &w)
    {
        return fmaxf(
            0.f,
            dot(normal, w)
        );
    }

    __device__ inline float absCosTheta(const Vec3 &normal, const Vec3 &w)
    {
        return fabsf(dot(normal, w));
    }
} };
