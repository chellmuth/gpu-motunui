#pragma once

#include "moana/core/vec3.hpp"

namespace moana { namespace TangentFrame {
    __device__ inline float cosTheta(const Vec3 &vector)
    {
        return vector.z();
    }

    __device__ inline float absCosTheta(const Vec3 &vector)
    {
        return fabsf(vector.z());
    }

    __device__ inline float cos2Theta(const Vec3 &vector)
    {
        return vector.z() * vector.z();
    }

    __device__ inline float sin2Theta(const Vec3 &vector)
    {
        return 1.f - cos2Theta(vector);
    }

    __device__ inline float tan2Theta(const Vec3 &vector)
    {
        return sin2Theta(vector) / cos2Theta(vector);
    }

    __device__ inline float sinTheta(const Vec3 &vector)
    {
        return sqrtf(fmaxf(0.f, 1.f - cos2Theta(vector)));
    }

    __device__ inline float tanTheta(const Vec3 &vector)
    {
        return sinTheta(vector) / cosTheta(vector);
    }

    __device__ inline float absTanTheta(const Vec3 &vector)
    {
        return fabsf(tanTheta(vector));
    }

} }
