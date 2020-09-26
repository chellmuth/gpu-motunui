#pragma once

namespace moana { namespace Trig {
    __device__ inline float sinFromCos(float cosTheta)
    {
        const float sin2Theta = 1.f - (cosTheta * cosTheta);
        return sqrtf(fmaxf(0.f, sin2Theta));
    }

    __device__ inline float sinThetaFromCosTheta(float cosTheta)
    {
        const float result = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
        return result;
    }

    __device__ inline float sin2FromCos(float cosTheta)
    {
        return fmaxf(0.f, 1.f - (cosTheta * cosTheta));
    }

    __device__ inline float cosFromSin2(float sin2Theta)
    {
        return sqrtf(
            fmaxf(0.f, 1.f - sin2Theta)
        );
    }
} }
