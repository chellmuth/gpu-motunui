#pragma once

#include "moana/cuda/snell.hpp"

namespace moana { namespace Fresnel {

__device__ static float divide(const float a, const float b)
{
    // assert(b != 0.f);

    return a / b;
}

__device__ inline float dielectricReflectance(
    float cosThetaIncident,
    float etaIncident,
    float etaTransmitted
) {
    const float sinThetaTransmitted = Snell::transmittedSinTheta(
        cosThetaIncident,
        etaIncident,
        etaTransmitted
    );

    if (sinThetaTransmitted > 1.f) {
        return 1.f; // Total internal reflection
    }

    const float cosThetaTransmitted = sqrtf(fmaxf(0.f, 1.f - sinThetaTransmitted * sinThetaTransmitted));

    const float rParallel = divide(
        etaTransmitted * cosThetaIncident - etaIncident * cosThetaTransmitted,
        etaTransmitted * cosThetaIncident + etaIncident * cosThetaTransmitted
    );

    const float rPerpendicular = divide(
        etaIncident * cosThetaIncident - etaTransmitted * cosThetaTransmitted,
        etaIncident * cosThetaIncident + etaTransmitted * cosThetaTransmitted
    );

    return 0.5f * (rParallel * rParallel + rPerpendicular * rPerpendicular);
}


} }
