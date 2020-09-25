#pragma once

#include <algorithm>

#include "moana/cuda/trig.hpp"
#include "moana/core/vec3.hpp"

namespace moana { namespace Snell {

__device__ inline bool refract(
    const Vec3 &incidentLocal,
    Vec3 *transmittedLocal,
    Vec3 normal,
    float etaIncident,
    float etaTransmitted
) {
    if (dot(incidentLocal, normal) < 0.f) {
        normal = normal * -1.f;

        const float temp = etaIncident;
        etaIncident = etaTransmitted;
        etaTransmitted = temp;
    }

    const Vec3 wIncidentPerpendicular = incidentLocal - (normal * incidentLocal.dot(normal));
    const Vec3 wTransmittedPerpendicular = -wIncidentPerpendicular * (etaIncident / etaTransmitted);

    const float transmittedPerpendicularLength2 = wTransmittedPerpendicular.length() * wTransmittedPerpendicular.length();
    const float wTransmittedParallelLength = sqrtf(fmaxf(0.f, 1.f - transmittedPerpendicularLength2));
    const Vec3 wTransmittedParallel = normal * -wTransmittedParallelLength;

    const float cosThetaIncident = absDot(incidentLocal, normal);
    const float sin2ThetaIncident = Trig::sin2FromCos(cosThetaIncident);
    const float eta2 = (etaIncident / etaTransmitted) * (etaIncident / etaTransmitted);
    const float sin2ThetaTransmitted = eta2 * sin2ThetaIncident;

    *transmittedLocal = normalized(wTransmittedParallel + wTransmittedPerpendicular);

    if (sin2ThetaTransmitted >= 1.f) { // total internal reflection
        return false;
    }
    return true;
}

__device__ inline bool refract(
    const Vec3 &incidentLocal,
    Vec3 *transmittedLocal,
    float etaIncident,
    float etaTransmitted
) {
    return refract(
        incidentLocal,
        transmittedLocal,
        Vec3(0.f, 0.f, 1.f),
        etaIncident,
        etaTransmitted
    );
}

__device__ inline float transmittedSinTheta(
    float cosThetaIncident,
    float etaIncident,
    float etaTransmitted
) {
    return (etaIncident / etaTransmitted) * Trig::sinThetaFromCosTheta(cosThetaIncident);
}

} }
