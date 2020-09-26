#pragma once

#include "moana/core/bsdf_sample_record.hpp"
#include "moana/core/frame.hpp"
#include "ray_data.hpp"
#include "sample.hpp"

namespace moana { namespace Lambertian {

__forceinline__ __device__ BSDFSampleRecord sample(
    uint3 index,
    uint3 dim,
    const PerRayData &prd,
    const float *xiBuffer
) {
    const int xiIndex = 2 * (index.y * dim.x + index.x);
    const float xi1 = xiBuffer[xiIndex + 0];
    const float xi2 = xiBuffer[xiIndex + 1];

    const Frame frame(prd.normal);
    const Vec3 wiLocal = Sample::uniformHemisphere(xi1, xi2);
    const float weight = 2.f;

    const BSDFSampleRecord record = {
        .isValid = true,
        .point = prd.point,
        .wiLocal = wiLocal,
        .normal = prd.normal,
        .frame = frame,
        .weight = weight
    };

    return record;
}

} }
