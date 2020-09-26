#include <optix.h>

#include <stdio.h>

#include "moana/driver.hpp"
#include "moana/core/bsdf_sample_record.hpp"
#include "moana/core/camera.hpp"
#include "moana/core/frame.hpp"
#include "moana/core/ray.hpp"
#include "moana/cuda/triangle.hpp"
#include "moana/renderer.hpp"
#include "optix_sdk.hpp"
#include "random.hpp"
#include "sample.hpp"
#include "util.hpp"

using namespace moana;

struct PerRayData {
    bool isHit;
};

extern "C" {
    __constant__ Renderer::Params params;
}

__forceinline__ __device__ static PerRayData *getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<PerRayData *>(util::unpackPointer(u0, u1));
}

extern "C" __global__ void __miss__ms()
{
    PerRayData *prd = getPRD();
    prd->isHit = false;
}

extern "C" __global__ void __closesthit__ch()
{
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const int shadowOcclusionIndex = 1 * (index.y * dim.x + index.x);

    const int sampleRecordIndex = 1 * (index.y * dim.x + index.x);
    const BSDFSampleRecord &sampleRecord = params.sampleRecordOutBuffer[sampleRecordIndex];
    if (!sampleRecord.isValid) {
        params.shadowOcclusionBuffer[shadowOcclusionIndex] = 1;

        return;
    }

    const int shadowWeightIndex = 1 * (index.y * dim.x + index.x);
    if (sampleRecord.isDelta) {
        params.shadowWeightBuffer[shadowWeightIndex] = 0.f;

        return;
    }

    unsigned int seed = tea<4>(
        index.y * dim.x + index.x,
        params.sampleCount + (dim.x * dim.y * params.bounce)
    );
    const float xi1 = rnd(seed);
    const float xi2 = rnd(seed);
    const float xi3 = rnd(seed);

    const Triangle t1(
        Vec3(101346.539, 202660.438, 189948.188),
        Vec3(106779.617, 187339.562, 201599.453),
        Vec3(83220.3828, 202660.438, 198400.547)
    );
    const Triangle t2(
        Vec3(101346.539, 202660.438, 189948.188),
        Vec3(88653.4609, 187339.562, 210051.812),
        Vec3(88653.4609, 187339.562, 210051.812)
    );

    const Triangle *sampleTriangle;
    if (xi1 < 0.5f) {
        sampleTriangle = &t1;
    } else {
        sampleTriangle = &t2;
    }

    const SurfaceSample lightSample = sampleTriangle->sample(xi2, xi3);

    const Vec3 origin(sampleRecord.point.x, sampleRecord.point.y, sampleRecord.point.z);
    const Vec3 lightPoint = lightSample.point;
    const Vec3 lightDirection = lightPoint - origin;
    const Vec3 wi = normalized(lightDirection);
    const float tMax = lightDirection.length();

    PerRayData prd;
    prd.isHit = true;

    unsigned int p0, p1;
    util::packPointer(&prd, p0, p1);

    optixTrace(
        params.handle,
        float3{ origin.x(), origin.y(), origin.z() },
        float3{ wi.x(), wi.y(), wi.z() },
        2e-3,
        tMax - 1e-4,
        0.f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
        0, 1, 0, // SBT params
        p0, p1
    );

    const Vec3 lightNormal = Vec3(-0.323744059f, -0.642787874f, -0.694271863f);

    if (prd.isHit) {
        params.shadowOcclusionBuffer[shadowOcclusionIndex] = 1;
    }

    params.shadowWeightBuffer[shadowWeightIndex] = 1.f
        * fabsf(dot(lightNormal, -wi))
        * fmaxf(0.f, dot(wi, sampleRecord.normal))
        * (20000.f * 20000.f) / (lightDirection.length() * lightDirection.length())
        * (1.f / M_PI)
    ;
}
