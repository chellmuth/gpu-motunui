#include <optix.h>

#include <stdio.h>

#include "moana/driver.hpp"
#include "moana/core/camera.hpp"
#include "moana/core/ray.hpp"
#include "util.hpp"

using namespace moana;

struct PerRayData {
    bool isHit;
};

extern "C" {
    __constant__ Params params;
}

__forceinline__ __device__ static PerRayData *getPRD()
{
    const unsigned int u0 = optixGetPayload_0();
    const unsigned int u1 = optixGetPayload_1();
    return reinterpret_cast<PerRayData *>(util::unpackPointer(u0, u1));
}

extern "C" __global__ void __closesthit__ch()
{
    PerRayData *prd = getPRD();
    prd->isHit = true;
}

extern "C" __global__ void __miss__ms()
{
    float3 direction = optixGetWorldRayDirection();
    PerRayData *prd = getPRD();
    prd->isHit = false;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const int row = index.y;
    const int col = index.x;

    const Ray cameraRay = params.camera.generateRay(
        row, col,
        float2{ 0.5f, 0.5f }
    );

    const Vec3 origin = cameraRay.origin();
    const Vec3 direction = cameraRay.direction();

    PerRayData prd;
    prd.isHit = false;

    unsigned int p0, p1;
    util::packPointer(&prd, p0, p1);
    optixTrace(
        params.handle,
        float3{ origin.x(), origin.y(), origin.z() },
        float3{ direction.x(), direction.y(), direction.z() },
        0.f,
        1e16f,
        0.f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // SBT params
        p0, p1
    );

    const int pixelIndex = 3 * (index.y * dim.x + index.x);
    if (prd.isHit) {
        params.outputBuffer[pixelIndex + 0] = 1.f;
    } else {
        params.outputBuffer[pixelIndex + 0] = 0.f;
    }
    params.outputBuffer[pixelIndex + 1] = 1.f;

}
