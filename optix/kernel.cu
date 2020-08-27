#include <optix.h>

#include <stdio.h>

#include "moana/driver.hpp"
#include "moana/core/camera.hpp"
#include "moana/core/ray.hpp"
#include "optix_sdk.hpp"
#include "util.hpp"

using namespace moana;

struct PerRayData {
    bool isHit;
    float t;
    Vec3 normal;
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
    prd->t = optixGetRayTmax();

    if (optixIsTriangleHit()) {
        OptixTraversableHandle gas = optixGetGASTraversableHandle();
        unsigned int primitiveIndex = optixGetPrimitiveIndex();
        unsigned int sbtIndex = optixGetSbtGASIndex();
        float time = optixGetRayTime();

        float3 vertices[3];
        optixGetTriangleVertexData(
            gas,
            primitiveIndex,
            sbtIndex,
            time,
            vertices
        );

        const Vec3 p0(vertices[0].x, vertices[0].y, vertices[0].z);
        const Vec3 p1(vertices[1].x, vertices[1].y, vertices[1].z);
        const Vec3 p2(vertices[2].x, vertices[2].y, vertices[2].z);

        const Vec3 e1 = p1 - p0;
        const Vec3 e2 = p2 - p0;
        const Vec3 normal = normalized(cross(e1, e2));

        prd->normal = normal;
    } else {
        const unsigned int primitiveIndex = optixGetPrimitiveIndex();
        const float3 normal = normalCubic(primitiveIndex);
        prd->normal = normalized(Vec3(normal.x, normal.y, normal.z));
    }
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

    const int depthIndex = index.y * dim.x + index.x;
    const float tMax = params.depthBuffer[depthIndex];

    unsigned int p0, p1;
    util::packPointer(&prd, p0, p1);
    optixTrace(
        params.handle,
        float3{ origin.x(), origin.y(), origin.z() },
        float3{ direction.x(), direction.y(), direction.z() },
        0.f,
        tMax,
        0.f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // SBT params
        p0, p1
    );

    const int pixelIndex = 3 * (index.y * dim.x + index.x);
    if (prd.isHit) {
        params.depthBuffer[depthIndex] = prd.t;

        const float cosTheta = fabs(-dot(prd.normal, direction));
        params.outputBuffer[pixelIndex + 0] = cosTheta;
        params.outputBuffer[pixelIndex + 1] = cosTheta;
        params.outputBuffer[pixelIndex + 2] = cosTheta;
    }
}
