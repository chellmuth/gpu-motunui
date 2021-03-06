#include <optix.h>

#include <stdio.h>

#include "bsdfs/lambertian.hpp"
#include "bsdfs/water.hpp"
#include "moana/core/bsdf_sample_record.hpp"
#include "moana/core/camera.hpp"
#include "moana/core/ray.hpp"
#include "moana/cuda/bsdf.hpp"
#include "moana/driver.hpp"
#include "moana/render/renderer.hpp"
#include "optix_sdk.hpp"
#include "random.hpp"
#include "ray_data.hpp"
#include "util.hpp"

using namespace moana;

extern "C" {
    __constant__ Renderer::Params params;
}

__forceinline__ __device__ static BSDFSampleRecord createSamplingRecord(
    const PerRayData &prd
) {
    const uint3 index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    if (prd.bsdfType == BSDFType::Water) {
        return Water::sample(index, dim, prd, params.xiBuffer);
    } else { // prd.bsdfType == BSDFType::Diffuse
        return Lambertian::sample(index, dim, prd, params.xiBuffer);
    }

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
    prd->point = getHitPoint();

    const float3 rayDirection = optixGetWorldRayDirection();
    prd->woWorld = -Vec3(rayDirection.x, rayDirection.y, rayDirection.z);

    HitGroupData *hitgroupData = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
    prd->baseColor = hitgroupData->baseColor;
    prd->materialID = hitgroupData->materialID;
    prd->bsdfType = hitgroupData->bsdfType;
    prd->textureIndex = hitgroupData->textureIndex;

    const unsigned int primitiveIndex = optixGetPrimitiveIndex();
    prd->primitiveID = primitiveIndex;

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

        vertices[0] = optixTransformPointFromObjectToWorldSpace(vertices[0]);
        vertices[1] = optixTransformPointFromObjectToWorldSpace(vertices[1]);
        vertices[2] = optixTransformPointFromObjectToWorldSpace(vertices[2]);

        int normalIndex0 = hitgroupData->normalIndices[primitiveIndex * 3 + 0];
        int normalIndex1 = hitgroupData->normalIndices[primitiveIndex * 3 + 1];
        int normalIndex2 = hitgroupData->normalIndices[primitiveIndex * 3 + 2];

        float n0x = hitgroupData->normals[normalIndex0 * 3 + 0];
        float n0y = hitgroupData->normals[normalIndex0 * 3 + 1];
        float n0z = hitgroupData->normals[normalIndex0 * 3 + 2];

        float n1x = hitgroupData->normals[normalIndex1 * 3 + 0];
        float n1y = hitgroupData->normals[normalIndex1 * 3 + 1];
        float n1z = hitgroupData->normals[normalIndex1 * 3 + 2];

        float n2x = hitgroupData->normals[normalIndex2 * 3 + 0];
        float n2y = hitgroupData->normals[normalIndex2 * 3 + 1];
        float n2z = hitgroupData->normals[normalIndex2 * 3 + 2];

        float3 n0Object{n0x, n0y, n0z};
        float3 n1Object{n1x, n1y, n1z};
        float3 n2Object{n2x, n2y, n2z};

        float3 n0World = optixTransformNormalFromObjectToWorldSpace(n0Object);
        float3 n1World = optixTransformNormalFromObjectToWorldSpace(n1Object);
        float3 n2World = optixTransformNormalFromObjectToWorldSpace(n2Object);

        const Vec3 n0 = normalized(Vec3(n0World.x, n0World.y, n0World.z));
        const Vec3 n1 = normalized(Vec3(n1World.x, n1World.y, n1World.z));
        const Vec3 n2 = normalized(Vec3(n2World.x, n2World.y, n2World.z));

        const float2 barycentrics = optixGetTriangleBarycentrics();
        const float alpha = barycentrics.x;
        const float beta = barycentrics.y;
        const float gamma = 1.f - alpha - beta;

        const Vec3 normal = gamma * n0
            + alpha * n1
            + beta * n2;

        // const Vec3 p0(vertices[0].x, vertices[0].y, vertices[0].z);
        // const Vec3 p1(vertices[1].x, vertices[1].y, vertices[1].z);
        // const Vec3 p2(vertices[2].x, vertices[2].y, vertices[2].z);

        // // Debug: face normals
        // const Vec3 e1 = p1 - p0;
        // const Vec3 e2 = p2 - p0;
        // const Vec3 normal = normalized(cross(e1, e2));

        prd->normal = normalized(normal);
        prd->barycentrics = optixGetTriangleBarycentrics();
    } else {
        const unsigned int primitiveIndex = optixGetPrimitiveIndex();
        const float3 normal = normalCubic(primitiveIndex);
        prd->normal = normalized(Vec3(normal.x, normal.y, normal.z));
        prd->barycentrics = float2{0.f, 0.f};
    }

    if (dot(prd->normal, prd->woWorld) < 0.f) {
        prd->normal = -1.f * prd->normal;
        prd->isInside = true;
    } else {
        prd->isInside = false;
    }
}

extern "C" __global__ void __miss__ms()
{
    float3 direction = optixGetWorldRayDirection();
    PerRayData *prd = getPRD();
    prd->isHit = false;
    prd->materialID = -1;
}

__forceinline__ __device__ static Ray raygenCamera()
{
    const uint3 index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const int row = index.y;
    const int col = index.x;

    const int xiIndex = 2 * (index.y * dim.x + index.x);
    float xi1 = params.xiBuffer[xiIndex + 0];
    float xi2 = params.xiBuffer[xiIndex + 1];
    if (xi1 < 0) {
        unsigned int seed = tea<4>(index.y * dim.x + index.x, params.sampleCount);
        xi1 = rnd(seed);
        xi2 = rnd(seed);

        params.xiBuffer[xiIndex + 0] = xi1;
        params.xiBuffer[xiIndex + 1] = xi2;
    }

    const Ray cameraRay = params.camera.generateRay(
        row, col,
        float2{ xi1, xi2 }
    );
    return cameraRay;
}

__forceinline__ __device__ static Ray raygenBounce(bool *quit)
{
    *quit = false;

    const uint3 index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const int sampleRecordIndex = 1 * (index.y * dim.x + index.x);

    const BSDFSampleRecord &sampleRecord = params.sampleRecordInBuffer[sampleRecordIndex];
    if (!sampleRecord.isValid) {
        *quit = true;
        return Ray();
    }

    const float3 origin = sampleRecord.point;

    const Vec3 &wiLocal = sampleRecord.wiLocal;
    const Vec3 wiWorld = sampleRecord.frame.toWorld(wiLocal);

    const Ray bounceRay(Vec3(origin.x, origin.y, origin.z), normalized(wiWorld));
    return bounceRay;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    bool quit = false;
    Ray ray = params.bounce == 0
        ? raygenCamera()
        : raygenBounce(&quit)
    ;

    if (quit) { return; }

    const Vec3 origin = ray.origin();
    const Vec3 direction = ray.direction();

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
        1e-4,
        tMax,
        0.f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0, // SBT params
        p0, p1
    );

    if (prd.isHit) {
        params.depthBuffer[depthIndex] = prd.t;

        const int occlusionIndex = 1 * (index.y * dim.x + index.x);
        params.occlusionBuffer[occlusionIndex + 0] = 1.f;

        const BSDFSampleRecord sampleRecord = createSamplingRecord(prd);
        const int sampleRecordIndex = 1 * (index.y * dim.x + index.x);
        params.sampleRecordOutBuffer[sampleRecordIndex] = sampleRecord;

        const int cosThetaWiIndex = index.y * dim.x + index.x;
        params.cosThetaWiBuffer[cosThetaWiIndex] = fabsf(sampleRecord.wiLocal.z());

        const float3 baseColor = prd.baseColor;
        const int colorIndex = 3 * (index.y * dim.x + index.x);
        params.colorBuffer[colorIndex + 0] = baseColor.x;
        params.colorBuffer[colorIndex + 1] = baseColor.y;
        params.colorBuffer[colorIndex + 2] = baseColor.z;

        const int barycentricIndex = 2 * (index.y * dim.x + index.x);
        params.barycentricBuffer[barycentricIndex + 0] = prd.barycentrics.x;
        params.barycentricBuffer[barycentricIndex + 1] = prd.barycentrics.y;

        const int idIndex = 3 * (index.y * dim.x + index.x);
        params.idBuffer[idIndex + 0] = prd.primitiveID;
        params.idBuffer[idIndex + 1] = prd.materialID;
        params.idBuffer[idIndex + 2] = prd.textureIndex;
    }

    const int missDirectionIndex = 3 * (index.y * dim.x + index.x);
    params.missDirectionBuffer[missDirectionIndex + 0] = direction.x();
    params.missDirectionBuffer[missDirectionIndex + 1] = direction.y();
    params.missDirectionBuffer[missDirectionIndex + 2] = direction.z();
}
