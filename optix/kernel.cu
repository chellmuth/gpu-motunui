#include <optix.h>

#include <stdio.h>

#include "moana/core/bsdf_sample_record.hpp"
#include "moana/core/camera.hpp"
#include "moana/core/frame.hpp"
#include "moana/core/ray.hpp"
#include "moana/cuda/triangle.hpp"
#include "moana/driver.hpp"
#include "moana/renderer.hpp"
#include "optix_sdk.hpp"
#include "random.hpp"
#include "sample.hpp"
#include "util.hpp"

using namespace moana;

struct PerRayData {
    bool isHit;
    float t;
    float3 point;
    Vec3 normal;
    float3 baseColor;
    int materialID;
    int primitiveID;
    int textureIndex;
    float2 barycentrics;
};

extern "C" {
    __constant__ Renderer::Params params;
}

__forceinline__ __device__ static BSDFSampleRecord createSamplingRecord(
    const PerRayData &prd,
    const Vec3 &wo
) {
    const uint3 index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    const int xiIndex = 2 * (index.y * dim.x + index.x);
    float xi1 = params.xiBuffer[xiIndex + 0];
    float xi2 = params.xiBuffer[xiIndex + 1];

    const Vec3 wiLocal = Sample::uniformHemisphere(xi1, xi2);
    const Frame frame(prd.normal);

    const BSDFSampleRecord record = {
        .isValid = true,
        .point = prd.point,
        .wiLocal = wiLocal,
        .normal = prd.normal,
        .frame = frame,
    };

    return record;
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

    HitGroupData *hitgroupData = reinterpret_cast<HitGroupData *>(optixGetSbtDataPointer());
    prd->baseColor = hitgroupData->baseColor;
    prd->materialID = hitgroupData->materialID;
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

    const float3 optixDirection = optixGetWorldRayDirection();
    const Vec3 direction(optixDirection.x, optixDirection.y, optixDirection.z);
    if (dot(prd->normal, direction) > 0.f) {
        prd->normal = -1.f * prd->normal;
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

__device__ static void raygenNormal()
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

        const BSDFSampleRecord sampleRecord = createSamplingRecord(prd, -direction);
        const int sampleRecordIndex = 1 * (index.y * dim.x + index.x);
        params.sampleRecordOutBuffer[sampleRecordIndex] = sampleRecord;

        const int cosThetaWiIndex = index.y * dim.x + index.x;
        params.cosThetaWiBuffer[cosThetaWiIndex] = sampleRecord.wiLocal.z();

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

__device__ static void raygenShadow()
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
    prd.isHit = false;

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
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        0, 1, 0, // SBT params
        p0, p1
    );

    const Vec3 lightNormal = Vec3(-0.323744059f, -0.642787874f, -0.694271863f);

    if (prd.isHit) {
        params.shadowOcclusionBuffer[shadowOcclusionIndex] = 1;
    }

    const int shadowWeightIndex = 1 * (index.y * dim.x + index.x);
    params.shadowWeightBuffer[shadowWeightIndex] = 1.f
        * fabsf(dot(lightNormal, -wi))
        * fmaxf(0.f, dot(wi, sampleRecord.normal))
        * (20000.f * 20000.f) / (lightDirection.length() * lightDirection.length())
        * (1.f / M_PI)
    ;
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 index = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // fixme: two pipelines or program groups
    if (params.rayType == 0) {
        raygenNormal();
    } else {
        raygenShadow();
    }
}
