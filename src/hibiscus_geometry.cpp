#include "hibiscus_geometry.hpp"

#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "assert_macros.hpp"
#include "moana/parsers/obj_parser.hpp"

namespace moana {

struct Instances {
    int count;
    float *transforms;
};

struct GASInfo {
    OptixTraversableHandle handle;
    CUdeviceptr gasOutputBuffer;
    size_t outputBufferSizeInBytes;
};

static Instances parseInstances(const std::string filepath)
{
    constexpr int transformSize = 12;
    std::ifstream instanceFile(filepath);

    Instances result;
    instanceFile.read((char *)&result.count, sizeof(int));

    int offset = 0;
    result.transforms = new float[transformSize * result.count];
    while (instanceFile.peek() != EOF) {
        instanceFile.read((char *)&result.transforms[offset], sizeof(float) * transformSize);
        offset += transformSize;
    }

    return result;
}

static GASInfo gasFromObj(OptixDeviceContext context, const ObjResult &model)
{
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS; // fixme; use user data
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD; // no updates

    CUdeviceptr d_vertices = 0;
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_vertices),
        model.vertexCount * 3 * sizeof(float)
    ));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_vertices),
        model.vertices.data(),
        model.vertexCount * 3 * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_indices = 0;
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_indices),
        model.indexTripletCount * 3 * sizeof(int)
    ));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_indices),
        model.indices.data(),
        model.indexTripletCount * 3 * sizeof(int),
        cudaMemcpyHostToDevice
    ));

    // Setup build input
    uint32_t inputFlags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.numVertices = model.vertexCount;
    triangleInput.triangleArray.vertexBuffers = &d_vertices;

    triangleInput.triangleArray.numIndexTriplets = model.indexTripletCount;
    triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexBuffer = d_indices;

    triangleInput.triangleArray.flags = inputFlags;
    triangleInput.triangleArray.numSbtRecords = 1;

    // Calculate max memory size
    OptixAccelBufferSizes gasBufferSizes;
    CHECK_OPTIX(optixAccelComputeMemoryUsage(
        context,
        &accelOptions,
        &triangleInput,
        1, // build input count
        &gasBufferSizes
    ));

    CUdeviceptr d_tempBufferGas;
    CUdeviceptr d_gasOutputBuffer;
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_tempBufferGas),
        gasBufferSizes.tempSizeInBytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_gasOutputBuffer),
        gasBufferSizes.outputSizeInBytes
    ));

    OptixTraversableHandle handle;
    CHECK_OPTIX(optixAccelBuild(
        context,
        0, // default CUDA stream
        &accelOptions,
        &triangleInput,
        1, // build input count
        d_tempBufferGas,
        gasBufferSizes.tempSizeInBytes,
        d_gasOutputBuffer,
        gasBufferSizes.outputSizeInBytes,
        &handle,
        nullptr, 0 // emitted property params
    ));

    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_vertices)));
    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_indices)));
    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_tempBufferGas)));

    return GASInfo{
        handle,
        d_gasOutputBuffer,
        gasBufferSizes.outputSizeInBytes
    };
}

static void createOptixInstanceRecords(
    OptixDeviceContext context,
    std::vector<OptixInstance> &records,
    const Instances &instances,
    const GASInfo &gasInfo
) {
    int offset = records.size();
    for (int i = 0; i < instances.count; i++) {
        OptixInstance instance;
        memcpy(
            instance.transform,
            &instances.transforms[i * 12],
            sizeof(float) * 12
        );

        instance.instanceId = offset + i;
        instance.visibilityMask = 255;
        instance.sbtOffset = 0;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = gasInfo.handle;

        records.push_back(instance);
    }
}

static OptixTraversableHandle iasFromInstanceRecords(
    OptixDeviceContext context,
    const std::vector<OptixInstance> &records
) {
    CUdeviceptr d_instances;
    const size_t instancesSizeInBytes = sizeof(OptixInstance) * records.size();
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_instances),
        instancesSizeInBytes
    ));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_instances),
        records.data(),
        instancesSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    OptixBuildInput instanceInput = {};
    instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceInput.instanceArray.instances = d_instances;
    instanceInput.instanceArray.numInstances = records.size();

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes iasBufferSizes;
    CHECK_OPTIX(optixAccelComputeMemoryUsage(
        context,
        &accelOptions,
        &instanceInput,
        1, // num build inputs
        &iasBufferSizes
    ));

    CUdeviceptr d_tempBuffer;
    CUdeviceptr d_iasOutputBuffer; // fixme (free)
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_tempBuffer),
        iasBufferSizes.tempSizeInBytes
    ));
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_iasOutputBuffer),
        iasBufferSizes.outputSizeInBytes
    ));

    OptixTraversableHandle handle;
    CHECK_OPTIX(optixAccelBuild(
        context,
        0, // CUDA stream
        &accelOptions,
        &instanceInput,
        1, // num build inputs
        d_tempBuffer,
        iasBufferSizes.tempSizeInBytes,
        d_iasOutputBuffer,
        iasBufferSizes.outputSizeInBytes,
        &handle,
        nullptr,
        0
    ));

    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_tempBuffer)));
    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_instances)));

    return handle;
}

GeometryResult HibiscusGeometry::buildAcceleration(OptixDeviceContext context)
{
    const std::string moanaRoot = MOANA_ROOT;

    const std::string baseObj = moanaRoot + "/island/obj/isHibiscus/isHibiscus.obj";

    const std::vector<std::string> archives = {
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0001_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusFlower0001_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0003_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0002_mod.obj"
    };

    const std::vector<std::string> instances = {
        "../scene/hibiscus-archiveHibiscusLeaf0001_mod.bin",
        "../scene/hibiscus-archiveHibiscusFlower0001_mod.bin",
        "../scene/hibiscus-archiveHibiscusLeaf0002_mod.bin",
        "../scene/hibiscus-archiveHibiscusLeaf0003_mod.bin",
    };

    std::vector<OptixInstance> records;

    {
        std::cout << "Processing base obj: " << baseObj << std::endl;

        std::cout << "  Geometry:" << std::endl;
        ObjParser objParser(baseObj);
        auto model = objParser.parse();
        std::cout << "    Vertex count: " << model.vertexCount << std::endl
                  << "    Index triplet count: " << model.indexTripletCount << std::endl;

        std::cout << "  GAS:" << std::endl;
        const GASInfo gasInfo = gasFromObj(context, model);
        std::cout << "    Output Buffer size(mb): "
                  << (gasInfo.outputBufferSizeInBytes / (1024. * 1024.))
                  << std::endl;

        float transform[12] = {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f
        };
        Instances instancesResult;
        instancesResult.transforms = transform;
        instancesResult.count = 1;

        createOptixInstanceRecords(
            context,
            records,
            instancesResult,
            gasInfo
        );
    }

    for (int i = 0; i < 4; i++) {
        const std::string archive = archives[i];

        std::cout << "Processing " << archive << std::endl;

        std::cout << "  Geometry:" << std::endl;
        ObjParser objParser(archive);
        auto model = objParser.parse();
        std::cout << "    Vertex count: " << model.vertexCount << std::endl
                  << "    Index triplet count: " << model.indexTripletCount << std::endl;

        std::cout << "  GAS:" << std::endl;
        const GASInfo gasInfo = gasFromObj(context, model);
        std::cout << "    Output Buffer size(mb): "
                  << (gasInfo.outputBufferSizeInBytes / (1024. * 1024.))
                  << std::endl;

        std::cout << "  Instances:" << std::endl;
        const std::string instance = instances[i];
        const Instances instancesResult = parseInstances(instances[i]);
        std::cout << "    Count: " << instancesResult.count << std::endl;

        createOptixInstanceRecords(
            context,
            records,
            instancesResult,
            gasInfo
        );
    }

    OptixTraversableHandle iasHandle = iasFromInstanceRecords(context, records);

    return GeometryResult{
        iasHandle,
        {}
    };

}

}
