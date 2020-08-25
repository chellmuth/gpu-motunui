#include "hibiscus_geometry.hpp"

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
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE; // no build flags
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

GeometryResult HibiscusGeometry::buildAcceleration(OptixDeviceContext context)
{
    const std::string moanaRoot = MOANA_ROOT;
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

    for (int i = 0; i < 4; i++) {
        const std::string archive = archives[i];
        const std::string instance = instances[i];

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
        const Instances instancesResult = parseInstances(instances[i]);
        std::cout << "    Count: " << instancesResult.count << std::endl;
    }

    return GeometryResult{};
}

}
