#include "scene/gas.hpp"

#include <iostream>

#include <string.h>
#include <cuda_runtime.h>
#include <optix_stubs.h>

#include "assert_macros.hpp"
#include "util/enumerate.hpp"

namespace moana { namespace GAS {

OptixTraversableHandle gasInfoFromMeshRecords(
    OptixDeviceContext context,
    ASArena &arena,
    const std::vector<MeshRecord> &records,
    int primitiveIndexOffset
) {
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS; // fixme; use user data
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD; // no updates

    const int buildInputCount = records.size();
    std::vector<OptixBuildInput> triangleInputs(buildInputCount); // fixme: So many buildInputCount > 100k
    memset(triangleInputs.data(), 0, sizeof(OptixBuildInput) * buildInputCount);

    std::vector<CUdeviceptr> verticesToFree(buildInputCount);
    std::vector<CUdeviceptr> vertexIndicesToFree(buildInputCount);

    uint32_t inputFlags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    for (const auto &[i, record] : enumerate(records)) {
        CUdeviceptr d_vertices = 0;
        size_t verticesSizeInBytes = record.vertices.size() * sizeof(float);
        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void **>(&d_vertices),
            verticesSizeInBytes
        ));
        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_vertices),
            record.vertices.data(),
            verticesSizeInBytes,
            cudaMemcpyHostToDevice
        ));

        CUdeviceptr d_vertexIndices = 0;
        size_t vertexIndicesSizeInBytes = record.vertexIndices.size() * sizeof(int);
        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void **>(&d_vertexIndices),
            vertexIndicesSizeInBytes
        ));
        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_vertexIndices),
            record.vertexIndices.data(),
            vertexIndicesSizeInBytes,
            cudaMemcpyHostToDevice
        ));

        verticesToFree.push_back(d_vertices);
        vertexIndicesToFree.push_back(d_vertexIndices);

        // Setup build input
        triangleInputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        triangleInputs[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInputs[i].triangleArray.numVertices = record.vertices.size() / 3;
        triangleInputs[i].triangleArray.vertexBuffers = &verticesToFree[verticesToFree.size() - 1];

        triangleInputs[i].triangleArray.numIndexTriplets = record.indexTripletCount;
        triangleInputs[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInputs[i].triangleArray.indexBuffer = d_vertexIndices;

        triangleInputs[i].triangleArray.flags = inputFlags;
        triangleInputs[i].triangleArray.primitiveIndexOffset = 0;
        triangleInputs[i].triangleArray.numSbtRecords = 1;
    }

    // Calculate max memory size
    OptixAccelBufferSizes gasBufferSizes;
    CHECK_OPTIX(optixAccelComputeMemoryUsage(
        context,
        &accelOptions,
        triangleInputs.data(),
        buildInputCount,
        &gasBufferSizes
    ));

    std::cout << "  GAS:" << std::endl;
    std::cout << "    Output Buffer size(mb): "
              << (gasBufferSizes.outputSizeInBytes / (1024. * 1024.))
              << std::endl
              << "    Temp Buffer size(mb): "
              << (gasBufferSizes.tempSizeInBytes / (1024. * 1024.))
              << std::endl
              << std::endl;

    CUdeviceptr d_tempBufferGas = arena.pushTemp(gasBufferSizes.tempSizeInBytes);
    CUdeviceptr d_gasOutputBuffer = arena.allocOutput(gasBufferSizes.outputSizeInBytes);

    OptixTraversableHandle handle;
    CHECK_OPTIX(optixAccelBuild(
        context,
        0, // default CUDA stream
        &accelOptions,
        triangleInputs.data(),
        buildInputCount,
        d_tempBufferGas,
        gasBufferSizes.tempSizeInBytes,
        d_gasOutputBuffer,
        gasBufferSizes.outputSizeInBytes,
        &handle,
        nullptr, 0 // emitted property params
    ));

    CHECK_CUDA(cudaDeviceSynchronize());

    for (auto d_vertices : verticesToFree) {
        CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_vertices)));
    }
    for (auto d_vertexIndices : vertexIndicesToFree) {
        CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_vertexIndices)));
    }
    arena.popTemp();

    return handle;
}

} }
