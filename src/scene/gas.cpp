#include "scene/gas.hpp"

#include <iostream>

#include <string.h>
#include <cuda_runtime.h>
#include <optix_stubs.h>

#include "assert_macros.hpp"
#include "enumerate.hpp"

namespace moana { namespace GAS {

OptixTraversableHandle gasInfoFromObjResult(
    OptixDeviceContext context,
    ASArena &arena,
    const ObjResult &model
) {
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS; // fixme; use user data
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD; // no updates

    CUdeviceptr d_vertices = 0;
    size_t verticesSizeInBytes = model.vertexCount * 3 * sizeof(float);
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_vertices),
        verticesSizeInBytes
    ));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_vertices),
        model.vertices.data(),
        verticesSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    const int buildInputCount = model.buildInputResults.size();
    OptixBuildInput triangleInputs[model.buildInputResults.size()];
    memset(triangleInputs, 0, sizeof(OptixBuildInput) * buildInputCount);

    std::vector<CUdeviceptr> indicesToFree;

    uint32_t inputFlags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    for (const auto &[i, buildInputResult] : enumerate(model.buildInputResults)) {
        CUdeviceptr d_indices = 0;
        size_t indicesSizeInBytes = buildInputResult.indexTripletCount * 3 * sizeof(int);
        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void **>(&d_indices),
            indicesSizeInBytes
        ));
        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_indices),
            buildInputResult.indices.data(),
            indicesSizeInBytes,
            cudaMemcpyHostToDevice
        ));

        indicesToFree.push_back(d_indices);

        // Setup build input
        triangleInputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        triangleInputs[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInputs[i].triangleArray.numVertices = model.vertexCount;
        triangleInputs[i].triangleArray.vertexBuffers = &d_vertices;

        triangleInputs[i].triangleArray.numIndexTriplets = buildInputResult.indexTripletCount;
        triangleInputs[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInputs[i].triangleArray.indexBuffer = d_indices;

        triangleInputs[i].triangleArray.flags = inputFlags;
        triangleInputs[i].triangleArray.numSbtRecords = 1;
    }

    // Calculate max memory size
    OptixAccelBufferSizes gasBufferSizes;
    CHECK_OPTIX(optixAccelComputeMemoryUsage(
        context,
        &accelOptions,
        triangleInputs,
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
              << "    Vertices size(mb): "
              << (verticesSizeInBytes / (1024. * 1024.))
              << std::endl;

    CUdeviceptr d_tempBufferGas;
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_tempBufferGas),
        gasBufferSizes.tempSizeInBytes
    ));

    CUdeviceptr d_gasOutputBuffer = arena.allocOutput(gasBufferSizes.outputSizeInBytes);

    OptixTraversableHandle handle;
    CHECK_OPTIX(optixAccelBuild(
        context,
        0, // default CUDA stream
        &accelOptions,
        triangleInputs,
        buildInputCount,
        d_tempBufferGas,
        gasBufferSizes.tempSizeInBytes,
        d_gasOutputBuffer,
        gasBufferSizes.outputSizeInBytes,
        &handle,
        nullptr, 0 // emitted property params
    ));

    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_vertices)));
    for (auto indices : indicesToFree) {
        CHECK_CUDA(cudaFree(reinterpret_cast<void *>(indices)));
    }
    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_tempBufferGas)));

    return handle;
}

} }
