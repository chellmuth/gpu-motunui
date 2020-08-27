#include "scene/gas.hpp"

#include <iostream>

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include "assert_macros.hpp"

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

    CUdeviceptr d_indices = 0;
    size_t indicesSizeInBytes = model.indexTripletCount * 3 * sizeof(int);
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_indices),
        indicesSizeInBytes
    ));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_indices),
        model.indices.data(),
        indicesSizeInBytes,
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

    std::cout << "  GAS:" << std::endl;
    std::cout << "    Output Buffer size(mb): "
              << (gasBufferSizes.outputSizeInBytes / (1024. * 1024.))
              << std::endl
              << "    Temp Buffer size(mb): "
              << (gasBufferSizes.tempSizeInBytes / (1024. * 1024.))
              << std::endl
              << "    Vertices size(mb): "
              << (verticesSizeInBytes / (1024. * 1024.))
              << std::endl
              << "    Indices size(mb): "
              << (indicesSizeInBytes / (1024. * 1024.))
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

    return handle;
}

} }
