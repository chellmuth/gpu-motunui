#include "parsers/curve_parser.hpp"

#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include "assert_macros.hpp"

namespace moana {

Curve::Curve(const std::string &filename)
    : m_filename(filename)
{}

OptixTraversableHandle Curve::gasFromCurve(
    OptixDeviceContext context,
    ASArena &arena
) {
    std::ifstream curveFile(m_filename);

    int strandCount;
    curveFile.read((char *)&strandCount, sizeof(int));
    std::cout << "Strand count: " << strandCount << std::endl;

    int verticesPerStrand;
    curveFile.read((char *)&verticesPerStrand, sizeof(int));
    std::cout << "Vertices per strand: " << verticesPerStrand << std::endl;

    const int degree = 3;
    const int segmentsPerStrand = verticesPerStrand - degree;
    const int segmentCount = strandCount * segmentsPerStrand;
    std::cout << "Segment count: " << segmentCount << std::endl;

    float rootWidth;
    float tipWidth;
    curveFile.read((char *)&rootWidth, sizeof(float));
    curveFile.read((char *)&tipWidth, sizeof(float));

    const float rootRadius = rootWidth / 2.f;
    const float tipRadius = tipWidth / 2.f;

    std::cout << "Root width: " << rootRadius << std::endl
              << "Tip width: " << tipRadius << std::endl;

    const int totalVertexCount = strandCount * verticesPerStrand;
    const int totalFloatCount = totalVertexCount * 3;
    std::cout << "Total vertices: " << totalVertexCount << std::endl;
    std::cout << "Total floats: " << totalFloatCount << std::endl;

    std::vector<float> controlPoints(totalFloatCount);
    curveFile.read(
        (char *)controlPoints.data(),
        sizeof(float) * totalFloatCount
    );

    CUdeviceptr d_vertices = 0;
    size_t verticesSizeInBytes = totalFloatCount * sizeof(float);
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_vertices),
        verticesSizeInBytes
    ));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_vertices),
        controlPoints.data(),
        verticesSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    const float range = tipRadius - rootRadius;
    std::vector<float> widths(totalVertexCount);

    // TODO: Which is right?
    // for (int j = 0; j < verticesPerStrand; j++) {
    //     const float lerpT = 1.f * (j - 1) / (verticesPerStrand - 1);
    //     widths[i * verticesPerStrand + j] = rootRadius + (range * lerpT);
    // }

    // Make sure the phantom points aren't part of the radius interpolation
    for (int i = 0; i < strandCount; i++) {
        widths[i * verticesPerStrand + 0] = rootRadius;

        for (int j = 1; j < verticesPerStrand - 1; j++) {
            const float lerpT = 1.f * (j - 1) / (verticesPerStrand - 3);
            widths[i * verticesPerStrand + j] = rootRadius + (range * lerpT);
        }

        widths[i * verticesPerStrand + verticesPerStrand - 1] = tipRadius;
    }

    CUdeviceptr d_widths = 0;
    size_t widthsSizeInBytes = totalVertexCount * sizeof(float);
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_widths),
        widthsSizeInBytes
    ));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_widths),
        widths.data(),
        widthsSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    std::vector<int> indices;
    for (int i = 0; i < strandCount; i++) {
        for (int j = 0; j < segmentsPerStrand; j++) {
            indices.push_back(i * verticesPerStrand + j);
        }
    }

    CUdeviceptr d_indices = 0;
    size_t indicesSizeInBytes = segmentCount * sizeof(int);
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_indices),
        indicesSizeInBytes
    ));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_indices),
        indices.data(),
        indicesSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS; // fixme; use user data
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD; // no updates

    // Setup build input
    uint32_t inputFlags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    OptixBuildInput curveInput = {};
    curveInput.type = OPTIX_BUILD_INPUT_TYPE_CURVES;

    curveInput.curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
    curveInput.curveArray.numPrimitives = segmentCount;
    curveInput.curveArray.vertexBuffers = &d_vertices;
    curveInput.curveArray.numVertices = static_cast<uint32_t>(totalVertexCount);
    curveInput.curveArray.vertexStrideInBytes = sizeof(float) * 3;
    curveInput.curveArray.widthBuffers = &d_widths;
    curveInput.curveArray.widthStrideInBytes = sizeof(float);
    curveInput.curveArray.normalBuffers = 0;
    curveInput.curveArray.normalStrideInBytes = 0;
    curveInput.curveArray.indexBuffer = d_indices;
    curveInput.curveArray.indexStrideInBytes = sizeof(int);
    curveInput.curveArray.flag = OPTIX_GEOMETRY_FLAG_NONE;
    curveInput.curveArray.primitiveIndexOffset = 0;

    // Calculate max memory size
    OptixAccelBufferSizes gasBufferSizes;
    CHECK_OPTIX(optixAccelComputeMemoryUsage(
        context,
        &accelOptions,
        &curveInput,
        1, // build input count
        &gasBufferSizes
    ));

    std::cout << "  Curve GAS:" << std::endl;
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
        &curveInput,
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
    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_widths)));
    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_tempBufferGas)));

    return handle;
}

}
