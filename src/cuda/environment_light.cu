#include "moana/cuda/environment_light.hpp"

#include "assert_macros.hpp"
#include "moana/core/coordinates.hpp"
#include "moana/core/vec3.hpp"

namespace moana {

// fixme
static constexpr float rotationOffset = 115.f / 180.f * M_PI;

__global__ static void environmentLightKernel(
    int width,
    int height,
    cudaTextureObject_t textureObject,
    float *occlusionBuffer,
    float *directionBuffer,
    float *outputBuffer
) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    if ((row >= height) || (col >= width)) { return; }

    const int directionIndex = 3 * (row * width + col);
    const int occlusionIndex = 1 * (row * width + col);
    if (occlusionBuffer[occlusionIndex] != 0.f) { return; }

    Vec3 direction(
        directionBuffer[directionIndex + 0],
        directionBuffer[directionIndex + 1],
        directionBuffer[directionIndex + 2]
    );

    // Pixels that have already been lit in previous bounces
    if (direction.isZero()) { return; }

    float phi, theta;
    Coordinates::cartesianToSpherical(direction, &phi, &theta);

    phi += rotationOffset;
    if (phi > 2.f * M_PI) {
        phi -= 2.f * M_PI;
    }

    float4 environment = tex2D<float4>(
        textureObject,
        phi / (M_PI * 2.f),
        theta / M_PI
    );

    const int outputIndex = 3 * (row * width + col);
    outputBuffer[outputIndex + 0] = environment.x;
    outputBuffer[outputIndex + 1] = environment.y;
    outputBuffer[outputIndex + 2] = environment.z;
}

void EnvironmentLight::calculateEnvironmentLighting(
    int width,
    int height,
    cudaTextureObject_t textureObject,
    float *devOcclusionBuffer,
    float *devDirectionBuffer,
    std::vector<float> &outputBuffer
) {
    const size_t outputBufferSizeInBytes = outputBuffer.size() * sizeof(float);
    CUdeviceptr d_outputBuffer = 0;
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_outputBuffer),
        outputBufferSizeInBytes
    ));
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(d_outputBuffer),
        0,
        outputBufferSizeInBytes
    ));

    const int blockWidth = 16;
    const int blockHeight = 16;

    const dim3 blocks(width / blockWidth + 1, height / blockHeight + 1);
    const dim3 threads(blockWidth, blockHeight);

    environmentLightKernel<<<blocks, threads>>>(
        width,
        height,
        textureObject,
        devOcclusionBuffer,
        devDirectionBuffer,
        reinterpret_cast<float *>(d_outputBuffer)
    );

    CHECK_CUDA(cudaMemcpy(
        outputBuffer.data(),
        reinterpret_cast<void *>(d_outputBuffer),
        outputBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_outputBuffer)));

    CHECK_CUDA(cudaDeviceSynchronize());
}

void EnvironmentLight::queryMemoryRequirements()
{
    std::string moanaRoot = MOANA_ROOT;
    m_texturePtr = std::make_unique<Texture>(moanaRoot + "/island/textures/islandsun.exr");
    m_texturePtr->determineAndSetPitch();
}

EnvironmentLightState EnvironmentLight::snapshotTextureObject(ASArena &arena)
{
    EnvironmentLightState environmentState;
    environmentState.textureObject = m_texturePtr->createTextureObject(arena);
    environmentState.snapshot = arena.createSnapshot();

    return environmentState;
}

}
