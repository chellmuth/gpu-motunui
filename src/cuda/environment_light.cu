#include "cuda/environment_light.hpp"

#include <iostream> // fixme

#include "assert_macros.hpp"

namespace moana {

__global__ static void testKernel(
    int width,
    int height,
    cudaTextureObject_t textureObject,
    float *outputBuffer
) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int col = threadIdx.x + blockIdx.x * blockDim.x;

    if ((row >= height) || (col >= width)) return;

    float4 environment = tex2D<float4>(
        textureObject,
        1.f * col / width,
        1.f * row / height
    );

    const int outputIndex = 3 * (row * width + col);
    outputBuffer[outputIndex + 0] = environment.x;
    outputBuffer[outputIndex + 1] = environment.y;
    outputBuffer[outputIndex + 2] = environment.z;

//         float phi, theta;
//         Coordinates::cartesianToSpherical(direction, &phi, &theta);

//         phi += 115.f / 180.f * M_PI;
//         if (phi > 2.f * M_PI) {
//             phi -= 2.f * M_PI;
//         }

//         float4 environment = tex2D<float4>(
//             params.environment,
//             phi / (M_PI * 2.f),
//             theta / M_PI
//         );

//         const int colorIndex = 3 * (index.y * dim.x + index.x);
//         params.colorBuffer[colorIndex + 0] = environment.x;
//         params.colorBuffer[colorIndex + 1] = environment.y;
//         params.colorBuffer[colorIndex + 2] = environment.z;
//     }
}

void runAKernel(
    int width,
    int height,
    cudaTextureObject_t textureObject,
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

    std::cout << "KERNEL" << std::endl;
    testKernel<<<blocks, threads>>>(
        width,
        height,
        textureObject,
        reinterpret_cast<float *>(d_outputBuffer)
    );

    CHECK_CUDA(cudaMemcpy(
        outputBuffer.data(),
        reinterpret_cast<void *>(d_outputBuffer),
        outputBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaDeviceSynchronize());
}

}
