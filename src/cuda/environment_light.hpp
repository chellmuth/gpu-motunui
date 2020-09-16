#pragma once

#include <vector>

#include <cuda.h>

namespace moana {

void runAKernel(
    int width,
    int height,
    cudaTextureObject_t textureObject,
    float *devDirectionBuffer,
    std::vector<float> &outputBuffer
);

}
