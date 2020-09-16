#pragma once

#include <vector>

#include <cuda.h>

namespace moana { namespace EnvironmentLight {

void calculateEnvironmentLighting(
    int width,
    int height,
    cudaTextureObject_t textureObject,
    float *devDirectionBuffer,
    std::vector<float> &outputBuffer
);

} }
