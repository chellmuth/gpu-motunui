#pragma once

#include <memory>
#include <vector>

#include <cuda.h>

#include "moana/core/texture.hpp"
#include "moana/scene/as_arena.hpp"

namespace moana {

struct EnvironmentLightState {
    cudaTextureObject_t textureObject;
    Snapshot snapshot;
};

class EnvironmentLight {
public:
    void queryMemoryRequirements();
    EnvironmentLightState snapshotTextureObject(ASArena &arena);

private:
    std::unique_ptr<Texture> m_texturePtr;
};

}

namespace moana { namespace EnvironmentLighting {

void calculateEnvironmentLighting(
    int width,
    int height,
    cudaTextureObject_t textureObject,
    float *devDirectionBuffer,
    std::vector<float> &outputBuffer
);

} }
