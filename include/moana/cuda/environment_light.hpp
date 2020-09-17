#pragma once

#include <memory>
#include <vector>

#include <cuda.h>

#include "moana/core/bsdf_sample_record.hpp"
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

    static void calculateEnvironmentLighting(
        int width,
        int height,
        cudaTextureObject_t textureObject,
        float *devOcclusionBuffer,
        float *devDirectionBuffer,
        std::vector<float> &outputBuffer
    );

    // fixme: duped
    static void calculateEnvironmentLighting(
        int width,
        int height,
        cudaTextureObject_t textureObject,
        BSDFSampleRecord *devDirectionBuffer,
        std::vector<float> &outputBuffer
    );

private:
    std::unique_ptr<Texture> m_texturePtr;
};

}
