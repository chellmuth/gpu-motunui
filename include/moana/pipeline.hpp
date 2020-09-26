#pragma once

#include <optix.h>

#include "moana/cuda/bsdf.hpp"
#include "moana/scene/types.hpp"

namespace moana {

struct RayGenData {};
struct MissData {};
struct HitGroupData {
    float3 baseColor;
    int textureIndex;
    int materialID;
    BSDFType bsdfType;

    float *normals;
    int *normalIndices;
};

struct OptixState {
    OptixPipeline pipeline = 0;
    OptixShaderBindingTable sbt = {};
};

}

namespace moana { namespace Pipeline {

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    SceneState &sceneState
);

} }

namespace moana { namespace ShadowPipeline {

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    SceneState &sceneState
);

} }
