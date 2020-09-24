#pragma once

#include <optix.h>

#include "moana/scene/types.hpp"

namespace moana {

struct RayGenData {};
struct MissData {};
struct HitGroupData {
    float3 baseColor;
    int textureIndex;
    int materialID;

    float *normals;
    int *normalIndices;
};

struct OptixState {
    OptixDeviceContext context = 0;
    std::vector<OptixTraversableHandle> gasHandles = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixModuleCompileOptions moduleCompileOptions = {};
    OptixModule module = 0;
    OptixProgramGroup raygenProgramGroup;
    OptixProgramGroup missProgramGroup;
    OptixProgramGroup hitgroupProgramGroup;
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
