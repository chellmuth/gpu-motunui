#pragma once

#include <string>
#include <vector>

#include <optix.h>

#include "moana/core/bsdf_sample_record.hpp"
#include "moana/core/camera.hpp"
#include "moana/cuda/environment_light.hpp"
#include "moana/scene.hpp"
#include "moana/scene/as_arena.hpp"
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

struct SceneState {
    ASArena arena;
    std::vector<GeometryResult> geometries;
    EnvironmentLightState environmentState;
};

class Driver {
public:
    void init();
    void launch(Cam cam, const std::string &exrFilename);

private:
    OptixState m_optixState;
    SceneState m_sceneState;
};

}
