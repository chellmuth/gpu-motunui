#pragma once

#include <string>
#include <vector>

#include <optix.h>

#include "moana/core/camera.hpp"
#include "moana/scene.hpp"
#include "moana/scene/as_arena.hpp"
#include "moana/scene/types.hpp"

namespace moana {

struct RayGenData {};
struct MissData {};
struct HitGroupData {
    float3 baseColor;
};

struct Params {
    OptixTraversableHandle handle;

    float *outputBuffer;
    float *depthBuffer;
    Camera camera;
};

struct OptixState {
    OptixDeviceContext context = 0;
    std::vector<OptixTraversableHandle> gasHandles = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixModule module = 0;
    OptixProgramGroup raygenProgramGroup;
    OptixProgramGroup missProgramGroup;
    OptixProgramGroup hitgroupProgramGroup;
    OptixPipeline pipeline = 0;
    OptixShaderBindingTable sbt = {};

    CUdeviceptr gasOutputBuffer;
    size_t outputBufferSizeInBytes;
    std::vector<void *> gasOutputs;

    ASArena arena;
    std::vector<GeometryResult> geometries;

    OptixModuleCompileOptions moduleCompileOptions = {};
};

class Driver {
public:
    void init();
    void launch(Cam cam, const std::string &exrFilename);

private:
    OptixState m_state;
    CUdeviceptr d_params;
};

}
