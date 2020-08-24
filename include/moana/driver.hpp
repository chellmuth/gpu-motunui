#pragma once

#include <optix.h>

#include "moana/core/camera.hpp"

namespace moana {

struct RayGenData {};
struct MissData {};
struct HitGroupData {};

struct Params {
    OptixTraversableHandle handle;

    float *outputBuffer;
    Camera camera;
};

struct OptixState {
    OptixDeviceContext context = 0;
    OptixTraversableHandle gasHandle = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixModule module = 0;
    OptixProgramGroup raygenProgramGroup;
    OptixProgramGroup missProgramGroup;
    OptixProgramGroup hitgroupProgramGroup;
    OptixPipeline pipeline = 0;
    OptixShaderBindingTable sbt = {};
};

class Driver {
public:
    void init();
    void launch();

private:
    OptixState m_state;
    CUdeviceptr d_params;
};

}
