#pragma once

#include <vector>

#include <optix.h>

#include "moana/core/camera.hpp"
#include "moana/parsers/obj_parser.hpp"

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
};

class Driver {
public:
    void init(const ObjResult &model);
    void launch();

private:
    OptixState m_state;
    CUdeviceptr d_params;
};

}
