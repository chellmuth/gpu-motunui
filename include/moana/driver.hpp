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

struct Params {
    OptixTraversableHandle handle;

    float *outputBuffer;
    float *depthBuffer;
    float *xiBuffer;
    float *cosThetaWiBuffer;
    BSDFSampleRecord *sampleRecordInBuffer;
    BSDFSampleRecord *sampleRecordOutBuffer;
    float *occlusionBuffer;
    float *missDirectionBuffer;
    float *colorBuffer;
    float *normalBuffer;
    float *barycentricBuffer;
    int *idBuffer;

    Camera camera;
    int bounce;

    int sampleCount;
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

    ASArena arena;
    std::vector<GeometryResult> geometries;

    EnvironmentLightState environmentState;

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
