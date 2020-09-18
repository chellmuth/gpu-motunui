#pragma once

#include <cuda.h>

#include "moana/driver.hpp"
#include "moana/scene.hpp"
#include "moana/scene/types.hpp"

namespace moana { namespace Renderer {

struct Params {
    OptixTraversableHandle handle;

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
    float *tempBuffer;

    Camera camera;
    int bounce;

    int sampleCount;
    int rayType;
};

void launch(
    OptixState &optixState,
    SceneState &sceneState,
    Cam cam,
    const std::string &exrFilename
);

} }
