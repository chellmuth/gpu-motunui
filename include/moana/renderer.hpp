#pragma once

#include <cuda.h>

#include "moana/driver.hpp"
#include "moana/scene.hpp"

namespace moana { namespace Renderer {

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

void launch(
    OptixState &state,
    Cam cam,
    const std::string &exrFilename
);

} }
