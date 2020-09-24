#pragma once

#include <cuda.h>

#include "moana/driver.hpp"
#include "moana/scene.hpp"
#include "moana/scene/types.hpp"
#include "moana/types.hpp"

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
    float *barycentricBuffer;
    int *idBuffer;
    char *shadowOcclusionBuffer;
    float *shadowWeightBuffer;

    Camera camera;
    int bounce;

    int sampleCount;
    int rayType;
};

void launch(
    RenderRequest request,
    OptixState &optixState,
    OptixState &optixStateShadow, // fixme
    SceneState &sceneState,
    Cam cam,
    const std::string &exrFilename
);

} }
