#pragma once

#include <string>
#include <vector>

#include <optix.h>

#include "moana/pipeline.hpp"

namespace moana { namespace PipelineHelper {

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    const SceneState &sceneState,
    const std::string &ptxSource
);

} }
