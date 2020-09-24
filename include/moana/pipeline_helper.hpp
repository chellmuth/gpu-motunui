#pragma once

#include <string>
#include <vector>

#include <optix.h>

#include "moana/pipeline.hpp"

namespace moana { namespace PipelineHelper {

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    SceneState &sceneState,
    const std::string &ptxSource
);

void linkPipeline(
    OptixState &state,
    const std::vector<OptixProgramGroup> programGroups
);

} }
