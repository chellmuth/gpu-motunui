#include "moana/pipeline.hpp"

#include "kernel.hpp"
#include "shadow_ray_kernel.hpp"
#include "pipeline_helper.hpp"

namespace moana { namespace Pipeline {

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    SceneState &sceneState
) {
    return PipelineHelper::initOptixState(
        optixState,
        context,
        sceneState,
        PTX::mainRaySource
    );
}

} }

namespace moana { namespace ShadowPipeline {

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    SceneState &sceneState
) {
    return PipelineHelper::initOptixState(
        optixState,
        context,
        sceneState,
        PTX::shadowRaySource
    );
}

} }
