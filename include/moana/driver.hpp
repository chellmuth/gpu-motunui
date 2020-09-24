#pragma once

#include <map>
#include <string>
#include <vector>

#include <optix.h>

#include "moana/core/bsdf_sample_record.hpp"
#include "moana/core/camera.hpp"
#include "moana/cuda/environment_light.hpp"
#include "moana/scene.hpp"
#include "moana/scene/as_arena.hpp"
#include "moana/scene/types.hpp"
#include "moana/pipeline.hpp"
#include "moana/types.hpp"

namespace moana {

enum class PipelineType {
    MainRay = 0,
    ShadowRay = 1
};

class Driver {
public:
    void init();
    void launch(
        RenderRequest renderRequest,
        Cam cam,
        const std::string &exrFilename
    );

private:
    std::map<PipelineType, OptixState> m_optixStates;
    SceneState m_sceneState;
};

}
