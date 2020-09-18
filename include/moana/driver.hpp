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
#include "moana/pipeline.hpp"

namespace moana {

class Driver {
public:
    void init();
    void launch(Cam cam, const std::string &exrFilename);

private:
    OptixState m_optixState;
    SceneState m_sceneState;
};

}
