#pragma once

#include <cuda.h>

#include "moana/driver.hpp"
#include "moana/scene.hpp"

namespace moana { namespace Renderer {

void launch(
    OptixState &state,
    CUdeviceptr &d_params,
    Cam cam,
    const std::string &exrFilename
);

} }
