#pragma once

#include <vector>
#include <string>

namespace moana { namespace Image {

// radiances[0] == top-left red
void save(
    int width,
    int height,
    const std::vector<float> &radiances,
    const std::string &filename
);

} }
