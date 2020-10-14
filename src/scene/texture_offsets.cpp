#include "scene/texture_offsets.hpp"

namespace moana { namespace Textures {

static const char *data[] = {
    #define X(x) x,
    #include "scene/data/texture_offsets_data.cpp"
    #undef X
};

std::vector<std::string> textureFilenames(std::begin(data), std::end(data));

} }
