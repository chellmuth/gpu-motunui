#include "scene/texture_lookup.hpp"

#include <map>
#include <utility>

namespace moana { namespace TextureLookup {

static const std::pair<std::tuple<const char *, const char *, const char *>, int> data[] = {
    #define X(k1, k2, k3, i) { std::make_tuple(k1, k2, k3), i },
    #include "scene/data/texture_lookup_data.cpp"
    #undef X
};

using TextureIndexKey = std::tuple<std::string, std::string, std::string>;
static const std::map<TextureIndexKey, int> lookup(std::begin(data), std::end(data));

int indexForMesh(
    const std::string &element,
    const std::string &material,
    const std::string &mesh
) {
    const TextureIndexKey key = std::make_tuple(element, material, mesh);
    if (lookup.count(key) > 0) {
        return lookup.at(key);
    }
    return -1;
}

} }
