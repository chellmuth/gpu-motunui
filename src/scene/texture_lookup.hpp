#pragma once

#include <map>
#include <string>

namespace moana { namespace TextureLookup {

using TextureIndexKey = std::tuple<std::string, std::string, std::string>;

int indexForMesh(
    const std::string &element,
    const std::string &material,
    const std::string &mesh
);

} }
