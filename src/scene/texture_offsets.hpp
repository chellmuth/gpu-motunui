#pragma once

#include <string>
#include <vector>

namespace moana {

struct TextureOffset {
    int startIndex; // primitive index; halve for face index
    int endIndex;
    int textureIndex;
    std::string debugName;
};

}

namespace moana { namespace Textures {

extern std::vector<std::string> textureFilenames;
extern std::vector<std::vector<TextureOffset> > offsets;

} }
