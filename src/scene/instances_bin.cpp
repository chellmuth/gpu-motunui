#include "scene/instances_bin.hpp"

#include <fstream>

namespace moana { namespace InstancesBin {

Instances parse(const std::string &filepath)
{
    constexpr int transformSize = 12;
    std::ifstream instanceFile(filepath);

    Instances result;
    instanceFile.read((char *)&result.count, sizeof(int));

    int offset = 0;
    result.transforms = new float[transformSize * result.count];
    while (instanceFile.peek() != EOF) {
        instanceFile.read((char *)&result.transforms[offset], sizeof(float) * transformSize);
        offset += transformSize;
    }

    return result;
}

} }
