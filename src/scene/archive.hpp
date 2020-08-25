#pragma once

#include <string>
#include <vector>

#include <optix.h>

namespace moana {

class Archive {
public:
    Archive(
        const std::vector<std::string> &binPaths,
        const std::vector<std::string> &objPaths
    );

    void processRecords(
        OptixDeviceContext context,
        std::vector<OptixInstance> &records
    ) const;

private:
    std::vector<std::string> m_binPaths;
    std::vector<std::string> m_objPaths;
};

}
