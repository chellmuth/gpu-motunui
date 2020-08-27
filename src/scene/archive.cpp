#include "scene/archive.hpp"

#include <assert.h>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>

#include "assert_macros.hpp"
#include "moana/parsers/obj_parser.hpp"
#include "scene/gas.hpp"
#include "scene/ias.hpp"
#include "scene/instances_bin.hpp"

namespace moana {

Archive::Archive(
    const std::vector<std::string> &binPaths,
    const std::vector<int> &handleIndices,
    const std::vector<OptixTraversableHandle> &handles
) : m_binPaths(binPaths),
    m_handleIndices(handleIndices),
    m_handles(handles)
{}

void Archive::processRecords(
    OptixDeviceContext context,
    ASArena &arena,
    std::vector<OptixInstance> &records
) const {
    assert(m_binPaths.size() == m_handleIndices.size());

    const int archivesSize = m_binPaths.size();
    for (int i = 0; i < archivesSize; i++) {
        const std::string binPath = m_binPaths[i];

        std::cout << "Processing " << binPath << std::endl;

        std::cout << "  Instances:" << std::endl;
        const Instances instancesResult = InstancesBin::parse(binPath);
        std::cout << "    Count: " << instancesResult.count << std::endl;

        const int handleIndex = m_handleIndices[i];
        IAS::createOptixInstanceRecords(
            context,
            records,
            instancesResult,
            m_handles[handleIndex]
        );
    }
}

}
