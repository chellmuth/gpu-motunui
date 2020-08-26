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
    const std::vector<std::string> &objPaths
) : m_binPaths(binPaths),
    m_objPaths(objPaths)
{}

void Archive::processRecords(
    OptixDeviceContext context,
    ASArena &arena,
    std::vector<OptixInstance> &records
) const {
    assert(m_binPaths.size() == m_objPaths.size());

    const int archivesSize = m_binPaths.size();
    for (int i = 0; i < archivesSize; i++) {
        const std::string objPath = m_objPaths[i];

        std::cout << "Processing " << objPath << std::endl;

        ObjParser objParser(objPath);
        auto model = objParser.parse();

        const auto gasHandle = GAS::gasInfoFromObjResult(context, arena, model);

        std::cout << "  Instances:" << std::endl;
        const std::string binPath = m_binPaths[i];
        const Instances instancesResult = InstancesBin::parse(binPath);
        std::cout << "    Count: " << instancesResult.count << std::endl;

        IAS::createOptixInstanceRecords(
            context,
            records,
            instancesResult,
            gasHandle
        );
    }
}

}
