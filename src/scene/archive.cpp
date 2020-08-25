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
#include "scene/types.hpp"

namespace moana {

Archive::Archive(
    const std::vector<std::string> &binPaths,
    const std::vector<std::string> &objPaths
) : m_binPaths(binPaths),
    m_objPaths(objPaths)
{}

void Archive::processRecords(
    OptixDeviceContext context,
    std::vector<OptixInstance> &records
) const {
    assert(m_binPaths.size() == m_objPaths.size());

    const int archivesSize = m_binPaths.size();
    for (int i = 0; i < archivesSize; i++) {
        const std::string objPath = m_objPaths[i];

        std::cout << "Processing " << objPath << std::endl;

        std::cout << "  Geometry:" << std::endl;
        ObjParser objParser(objPath);
        auto model = objParser.parse();
        std::cout << "    Vertex count: " << model.vertexCount << std::endl
                  << "    Index triplet count: " << model.indexTripletCount << std::endl;

        std::cout << "  GAS:" << std::endl;
        const GASInfo gasInfo = GAS::gasInfoFromObjResult(context, model);
        std::cout << "    Output Buffer size(mb): "
                  << (gasInfo.outputBufferSizeInBytes / (1024. * 1024.))
                  << std::endl;

        std::cout << "  Instances:" << std::endl;
        const std::string binPath = m_binPaths[i];
        const Instances instancesResult = InstancesBin::parse(binPath);
        std::cout << "    Count: " << instancesResult.count << std::endl;

        IAS::createOptixInstanceRecords(
            context,
            records,
            instancesResult,
            gasInfo.handle
        );
    }
}

}
