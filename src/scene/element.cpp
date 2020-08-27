#include "scene/element.hpp"

#include <iostream>

#include <cuda_runtime.h>

#include "assert_macros.hpp"
#include "moana/parsers/obj_parser.hpp"
#include "scene/archive.hpp"
#include "scene/gas.hpp"
#include "scene/ias.hpp"
#include "scene/instances_bin.hpp"

namespace moana {

GeometryResult Element::buildAcceleration(
    OptixDeviceContext context,
    ASArena &arena
) {
    const std::string moanaRoot = MOANA_ROOT;

    std::vector<OptixInstance> records;
    {
        std::cout << "Processing base obj: " << m_baseObj << std::endl;

        ObjParser objParser(m_baseObj);
        auto model = objParser.parse();

        const auto gasHandle = GAS::gasInfoFromObjResult(context, arena, model);

        float transform[12] = {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f
        };
        Instances instancesResult;
        instancesResult.transforms = transform;
        instancesResult.count = 1;

        IAS::createOptixInstanceRecords(
            context,
            records,
            instancesResult,
            gasHandle
        );
    }

    Archive archive(m_binPaths, m_objPaths);
    archive.processRecords(context, arena, records);

    auto iasObjectHandle = IAS::iasFromInstanceRecords(context, arena, records);

    std::vector<OptixInstance> rootRecords;
    std::cout << "Processing: root" << std::endl;

    std::cout << "  Instances:" << std::endl;
    const std::string rootInstances = m_elementInstancesBinPath;
    const Instances instancesResult = InstancesBin::parse(rootInstances);
    std::cout << "    Count: " << instancesResult.count << std::endl;

    IAS::createOptixInstanceRecords(
        context,
        rootRecords,
        instancesResult,
        iasObjectHandle
    );

    auto iasHandle = IAS::iasFromInstanceRecords(context, arena, rootRecords);

    Snapshot snapshot = arena.createSnapshot();
    arena.releaseAll();

    return GeometryResult{
        iasHandle,
        snapshot
    };
}

}
