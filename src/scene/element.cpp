#include "scene/element.hpp"

#include <iostream>

#include <cuda_runtime.h>

#include "assert_macros.hpp"
#include "enumerate.hpp"
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

    std::cout << "Processing " << m_elementName << std::endl;
    std::vector<OptixInstance> rootRecords;

    std::cout << "  Processing primitive archives" << std::endl;
    std::vector<OptixTraversableHandle> archiveHandles;
    for (auto [i, objArchivePath] : enumerate(m_objArchivePaths)) {
        std::cout << "    Processing: " << objArchivePath << std::endl;

        ObjParser objParser(objArchivePath);
        auto model = objParser.parse();

        const auto gasHandle = GAS::gasInfoFromObjResult(context, arena, model);
        archiveHandles.push_back(gasHandle);
    }

    const int uniqueElementCopyCount = m_baseObjs.size();
    for (int i = 0; i < uniqueElementCopyCount; i++) {
        std::vector<OptixInstance> records;

        const std::string baseObj = m_baseObjs[i];
        std::cout << "  Processing base obj: " << baseObj << std::endl;

        ObjParser objParser(baseObj);
        auto model = objParser.parse();

        const auto gasHandle = GAS::gasInfoFromObjResult(context, arena, model);

        float transform[12] = {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f
        };
        Instances elementGeometryInstances;
        elementGeometryInstances.transforms = transform;
        elementGeometryInstances.count = 1;

        IAS::createOptixInstanceRecords(
            context,
            records,
            elementGeometryInstances,
            gasHandle
        );

        Archive archive(
            m_primitiveInstancesBinPaths[i],
            m_primitiveInstancesHandleIndices[i],
            archiveHandles
        );
        archive.processRecords(context, arena, records);

        auto iasObjectHandle = IAS::iasFromInstanceRecords(context, arena, records);

        std::cout << "  Processing element instances" << std::endl;

        const std::string instancesPath = m_elementInstancesBinPaths[i];
        const Instances instancesResult = InstancesBin::parse(instancesPath);
        std::cout << "    Count: " << instancesResult.count << std::endl;

        IAS::createOptixInstanceRecords(
            context,
            rootRecords,
            instancesResult,
            iasObjectHandle
        );

    }

    auto iasHandle = IAS::iasFromInstanceRecords(context, arena, rootRecords);

    Snapshot snapshot = arena.createSnapshot();
    arena.releaseAll();

    return GeometryResult{
        iasHandle,
        snapshot
    };
}

}
