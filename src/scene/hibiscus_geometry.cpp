#include "scene/hibiscus_geometry.hpp"

#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "assert_macros.hpp"
#include "moana/parsers/obj_parser.hpp"
#include "scene/archive.hpp"
#include "scene/gas.hpp"
#include "scene/ias.hpp"
#include "scene/instances_bin.hpp"

namespace moana {

GeometryResult HibiscusElement::buildAcceleration(OptixDeviceContext context, ASArena &arena)
{
    const std::string moanaRoot = MOANA_ROOT;

    const std::string baseObj = moanaRoot + "/island/obj/isHibiscus/isHibiscus.obj";

    std::vector<OptixInstance> records;
    {
        std::cout << "Processing base obj: " << baseObj << std::endl;

        ObjParser objParser(baseObj);
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

    const std::vector<std::string> objPaths = {
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0001_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusFlower0001_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0003_mod.obj",
        moanaRoot + "/island/obj/isHibiscus/archives/archiveHibiscusLeaf0002_mod.obj"
    };

    const std::vector<std::string> binPaths = {
        "../scene/hibiscus-archiveHibiscusLeaf0001_mod.bin",
        "../scene/hibiscus-archiveHibiscusFlower0001_mod.bin",
        "../scene/hibiscus-archiveHibiscusLeaf0002_mod.bin",
        "../scene/hibiscus-archiveHibiscusLeaf0003_mod.bin",
    };

    Archive archive(binPaths, objPaths);
    archive.processRecords(context, arena, records);

    auto iasObjectHandle = IAS::iasFromInstanceRecords(context, arena, records);

    std::vector<OptixInstance> rootRecords;
    {
        std::cout << "Processing: root" << std::endl;

        std::cout << "  Instances:" << std::endl;
        const std::string rootInstances = "../scene/hibiscus-root.bin";
        const Instances instancesResult = InstancesBin::parse(rootInstances);
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
