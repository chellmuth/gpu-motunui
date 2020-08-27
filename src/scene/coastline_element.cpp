#include "scene/coastline_element.hpp"

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

GeometryResult CoastlineElement::buildAcceleration(
    OptixDeviceContext context,
    ASArena &arena
) {
    const std::string moanaRoot = MOANA_ROOT;

    const std::string baseObj = moanaRoot + "/island/obj/isCoastline/isCoastline.obj";

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
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0125_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0127_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0124_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0123_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgPalmDebris_archiveLeaflet0126_geo.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgFibers_archiveseedpodb_mod.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgFibers_archivepineneedle0002_mod.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgFibers_archivepineneedle0003_mod.obj",
        moanaRoot + "/island/obj/isCoastline/archives/xgFibers_archivepineneedle0001_mod.obj",
    };

    const std::vector<std::string> binPaths = {
        "../scene/coastline-xgPalmDebris_archiveLeaflet0125_geo.bin",
        "../scene/coastline-xgPalmDebris_archiveLeaflet0127_geo.bin",
        "../scene/coastline-xgPalmDebris_archiveLeaflet0124_geo.bin",
        "../scene/coastline-xgPalmDebris_archiveLeaflet0123_geo.bin",
        "../scene/coastline-xgPalmDebris_archiveLeaflet0126_geo.bin",
        "../scene/coastline-xgFibers_archiveseedpodb_mod.bin",
        "../scene/coastline-xgFibers_archivepineneedle0002_mod.bin",
        "../scene/coastline-xgFibers_archivepineneedle0003_mod.bin",
        "../scene/coastline-xgFibers_archivepineneedle0001_mod.bin",
    };

    Archive archive(binPaths, objPaths);
    archive.processRecords(context, arena, records);

    auto iasObjectHandle = IAS::iasFromInstanceRecords(context, arena, records);

    Snapshot snapshot = arena.createSnapshot();
    arena.releaseAll();
    return GeometryResult{
        iasObjectHandle,
        snapshot
    };
}

}
