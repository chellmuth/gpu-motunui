#include "scene/dunes_a_geometry.hpp"

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

GeometryResult DunesAElement::buildAcceleration(OptixDeviceContext context, ASArena &arena)
{
    const std::string moanaRoot = MOANA_ROOT;

    const std::string baseObj = moanaRoot + "/island/obj/isDunesA/isDunesA.obj";

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
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archivePalmdead0004_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0124_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0126_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0123_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0127_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgPalmDebris_archiveLeaflet0125_geo.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archiveseedpodb_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archiveseedpoda_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archivepineneedle0003_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archivepineneedle0002_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgDebris_archivepineneedle0001_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0009_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0005_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0008_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0004_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0007_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0006_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0003_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgHibiscusFlower_archiveHibiscusFlower0002_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0011_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0002_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0012_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0004_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0005_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0010_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0006_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0008_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0009_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0007_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0003_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0013_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0001_mod.obj",
        moanaRoot + "/island/obj/isDunesA/archives/xgMuskFern_fern0014_mod.obj",
    };

    const std::vector<std::string> binPaths = {
        "../scene/dunesA-xgPalmDebris_archivePalmdead0004_mod.bin",
        "../scene/dunesA-xgPalmDebris_archiveLeaflet0124_geo.bin",
        "../scene/dunesA-xgPalmDebris_archiveLeaflet0126_geo.bin",
        "../scene/dunesA-xgPalmDebris_archiveLeaflet0123_geo.bin",
        "../scene/dunesA-xgPalmDebris_archiveLeaflet0127_geo.bin",
        "../scene/dunesA-xgPalmDebris_archiveLeaflet0125_geo.bin",
        "../scene/dunesA-xgDebris_archiveseedpodb_mod.bin",
        "../scene/dunesA-xgDebris_archiveseedpoda_mod.bin",
        "../scene/dunesA-xgDebris_archivepineneedle0003_mod.bin",
        "../scene/dunesA-xgDebris_archivepineneedle0002_mod.bin",
        "../scene/dunesA-xgDebris_archivepineneedle0001_mod.bin",
        "../scene/dunesA-xgHibiscusFlower_archiveHibiscusFlower0009_mod.bin",
        "../scene/dunesA-xgHibiscusFlower_archiveHibiscusFlower0005_mod.bin",
        "../scene/dunesA-xgHibiscusFlower_archiveHibiscusFlower0008_mod.bin",
        "../scene/dunesA-xgHibiscusFlower_archiveHibiscusFlower0004_mod.bin",
        "../scene/dunesA-xgHibiscusFlower_archiveHibiscusFlower0007_mod.bin",
        "../scene/dunesA-xgHibiscusFlower_archiveHibiscusFlower0006_mod.bin",
        "../scene/dunesA-xgHibiscusFlower_archiveHibiscusFlower0003_mod.bin",
        "../scene/dunesA-xgHibiscusFlower_archiveHibiscusFlower0002_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0011_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0002_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0012_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0004_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0005_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0010_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0006_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0008_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0009_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0007_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0003_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0013_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0001_mod.bin",
        "../scene/dunesA-xgMuskFern_fern0014_mod.bin",
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
