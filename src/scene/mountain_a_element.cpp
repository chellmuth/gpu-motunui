#include "scene/mountain_a_element.hpp"

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

GeometryResult MountainAElement::buildAcceleration(
    OptixDeviceContext context,
    ASArena &arena
) {
    const std::string moanaRoot = MOANA_ROOT;

    const std::string baseObj = moanaRoot + "/island/obj/isMountainA/isMountainA.obj";

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
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig8.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig4.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig1.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig12.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig3.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig7.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig15.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig16.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig14.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig17.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig6.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig5.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig13.obj",
        moanaRoot + "/island/obj/isMountainA/archives/xgCocoPalms_isPalmRig2.obj",

        moanaRoot + "/island/obj/isMountainA/archives/xgBreadFruit_archiveBreadFruitBaked.obj"
    };


    const std::vector<std::string> binPaths = {
        "../scene/mountainA-xgCocoPalms_isPalmRig8.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig4.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig1.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig12.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig3.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig7.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig15.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig16.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig14.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig17.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig6.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig5.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig13.bin",
        "../scene/mountainA-xgCocoPalms_isPalmRig2.bin",

        "../scene/mountainA-xgBreadFruit_archiveBreadFruitBaked.bin",
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
