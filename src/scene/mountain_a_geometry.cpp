#include "scene/mountain_a_geometry.hpp"

#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "assert_macros.hpp"
#include "moana/parsers/obj_parser.hpp"
#include "scene/archive.hpp"
#include "scene/gas.hpp"
#include "scene/ias.hpp"
#include "scene/instances_bin.hpp"
#include "scene/types.hpp"

namespace moana {

GeometryResult MountainAGeometry::buildAcceleration(OptixDeviceContext context)
{
    const std::string moanaRoot = MOANA_ROOT;

    const std::string baseObj = moanaRoot + "/island/obj/isMountainA/isMountainA.obj";

    std::vector<OptixInstance> records;
    {
        std::cout << "Processing base obj: " << baseObj << std::endl;

        std::cout << "  Geometry:" << std::endl;
        ObjParser objParser(baseObj);
        auto model = objParser.parse();
        std::cout << "    Vertex count: " << model.vertexCount << std::endl
                  << "    Index triplet count: " << model.indexTripletCount << std::endl;

        std::cout << "  GAS:" << std::endl;
        const GASInfo gasInfo = GAS::gasInfoFromObjResult(context, model);
        std::cout << "    Output Buffer size(mb): "
                  << (gasInfo.outputBufferSizeInBytes / (1024. * 1024.))
                  << std::endl;

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
            gasInfo.handle
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
    archive.processRecords(context, records);

    OptixTraversableHandle iasObjectHandle = IAS::iasFromInstanceRecords(context, records);

    return GeometryResult{
        iasObjectHandle,
        {}
    };
}

}
