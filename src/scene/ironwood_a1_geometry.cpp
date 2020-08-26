#include "scene/ironwood_a1_geometry.hpp"

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

GeometryResult IronwoodA1Geometry::buildAcceleration(OptixDeviceContext context)
{
    const std::string moanaRoot = MOANA_ROOT;

    const std::string baseObj = moanaRoot + "/island/obj/isIronwoodA1/isIronwoodA1.obj";

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
        moanaRoot + "/island/obj/isIronwoodA1/archives/archiveseedpodb_mod.obj",
    };

    const std::vector<std::string> binPaths = {
        "../scene/ironwoodA1-archiveseedpodb_mod.bin",
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
