#include "hibiscus_geometry.hpp"

#include <cstring>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "assert_macros.hpp"
#include "moana/parsers/obj_parser.hpp"
#include "scene/archive.hpp"
#include "scene/gas.hpp"
#include "scene/instances_bin.hpp"
#include "scene/types.hpp"

namespace moana {

static void createOptixInstanceRecords(
    OptixDeviceContext context,
    std::vector<OptixInstance> &records,
    const Instances &instances,
    const OptixTraversableHandle &traversableHandle
) {
    int offset = records.size();
    for (int i = 0; i < instances.count; i++) {
        OptixInstance instance;
        memcpy(
            instance.transform,
            &instances.transforms[i * 12],
            sizeof(float) * 12
        );

        instance.instanceId = offset + i;
        instance.visibilityMask = 255;
        instance.sbtOffset = 0;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = traversableHandle;

        records.push_back(instance);
    }
}

static OptixTraversableHandle iasFromInstanceRecords(
    OptixDeviceContext context,
    const std::vector<OptixInstance> &records
) {
    CUdeviceptr d_instances;
    const size_t instancesSizeInBytes = sizeof(OptixInstance) * records.size();
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_instances),
        instancesSizeInBytes
    ));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_instances),
        records.data(),
        instancesSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    OptixBuildInput instanceInput = {};
    instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceInput.instanceArray.instances = d_instances;
    instanceInput.instanceArray.numInstances = records.size();

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes iasBufferSizes;
    CHECK_OPTIX(optixAccelComputeMemoryUsage(
        context,
        &accelOptions,
        &instanceInput,
        1, // num build inputs
        &iasBufferSizes
    ));

    CUdeviceptr d_tempBuffer;
    CUdeviceptr d_iasOutputBuffer; // fixme (free)
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_tempBuffer),
        iasBufferSizes.tempSizeInBytes
    ));
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_iasOutputBuffer),
        iasBufferSizes.outputSizeInBytes
    ));

    OptixTraversableHandle handle;
    CHECK_OPTIX(optixAccelBuild(
        context,
        0, // CUDA stream
        &accelOptions,
        &instanceInput,
        1, // num build inputs
        d_tempBuffer,
        iasBufferSizes.tempSizeInBytes,
        d_iasOutputBuffer,
        iasBufferSizes.outputSizeInBytes,
        &handle,
        nullptr,
        0
    ));

    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_tempBuffer)));
    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_instances)));

    return handle;
}

GeometryResult HibiscusGeometry::buildAcceleration(OptixDeviceContext context)
{
    const std::string moanaRoot = MOANA_ROOT;

    const std::string baseObj = moanaRoot + "/island/obj/isHibiscus/isHibiscus.obj";

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

        createOptixInstanceRecords(
            context,
            records,
            instancesResult,
            gasInfo.handle
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
    archive.processRecords(context, records);
    OptixTraversableHandle iasObjectHandle = iasFromInstanceRecords(context, records);

    std::vector<OptixInstance> rootRecords;
    {
        std::cout << "Processing: root" << std::endl;

        std::cout << "  Instances:" << std::endl;
        const std::string rootInstances = "../scene/hibiscus-root.bin";
        const Instances instancesResult = InstancesBin::parse(rootInstances);
        std::cout << "    Count: " << instancesResult.count << std::endl;

        createOptixInstanceRecords(
            context,
            rootRecords,
            instancesResult,
            iasObjectHandle
        );

    }
    OptixTraversableHandle iasHandle = iasFromInstanceRecords(context, rootRecords);

    return GeometryResult{
        iasHandle,
        {}
    };
}

}
