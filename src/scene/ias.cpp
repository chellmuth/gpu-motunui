#include "scene/ias.hpp"

#include <cstring>

#include <cuda_runtime.h>
#include <optix_stubs.h>

#include "assert_macros.hpp"

namespace moana { namespace IAS {

void createOptixInstanceRecords(
    OptixDeviceContext context,
    std::vector<OptixInstance> &records,
    const Instances &instances,
    const OptixTraversableHandle &traversableHandle,
    int sbtOffset
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
        instance.sbtOffset = sbtOffset;
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.traversableHandle = traversableHandle;

        records.push_back(instance);
    }
}

OptixTraversableHandle iasFromInstanceRecords(
    OptixDeviceContext context,
    ASArena &arena,
    const std::vector<OptixInstance> &records,
    bool shouldCompact
) {
    CUdeviceptr d_instances;
    const size_t instancesSizeInBytes = sizeof(OptixInstance) * records.size();
    std::cout << "IAS:" << std::endl
              << "  Records: " << records.size() << std::endl
              << "  Records size(mb): " << (instancesSizeInBytes / (1024. * 1024.)) << std::endl;

    d_instances = arena.pushTemp(instancesSizeInBytes);
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
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes iasBufferSizes;
    CHECK_OPTIX(optixAccelComputeMemoryUsage(
        context,
        &accelOptions,
        &instanceInput,
        1, // num build inputs
        &iasBufferSizes
    ));

    std::cout << "  Output Buffer size(mb): "
              << (iasBufferSizes.outputSizeInBytes / (1024. * 1024.)) << std::endl
              << "  Temp Buffer size(mb): "
              << (iasBufferSizes.tempSizeInBytes / (1024. * 1024.)) << std::endl;

    CUdeviceptr d_tempBuffer = arena.pushTemp(iasBufferSizes.tempSizeInBytes);
    CUdeviceptr d_iasOutputBuffer = arena.allocOutput(iasBufferSizes.outputSizeInBytes);

    CUdeviceptr d_compactedSize = arena.pushTemp(sizeof(size_t));
    OptixAccelEmitDesc property;
    property.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    property.result = d_compactedSize;

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
        &property,
        1
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    arena.popTemp(); // tempBuffer
    arena.popTemp(); // OptixInstance records

    size_t compactedSize;
    CHECK_CUDA(cudaMemcpy(
        &compactedSize,
        reinterpret_cast<void *>(d_compactedSize),
        sizeof(size_t),
        cudaMemcpyDeviceToHost
    ));

    arena.popTemp(); // compactedSize

    if (!shouldCompact) { return handle; }

    std::cout << "  Compacted Buffer size(mb): " << compactedSize / (1024. * 1024.) << std::endl;

    CUdeviceptr d_compactedOutputBuffer = arena.pushTemp(compactedSize);
    OptixTraversableHandle compactedHandle;
    CHECK_OPTIX(optixAccelCompact(
        context,
        0,
        handle,
        d_compactedOutputBuffer,
        compactedSize,
        &compactedHandle
     ));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_iasOutputBuffer),
        reinterpret_cast<void *>(d_compactedOutputBuffer),
        compactedSize,
        cudaMemcpyDeviceToDevice
    ));

    OptixAccelRelocationInfo relocationInfo;
    CHECK_OPTIX(optixAccelGetRelocationInfo(context, compactedHandle, &relocationInfo));

    OptixTraversableHandle relocatedHandle;
    CHECK_OPTIX(optixAccelRelocate(
        context,
        0,
        &relocationInfo,
        0,
        0,
        d_iasOutputBuffer,
        compactedSize,
        &relocatedHandle
     ));

    arena.returnCompactedOutput(iasBufferSizes.outputSizeInBytes - compactedSize);
    arena.popTemp(); // compactedOutputBuffer

    return relocatedHandle;
}

} }
