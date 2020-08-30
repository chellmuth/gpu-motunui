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
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_tempBuffer),
        iasBufferSizes.tempSizeInBytes
    ));

    CUdeviceptr d_iasOutputBuffer = arena.allocOutput(iasBufferSizes.outputSizeInBytes);

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
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_tempBuffer)));
    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_instances)));

    return handle;
}

} }
