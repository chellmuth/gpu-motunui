#include "pipeline_helper.hpp"

#include <optix_stack_size.h>
#include <optix_stubs.h>

#include "assert_macros.hpp"
#include "scene/materials.hpp"

namespace moana { namespace PipelineHelper {

struct OptixInternalState {
    OptixDeviceContext context = 0;
    std::vector<OptixTraversableHandle> gasHandles = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixModuleCompileOptions moduleCompileOptions = {};
    OptixModule module = 0;
    OptixProgramGroup raygenProgramGroup;
    OptixProgramGroup missProgramGroup;
    OptixProgramGroup hitgroupProgramGroup;

    OptixPipeline pipeline = 0;
    OptixShaderBindingTable sbt = {};
};

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

static void createModule(
    OptixInternalState &state,
    const std::string &ptxSource
) {
    state.moduleCompileOptions = {};
    state.moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    state.moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    state.moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    state.pipelineCompileOptions.usesMotionBlur = false;
    state.pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    state.pipelineCompileOptions.numPayloadValues = 3;
    state.pipelineCompileOptions.numAttributeValues = 3;
#ifdef DEBUG
    state.pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    state.pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
#endif
    state.pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
    state.pipelineCompileOptions.usesPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;

    std::string ptx(ptxSource);

    char log[2048];
    size_t sizeofLog = sizeof(log);

    CHECK_OPTIX(optixModuleCreateFromPTX(
        state.context,
        &state.moduleCompileOptions,
        &state.pipelineCompileOptions,
        ptx.c_str(),
        ptx.size(),
        log,
        &sizeofLog,
        &state.module
    ));
}

static void createProgramGroups(OptixInternalState &state)
{
    OptixProgramGroupOptions programGroupOptions = {};

    OptixProgramGroupDesc raygenProgramGroupDesc = {};
    raygenProgramGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygenProgramGroupDesc.raygen.module = state.module;
    raygenProgramGroupDesc.raygen.entryFunctionName = "__raygen__rg";

    char log[2048];
    size_t sizeofLog = sizeof(log);

    CHECK_OPTIX(optixProgramGroupCreate(
        state.context,
        &raygenProgramGroupDesc,
        1, // program group count
        &programGroupOptions,
        log,
        &sizeofLog,
        &state.raygenProgramGroup
    ));

    OptixProgramGroupDesc missProgramGroupDesc = {};
    missProgramGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missProgramGroupDesc.miss.module = state.module;
    missProgramGroupDesc.miss.entryFunctionName = "__miss__ms";

    CHECK_OPTIX(optixProgramGroupCreate(
        state.context,
        &missProgramGroupDesc,
        1, // program group count
        &programGroupOptions,
        log,
        &sizeofLog,
        &state.missProgramGroup
    ));

    OptixBuiltinISOptions builtinISOptions = {};
    OptixModule geometryModule = nullptr;

    builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
    CHECK_OPTIX(optixBuiltinISModuleGet(
        state.context,
        &state.moduleCompileOptions,
        &state.pipelineCompileOptions,
        &builtinISOptions,
        &geometryModule
    ));

    OptixProgramGroupDesc hitgroupProgramGroupDesc = {};
    hitgroupProgramGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupProgramGroupDesc.hitgroup.moduleCH = state.module;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    hitgroupProgramGroupDesc.hitgroup.moduleIS = geometryModule;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameIS = 0;

    CHECK_OPTIX(optixProgramGroupCreate(
        state.context,
        &hitgroupProgramGroupDesc,
        1, // program group count
        &programGroupOptions,
        log,
        &sizeofLog,
        &state.hitgroupProgramGroup
    ));
}

static void createShaderBindingTable(
    OptixInternalState &optixState,
    const SceneState &sceneState
) {
    CUdeviceptr raygenRecord;
    const size_t raygenRecordSize = sizeof(RayGenSbtRecord);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&raygenRecord), raygenRecordSize));

    RayGenSbtRecord raygenSbt;
    CHECK_OPTIX(optixSbtRecordPackHeader(optixState.raygenProgramGroup, &raygenSbt));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(raygenRecord),
        &raygenSbt,
        raygenRecordSize,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr missRecord;
    size_t missRecordSize = sizeof(MissSbtRecord);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&missRecord), missRecordSize));

    MissSbtRecord missSbt;
    CHECK_OPTIX(optixSbtRecordPackHeader(optixState.missProgramGroup, &missSbt));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(missRecord),
        &missSbt,
        missRecordSize,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr d_hitgroupRecords;

    std::vector<HitGroupSbtRecord> hitgroupRecords;
    for (const auto &geometryResult : sceneState.geometries) {
        for (const auto &record : geometryResult.hostSBTRecords) {
            HitGroupSbtRecord hitgroupSbt;
            CHECK_OPTIX(optixSbtRecordPackHeader(optixState.hitgroupProgramGroup, &hitgroupSbt));
            hitgroupSbt.data.baseColor = Materials::baseColors[record.materialID];
            hitgroupSbt.data.textureIndex = record.textureIndex;
            hitgroupSbt.data.materialID = record.materialID;
            hitgroupSbt.data.normals = reinterpret_cast<float *>(record.d_normals);
            hitgroupSbt.data.normalIndices = reinterpret_cast<int *>(record.d_normalIndices);

            hitgroupRecords.push_back(hitgroupSbt);
        }
    }

    size_t hitgroupRecordSize = sizeof(HitGroupSbtRecord) * hitgroupRecords.size();

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_hitgroupRecords),
        hitgroupRecordSize
    ));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_hitgroupRecords),
        hitgroupRecords.data(),
        hitgroupRecordSize,
        cudaMemcpyHostToDevice
    ));

    optixState.sbt.raygenRecord = raygenRecord;
    optixState.sbt.missRecordBase = missRecord;
    optixState.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    optixState.sbt.missRecordCount = 1;
    optixState.sbt.hitgroupRecordBase = d_hitgroupRecords;
    optixState.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    optixState.sbt.hitgroupRecordCount = hitgroupRecords.size();
}

void linkPipeline(
    OptixInternalState &state,
    const std::vector<OptixProgramGroup> programGroups
) {
    const uint32_t maxTraceDepth = 1;

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = maxTraceDepth;
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char log[2048];
    size_t sizeofLog = sizeof(log);

    CHECK_OPTIX(optixPipelineCreate(
        state.context,
        &state.pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups.data(),
        programGroups.size(),
        log,
        &sizeofLog,
        &state.pipeline
    ));

    OptixStackSizes stackSizes = {};
    for(const auto &progGroup : programGroups) {
        CHECK_OPTIX(optixUtilAccumulateStackSizes(progGroup, &stackSizes));
    }

    uint32_t directCallableStackSizeFromTraversal;
    uint32_t directCallableStackSizeFromState;
    uint32_t continuationStackSize;
    CHECK_OPTIX(optixUtilComputeStackSizes(
        &stackSizes,
        maxTraceDepth,
        0, // maxCCDepth
        0, // maxDCDEpth
        &directCallableStackSizeFromTraversal,
        &directCallableStackSizeFromState,
        &continuationStackSize
    ));
    CHECK_OPTIX(optixPipelineSetStackSize(
        state.pipeline,
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState,
        continuationStackSize,
        // 1 = obj for small details
        // 2 = instanced details yields element
        // 3 = instanced element yields scene object
        3 // maxTraversableDepth
    ));
}

void initOptixState(
    OptixState &optixState,
    OptixDeviceContext context,
    const SceneState &sceneState,
    const std::string &ptxSource
) {
    OptixInternalState internalState;
    internalState.context = context;

    createModule(internalState, ptxSource);
    createProgramGroups(internalState);
    std::vector<OptixProgramGroup> programGroups = {
        internalState.raygenProgramGroup,
        internalState.missProgramGroup,
        internalState.hitgroupProgramGroup
    };
    linkPipeline(internalState, programGroups);
    createShaderBindingTable(internalState, sceneState);

    optixState.pipeline = internalState.pipeline;
    optixState.sbt = internalState.sbt;
}

} }
