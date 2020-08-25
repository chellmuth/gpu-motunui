#include "moana/driver.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include "assert_macros.hpp"
#include "enumerate.hpp"
#include "hibiscus_geometry.hpp"
#include "kernel.hpp"
#include "moana/core/vec3.hpp"
#include "moana/io/image.hpp"
#include "moana/parsers/obj_parser.hpp"
#include "scene.hpp"

namespace moana {

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

static void contextLogCallback(
    unsigned int level,
    const char *tag,
    const char *message,
    void * /*cbdata */
)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
              << message << std::endl;
}

static void createContext(OptixState &state)
{
    // initialize CUDA
    CHECK_CUDA(cudaFree(0));

    CHECK_OPTIX(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &contextLogCallback;
    options.logCallbackLevel = 4;

    CUcontext cuContext = 0; // current context
    CHECK_OPTIX(optixDeviceContextCreate(cuContext, &options, &state.context));
}

static void createGeometry(OptixState &state, const ObjResult &model)
{
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE; // no build flags
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD; // no updates

    size_t maxTempSizeInBytes = 0;
    size_t maxOutputSizeInBytes = 0;

    std::vector<ObjResult> models = { model };
    for (const auto &model : models) {
        CUdeviceptr d_vertices = 0;
        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void **>(&d_vertices),
            model.vertexCount * 3 * sizeof(float)
        ));
        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_vertices),
            model.vertices.data(),
            model.vertexCount * 3 * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        CUdeviceptr d_indices = 0;
        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void **>(&d_indices),
            model.indexTripletCount * 3 * sizeof(int)
        ));
        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_indices),
            model.indices.data(),
            model.indexTripletCount * 3 * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        // Setup build input
        uint32_t inputFlags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
        OptixBuildInput triangleInput = {};
        triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.numVertices = model.vertexCount;
        triangleInput.triangleArray.vertexBuffers = &d_vertices;

        triangleInput.triangleArray.numIndexTriplets = model.indexTripletCount;
        triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput.triangleArray.indexBuffer = d_indices;

        triangleInput.triangleArray.flags = inputFlags;
        triangleInput.triangleArray.numSbtRecords = 1;

        // Calculate max memory size
        OptixAccelBufferSizes gasBufferSizes;
        CHECK_OPTIX(optixAccelComputeMemoryUsage(
            state.context,
            &accelOptions,
            &triangleInput,
            1, // build input count
            &gasBufferSizes
        ));

        std::cout << "Buffer sizes:"
                  << " temp=" << (gasBufferSizes.tempSizeInBytes / (1024.f * 1024.f))
                  << " output(mb)=" << (gasBufferSizes.outputSizeInBytes / (1024.f * 1024.f))
                  << std::endl;

        maxTempSizeInBytes = std::max(
            gasBufferSizes.tempSizeInBytes,
            maxTempSizeInBytes
        );
        maxOutputSizeInBytes = std::max(
            gasBufferSizes.outputSizeInBytes,
            maxOutputSizeInBytes
        );

        CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_vertices)));
    }

    std::cout << "Final buffer sizes:"
              << " temp(mb)=" << (maxTempSizeInBytes / (1024.f * 1024.f))
              << " output(mb)=" << (maxOutputSizeInBytes / (1024.f * 1024.f))
              << std::endl;

    // Allocate enough for biggest structure
    state.outputBufferSizeInBytes = maxOutputSizeInBytes;
    CUdeviceptr d_tempBufferGas;
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_tempBufferGas),
        maxTempSizeInBytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&state.gasOutputBuffer),
        state.outputBufferSizeInBytes
    ));

    for (const auto &model : models) {
        CUdeviceptr d_vertices = 0;
        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void **>(&d_vertices),
            model.vertexCount * 3 * sizeof(float)
        ));
        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_vertices),
            model.vertices.data(),
            model.vertexCount * 3 * sizeof(float),
            cudaMemcpyHostToDevice
        ));

        CUdeviceptr d_indices = 0;
        CHECK_CUDA(cudaMalloc(
            reinterpret_cast<void **>(&d_indices),
            model.indexTripletCount * 3 * sizeof(int)
        ));
        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_indices),
            model.indices.data(),
            model.indexTripletCount * 3 * sizeof(int),
            cudaMemcpyHostToDevice
        ));

        // Setup build input
        uint32_t inputFlags[] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
        OptixBuildInput triangleInput = {};
        triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.numVertices = model.vertexCount;
        triangleInput.triangleArray.vertexBuffers = &d_vertices;

        triangleInput.triangleArray.numIndexTriplets = model.indexTripletCount;
        triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput.triangleArray.indexBuffer = d_indices;

        triangleInput.triangleArray.flags = inputFlags;
        triangleInput.triangleArray.numSbtRecords = 1;

        OptixTraversableHandle handle;
        CHECK_OPTIX(optixAccelBuild(
            state.context,
            0, // default CUDA stream
            &accelOptions,
            &triangleInput,
            1, // build input count
            d_tempBufferGas,
            maxTempSizeInBytes,
            state.gasOutputBuffer,
            state.outputBufferSizeInBytes,
            &handle,
            nullptr, 0 // emitted property params
        ));
        state.gasHandles.push_back(handle);

        void *gasOutput = malloc(state.outputBufferSizeInBytes);
        CHECK_CUDA(cudaMemcpy(
            gasOutput,
            reinterpret_cast<void *>(state.gasOutputBuffer),
            state.outputBufferSizeInBytes,
            cudaMemcpyDeviceToHost
        ));
        state.gasOutputs.push_back(gasOutput);
        CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_vertices)));
    }

    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_tempBufferGas)));
}

static void createInstances(OptixState &state)
{
    std::vector<float> transform1 = {
        0.515481612437928,
        0.08516036166090636,
        0.85265844277775,
        0.0,
        0.009029136474629525,
        0.9944543262499203,
        -0.10478104646041478,
        0.0,
        -0.8568530690456307,
        0.06171147222843346,
        0.511853995063517,
        0.0,
        183.44022,
        30.927343,
        -174.708914,
        1.0
    };
    std::vector<float> transform2 = {
        -0.8205686666791795,
        -0.010472159890024308,
        0.36953673312891555,
        0.0,
        0.016643853165337303,
        0.8976801022670159,
        0.062397244696305885,
        0.0,
        -0.369310229233045,
        0.06372415445588704,
        -0.8182598528112679,
        0.0,
        5383.439475,
        13.468052,
        641.810391,
        1.0
    };
    std::vector<float> transform3 = {
        0.5832246199874493,
        -0.010472159890024306,
        0.6853753544647874,
        0.0,
        0.055303979870151725,
        0.8976801022670159,
        -0.033345221612726364,
        0.0,
        -0.6832206908832731,
        0.06372415445588704,
        0.5823647651496999,
        0.0,
        1117.658295,
        13.468052,
        -148.996174,
        1.0
    };
    std::vector<float> transform4 = {
        -0.8051486219209945,
        0.08516036166090636,
        0.5869270903787858,
        0.0,
        0.10516935220464743,
        0.9944543262499203,
        -1.896501794359726e-05,
        0.0,
        -0.5836737992882484,
        0.061711472228433456,
        -0.8096397904128854,
        0.0,
        -435.311899,
        30.927343,
        -174.708914,
        1.0
    };
    std::vector<float> transform5 = {
        -0.35487126530357416,
        0.0,
        0.988213734503746,
        0.0,
        0.0,
        1.05,
        0.0,
        0.0,
        -0.988213734503746,
        0.0,
        -0.35487126530357416,
        0.0,
        4023.078545,
        25.852035,
        -331.751723,
        1.0
    };

    constexpr int numInstances = 5;

    CUdeviceptr d_instances;
    size_t instanceSizeInBytes = sizeof(OptixInstance) * numInstances;
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&d_instances),
        instanceSizeInBytes
    ));

    OptixBuildInput instanceInput = {};
    instanceInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceInput.instanceArray.instances = d_instances;
    instanceInput.instanceArray.numInstances = numInstances;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes iasBufferSizes;
    CHECK_OPTIX(optixAccelComputeMemoryUsage(
        state.context,
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

    float identity[12] = {
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f
    };

    OptixInstance optixInstances[numInstances];
    memset(optixInstances, 0, instanceSizeInBytes);

    std::vector<std::vector<float> > transforms = {
        transform1,
        transform2,
        transform3,
        transform4,
        transform5
    };

    for (auto [i, transformColMajor] : enumerate(transforms)) {
        optixInstances[i].traversableHandle = state.gasHandles[0];
        optixInstances[i].flags = OPTIX_INSTANCE_FLAG_NONE;
        optixInstances[i].instanceId = 0;
        optixInstances[i].sbtOffset = 0;
        optixInstances[i].visibilityMask = 255;

        float transform[12] = {
            transformColMajor[0],
            transformColMajor[4],
            transformColMajor[8],
            transformColMajor[12],
            transformColMajor[1],
            transformColMajor[5],
            transformColMajor[9],
            transformColMajor[13],
            transformColMajor[2],
            transformColMajor[6],
            transformColMajor[10],
            transformColMajor[14],
        };
        memcpy(optixInstances[i].transform, transform, sizeof(float) * 12);
    }

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(d_instances),
        &optixInstances,
        instanceSizeInBytes,
        cudaMemcpyHostToDevice
    ));

    CHECK_OPTIX(optixAccelBuild(
        state.context,
        0, // CUDA stream
        &accelOptions,
        &instanceInput,
        1, // num build inputs
        d_tempBuffer,
        iasBufferSizes.tempSizeInBytes,
        d_iasOutputBuffer,
        iasBufferSizes.outputSizeInBytes,
        &state.iasHandle,
        nullptr,
        0
    ));

    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_tempBuffer)));
    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_instances)));
}

static void createModule(OptixState &state)
{
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

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
    state.pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    std::string ptx(ptxSource);

    char log[2048];
    size_t sizeofLog = sizeof(log);

    CHECK_OPTIX(optixModuleCreateFromPTX(
        state.context,
        &moduleCompileOptions,
        &state.pipelineCompileOptions,
        ptx.c_str(),
        ptx.size(),
        log,
        &sizeofLog,
        &state.module
    ));
}

static void createProgramGroups(OptixState &state)
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

    OptixProgramGroupDesc hitgroupProgramGroupDesc = {};
    hitgroupProgramGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroupProgramGroupDesc.hitgroup.moduleCH = state.module;
    hitgroupProgramGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

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

static void linkPipeline(OptixState &state)
{
    const uint32_t maxTraceDepth = 1;
    OptixProgramGroup programGroups[] = {
        state.raygenProgramGroup,
        state.missProgramGroup,
        state.hitgroupProgramGroup
    };

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = maxTraceDepth;
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    char log[2048];
    size_t sizeofLog = sizeof(log);

    CHECK_OPTIX(optixPipelineCreate(
        state.context,
        &state.pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups,
        sizeof(programGroups) / sizeof(programGroups[0]),
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
        3 // maxTraversableDepth
    ));
}

static void createShaderBindingTable(OptixState &state)
{
    CUdeviceptr raygenRecord;
    const size_t raygenRecordSize = sizeof(RayGenSbtRecord);
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&raygenRecord), raygenRecordSize));

    RayGenSbtRecord raygenSbt;
    CHECK_OPTIX(optixSbtRecordPackHeader(state.raygenProgramGroup, &raygenSbt));
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
    CHECK_OPTIX(optixSbtRecordPackHeader(state.missProgramGroup, &missSbt));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(missRecord),
        &missSbt,
        missRecordSize,
        cudaMemcpyHostToDevice
    ));

    CUdeviceptr hitgroupRecord;
    size_t hitgroupRecordSize = sizeof(HitGroupSbtRecord);
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&hitgroupRecord),
        hitgroupRecordSize
    ));

    HitGroupSbtRecord hitgroupSbt;
    CHECK_OPTIX(optixSbtRecordPackHeader(state.hitgroupProgramGroup, &hitgroupSbt));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(hitgroupRecord),
        &hitgroupSbt,
        hitgroupRecordSize,
        cudaMemcpyHostToDevice
    ));

    state.sbt.raygenRecord = raygenRecord;
    state.sbt.missRecordBase = missRecord;
    state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = hitgroupRecord;
    state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    state.sbt.hitgroupRecordCount = 1;
}

void Driver::init()
{
    createContext(m_state);

    HibiscusGeometry hibiscus;
    auto result = hibiscus.buildAcceleration(m_state.context);
    // createGeometry(m_state, model);
    m_state.gasHandles.push_back(result.handle);

    createInstances(m_state);
    createModule(m_state);
    createProgramGroups(m_state);
    linkPipeline(m_state);
    createShaderBindingTable(m_state);

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));
    CHECK_CUDA(cudaDeviceSynchronize());
}

void Driver::launch()
{
    CUstream stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const int width = 952;
    const int height = 400;

    Params params;

    const size_t outputBufferSizeInBytes = width * height * 3 * sizeof(float);
    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&params.outputBuffer),
        outputBufferSizeInBytes
    ));
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.outputBuffer),
        0,
        outputBufferSizeInBytes
    ));

    // Camera camera(
    //     Vec3(60.f, 0.f, 700.f),
    //     Vec3(0.f, 80.f, 0.f),
    //     Vec3(0.f, 1.f, 0.f),
    //     33.f / 180.f * M_PI,
    //     Resolution{ width, height },
    //     false
    // );

    Scene scene(Cam::ShotCam);
    Camera camera = scene.getCamera(width, height);

    params.camera = camera;

    for (auto [i, handle] : enumerate(m_state.gasHandles)) {
        params.handle = handle;

        // CHECK_CUDA(cudaMemcpy(
        //     reinterpret_cast<void *>(m_state.gasOutputBuffer),
        //     m_state.gasOutputs[i],
        //     m_state.outputBufferSizeInBytes,
        //     cudaMemcpyHostToDevice
        // ));

        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_params),
            &params,
            sizeof(params),
            cudaMemcpyHostToDevice
        ));

        CHECK_OPTIX(optixLaunch(
            m_state.pipeline,
            stream,
            d_params,
            sizeof(Params),
            &m_state.sbt,
            width,
            height,
            /*depth=*/1
        ));

        CHECK_CUDA(cudaDeviceSynchronize());
    }

    std::vector<float> outputBuffer(width * height * 3);
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(outputBuffer.data()),
        params.outputBuffer,
        outputBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    outputBuffer[0] = 1.f;
    Image::save(
        width,
        height,
        outputBuffer,
        "out.exr"
    );

    // CHECK_CUDA(cudaFree(reinterpret_cast<void *>(m_state.gasOutputBuffer)));
    CHECK_CUDA(cudaFree(params.outputBuffer));
}

}
