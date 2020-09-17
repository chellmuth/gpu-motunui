#include "moana/driver.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>

#include <cuda_runtime.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include "assert_macros.hpp"
#include "core/ptex_texture.hpp"
#include "kernel.hpp"
#include "moana/core/vec3.hpp"
#include "moana/io/image.hpp"
#include "scene/container.hpp"
#include "scene/materials.hpp"
#include "scene/texture_offsets.hpp"
#include "util/color_map.hpp"
#include "util/enumerate.hpp"

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

static void createModule(OptixState &state)
{
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
        // 1 = obj for small details
        // 2 = instanced details yields element
        // 3 = instanced element yields scene object
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

    CUdeviceptr d_hitgroupRecords;

    std::vector<HitGroupSbtRecord> hitgroupRecords;
    for (const auto &geometryResult : state.geometries) {
        for (const auto &record : geometryResult.hostSBTRecords) {
            HitGroupSbtRecord hitgroupSbt;
            CHECK_OPTIX(optixSbtRecordPackHeader(state.hitgroupProgramGroup, &hitgroupSbt));
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

    state.sbt.raygenRecord = raygenRecord;
    state.sbt.missRecordBase = missRecord;
    state.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    state.sbt.missRecordCount = 1;
    state.sbt.hitgroupRecordBase = d_hitgroupRecords;
    state.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    state.sbt.hitgroupRecordCount = hitgroupRecords.size();
}

void Driver::init()
{
    createContext(m_state);

    EnvironmentLight environmentLight;
    environmentLight.queryMemoryRequirements();

    size_t gb = 1024 * 1024 * 1024;
    m_state.arena.init(6.8 * gb);

    m_state.environmentState = environmentLight.snapshotTextureObject(m_state.arena);

    m_state.geometries = Container::createGeometryResults(m_state.context, m_state.arena);

    createModule(m_state);
    createProgramGroups(m_state);
    linkPipeline(m_state);
    createShaderBindingTable(m_state);

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(Params)));
    CHECK_CUDA(cudaDeviceSynchronize());
}

struct HostBuffers {
    std::vector<float> betaBuffer;
};

struct OutputBuffers {
    std::vector<float> outputBuffer;
    std::vector<float> cosThetaWiBuffer;
    std::vector<float> barycentricBuffer;
    std::vector<int> idBuffer;
    std::vector<float> colorBuffer;
};

// struct Images {
//     std::vector<float> outputImage;
//     std::vector<float> textureImage;
//     std::vector<float> occlusionImage;
//     std::vector<float> normalImage;
//     std::vector<float> faceImage;
//     std::vector<float> uvImage;
// };

struct BufferManager {
    size_t outputBufferSizeInBytes;
    size_t depthBufferSizeInBytes;
    size_t xiBufferSizeInBytes;
    size_t cosThetaWiBufferSizeInBytes;
    size_t sampleRecordBufferSizeInBytes;
    size_t occlusionBufferSizeInBytes;
    size_t missDirectionBufferSizeInBytes;
    size_t barycentricBufferSizeInBytes;
    size_t idBufferSizeInBytes;
    size_t colorBufferSizeInBytes;

    HostBuffers host;
    OutputBuffers output;
};

static void copyOutputBuffers(
    BufferManager &buffers,
    int width,
    int height,
    Params &params
) {
    buffers.output.outputBuffer.resize(width * height * 3, 0.f);
    buffers.output.cosThetaWiBuffer.resize(width * height * 1, 0.f);
    buffers.output.barycentricBuffer.resize(width * height * 2, 0.f);
    buffers.output.idBuffer.resize(width * height * 3, 0);
    buffers.output.colorBuffer.resize(width * height * 3, 0.f);

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(buffers.output.outputBuffer.data()),
        params.outputBuffer,
        buffers.outputBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(buffers.output.cosThetaWiBuffer.data()),
        params.cosThetaWiBuffer,
        buffers.cosThetaWiBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(buffers.output.barycentricBuffer.data()),
        params.barycentricBuffer,
        buffers.barycentricBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(buffers.output.idBuffer.data()),
        params.idBuffer,
        buffers.idBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(buffers.output.colorBuffer.data()),
        params.colorBuffer,
        buffers.colorBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));
}

static void resetBuffers(
    BufferManager &buffers,
    int width,
    int height,
    Params &params
) {
    buffers.host.betaBuffer.resize(width * height * 3);
    std::fill(
        buffers.host.betaBuffer.begin(),
        buffers.host.betaBuffer.end(),
        1.f
    );

    std::vector<float> depthBuffer(width * height, std::numeric_limits<float>::max());
    std::vector<float> xiBuffer(width * height * 2, -1.f);

    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.outputBuffer),
        0,
        buffers.outputBufferSizeInBytes
    ));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(params.depthBuffer),
        depthBuffer.data(),
        buffers.depthBufferSizeInBytes,
        cudaMemcpyHostToDevice
    ));
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(params.xiBuffer),
        xiBuffer.data(),
        buffers.xiBufferSizeInBytes,
        cudaMemcpyHostToDevice
    ));
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.cosThetaWiBuffer),
        0,
        buffers.cosThetaWiBufferSizeInBytes
    ));
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.sampleRecordBuffer),
        0,
        buffers.sampleRecordBufferSizeInBytes
    ));
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.occlusionBuffer),
        0,
        buffers.occlusionBufferSizeInBytes
    ));
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.missDirectionBuffer),
        0,
        buffers.missDirectionBufferSizeInBytes
    ));
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.barycentricBuffer),
        0,
        buffers.barycentricBufferSizeInBytes
    ));
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.idBuffer),
        0,
        buffers.idBufferSizeInBytes
    ));
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.colorBuffer),
        0,
        buffers.colorBufferSizeInBytes
    ));
}

static void mallocBuffers(
    BufferManager &buffers,
    int width,
    int height,
    Params &params
) {
    buffers.outputBufferSizeInBytes = width * height * 3 * sizeof(float);
    buffers.depthBufferSizeInBytes = width * height * sizeof(float);
    buffers.xiBufferSizeInBytes = width * height * 2 * sizeof(float);
    buffers.cosThetaWiBufferSizeInBytes = width * height * 1 * sizeof(float);
    buffers.sampleRecordBufferSizeInBytes = width * height * sizeof(BSDFSampleRecord);
    buffers.occlusionBufferSizeInBytes = width * height * 1 * sizeof(float);
    buffers.missDirectionBufferSizeInBytes = width * height * 3 * sizeof(float);
    buffers.barycentricBufferSizeInBytes = width * height * 2 * sizeof(float);
    buffers.idBufferSizeInBytes = width * height * sizeof(int) * 3;
    buffers.colorBufferSizeInBytes = width * height * sizeof(float) * 3;

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&params.outputBuffer),
        buffers.outputBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&params.depthBuffer),
        buffers.depthBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&params.xiBuffer),
        buffers.xiBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&params.cosThetaWiBuffer),
        buffers.cosThetaWiBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&params.sampleRecordBuffer),
        buffers.sampleRecordBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&params.occlusionBuffer),
        buffers.occlusionBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&params.missDirectionBuffer),
        buffers.missDirectionBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&params.barycentricBuffer),
        buffers.barycentricBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&params.idBuffer),
        buffers.idBufferSizeInBytes
    ));

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&params.colorBuffer),
        buffers.colorBufferSizeInBytes
    ));

}

static void updateBetaWithTextureAlbedos(
    BufferManager &buffers,
    std::vector<PtexTexture> &textures,
    int width,
    int height,
    int spp,
    std::vector<float> &textureImage
) {
   ColorMap faceMap;
   ColorMap materialMap;
   std::vector<float> faceImage(width * height * 3, 0.f);
   std::vector<float> uvImage(width * height * 3, 0.f);
   for (int row = 0; row < height; row++) {
       for (int col = 0; col < width; col++) {
           const int pixelIndex = 3 * (row * width + col);

           const int idIndex = 3 * (row * width + col);
           const int primitiveID = buffers.output.idBuffer[idIndex + 0];
           const int materialID = buffers.output.idBuffer[idIndex + 1];
           const int textureIndex = buffers.output.idBuffer[idIndex + 2];
           const int faceID = primitiveID / 2;

           const int barycentricIndex = 2 * (row * width + col);
           const float alpha = buffers.output.barycentricBuffer[barycentricIndex + 0];
           const float beta = buffers.output.barycentricBuffer[barycentricIndex + 1];

           float u, v;
           if (primitiveID % 2 == 0) {
               u = alpha + beta;
               v = beta;
           } else {
               u = alpha;
               v = alpha + beta;
           }
           uvImage[pixelIndex + 0] = u;
           uvImage[pixelIndex + 1] = v;
           uvImage[pixelIndex + 2] = materialID;

           if (materialID > 0) {
               float3 color = faceMap.get(faceID);
               faceImage[pixelIndex + 0] = color.x;
               faceImage[pixelIndex + 1] = color.y;
               faceImage[pixelIndex + 2] = color.z;
           }

           float textureX = 0;
           float textureY = 0;
           float textureZ = 0;
           if (textureIndex >= 0) {
               PtexTexture texture = textures[textureIndex];

               Vec3 color = texture.lookup(
                   float2{ u, v },
                   faceID
               );
               textureX = color.x();
               textureY = color.y();
               textureZ = color.z();
           } else if (materialID > 0) {
               float3 color = materialMap.get(materialID);

               textureX = buffers.output.colorBuffer[pixelIndex + 0];
               textureY = buffers.output.colorBuffer[pixelIndex + 1];
               textureZ = buffers.output.colorBuffer[pixelIndex + 2];
           }

           const int cosThetaWiIndex = row * width + col;
           const float cosThetaWi = buffers.output.cosThetaWiBuffer[cosThetaWiIndex];
           buffers.host.betaBuffer[pixelIndex + 0] *= cosThetaWi * textureX;
           buffers.host.betaBuffer[pixelIndex + 1] *= cosThetaWi * textureY;
           buffers.host.betaBuffer[pixelIndex + 2] *= cosThetaWi * textureZ;

           textureImage[pixelIndex + 0] += (1.f / spp) * textureX * buffers.output.outputBuffer[pixelIndex + 0];
           textureImage[pixelIndex + 1] += (1.f / spp) * textureY * buffers.output.outputBuffer[pixelIndex + 1];
           textureImage[pixelIndex + 2] += (1.f / spp) * textureZ * buffers.output.outputBuffer[pixelIndex + 2];
       }
   }
}

static void runSample(
    int sample,
    OptixState &state,
    BufferManager &buffers,
    std::vector<PtexTexture> &textures,
    CUstream stream,
    int width,
    int height,
    int spp,
    Params &params,
    CUdeviceptr d_params,
    std::vector<float> &outputImage,
    std::vector<float> &textureImage
) {
    std::cout << "Sample #" << sample << std::endl;

    params.bounce = 0;
    params.sampleCount = sample;

    resetBuffers(buffers, width, height, params);

    // Run intersection
    for (const auto &[i, geometry] : enumerate(state.geometries)) {
        state.arena.restoreSnapshot(geometry.snapshot);

        params.handle = geometry.handle;
        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_params),
            &params,
            sizeof(params),
            cudaMemcpyHostToDevice
        ));

        CHECK_OPTIX(optixLaunch(
            state.pipeline,
            stream,
            d_params,
            sizeof(Params),
            &state.sbt,
            width,
            height,
            /*depth=*/1
        ));

        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Copy buffers to host for texture calculations
    copyOutputBuffers(buffers, width, height, params);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Lookup ptex textures
    updateBetaWithTextureAlbedos(
        buffers,
        textures,
        width,
        height,
        spp,
        textureImage
    );

    // Bounce
    CHECK_CUDA(cudaMemset(
        reinterpret_cast<void *>(params.occlusionBuffer),
        0,
        buffers.occlusionBufferSizeInBytes
    ));
    for (const auto &[i, geometry] : enumerate(state.geometries)) {
        state.arena.restoreSnapshot(geometry.snapshot);

        params.handle = geometry.handle;
        params.bounce = 1;
        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_params),
            &params,
            sizeof(params),
            cudaMemcpyHostToDevice
        ));

        CHECK_OPTIX(optixLaunch(
            state.pipeline,
            stream,
            d_params,
            sizeof(Params),
            &state.sbt,
            width,
            height,
            /*depth=*/1
        ));

        CHECK_CUDA(cudaDeviceSynchronize());
    }

    std::vector<float> occlusionBuffer(width * height * 1);
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(occlusionBuffer.data()),
        params.occlusionBuffer,
        buffers.occlusionBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Lookup L for misses
    state.arena.restoreSnapshot(state.environmentState.snapshot);
    std::vector<float> environmentLightBuffer(width * height * 3, 0.f);
    EnvironmentLight::calculateEnvironmentLighting(
        width,
        height,
        state.environmentState.textureObject,
        params.missDirectionBuffer,
        environmentLightBuffer
    );

    std::vector<BSDFSampleRecord> sampleRecordBuffer(width * height);
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(sampleRecordBuffer.data()),
        params.sampleRecordBuffer,
        buffers.sampleRecordBufferSizeInBytes,
        cudaMemcpyDeviceToHost
    ));

    // Lookup L for direct lighting
    EnvironmentLight::calculateEnvironmentLighting(
        width,
        height,
        state.environmentState.textureObject,
        params.sampleRecordBuffer,
        environmentLightBuffer
    );

    // Calculate Li
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            const int pixelIndex = 3 * (row * width + col);

            const int sampleIndex = 1 * (row * width + col);
            const BSDFSampleRecord sampleRecord = sampleRecordBuffer[sampleIndex];
            if (sampleRecord.isValid) {
                const int occlusionIndex = 1 * (row * width + col);

                for (int i = 0; i < 3; i++) {
                    outputImage[pixelIndex + i] += 1.f
                        * buffers.host.betaBuffer[pixelIndex + i]
                        * environmentLightBuffer[pixelIndex + i]
                        * (1.f - occlusionBuffer[occlusionIndex])
                        * (1.f / spp);
                }
            } else {
                const int environmentIndex = 3 * (row * width + col);
                outputImage[pixelIndex + 0] += environmentLightBuffer[environmentIndex + 0] * (1.f / spp);
                outputImage[pixelIndex + 1] += environmentLightBuffer[environmentIndex + 1] * (1.f / spp);
                outputImage[pixelIndex + 2] += environmentLightBuffer[environmentIndex + 2] * (1.f / spp);
            }
        }
    }
}

void Driver::launch(Cam cam, const std::string &exrFilename)
{
    std::cout << "Rendering: " << exrFilename << std::endl;

    std::vector<PtexTexture> textures;
    for (const auto &filename : Textures::textureFilenames) {
        PtexTexture texture(MOANA_ROOT + std::string("/island/") + filename);
        textures.push_back(texture);
    }

    CUstream stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const int width = 1024;
    const int height = 429;

    std::vector<float> outputImage(width * height * 3, 0.f);

    Params params;

    BufferManager buffers;
    mallocBuffers(buffers, width, height, params);

    Scene scene(cam);
    Camera camera = scene.getCamera(width, height);

    params.camera = camera;

    std::vector<float> textureImage(width * height * 3, 0.f);

    const int spp = 4;

    for (int sample = 0; sample < spp; sample++) {
        runSample(
            sample,
            m_state,
            buffers,
            textures,
            stream,
            width,
            height,
            spp,
            params,
            d_params,
            outputImage,
            textureImage
        );
    }

    Image::save(
        width,
        height,
        outputImage,
        exrFilename
    );

    // Image::save(
    //     width,
    //     height,
    //     faceImage,
    //     "face-buffer_" + exrFilename
    // );

    Image::save(
        width,
        height,
        textureImage,
        "texture-buffer_" + exrFilename
    );

    // Image::save(
    //     width,
    //     height,
    //     uvImage,
    //     "uv-buffer_" + exrFilename
    // );

    CHECK_CUDA(cudaFree(params.outputBuffer));
    CHECK_CUDA(cudaFree(params.depthBuffer));
    CHECK_CUDA(cudaFree(params.xiBuffer));
    CHECK_CUDA(cudaFree(params.cosThetaWiBuffer));
    CHECK_CUDA(cudaFree(params.sampleRecordBuffer));
    CHECK_CUDA(cudaFree(params.missDirectionBuffer));
    CHECK_CUDA(cudaFree(params.barycentricBuffer));
    CHECK_CUDA(cudaFree(params.idBuffer));
    CHECK_CUDA(cudaFree(params.colorBuffer));
}

}
