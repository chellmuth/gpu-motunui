#include "scene/element.hpp"

#include <iostream>

#include <cuda_runtime.h>

#include "assert_macros.hpp"
#include "util/enumerate.hpp"
#include "moana/parsers/obj_parser.hpp"
#include "parsers/curve_parser.hpp"
#include "scene/archive.hpp"
#include "scene/gas.hpp"
#include "scene/ias.hpp"
#include "scene/instances_bin.hpp"
#include "scene/texture_lookup.hpp"

namespace moana {

static std::vector<HostSBTRecord> createSBTRecords(
    const std::vector<MeshRecord> &meshRecords,
    ASArena &arena,
    const std::string &element,
    const std::vector<std::string> &mtlLookup,
    int materialOffset
) {
    std::vector<HostSBTRecord> sbtRecords;

    size_t totalNormalBuffersSizeInBytes = 0;

    std::cout << "Processing " << meshRecords.size() << " mesh records" << std::endl;
    for (const auto &meshRecord : meshRecords) {
        size_t normalsSizeInBytes = meshRecord.normals.size() * sizeof(float);
        CUdeviceptr d_normals = arena.allocOutput(normalsSizeInBytes);

        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_normals),
            meshRecord.normals.data(),
            normalsSizeInBytes,
            cudaMemcpyHostToDevice
        ));

        size_t normalIndicesSizeInBytes = meshRecord.normalIndices.size() * sizeof(int);
        CUdeviceptr d_normalIndices = arena.allocOutput(normalIndicesSizeInBytes);

        CHECK_CUDA(cudaMemcpy(
            reinterpret_cast<void *>(d_normalIndices),
            meshRecord.normalIndices.data(),
            normalIndicesSizeInBytes,
            cudaMemcpyHostToDevice
        ));

        totalNormalBuffersSizeInBytes += normalsSizeInBytes + normalIndicesSizeInBytes;

        const int textureIndex = TextureLookup::indexForMesh(
            element,
            mtlLookup[meshRecord.materialIndex],
            meshRecord.name
        );
        HostSBTRecord sbtRecord{
            .d_normals = d_normals,
            .d_normalIndices = d_normalIndices,
            .materialID = materialOffset + meshRecord.materialIndex,
            .textureIndex = textureIndex
        };
        sbtRecords.push_back(sbtRecord);
    }

    std::cout << "    Cumulative normal buffers size(mb): "
              << (totalNormalBuffersSizeInBytes / (1024. * 1024.))
              << std::endl;

    return sbtRecords;
}

void createRecordsForElementInstances(
    bool shouldSplitRecords,
    OptixDeviceContext context,
    ASArena &arena,
    const std::string &instancesBinPath,
    const std::vector<OptixInstance> &primitiveRecords,
    std::vector<OptixInstance> &elementInstanceRecords
) {
    std::vector<OptixInstance> records1;
    std::vector<OptixInstance> records2;

    if (shouldSplitRecords) {
        const int recordSplit = int(primitiveRecords.size() / 2.f);
        const int recordSize = primitiveRecords.size();

        for (int i = 0; i < recordSplit; i++) {
            records1.push_back(primitiveRecords[i]);
        }
        for (int i = recordSplit; i < recordSize; i++) {
            records2.push_back(primitiveRecords[i]);
        }
    } else {
        records1 = primitiveRecords;
    }

    {
        // Build IAS records for each instance with this geometry
        auto iasObjectHandle = IAS::iasFromInstanceRecords(
            context,
            arena,
            records1,
            shouldSplitRecords
        );

        std::cout << "  Processing element instances" << std::endl;

        const Instances instancesResult = InstancesBin::parse(instancesBinPath);
        std::cout << "    Count: " << instancesResult.count << std::endl;

        IAS::createOptixInstanceRecords(
            context,
            elementInstanceRecords,
            instancesResult,
            iasObjectHandle
        );
    }
    if (records2.size() > 0) {
        // Build IAS records for each instance with this geometry
        auto iasObjectHandle = IAS::iasFromInstanceRecords(context, arena, records2);

        std::cout << "  Processing element instances" << std::endl;

        const Instances instancesResult = InstancesBin::parse(instancesBinPath);
        std::cout << "    Count: " << instancesResult.count << std::endl;

        IAS::createOptixInstanceRecords(
            context,
            elementInstanceRecords,
            instancesResult,
            iasObjectHandle
        );
    }
}

GeometryResult Element::buildAcceleration(
    OptixDeviceContext context,
    ASArena &arena,
    int elementSBTOffset
) {
    const std::string moanaRoot = MOANA_ROOT;

    std::cout << "Processing " << m_elementName << std::endl;
    std::cout << "  SBT Offset: " << elementSBTOffset << std::endl;
    std::vector<OptixInstance> rootRecords;
    std::vector<HostSBTRecord> hostSBTRecords;
    std::vector<int> archiveSBTOffsets;

    std::cout << "  Processing primitive archives" << std::endl;

    // Process the objs needed for archive instancing later
    std::vector<OptixTraversableHandle> archiveHandles;
    for (auto [i, objArchivePath] : enumerate(m_objArchivePaths)) {
        std::cout << "    Processing: " << objArchivePath << std::endl;

        ObjParser objParser(objArchivePath, m_mtlLookup);
        auto meshRecords = objParser.parse();

        const auto gasHandle = GAS::gasInfoFromMeshRecords(
            context,
            arena,
            meshRecords
        );

        archiveSBTOffsets.push_back(hostSBTRecords.size() + elementSBTOffset);

        auto meshHostSBTRecords = createSBTRecords(
            meshRecords,
            arena,
            m_elementName,
            m_mtlLookup,
            m_materialOffset
        );
        hostSBTRecords.insert(
            hostSBTRecords.end(),
            meshHostSBTRecords.begin(),
            meshHostSBTRecords.end()
        );

        archiveHandles.push_back(gasHandle);
    }

    // Loop over element instances with unique geometry
    const int uniqueElementCopyCount = m_baseObjs.size();
    for (int i = 0; i < uniqueElementCopyCount; i++) {
        std::vector<OptixInstance> records;

        // Process element instance archives
        Archive archive(
            m_primitiveInstancesBinPaths[i],
            m_primitiveInstancesHandleIndices[i],
            archiveHandles
        );
        archive.processRecords(context, arena, records, archiveSBTOffsets);

        // Process element instance curves
        const auto &curveBinPaths = m_curveBinPathsByElementInstance[i];
        const auto &curveMtlIndices = m_curveMtlIndicesByElementInstance[i];
        for (const auto &[j, curveBinPath] : enumerate(curveBinPaths)) {
            std::cout << "Processing " << curveBinPath << std::endl;

            Curve curve(curveBinPath);
            auto curveHandle = curve.gasFromCurve(context, arena);
            {
                float transform[12] = {
                    1.f, 0.f, 0.f, 0.f,
                    0.f, 1.f, 0.f, 0.f,
                    0.f, 0.f, 1.f, 0.f
                };
                Instances curveInstances;
                curveInstances.transforms = transform;
                curveInstances.count = 1;

                IAS::createOptixInstanceRecords(
                    context,
                    records,
                    curveInstances,
                    curveHandle,
                    hostSBTRecords.size() + elementSBTOffset
                );

                HostSBTRecord curveRecord = {
                    .d_normals = 0,
                    .d_normalIndices = 0,
                    .materialID = m_materialOffset + curveMtlIndices[j],
                    .textureIndex = -1,
                };
                hostSBTRecords.push_back(curveRecord);
            }
        }

        const std::string baseObj = m_baseObjs[i];
        std::cout << "  Processing base obj: " << baseObj << std::endl;

        ObjParser objParser(baseObj, m_mtlLookup);
        auto meshRecords = objParser.parse();

        if (meshRecords.size() > 0) {
            const auto gasHandle = GAS::gasInfoFromMeshRecords(
                context,
                arena,
                meshRecords
            );

            // Process element instance geometry
            {
                float transform[12] = {
                    1.f, 0.f, 0.f, 0.f,
                    0.f, 1.f, 0.f, 0.f,
                    0.f, 0.f, 1.f, 0.f
                };
                Instances elementGeometryInstances;
                elementGeometryInstances.transforms = transform;
                elementGeometryInstances.count = 1;

                IAS::createOptixInstanceRecords(
                    context,
                    records,
                    elementGeometryInstances,
                    gasHandle,
                    hostSBTRecords.size() + elementSBTOffset
                );
            }

            std::vector<HostSBTRecord> meshHostSBTRecords = createSBTRecords(
                meshRecords,
                arena,
                m_elementName,
                m_mtlLookup,
                m_materialOffset
            );
            hostSBTRecords.insert(
                hostSBTRecords.end(),
                meshHostSBTRecords.begin(),
                meshHostSBTRecords.end()
            );
        }

        createRecordsForElementInstances(
            m_shouldSplitPrimitiveInstances,
            context,
            arena,
            m_elementInstancesBinPaths[i],
            records,
            rootRecords
        );
    }

    // Generate and return the top-level IAS handle that will be snapshotted
    auto iasHandle = IAS::iasFromInstanceRecords(context, arena, rootRecords);

    Snapshot snapshot = arena.createSnapshot();
    arena.releaseAll();

    return GeometryResult{
        iasHandle,
        snapshot,
        hostSBTRecords
    };
}

}
