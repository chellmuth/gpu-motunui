#include "scene/element.hpp"

#include <iostream>

#include <cuda_runtime.h>

#include "assert_macros.hpp"
#include "enumerate.hpp"
#include "moana/parsers/obj_parser.hpp"
#include "parsers/curve_parser.hpp"
#include "scene/archive.hpp"
#include "scene/gas.hpp"
#include "scene/ias.hpp"
#include "scene/instances_bin.hpp"

namespace moana {

GeometryResult Element::buildAcceleration(
    OptixDeviceContext context,
    ASArena &arena
) {
    const std::string moanaRoot = MOANA_ROOT;

    std::cout << "Processing " << m_elementName << std::endl;
    std::vector<OptixInstance> rootRecords;

    std::cout << "  Processing primitive archives" << std::endl;

    // Process the objs needed for archive instancing later
    std::vector<OptixTraversableHandle> archiveHandles;
    for (auto [i, objArchivePath] : enumerate(m_objArchivePaths)) {
        std::cout << "    Processing: " << objArchivePath << std::endl;

        ObjParser objParser(objArchivePath, m_mtlLookup);
        auto model = objParser.parse();

        const auto gasHandle = GAS::gasInfoFromObjResult(context, arena, model);
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
        archive.processRecords(context, arena, records, m_sbtOffset);

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
                    m_sbtOffset + curveMtlIndices[j]
                );
            }
        }

        const std::string baseObj = m_baseObjs[i];
        std::cout << "  Processing base obj: " << baseObj << std::endl;

        ObjParser objParser(baseObj, m_mtlLookup);
        auto model = objParser.parse();

        const auto gasHandle = GAS::gasInfoFromObjResult(context, arena, model);

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
                m_sbtOffset
            );
        }

        // Build IAS records for each instance with this geometry
        auto iasObjectHandle = IAS::iasFromInstanceRecords(context, arena, records);

        std::cout << "  Processing element instances" << std::endl;

        const std::string instancesPath = m_elementInstancesBinPaths[i];
        const Instances instancesResult = InstancesBin::parse(instancesPath);
        std::cout << "    Count: " << instancesResult.count << std::endl;

        IAS::createOptixInstanceRecords(
            context,
            rootRecords,
            instancesResult,
            iasObjectHandle
        );
    }

    // Generate and return the top-level IAS handle that will be snapshotted
    auto iasHandle = IAS::iasFromInstanceRecords(context, arena, rootRecords);

    Snapshot snapshot = arena.createSnapshot();
    arena.releaseAll();

    return GeometryResult{
        iasHandle,
        snapshot
    };
}

}
