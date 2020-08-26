#include "scene/container.hpp"

#include "scene/dunes_a_geometry.hpp"
#include "scene/hibiscus_geometry.hpp"
#include "scene/ias.hpp"
#include "scene/ironwood_a1_geometry.hpp"
#include "scene/mountain_a_geometry.hpp"
#include "scene/mountain_b_geometry.hpp"
#include "scene/types.hpp"

namespace moana { namespace Container {

OptixTraversableHandle createHandle(OptixDeviceContext context)
{
    std::vector<OptixInstance> records;

    {
        DunesAGeometry geometry;
        auto result = geometry.buildAcceleration(context);

        float transform[12] = {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f
        };
        Instances instance;
        instance.transforms = transform;
        instance.count = 1;

        IAS::createOptixInstanceRecords(
            context,
            records,
            instance,
            result.handle
        );
    }
    {
        HibiscusGeometry geometry;
        auto result = geometry.buildAcceleration(context);

        float transform[12] = {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f
        };
        Instances instance;
        instance.transforms = transform;
        instance.count = 1;

        IAS::createOptixInstanceRecords(
            context,
            records,
            instance,
            result.handle
        );
    }
    {
        MountainAGeometry geometry;
        auto result = geometry.buildAcceleration(context);

        float transform[12] = {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f
        };
        Instances instance;
        instance.transforms = transform;
        instance.count = 1;

        IAS::createOptixInstanceRecords(
            context,
            records,
            instance,
            result.handle
        );
    }
    {
        MountainBGeometry geometry;
        auto result = geometry.buildAcceleration(context);

        float transform[12] = {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f
        };
        Instances instance;
        instance.transforms = transform;
        instance.count = 1;

        IAS::createOptixInstanceRecords(
            context,
            records,
            instance,
            result.handle
        );
    }
    // {
    //     IronwoodA1Geometry geometry;
    //     auto result = geometry.buildAcceleration(context);

    //     float transform[12] = {
    //         1.f, 0.f, 0.f, 0.f,
    //         0.f, 1.f, 0.f, 0.f,
    //         0.f, 0.f, 1.f, 0.f
    //     };
    //     Instances instance;
    //     instance.transforms = transform;
    //     instance.count = 1;

    //     IAS::createOptixInstanceRecords(
    //         context,
    //         records,
    //         instance,
    //         result.handle
    //     );
    // }

    return IAS::iasFromInstanceRecords(context, records);
}

} }
