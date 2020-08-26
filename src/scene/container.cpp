#include "scene/container.hpp"

#include "scene/dunes_a_geometry.hpp"
#include "scene/hibiscus_geometry.hpp"
#include "scene/ias.hpp"
#include "scene/ironwood_a1_geometry.hpp"
#include "scene/mountain_a_geometry.hpp"
#include "scene/mountain_b_geometry.hpp"

namespace moana { namespace Container {

std::vector<GeometryResult> createGeometryResults(
    OptixDeviceContext context,
    ASArena &arena
) {
    std::vector<GeometryResult> geometries;
    {
        HibiscusGeometry geometry;
        auto result = geometry.buildAcceleration(context, arena);

        geometries.push_back(result);
    }
    {
        DunesAGeometry geometry;
        auto result = geometry.buildAcceleration(context, arena);

        geometries.push_back(result);
    }
    {
        MountainAGeometry geometry;
        auto result = geometry.buildAcceleration(context, arena);

        geometries.push_back(result);
    }
    {
        MountainBGeometry geometry;
        auto result = geometry.buildAcceleration(context, arena);

        geometries.push_back(result);
    }
    {
        IronwoodA1Geometry geometry;
        auto result = geometry.buildAcceleration(context, arena);

        geometries.push_back(result);
    }

    return geometries;
}

} }
