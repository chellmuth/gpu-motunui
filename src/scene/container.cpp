#include "scene/container.hpp"

#include <memory>

#include "scene/bay_cedar_a1_element.hpp"
#include "scene/beach_element.hpp"
#include "scene/coastline_element.hpp"
#include "scene/coral_element.hpp"
#include "scene/dunes_a_element.hpp"
#include "scene/dunes_b_element.hpp"
#include "scene/element.hpp"
#include "scene/gardenia_a_element.hpp"
#include "scene/hibiscus_element.hpp"
#include "scene/hibiscus_young_element.hpp"
#include "scene/ias.hpp"
#include "scene/ironwood_a1_element.hpp"
#include "scene/ironwood_b_element.hpp"
#include "scene/kava_element.hpp"
#include "scene/lava_rocks_element.hpp"
#include "scene/mountain_a_element.hpp"
#include "scene/mountain_b_element.hpp"
#include "scene/naupaka_a_element.hpp"
#include "scene/ocean_element.hpp"
#include "scene/palm_dead_element.hpp"
#include "scene/palm_rig_element.hpp"
#include "scene/pandanus_a_element.hpp"

#include <map>
#include <utility>

namespace moana { namespace Container {

std::vector<GeometryResult> createGeometryResults(
    OptixDeviceContext context,
    ASArena &arena
) {
    std::vector<GeometryResult> geometries;

    std::unique_ptr<Element> elementPtrs[] = {
        std::make_unique<BayCedarA1Element>(),
        // std::make_unique<BeachElement>(),
        // std::make_unique<CoastlineElement>(),
        // std::make_unique<CoralElement>(),
        std::make_unique<DunesAElement>(),
        std::make_unique<DunesBElement>(),
        std::make_unique<GardeniaAElement>(),
        // std::make_unique<HibiscusElement>(),
        // std::make_unique<HibiscusYoungElement>(),
        // std::make_unique<IronwoodA1Element>(),
        // std::make_unique<IronwoodA1ElementOverflow>(),
        // std::make_unique<IronwoodBElement>(),
        // std::make_unique<IronwoodBElementOverflow>(),
        // std::make_unique<KavaElement>(),
        // std::make_unique<LavaRocksElement>(),
        // std::make_unique<MountainAElement>(),
        // std::make_unique<MountainBElement>(),
        // std::make_unique<NaupakaAElement>(),
        // std::make_unique<PalmDeadElement>(),
        // std::make_unique<PalmRigElement>(),
        // std::make_unique<PandanusAElement>(),
        // std::make_unique<OceanElement>(),
    };

    int elementSBTOffset = 0;
    for (const auto &elementPtr : elementPtrs) {
        auto result = elementPtr->buildAcceleration(context, arena, elementSBTOffset);
        elementSBTOffset += result.hostSBTRecords.size();

        geometries.push_back(result);
    }

    return geometries;
}

} }
