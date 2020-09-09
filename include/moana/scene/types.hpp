#pragma once

#include <vector>

#include <optix.h>
#include <cuda.h>

#include "moana/scene/as_arena.hpp"
#include "moana/parsers/obj_parser.hpp"

namespace moana {

struct HostSBTRecord {
    CUdeviceptr d_normals;
    CUdeviceptr d_normalIndices;
    int materialID;
    int textureIndex;
};

struct GeometryResult {
    OptixTraversableHandle handle;
    Snapshot snapshot;
    std::vector<HostSBTRecord> hostSBTRecords = {};
};

struct Instances {
    int count;
    float *transforms;
};

}
