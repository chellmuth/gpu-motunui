#pragma once

#include <cuda_runtime.h>

#include "moana/core/vec3.hpp"

namespace moana {

class Ray {
public:
    __host__ __device__ Ray() {}
    __device__ Ray(const Vec3 &origin, const Vec3 &direction)
        : m_origin(origin),
          m_direction(direction)
    {}

    __device__ Vec3 origin() const { return m_origin; }
    __device__ Vec3 direction() const { return m_direction; }
    __device__ Vec3 at(float t) const { return m_origin + t * m_direction; }

private:
    Vec3 m_origin;
    Vec3 m_direction;
};

}
