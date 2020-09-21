#pragma once

#include "moana/core/vec3.hpp"
#include "moana/cuda/surface_sample.hpp"

namespace moana {

class Triangle {
public:
    __host__ __device__ Triangle(
        const Vec3 &p0,
        const Vec3 &p1,
        const Vec3 &p2
    ) : m_p0(p0),
        m_p1(p1),
        m_p2(p2)
    {}

    __device__ float area() const {
        const Vec3 e1 = m_p1 - m_p0;
        const Vec3 e2 = m_p2 - m_p0;

        const Vec3 crossed = cross(e1, e2);
        return fabsf(crossed.length() / 2.f);
    }

    __device__ SurfaceSample sample(float xi1, float xi2) const {
        const float r1 = xi1;
        const float r2 = xi2;

        const float a = 1 - sqrtf(r1);
        const float b = sqrtf(r1) * (1 - r2);
        const float c = 1 - a - b;

        const Vec3 point = m_p0 * a + m_p1 * b + m_p2 * c;

        const Vec3 e1 = m_p1 - m_p0;
        const Vec3 e2 = m_p2 - m_p0;
        const Vec3 normal = normalized(cross(e1, e2));

        SurfaceSample sample = {
            .point = point,
            .normal = normal,
            .areaPDF = 1.f / area()
        };
        return sample;
    }

private:
    Vec3 m_p0, m_p1, m_p2;
};

}
