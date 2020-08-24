#pragma once

#include <cuda_runtime.h>

#include "moana/core/ray.hpp"
#include "moana/core/vec3.hpp"

namespace moana {

class Transform {
public:
    __host__ __device__ Transform() {}

    Transform(const float matrix[4][4])
    {
        for (int row = 0; row < 4; row++ ) {
            for (int col = 0; col < 4; col++ ) {
                m_matrix[row][col] = matrix[row][col];
            }
        }
    }

    __device__ Vec3 applyPoint(const Vec3 &point) const
    {
        const float x = point.x();
        const float y = point.y();
        const float z = point.z();

        return Vec3(
            m_matrix[0][0] * x + m_matrix[0][1] * y + m_matrix[0][2] * z + m_matrix[0][3],
            m_matrix[1][0] * x + m_matrix[1][1] * y + m_matrix[1][2] * z + m_matrix[1][3],
            m_matrix[2][0] * x + m_matrix[2][1] * y + m_matrix[2][2] * z + m_matrix[2][3]
        );
    }

    __device__ Vec3 applyVector(const Vec3 &vector) const
    {
        const float x = vector.x();
        const float y = vector.y();
        const float z = vector.z();

        return Vec3(
            m_matrix[0][0] * x + m_matrix[0][1] * y + m_matrix[0][2] * z,
            m_matrix[1][0] * x + m_matrix[1][1] * y + m_matrix[1][2] * z,
            m_matrix[2][0] * x + m_matrix[2][1] * y + m_matrix[2][2] * z
        );
    }

    __device__ Ray apply(const Ray &ray) const
    {
        return Ray(
            applyPoint(ray.origin()),
            applyVector(ray.direction())
        );
    }

private:
    float m_matrix[4][4];
};

Transform lookAt(
    const Vec3 &source,
    const Vec3 &target,
    const Vec3 &up,
    bool flipHandedness = false
);

}
