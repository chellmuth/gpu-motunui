#pragma once

#include <cuda_runtime.h>

#include "moana/core/ray.hpp"
#include "moana/core/transform.hpp"
#include "moana/core/vec3.hpp"

namespace moana {

struct Resolution {
    int x;
    int y;
};

class Camera {
public:
    __device__ Camera() {}

    Camera(
        const Vec3 &origin,
        const Vec3 &target,
        const Vec3 &up,
        float verticalFOV,
        const Resolution &resolution,
        bool flipHandedness
    );

    __device__ Ray generateRay(int row, int col, float2 samples) const {
        const float top = std::tan(m_verticalFOV / 2.f);
        const float height = top * 2.f;

        const float aspectRatio = 1.f * m_resolution.x / m_resolution.y;
        const float width = height * aspectRatio;
        const float right = width / 2.f;

        const float xCanonical = (col + samples.x) / m_resolution.x;
        const float yCanonical = (row + samples.y) / m_resolution.y;

        const float y = yCanonical * height - top;
        const float x = xCanonical * width - right;

        const Vec3 direction = normalized(Vec3(x, y, -1));

        const Vec3 origin(0.f);
        const Ray transformedRay = m_cameraToWorld.apply(Ray(origin, direction));
        return transformedRay;
    }

private:
    Transform m_cameraToWorld;

    Vec3 m_origin;
    Vec3 m_target;
    Vec3 m_up;
    float m_verticalFOV;
    Resolution m_resolution;
    bool m_flipHandedness;
};

}
