#pragma once

#include "moana/core/vec3.hpp"

namespace moana {

__device__ inline void coordinateSystem(const Vec3 &a, Vec3 &b, Vec3 &c)
{
    if (fabsf(a.x()) > fabsf(a.y())) {
        float invLen = 1.0f / sqrtf(a.x() * a.x() + a.z() * a.z());
        c = Vec3(a.z() * invLen, 0.0f, -a.x() * invLen);
    } else {
        float invLen = 1.0f / sqrtf(a.y() * a.y() + a.z() * a.z());
        c = Vec3(0.0f, a.z() * invLen, -a.y() * invLen);
    }
    b = cross(c, a);
}

class Frame {
public:
    __device__ Frame() {}

    __device__ Frame(const Vec3 &normal) : n(normal)
    {
        coordinateSystem(n, s, t);
    }

    __device__ Vec3 toWorld(const Vec3 &local) const
    {
        return Vec3(
            s.x() * local.x() + t.x() * local.y() + n.x() * local.z(),
            s.y() * local.x() + t.y() * local.y() + n.y() * local.z(),
            s.z() * local.x() + t.z() * local.y() + n.z() * local.z()
        );
    }

    __device__ Vec3 toLocal(const Vec3 &world) const
    {
        return Vec3(
            s.x() * world.x() + s.y() * world.y() + s.z() * world.z(),
            t.x() * world.x() + t.y() * world.y() + t.z() * world.z(),
            n.x() * world.x() + n.y() * world.y() + n.z() * world.z()
        );
    }

    __device__ float cosTheta(const Vec3 &wi) const
    {
        return wi.z();
    }

    __device__ float absCosTheta(const Vec3 &wi) const
    {
        return fabsf(wi.z());
    }

    Vec3 s;
    Vec3 t;
    Vec3 n;
};

}
