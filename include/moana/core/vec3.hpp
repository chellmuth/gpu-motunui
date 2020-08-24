#pragma once

#include <cmath>
#include <ostream>

#include <cuda_runtime.h>

namespace moana {

class Vec3 {
public:
    __host__ __device__ Vec3() {}
    __host__ __device__ Vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ Vec3(float ee) { e[0] = ee; e[1] = ee; e[2] = ee; }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const Vec3& operator+() const { return *this; }
    __host__ __device__ inline Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline Vec3 operator*(float t) const { return Vec3(t * e[0], t * e[1], t * e[2]); }
    __host__ __device__ inline Vec3 operator*(const Vec3 &v2) const { return Vec3(v2[0] * e[0], v2[1] * e[1], v2[2] * e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline bool operator==(const Vec3 &v2) const {
        return e[0] == v2.e[0]
            && e[1] == v2.e[1]
            && e[2] == v2.e[2]
        ;
    }

    __host__ __device__ inline Vec3& operator+=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator-=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator*=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator/=(const Vec3 &v2);
    __host__ __device__ inline Vec3& operator*=(const float t);
    __host__ __device__ inline Vec3& operator/=(const float t);

    __host__ __device__ inline Vec3 operator-(const Vec3 &v) const {
        return Vec3(e[0] - v.e[0], e[1] - v.e[1], e[2] - v.e[2]);
    }

    __host__ __device__ inline float length() const {
        return sqrtf(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    }

    __host__ __device__ inline bool isZero() const {
        return e[0] == 0.f && e[1] == 0.f && e[2] == 0.f;
    }

    __host__ __device__ inline Vec3 dot(const Vec3 &v) const {
        return e[0] * v.e[0] + e[1] * v.e[1] + e[2] * v.e[2];
    }

    __host__ __device__ inline Vec3 reflect(const Vec3 &normal) const {
        return normal * dot(normal) * 2.f - *this;
    }

    float e[3];
};

inline Vec3& Vec3::operator+=(const Vec3 &v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

inline Vec3& Vec3::operator*=(const Vec3 &v2)
{
    e[0] *= v2[0];
    e[1] *= v2[1];
    e[2] *= v2[2];

    return *this;
}

__host__ __device__ inline Vec3 cross(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(
        (v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
        (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
        (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0])
    );
}

__host__ __device__ inline float dot(const Vec3 &v1, const Vec3 &v2)
{
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline float absDot(const Vec3 &v1, const Vec3 &v2)
{
    return fabsf(v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2]);
}

__host__ __device__ inline Vec3 operator+(const Vec3 &v1, const Vec3 &v2)
{
    return Vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3 &v)
{
    return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline Vec3 operator/(Vec3 v, float t)
{
    return Vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline Vec3 normalized(Vec3 v)
{
    return v / v.length();
}

inline std::ostream &operator<<(std::ostream &os, const Vec3 &v)
{
    return os << "Vec3: " << v.x() << " " << v.y() << " " << v.z();
}

}
