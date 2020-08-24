#pragma once

#include <cmath>

namespace moana {

struct Point {
    Point(float _x, float _y, float _z)
        : x(_x), y(_y), z(_z)
    {}

    float x;
    float y;
    float z;

    Point operator-(const Point &other) const {
        return Point(
            x - other.x,
            y - other.y,
            z - other.z
        );
    }

    Point operator/(const float denominator) const {
        return Point(
            x / denominator,
            y / denominator,
            z / denominator
        );
    }

    bool operator==(const Point &other) const {
        return x == other.x && y == other.y && z == other.z;
    }

    float length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    Point normalized() const {
        return *this / length();
    }

    Point cross(const Point &other) const {
        return Point(
            y * other.z - z * other.y,
            -(x * other.z - z * other.x),
            x * other.y - y * other.x
        );
    }
};

struct Face {
    Face(int _v0, int _v1, int _v2,
         int _n0, int _n1, int _n2
    ) : v0(_v0), v1(_v1), v2(_v2),
        n0(_n0), n1(_n1), n2(_n2)
    {}

    int v0;
    int v1;
    int v2;

    int n0;
    int n1;
    int n2;
};

}
