#include "moana/core/transform.hpp"

#include <iostream>

namespace moana {

Transform lookAt(
    const Vec3 &source,
    const Vec3 &target,
    const Vec3 &up,
    bool flipHandedness
) {
    const Vec3 direction = normalized(source - target);

    if (direction == up) {
        std::cerr << "Look direction cannot equal up vector - quitting!" << std::endl;
        exit(1);
    }

    const Vec3 xAxis = normalized(cross(normalized(up), direction));
    const Vec3 yAxis = normalized(cross(direction, xAxis));

    const float sign = flipHandedness ? -1.f : 1.f;

    float matrix[4][4] {
        { sign * xAxis.x(), yAxis.x(), direction.x(), source.x() },
        { sign * xAxis.y(), yAxis.y(), direction.y(), source.y() },
        { sign * xAxis.z(), yAxis.z(), direction.z(), source.z() },
        { 0.f, 0.f, 0.f, 1.f }
    };

    return Transform(matrix);
}

}
