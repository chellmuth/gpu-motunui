#pragma once

#include <map>
#include <random>

#include <cuda_runtime.h>

namespace moana {

class RandomGenerator {
public:
    RandomGenerator()
        : m_generator(m_device()),
          m_distribution(0.f, 1.f - std::numeric_limits<float>::epsilon())
    {}

    float next() {
        return m_distribution(m_generator);
    }

private:
    std::random_device m_device;
    std::mt19937 m_generator;
    std::uniform_real_distribution<float> m_distribution;
};

class ColorMap {
public:
    float3 get(int index);

private:
    std::map<int, float3> m_colorMap;
    RandomGenerator m_rng;
};

}
