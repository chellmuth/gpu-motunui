#include "ptex_texture.hpp"

#include <array>
#include <cmath>
#include <iostream>

#include <omp.h>
#include <Ptexture.h>

namespace moana {

constexpr int cacheCount = 3;
static std::array<Ptex::PtexCache *, cacheCount> caches;

struct : public PtexErrorHandler {
    void reportError(const char *error) override { std::cout << error << std::endl; }
} errorHandler;

PtexTexture::PtexTexture(const std::string &texturePath)
    : m_texturePath(texturePath)
{
    if (!caches[0]) {
        for (int i = 0; i < cacheCount; i++) {
            caches[i] = Ptex::PtexCache::create(100, 1ull << 32, true, nullptr, &errorHandler);
        }
    }
}

Vec3 PtexTexture::lookup(float2 uv, int faceIndex) const
{
    // Handle wrapping
    float u = uv.x - (int)floorf(uv.x);
    float v = uv.y - (int)floorf(uv.y);

    const int threadNum = omp_get_thread_num();
    Ptex::PtexCache *cache = caches[threadNum % cacheCount];

    Ptex::String error;
    Ptex::PtexTexture *texture = cache->get(m_texturePath.c_str(), error);
    Ptex::PtexFilter::Options opts(Ptex::PtexFilter::FilterType::f_bspline);
    Ptex::PtexFilter *filter = Ptex::PtexFilter::getFilter(texture, opts);

    float result[3];
    filter->eval(
        result,
        0,
        texture->numChannels(),
        faceIndex,
        uv.x,
        uv.y,
        0.f,
        0.f,
        0.f,
        0.f
    );

    filter->release();
    texture->release();

    return Vec3(
        std::pow(result[0], 2.2f),
        std::pow(result[1], 2.2f),
        std::pow(result[2], 2.2f)
    );
}

}
