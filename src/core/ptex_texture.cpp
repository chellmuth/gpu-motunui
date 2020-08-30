#include "ptex_texture.hpp"

#include <cmath>
#include <iostream>

#include <Ptexture.h>

namespace moana {

static Ptex::PtexCache *cache;

struct : public PtexErrorHandler {
    void reportError(const char *error) override { std::cout << error << std::endl; }
} errorHandler;

PtexTexture::PtexTexture(const std::string &texturePath)
    : m_texturePath(texturePath)
{
    if (!cache) {
        cache = Ptex::PtexCache::create(100, 1ull << 32, true, nullptr, &errorHandler);
    }
}

Vec3 PtexTexture::lookup(float2 uv, int faceIndex) const
{
    // Handle wrapping
    float u = uv.x - (int)floorf(uv.x);
    float v = uv.y - (int)floorf(uv.y);

    Ptex::String error;
    Ptex::PtexTexture *texture = cache->get(m_texturePath.c_str(), error);
    Ptex::PtexFilter::Options opts(Ptex::PtexFilter::FilterType::f_bspline);
    Ptex::PtexFilter *filter = Ptex::PtexFilter::getFilter(texture, opts);

    float result[3];
    filter->eval(result, 0, texture->numChannels(), faceIndex, uv.x, uv.y, 0.f, 0.f, 0.f, 0.f);

    filter->release();
    texture->release();

    return Vec3(
        std::pow(result[0], 2.2f),
        std::pow(result[1], 2.2f),
        std::pow(result[2], 2.2f)
    );
}

}
