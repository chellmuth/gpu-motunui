#pragma once

#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include "moana/scene/as_arena.hpp"

namespace moana {

class Texture {
public:
    Texture(const std::string &filename);
    ~Texture();

    cudaTextureObject_t createTextureObject(ASArena &arena);

private:
    void loadImage();
    void determineAndSetPitch();

    std::string m_filename;
    float *m_data = nullptr;
    int m_width = 0;
    int m_height = 0;
    size_t m_pitch;
};

}
