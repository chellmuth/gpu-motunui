#include "texture.hpp"

#include <iostream>
#include <string.h>

#include "tinyexr.h"

#include "assert_macros.hpp"

namespace moana {

Texture::Texture(const std::string &filename)
    : m_filename(filename)
{}

void Texture::determineAndSetPitch()
{
    CUdeviceptr d_environment;
    const size_t bufferSizeInBytes = sizeof(float) * 4 * m_width * m_height;

    size_t pitch;
    CHECK_CUDA(cudaMallocPitch(
        reinterpret_cast<void **>(&d_environment),
        &pitch,
        sizeof(float) * 4 * m_width,
        m_height
    ));

    m_pitch = pitch;

    CHECK_CUDA(cudaFree(reinterpret_cast<void *>(d_environment)));
}

void Texture::loadImage()
{
    const char *error = nullptr;
    const int code = LoadEXR(&m_data, &m_width, &m_height, m_filename.c_str(), &error);
    if (code == TINYEXR_SUCCESS) {
        std::cout << "Loaded texture {"
                  << " width: " << m_width
                  << " height: " << m_height
                  << " }" << std::endl;
    } else {
        fprintf(stderr, "ENVIRONMENT MAP ERROR: %s\n", error);
        FreeEXRErrorMessage(error);
    }
}

cudaTextureObject_t Texture::createTextureObject(ASArena &arena)
{
    loadImage();
    determineAndSetPitch();

    const int width = m_width;
    const int height = m_height;

    const size_t hostBufferSizeInBytes = sizeof(float) * 4 * width * height;
    const size_t deviceBufferSizeInBytes = m_pitch * height;

    std::cout << m_data[0*4 + 0] << " " << m_data[0*4 + 1] << " " << m_data[0*4 + 2] << " " << m_data[0*4 + 3] << std::endl
              << m_data[1*4 + 4] << " " << m_data[1*4 + 5] << " " << m_data[1*4 + 6] << " " << m_data[1*4 + 7] << std::endl
              << m_data[2*4 + 4] << " " << m_data[2*4 + 5] << " " << m_data[2*4 + 6] << " " << m_data[2*4 + 7] << std::endl
              << m_data[3*4 + 4] << " " << m_data[3*4 + 5] << " " << m_data[3*4 + 6] << " " << m_data[3*4 + 7] << std::endl
              << m_data[4*4 + 4] << " " << m_data[4*4 + 5] << " " << m_data[4*4 + 6] << " " << m_data[4*4 + 7] << std::endl;

    CUdeviceptr d_environment = arena.allocOutput(deviceBufferSizeInBytes);

    CHECK_CUDA(cudaMemcpy2D(
        reinterpret_cast<void *>(d_environment),
        m_pitch,
        m_data,
        sizeof(float) * 4 * width,
        sizeof(float) * 4 * width,
        height,
        cudaMemcpyHostToDevice
    ));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = reinterpret_cast<void *>(d_environment);
    resDesc.res.pitch2D.desc.x = 32;
    resDesc.res.pitch2D.desc.y = 32;
    resDesc.res.pitch2D.desc.z = 32;
    resDesc.res.pitch2D.desc.w = 32;
    resDesc.res.pitch2D.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.pitch2D.width = width;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = m_pitch;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    CHECK_CUDA(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    return texObj;
}

Texture::~Texture()
{
    free(m_data);
}

}
