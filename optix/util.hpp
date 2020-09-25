#pragma once

namespace moana { namespace util {

__forceinline__ __device__ void *unpackPointer(unsigned int i0, unsigned int i1)
{
    const unsigned long long uptr = static_cast<unsigned long long>(i0) << 32 | i1;
    void *ptr = reinterpret_cast<void *>(uptr);
    return ptr;
}

__forceinline__ __device__ void packPointer(
    void *ptr,
    unsigned int &i0,
    unsigned int &i1
) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
}

__forceinline__ __device__ float square(float x) {
    return x * x;
}

} }
