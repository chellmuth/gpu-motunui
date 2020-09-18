#include "moana/scene/as_arena.hpp"

#include <assert.h>

#include "assert_macros.hpp"

namespace moana {

static constexpr size_t ByteAlignment = 128;

static size_t roundUp(size_t location, size_t alignment)
{
    if (location % alignment == 0) { return location; }

    return location + alignment - (location % alignment);
}

ASArena::ASArena()
{}

void ASArena::init(size_t poolSizeInBytes)
{
    poolSizeInBytes = roundUp(poolSizeInBytes, ByteAlignment);

    assert(poolSizeInBytes % ByteAlignment == 0);

    CHECK_CUDA(cudaMalloc(
        reinterpret_cast<void **>(&m_basePtr),
        poolSizeInBytes
    ));
    CHECK_CUDA(cudaDeviceSynchronize());
    assert(m_basePtr % ByteAlignment == 0);

    m_poolSizeInBytes = poolSizeInBytes;
    m_outputOffset = 0;
    m_tempOffset = poolSizeInBytes;
}

CUdeviceptr ASArena::allocOutput(size_t bytes)
{
    m_outputOffset = roundUp(m_outputOffset, ByteAlignment);

    CUdeviceptr pointer = m_basePtr + m_outputOffset;

    m_outputOffset += bytes;
    if (m_outputOffset >= m_tempOffset) {
        throw std::runtime_error("Not enough arena memory");
    }

    return pointer;
}

CUdeviceptr ASArena::pushTemp(size_t bytes)
{
    bytes = roundUp(bytes, ByteAlignment);

    m_tempOffset -= bytes;
    m_tempOffsetStack.push_back(bytes);

    if (m_tempOffset <= m_outputOffset) {
        throw std::runtime_error("Not enough arena memory");
    }

    CUdeviceptr pointer = m_basePtr + m_tempOffset;
    return pointer;
}

void ASArena::popTemp()
{
    if (m_tempOffsetStack.empty()) {
        throw std::runtime_error("No temp memory to pop!");
    }

    size_t currentTempSize = m_tempOffsetStack.back();
    m_tempOffset += currentTempSize;

    m_tempOffsetStack.pop_back();
}

Snapshot ASArena::createSnapshot()
{
    Snapshot snapshot;
    snapshot.dataPtr = malloc(m_outputOffset);
    snapshot.sizeInBytes = m_outputOffset;

    CHECK_CUDA(cudaMemcpy(
        snapshot.dataPtr,
        reinterpret_cast<void *>(m_basePtr),
        m_outputOffset,
        cudaMemcpyDeviceToHost
    ));
    CHECK_CUDA(cudaDeviceSynchronize());

    return snapshot;
}

void ASArena::restoreSnapshot(Snapshot snapshot)
{
    assert(snapshot.sizeInBytes <= m_poolSizeInBytes);

    m_outputOffset = snapshot.sizeInBytes;
    CHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void *>(m_basePtr),
        snapshot.dataPtr,
        snapshot.sizeInBytes,
        cudaMemcpyHostToDevice
    ));
    CHECK_CUDA(cudaDeviceSynchronize());
}

void ASArena::releaseAll()
{
    m_outputOffset = 0;

    assert(m_tempOffsetStack.empty());
}

}
