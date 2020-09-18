#pragma once

#include <vector>

#include <cuda.h>

namespace moana {

struct Snapshot {
    void *dataPtr;
    size_t sizeInBytes;
};

class ASArena {
public:

    ASArena();

    void init(size_t bytes);

    CUdeviceptr allocOutput(size_t bytes);
    void returnCompactedOutput(size_t bytes);

    Snapshot createSnapshot();
    void restoreSnapshot(Snapshot snapshot);

    CUdeviceptr pushTemp(size_t bytes);
    void popTemp();

    void releaseAll();

private:
    CUdeviceptr m_basePtr;
    size_t m_poolSizeInBytes;

    size_t m_outputOffset;
    size_t m_tempOffset;
    std::vector<size_t> m_tempOffsetStack;
};

}
