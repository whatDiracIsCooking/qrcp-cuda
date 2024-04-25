#pragma once

// includes, project
#include <helper_cuda.h>
#include <streamEvent.h>

namespace qrcp {

//----------------------------------------------------------------------------//
template <typename T>
inline void memset(T* dstPtr,
                   const int val,
                   const size_t numElts,
                   cuStream& stream)
{
    void* dst = static_cast<void*>(dstPtr);
    const size_t numBytes = numElts * sizeof(T);
    CUDA_CHECK(cudaMemsetAsync(dst, val, numBytes, !stream));
}

} // namespace qrcp
