#pragma once

// includes, system
#include <cuda.h>
#include <cuda_runtime.h>

namespace qrcp {

//----------------------------------------------------------------------------//
// Create an identity matrix of size dim*dim.
//----------------------------------------------------------------------------//
template <typename T>
void setIdentity(T* d_eye,
                 const size_t dim,
                 cudaStream_t &stream);

namespace kernels {

//----------------------------------------------------------------------------//
template <typename T>
__global__
void setIdentity(T* eye,
                 const size_t dim);

} // namespace kernels
} // namespace qrcp
