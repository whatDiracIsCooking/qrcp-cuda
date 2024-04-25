#pragma once

// includes, system
#include <cuda.h>
#include <cuda_runtime.h>

namespace qrcp {

//----------------------------------------------------------------------------//
// Create a Householder reflector and store its tau value.
// Note that the initial vector must already be stored in d_reflector.
//----------------------------------------------------------------------------//
template <typename T>
void setReflectorAndTau(T* d_reflector,
                        T* d_tau,
                        const T* d_magnitude,
                        cudaStream_t &stream);

namespace kernels {

//----------------------------------------------------------------------------//
template <typename T>
__global__
void setReflectorAndTau(T* reflector,
                        T* tau,
                        const T* magnitude);

} // namespace kernels
} // namespace qrcp
