// includes, system
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>

// includes, project
#include "householder.h"


namespace qrcp {

//----------------------------------------------------------------------------//
template <typename T>
void setReflectorAndTau(T* d_reflector,
                        T* d_tau,
                        const T* d_magnitude,
                        cudaStream_t &stream)
{
    // Launch with one block of two warps.
    const dim3 blockDims(32, 2);
    kernels::setReflectorAndTau<T><<<1, blockDims, 0, stream>>>(d_reflector, d_tau,
        d_magnitude);
}

#define FP_DATATYPES(_T) \
template void setReflectorAndTau(_T*, _T*, const _T*, cudaStream_t&);
#include <fpDataTypes.inc>
#undef FP_DATATYPES

namespace kernels {

//----------------------------------------------------------------------------//
__device__ __forceinline__
float copyArg(const float magnitude, const float argVal)
{
    return copysignf(magnitude, argVal);
}

__device__ __forceinline__
double copyArg(const double magnitude, const double argVal)
{
    return copysign(magnitude, argVal);
}

__device__ __forceinline__
cuComplex copyArg(const float magnitude, const cuComplex argVal)
{
    const float arg = atan2f(argVal.y, argVal.x);
    return { magnitude * sinf(arg), magnitude * cosf(arg) };
}

__device__ __forceinline__
cuDoubleComplex copyArg(const double magnitude, const cuDoubleComplex argVal)
{
    const double arg = atan2(argVal.y, argVal.x);
    return { magnitude * sin(arg), magnitude * cos(arg) };
}

//----------------------------------------------------------------------------//
template <typename T>
__global__
void setReflectorAndTau(T* reflector,
                        T* tau,
                        const T* magnitude)
{
    const T norm = magnitude[0];
    const T vecElt = reflector[0];
    const T alpha = copyArg(norm, vecElt) * -1;
    __syncthreads();

    // 0th warp.
    if (0 == threadIdx.y && 0 == threadIdx.x)
    {
        tau[0] = -1. / ((norm * norm) + (vecElt * alpha));
    }

    // 1st warp.
    if (1 == threadIdx.y && 0 == threadIdx.x)
    {
        reflector[0] = vecElt + alpha;
    }
}

#define FP_DATATYPES(_T) \
template __global__ void setReflectorAndTau(_T*, _T*, const _T*);
#include <fpDataTypes.inc>
#undef FP_DATATYPES

} // namespace kernels

} // namespace qrcp
