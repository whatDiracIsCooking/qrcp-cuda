// includes, system
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

// includes, project
#include "identity.h"
#include <common/helper_cuda.h>

namespace qrcp {

//----------------------------------------------------------------------------//
template <typename T>
void setIdentity(T* d_eye,
                 const size_t dim,
                 cudaStream_t& stream)
{
    CUDA_CHECK(cudaMemsetAsync(d_eye, 0, sizeof(T) * dim * dim, stream));

    // Launch with dim blocks of one thread each.
    kernels::setIdentity<T><<<dim, 1, 0, stream>>>(d_eye, dim);
}

#define FP_DATATYPES(_T) \
template void setIdentity(_T*, const size_t, cudaStream_t&);
#include <fpDataTypes.inc>
#undef FP_DATATYPES

namespace kernels {

//----------------------------------------------------------------------------//
template <typename T>
__global__
void setIdentity(T* eye,
                 const size_t dim)
{
    eye[blockIdx.x + blockIdx.x * dim] = 1.;
}

#define FP_DATATYPES(_T) \
template __global__ void setIdentity(_T*, const size_t);
#include <fpDataTypes.inc>
#undef FP_DATATYPES

} // namespace kernels
} // namespace qrcp
