#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#include "test_tools.h"
#include <common/helper_cuda.h>

namespace qrcp {
namespace test {

//----------------------------------------------------------------------------//
void initMemPools()
{
    const size_t numBytes = MAX_SIZE * sizeof(double);

    // Device pool.
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_pool), numBytes));
    CUDA_CHECK(cudaMemset(d_pool, 0, numBytes));

    // Host pool.
    h_pool = static_cast<void*>(std::malloc(numBytes));
    std::memset(h_pool, 0, numBytes);
}

//----------------------------------------------------------------------------//
void releaseMemPools()
{
    CUDA_CHECK(cudaFree(d_pool));
    std::free(h_pool);
}

//----------------------------------------------------------------------------//
template <typename T>
void makeIdentityMatrix(const size_t dim,
                        T* h_identity)
{
    for (uint ii = 0; ii < dim; ++ii)
    {
        for (uint jj = 0; jj < dim; ++jj)
        {
            h_identity[ii + (jj * dim)] = (ii == jj ? 1. : 0.);
        }
    }
}

#define FP_DATATYPES(_T) \
template void makeIdentityMatrix<_T>(const size_t, _T*);
#include <fpDataTypes.inc>
#undef FP_DATATYPES

//----------------------------------------------------------------------------//
template <typename T>
void numRowNormMatrix(const size_t numRows,
                      const size_t numCols,
                      T* h_matrix,
                      const bool reversePivots)
{
    for (uint ii = 0; ii < numRows; ++ii)
    {
        for (uint jj = 0; jj < numCols; ++jj)
        {
            const bool diag = (ii == jj);
            const bool antiDiag = (ii == numRows - 1 - jj);
            const size_t orderTerm = reversePivots ? (jj + 1) : (numCols - jj + 1);
            const T val = static_cast<T>(orderTerm) / pow(2, .5);
            h_matrix[ii + (jj * numRows)] = jj;
            h_matrix[ii + (jj * numRows)] += (antiDiag ? val : 0.);
            h_matrix[ii + (jj * numRows)] += (diag ? val : 0.);
        }
    }
}

#define FP_DATATYPES(_T) \
template void numRowNormMatrix<_T>(const size_t, const size_t, _T*, const bool);
#include <fpDataTypes.inc>
#undef FP_DATATYPES

//----------------------------------------------------------------------------//
template <typename T>
void compareHostPointers(const T* x,
                         const T* y,
                         const size_t size,
                         const T tol)
{
    uint firstDiff,
         numDiffs = 0;

    // Identify diffs, if any.
    for (uint ii = 0; ii < size; ++ii)
    {
        if (std::fabs(x[ii] - y[ii]) > tol)
        {
            if (0 == numDiffs) firstDiff = ii;
            ++numDiffs;
        }
    }

    // If a nonzero amount of diffs were found, report results.
    if (0 < numDiffs)
    {
        std::cout << numDiffs << " difference(s) found" << std::endl;
        std::cout << "First difference found at index " << firstDiff << std::endl;
        std::cout << x[firstDiff] << " != " << y[firstDiff] << std::endl;
        assert(std::fabs(x[firstDiff] - y[firstDiff]) < tol);
    }
}

#define FP_DATATYPES(_T) \
template void compareHostPointers<_T>(const _T*, const _T*, const size_t, const _T);
#include <fpDataTypes.inc>
#undef FP_DATATYPES

//------------------------------------------------------------------------//
// https://developer.nvidia.com/blog/how-query-device-properties-and-handle-errors-cuda-cc/
void gpuInfo()
{
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("\n");

        printf("  Max Grid Size: (%i, %i, %i)\n",
                prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Max 1D Texture Size: %d\n", prop.maxTexture1D);
        printf("  Max 1D Surface Size: %d\n", prop.maxSurface1D);
        printf("  L2 Cache Size (bytes): %i\n", prop.l2CacheSize);
        printf("  Max L2 Window Size (bytes): %i\n", prop.accessPolicyMaxWindowSize);
        printf("  Total global Memory (bytes): %lu\n", prop.totalGlobalMem);
        printf("\n");

        printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
        printf("  Cooperative launch: %s\n", prop.cooperativeLaunch ? "yes" : "no");
        printf("  Managed Memory: %s\n", prop.managedMemory ? "yes" : "no");
        printf("\n");

        printf("  Number of SM's: %d\n", prop.multiProcessorCount);
        printf("  Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("  Max Concurrent Blocks: %d\n", prop.maxBlocksPerMultiProcessor * prop.multiProcessorCount);
        printf("  Max Concurrent Threads: %d\n", prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount);
        printf("\n");
        printf("\n");
    }
}

}} // namespace qrcp::test
