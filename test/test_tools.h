#ifndef _TEST_TOOLS_H_
#define _TEST_TOOLS_H_

#include <common/streamEvent.h>
#include <common/cbHandle.h>

namespace qrcp {
namespace test {

//----------------------------------------------------------------------------//
constexpr size_t MAX_SIZE = 1 << 30;
constexpr float TOL = 1. / (1 << 20);
char* d_pool;
char* h_pool;

//----------------------------------------------------------------------------//
// Initialize memory pools at the start of a suite.
//----------------------------------------------------------------------------//
void initMemPools();

//----------------------------------------------------------------------------//
// Free memory pools.
//----------------------------------------------------------------------------//
void releaseMemPools();

//----------------------------------------------------------------------------//
// Create an identity matrix.
//----------------------------------------------------------------------------//
template <typename T>
void makeIdentityMatrix(const size_t dim,
                        T* h_identity);

//----------------------------------------------------------------------------//
// Create a host matrix for which the norm of each column (excluding the
// super-diagonal entries) is equal to the number of rows.
//----------------------------------------------------------------------------//
template <typename T>
void numRowNormMatrix(const size_t numRows,
                      const size_t numCols,
                      T* h_matrix,
                      const bool reversePivots = false);

//----------------------------------------------------------------------------//
// Print GPU info.
//----------------------------------------------------------------------------//
void gpuInfo();

//----------------------------------------------------------------------------//
// Compare pointers on host.
//----------------------------------------------------------------------------//
template <typename T>
void compareHostPointers(const T* h_x,
                         const T* h_y,
                         const size_t size,
                         const T tol = TOL);

//----------------------------------------------------------------------------//
// Random floating point from unit interval.
//----------------------------------------------------------------------------//
template <typename T>
inline T randUnitInterval()
{
    return static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
}
template float  randUnitInterval();
template double randUnitInterval();

//----------------------------------------------------------------------------//
// Random size with bounds.
//----------------------------------------------------------------------------//
inline size_t randomSize(const size_t maxSize)
{
    return static_cast<size_t>(std::ceil(randUnitInterval<float>()
                *static_cast<float>(maxSize)));
}

inline size_t randomSize(const size_t minSize,
                         const size_t maxSize)
{
    const size_t range = maxSize - minSize;
    return minSize + static_cast<size_t>(std::ceil(
                randUnitInterval<float>()*static_cast<float>(range)));
}



}} // namespace qrcp::test

#endif // #ifndef _TEST_TOOLS_H_
