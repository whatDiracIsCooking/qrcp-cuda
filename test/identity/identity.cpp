// includes, system
#include <algorithm>
#include <cassert>
#include <iostream>

// includes, project
#include <identity/identity.h>
#include <common/helper_cuda.h>
#include <common/streamEvent.h>
#include <common/cbHandle.h>
#include <wrappers/memcpy.h>
#include <wrappers/memset.h>
#include <test_tools.h>

using namespace qrcp;
using namespace qrcp::test;

namespace {

static constexpr uint MIN_SIZE_SHIFT = 5;
static constexpr uint MAX_SIZE_SHIFT = 15;

//----------------------------------------------------------------------------//
template<typename T>
void test_setIdentity(const size_t dim)
{
    cuStream stream;

    // Set identity matrix on device.
    T* d_eye = reinterpret_cast<T*>(d_pool);
    setIdentity(d_eye, dim, !stream);

    // Copy results to host.
    T* h_eye = reinterpret_cast<T*>(h_pool);
    memcpy<T, D2H>(h_eye, d_eye, dim * dim, stream);

    // Create validation matrix.
    T* h_validate = reinterpret_cast<T*>(h_pool) + (dim * dim);
    makeIdentityMatrix<T>(dim, h_validate);

    // Validate output.
    stream.sync();
    compareHostPointers<T>(h_eye, h_validate, dim * dim);
}

#define FP_DATATYPES(_T) \
template void test_setIdentity<_T>(const size_t);
#include <fpDataTypes.inc>
#undef FP_DATATYPES

//----------------------------------------------------------------------------//
void suite_setIdentity()
{
    for (uint shift = MIN_SIZE_SHIFT; shift < MAX_SIZE_SHIFT; ++shift)
    {
        const size_t dim = randomSize(1 << (shift - 1), 1 << shift);

#define FP_DATATYPES(_T) \
        test_setIdentity<_T>(dim);
#include <fpDataTypes.inc>
#undef FP_DATATYPES
    }
}

} // anonymous namespace

int main()
{
    initMemPools();
    gpuInfo();

    suite_setIdentity();
    std::cout << "PASSED: suite_setIdentity" << std::endl;

    releaseMemPools();
    return 0;
}
