#include <algorithm>
#include <cassert>
#include <iostream>
#include <cublas_v2.h>

#include <householder/householder.h>
#include <householder/identity.h>
#include <common/helper_cuda.h>
#include <common/streamEvent.h>
#include <common/cublasWrappers.h>
#include <common/cbHandle.h>
#include <common/memcpyWrappers.h>
#include <common/memsetWrappers.h>
#include "test_tools.h"


using namespace qrcp;
using namespace qrcp::test;

namespace {

//----------------------------------------------------------------------------//
template<typename T>
void test_setIdentity(const size_t dim)
{
    assert(dim * dim < MAX_SIZE && "Test array size cannot exceed MAX_SIZE");
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
    for (uint shift = 5; shift < 15; ++shift)
    {
        const size_t dim = randomSize(1 << (shift - 1), 1 << shift);

#define FP_DATATYPES(_T) \
        test_setIdentity<_T>(dim);
#include <fpDataTypes.inc>
#undef FP_DATATYPES
    }
}

//----------------------------------------------------------------------------//
template<typename T>
void test_singleReflectorGer(const size_t numRows,
                             const size_t numCols)
{
    const size_t matrixSize = numRows * numCols;
    const size_t householderSize = numRows * numRows;

    cuStream cbStream;
    cbHandle cbHandle(cbStream);
    ringContainer<cuStream, 4> streams;
    ringContainer<cuEvent, 4> events;

    // Device setup.
    T* d_matrix = reinterpret_cast<T*>(d_pool);
    T* d_reflectors = d_matrix + matrixSize;
    T* d_householder = d_reflectors + matrixSize;
    T* d_tau = d_householder + householderSize;
    T* d_mag = d_tau + numCols;
    T* d_zero = d_mag + 1;
    T* d_one = d_zero + 1;
    T* d_work = d_one + 1;
    T* d_firstCol = d_matrix;

    // Host setup.
    T* h_matrix = reinterpret_cast<T*>(h_pool);
    T* h_reflectors = h_matrix + matrixSize;
    T* h_householder = h_reflectors + matrixSize;
    T* h_tau = h_householder + householderSize;
    T* h_mag = h_tau + numCols;
    T* h_zero = h_mag + 1;
    T* h_one = h_zero + 1;
    T* h_work = h_one + 1;
    T* h_firstCol = h_matrix;

    // cbStream: Copy scalars.
    *h_zero = 0.;
    *h_one = 1.;
    memcpy<T, H2D>(d_zero, h_zero, 2, cbStream);

    // Create matrix on host.
    numRowNormMatrix<T>(numRows, numCols, h_matrix);

    // cbStream: Copy matrix D2H.
    memcpy<T, H2D>(d_matrix, h_matrix, matrixSize, cbStream);
    cbStream.sync();

    // streams[]: Copy column to reflector arrray D2D.
    cuStream& st_cpyColToRefl = ++streams;
    cuEvent& ev_cpyColToRefl = ++events;
    memcpy<T, D2D>(d_reflectors, d_firstCol, numRows, st_cpyColToRefl);
    st_cpyColToRefl.record(ev_cpyColToRefl);

    // streams[]: Copy matrix to work D2D.
    cuStream& st_cpyMatToWork = ++streams;
    cuEvent& ev_cpyMatToWork = ++events;
    memcpy<T, D2D>(d_work, d_matrix, matrixSize, st_cpyMatToWork);
    st_cpyMatToWork.record(ev_cpyMatToWork);

    // cbStream: Compute 2-norm of column.
    nrm2(!cbHandle, numRows, d_firstCol, 1, d_mag);

    // cbStream: Set reflector and tau.
    cbStream.wait(ev_cpyColToRefl);
    setReflectorAndTau(d_reflectors, d_tau, d_mag, !cbStream);

    // cbStream: Apply some of transformation via matrix-vector product.
    gemv(!cbHandle, CUBLAS_OP_T,
        numRows, numCols,
        d_tau,
        d_matrix, numRows,
        d_reflectors, 1,
        d_zero,
        d_householder, 1);

    // cbStream: Finish applying transformation via rank-1 update.
    cbStream.wait(ev_cpyMatToWork);
    ger(!cbHandle, numRows, numCols,
        d_one,
        d_reflectors, 1,
        d_householder, 1,
        d_work, numRows);

    //------------------------------------------------------------------------//
    // Verification.
    // cbStream: Compute magnitude of subdiagonal after transformation.
    nrm2(!cbHandle, numRows - 1, d_work + 1, 1, d_mag);

    // cbStream: Copy magnitude to host.
    memcpy<T, D2H>(h_mag, d_mag, 1, cbStream);
    memcpy<T, D2H>(h_work, d_work, matrixSize, cbStream);
    cbStream.sync();
    assert(std::fabs(h_mag[0]) < TOL);
}

//----------------------------------------------------------------------------//
template<typename T>
void test_makeSingleReflector(const size_t numRows,
                              const size_t numCols)
{
    const size_t matrixSize = numRows * numCols;
    const size_t householderSize = numRows * numRows;

    cuStream cbStream;
    cbHandle cbHandle(cbStream);
    ringContainer<cuStream, 4> streams;
    ringContainer<cuEvent, 4> events;

    // Device setup.
    T* d_matrix = reinterpret_cast<T*>(d_pool);
    T* d_reflectors = d_matrix + matrixSize;
    T* d_householder = d_reflectors + matrixSize;
    T* d_tau = d_householder + householderSize;
    T* d_mag = d_tau + numCols;
    T* d_zero = d_mag + 1;
    T* d_one = d_zero + 1;
    T* d_work = d_one + 1;
    T* d_firstCol = d_matrix;

    // Host setup.
    T* h_matrix = reinterpret_cast<T*>(h_pool);
    T* h_reflectors = h_matrix + matrixSize;
    T* h_householder = h_reflectors + matrixSize;
    T* h_tau = h_householder + householderSize;
    T* h_mag = h_tau + numCols;
    T* h_zero = h_mag + 1;
    T* h_one = h_zero + 1;
    T* h_work = h_one + 1;
    T* h_firstCol = h_matrix;

    // cbStream: Zero out householder matrix.
    memset(d_householder, 0, householderSize, cbStream);
    cbStream.sync();

    // cbStream: Copy scalars.
    *h_zero = 0.;
    *h_one = 1.;
    memcpy<T, H2D>(d_zero, h_zero, 2, cbStream);

    // Create matrix on host.
    numRowNormMatrix<T>(numRows, numCols, h_matrix);

    // cbStream: Copy matrix D2H.
    memcpy<T, H2D>(d_matrix, h_matrix, matrixSize, cbStream);
    cbStream.sync();

    // streams[]: Copy column to reflector arrray D2D.
    cuStream& st_cpyColToRefl = ++streams;
    cuEvent& ev_cpyColToRefl = ++events;
    memcpy<T, D2D>(d_reflectors, d_firstCol, numRows, st_cpyColToRefl);
    st_cpyColToRefl.record(ev_cpyColToRefl);

    // streams[]: Copy matrix to work D2D.
    cuStream& st_cpyMatToWork = ++streams;
    cuEvent& ev_cpyMatToWork = ++events;
    memcpy<T, D2D>(d_work, d_matrix, matrixSize, st_cpyMatToWork);
    st_cpyMatToWork.record(ev_cpyMatToWork);

    // cbStream: Compute 2-norm of column.
    nrm2(!cbHandle, numRows, d_firstCol, 1, d_mag);

    // cbStream: Set reflector and tau.
    cbStream.wait(ev_cpyColToRefl);
    setReflectorAndTau(d_reflectors, d_tau, d_mag, !cbStream);

    // cbStream: Form transformation via rank-1 update.
    ger(!cbHandle, numRows, numRows,
        d_tau,
        d_reflectors, 1,
        d_reflectors, 1,
        d_householder, numRows);

    // cbStream: Apply reflector, store result in work.
    cbStream.wait(ev_cpyMatToWork);
    gemm(!cbHandle, CUBLAS_OP_N, CUBLAS_OP_N,
        numRows, numCols, numRows,
        d_one,
        d_householder, numRows,
        d_matrix, numRows,
        d_one,
        d_work, numRows);

    //------------------------------------------------------------------------//
    // Verification.
    // cbStream: Compute magnitude of subdiagonal after transformation.
    nrm2(!cbHandle, numRows - 1, d_work + 1, 1, d_mag);

    // cbStream: Copy magnitude to host.
    memcpy<T, D2H>(h_mag, d_mag, 1, cbStream);
    memcpy<T, D2H>(h_work, d_work, matrixSize, cbStream);
    cbStream.sync();
    assert(std::fabs(h_mag[0]) < TOL);
}

} // anonymous namespace

int main()
{
    initMemPools();


    gpuInfo();
    //suite_setIdentity();
    //std::cout << "PASSED: suite_setIdentity" << std::endl;

    test_makeSingleReflector<double>(4096, 512);
    test_singleReflectorGer<double>(4096, 512);
    std::cout << "PASSED: test_makeSingleReflector" << std::endl;


    releaseMemPools();
    return 0;
}
