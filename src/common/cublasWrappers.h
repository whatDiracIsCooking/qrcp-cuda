#ifndef _CUBLAS_WRAPPERS_H_
#define _CUBLAS_WRAPPERS_H_

// includes, system
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// includes, project
#include <common/helper_cuda.h>

namespace qrcp {

//----------------------------------------------------------------------------//
// nrm2
//----------------------------------------------------------------------------//
void inline nrm2(cublasHandle_t& handle,
                 const int n,
                 const float* x,
                 const int incx,
                 float* result)
{
    CUBLAS_CHECK(cublasSnrm2(handle, n, x, incx, result));
}

void inline nrm2(cublasHandle_t& handle,
                 const int n,
                 const double* x,
                 const int incx,
                 double* result)
{
    CUBLAS_CHECK(cublasDnrm2(handle, n, x, incx, result));
}

//----------------------------------------------------------------------------//
// ger
//----------------------------------------------------------------------------//
void inline ger(cublasHandle_t& handle,
                const int m,
                const int n,
                const float* alpha,
                const float* x,
                const int incx,
                const float* y,
                const int incy,
                float* A,
                const int lda)
{
    CUBLAS_CHECK(cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

void inline ger(cublasHandle_t& handle,
                const int m,
                const int n,
                const double* alpha,
                const double* x,
                const int incx,
                const double* y,
                const int incy,
                double* A,
                const int lda)
{
    CUBLAS_CHECK(cublasDger(handle, m, n, alpha, x, incx, y, incy, A, lda));
}

//----------------------------------------------------------------------------//
// gemv
//----------------------------------------------------------------------------//
void inline gemv(cublasHandle_t& handle, const cublasOperation_t trans,
                 const int m, const int n,
                 const float *alpha,
                 const float *A, const int lda,
                 const float *x, const int incx,
                 const float *beta,
                 float *y, const int incy)
{
    CUBLAS_CHECK(cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
}

void inline gemv(cublasHandle_t& handle, const cublasOperation_t trans,
                 const int m, const int n,
                 const double *alpha,
                 const double *A, const int lda,
                 const double *x, const int incx,
                 const double *beta,
                 double *y, const int incy)
{
    CUBLAS_CHECK(cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy));
}

//----------------------------------------------------------------------------//
// gemm
//----------------------------------------------------------------------------//
void inline gemm(cublasHandle_t& handle,
                 const cublasOperation_t transa,
                 const cublasOperation_t transb,
                 const int m, const int n, const int k,
                 const float *alpha,
                 const float *A, const int lda,
                 const float *B, const int ldb,
                 const float *beta,
                 float *C, const int ldc)
{
    CUBLAS_CHECK(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}

void inline gemm(cublasHandle_t& handle,
                 const cublasOperation_t transa,
                 const cublasOperation_t transb,
                 const int m, const int n, const int k,
                 const double *alpha,
                 const double *A, const int lda,
                 const double *B, const int ldb,
                 const double *beta,
                 double *C, const int ldc)
{
    CUBLAS_CHECK(cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
}


} // namespace qrcp

#endif  // #ifndef _CUBLAS_WRAPPERS_H_
