#ifndef _CB_HANDLE_H_
#define _CB_HANDLE_H_

// includes, system
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// includes, project
#include "helper_cuda.h"
#include "streamEvent.h"

namespace qrcp {

//----------------------------------------------------------------------------//
class cbHandle
{
    public:
        inline cbHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }
        inline ~cbHandle() { CUBLAS_CHECK(cublasDestroy(handle_)); }
        inline cublasHandle_t& operator!() { return handle_; }

        // Constructor for assigning a stream and pointer mode.
        inline cbHandle(cuStream& stream,
                        const cublasPointerMode_t pointerMode = CUBLAS_POINTER_MODE_DEVICE)
        {
            CUBLAS_CHECK(cublasCreate(&handle_));
            CUBLAS_CHECK(cublasSetStream(handle_, !stream));
            CUBLAS_CHECK(cublasSetPointerMode(handle_, pointerMode));
        }

    private:
        //--------------------------------------------------------------------//
        // Copy and move constructors/assignment operators are not supported
        cbHandle(const cbHandle &other) = delete;
        cbHandle& operator=(const cbHandle &other) = delete;
        cbHandle(cbHandle &&other) = delete;
        cbHandle& operator=(cbHandle &&other) = delete;

        cublasHandle_t handle_;
};

} // namespace qrcp

#endif  // #ifndef _CB_HANDLE_H_
