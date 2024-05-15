#pragma once

// includes, system
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

// includes, project
#include "cusolverCheck.h"
#include "streamEvent.h"

namespace HelperCuda {

using namespace qrcp;

//----------------------------------------------------------------------------//
class dnHandle
{
    public:
        inline dnHandle() { CUSOLVER_CHECK(cusolverDnCreate(&handle_)); }
        inline ~dnHandle() { CUSOLVER_CHECK(cusolverDnDestroy(handle_)); }
        inline cusolverDnHandle_t& operator!() { return handle_; }

        // Constructor for assigning a stream.
        inline dnHandle(cuStream& stream)
        {
            CUSOLVER_CHECK(cusolverDnCreate(&handle_));
            CUSOLVER_CHECK(cusolverDnSetStream(handle_, !stream));
        }

    private:
        //--------------------------------------------------------------------//
        // Copy and move constructors/assignment operators are not supported
        dnHandle(const dnHandle &other) = delete;
        dnHandle& operator=(const dnHandle &other) = delete;
        dnHandle(dnHandle &&other) = delete;
        dnHandle& operator=(dnHandle &&other) = delete;

        cusolverDnHandle_t handle_;
};

} // namespace HelperCuda
