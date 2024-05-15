/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

// includes, system
#include <string>
#include <cusolverDn.h>

namespace HelperCuda {

inline const char* cusolverGetErrorName(cusolverStatus_t status)
{
    std::string msg;
    switch (status)
    {
        case CUSOLVER_STATUS_NOT_INITIALIZED: msg = "CUSOLVER_STATUS_NOT_INITIALIZED"; break;
        case CUSOLVER_STATUS_ALLOC_FAILED: msg = "CUSOLVER_STATUS_ALLOC_FAILED"; break;
        case CUSOLVER_STATUS_INVALID_VALUE: msg = "CUSOLVER_STATUS_INVALID_VALUE"; break;
        case CUSOLVER_STATUS_ARCH_MISMATCH: msg = "CUSOLVER_STATUS_ARCH_MISMATCH"; break;
        case CUSOLVER_STATUS_EXECUTION_FAILED: msg = "CUSOLVER_STATUS_EXECUTION_FAILED"; break;
        case CUSOLVER_STATUS_INTERNAL_ERROR: msg = "CUSOLVER_STATUS_INTERNAL_ERROR"; break;
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: msg = "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED"; break;
        case CUSOLVER_STATUS_NOT_SUPPORTED: msg = "CUSOLVER_STATUS_NOT_SUPPORTED"; break;
        default: msg = "Unknown cusolver error";
    }
    return msg.c_str();
}

//----------------------------------------------------------------------------//
inline void cusolverCheck(cusolverStatus_t result,
                          char const *const func,
                          const char *const file,
                          int const line)
{
    if (result) {
        fprintf(stderr, "CUSOLVER error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), cusolverGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

#define CUSOLVER_CHECK(val) HelperCuda::cusolverCheck((val), #val, __FILE__, __LINE__)

} // namespace HelperCuda
