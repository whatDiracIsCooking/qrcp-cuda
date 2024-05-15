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

/* These are CUDA helper functions for error checking
 * This is from helper_cuda.h in https://github.com/NVIDIA/cuda-samples/tree/master/Common
 */

#ifndef _HELPER_CUDA_H_
#define _HELPER_CUDA_H_

// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <cublas_v2.h>


namespace qrcp {

//----------------------------------------------------------------------------//
using uint = unsigned int;

//----------------------------------------------------------------------------//
template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), cudaGetErrorName(result), func);
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(val) qrcp::check((val), #val, __FILE__, __LINE__)

//----------------------------------------------------------------------------//
template <typename T>
void cublasCheck(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUBLAS error at %s:%d code=%d \"%s\" \n", file, line,
            static_cast<unsigned int>(result), func);
    exit(EXIT_FAILURE);
  }
}

#define CUBLAS_CHECK(val) qrcp::cublasCheck((val), #val, __FILE__, __LINE__)

//----------------------------------------------------------------------------//
inline size_t padDim(const size_t numElts,
                     const uint divisor = 32)
{
    return static_cast<size_t>(std::ceil(static_cast<float>(numElts) / divisor)
                                    * divisor);
}


//----------------------------------------------------------------------------//
inline uint computeMaxBlocks(const uint maxConcurrentBlocks,
                               const float maxOccupancyPerc)
{
    return static_cast<uint>(std::ceil(maxOccupancyPerc
                * maxConcurrentBlocks));
}


} // namespace qrcp

 

#endif  // #ifndef _HELPER_CUDA_H_
