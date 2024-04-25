// includes, system
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

// includes, project
#include "helper_pointer.h"
#include <aliases.h>
#include <helper_cuda.h>
#include <memcpy.h>

namespace cudaKitchen {

//----------------------------------------------------------------------------//
template <typename T,
          PointerType ptrType>
inline T* allocPtr(const size_t numElts)
{
    T* ptr = nullptr;
    const size_t numBytes = sizeof(T) * numElts;
    switch (ptrType)
    {
        case CK_HST: ptr = static_cast<T*>(std::malloc(numBytes));
                     break;

        case CK_PLK: CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr),
                                               numBytes));
                     break;

        case CK_UNI: CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>(
                                                  &ptr), numBytes));
                     break;

        case CK_DEV: CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr),
                                           numBytes));
                     break;
    }
    return ptr;
}

#define DATATYPES(_T)                                \
    template _T* allocPtr<_T, CK_HST>(const size_t); \
    template _T* allocPtr<_T, CK_PLK>(const size_t); \
    template _T* allocPtr<_T, CK_UNI>(const size_t); \
    template _T* allocPtr<_T, CK_DEV>(const size_t);
    #include <datatypes.inc>
#undef DATATYPES

//----------------------------------------------------------------------------//
#define DATATYPES(_T)                                   \
    template <PointerType ptrType>                      \
    void freePtr(_T* ptr)                               \
    {                                                   \
        switch (ptrType)                                \
        {                                               \
            case CK_HST: std::free(ptr);                \
                         break;                         \
                                                        \
            case CK_PLK: CUDA_CHECK(cudaFreeHost(ptr)); \
                         break;                         \
                                                        \
            case CK_UNI: CUDA_CHECK(cudaFree(ptr));     \
                         break;                         \
                                                        \
            case CK_DEV: CUDA_CHECK(cudaFree(ptr));     \
                         break;                         \
        }                                               \
    }                                                   \
    template void freePtr<CK_HST>(_T*);                 \
    template void freePtr<CK_PLK>(_T*);                 \
    template void freePtr<CK_UNI>(_T*);                 \
    template void freePtr<CK_DEV>(_T*);
    #include <datatypes.inc>
#undef DATATYPES

} // namespace cudaKitchen
