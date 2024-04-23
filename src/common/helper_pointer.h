#ifndef _HELPER_POINTER_H_
#define _HELPER_POINTER_H_

// includes, system
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>

// includes, project
#include <aliases.h>
#include <helper_cuda.h>
#include <memcpy.h>


namespace cudaKitchen {

//----------------------------------------------------------------------------//
// Types of pointers and their allocation/free routines:
//      Host:       std::malloc/std:free,
//      PageLocked: cudaMallocHost/cudaFreeHost,
//      Unified:    cudaMallocManaged/cudaFree,
//      Device:     cudaMalloc/cudaFree
enum class PointerType : uint8
{
    Host,
    PageLocked,
    Unified,
    Device
};
constexpr PointerType CK_HST = PointerType::Host;
constexpr PointerType CK_PLK = PointerType::PageLocked;
constexpr PointerType CK_UNI = PointerType::Unified;
constexpr PointerType CK_DEV = PointerType::Device;

//----------------------------------------------------------------------------//
// Get a pointer to allocated memory.
//----------------------------------------------------------------------------//
template <typename T,
          PointerType ptrType>
T* allocPtr(const size_t numElts);

//----------------------------------------------------------------------------//
// Free a pointer.
//----------------------------------------------------------------------------//
#define DATATYPES(_T)              \
    template <PointerType ptrType> \
    void freePtr(_T* ptr);
    #include <datatypes.inc>
#undef DATATYPES

} // namespace cudaKitchen

#endif  // #ifndef _HELPER_POINTER_H_
