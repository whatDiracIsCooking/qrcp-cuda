#pragma

// includes, system
#include <cuda.h>
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>


namespace qrcp {

//----------------------------------------------------------------------------//
// Types of pointers and their allocation/free routines:
//      Host:       std::malloc/std:free,
//      PageLocked: cudaMallocHost/cudaFreeHost,
//      Unified:    cudaMallocManaged/cudaFree,
//      Device:     cudaMalloc/cudaFree
enum class PointerType : uint
{
    HostPointer,
    PageLockedPointer,
    UnifiedPointer,
    DevicePointer
};

constexpr PointerType HostPtr = PointerType::HostPointer;
constexpr PointerType PlockedPtr = PointerType::PageLockedPointer;
constexpr PointerType UnifiedPtr = PointerType::UnifiedPointer;
constexpr PointerType DevPtr = PointerType::DevicePointer;

//----------------------------------------------------------------------------//
// Get a pointer to allocated memory.
//----------------------------------------------------------------------------//
template <typename T,
          PointerType ptrType>
inline T* allocPtr(const size_t numElts)
{
    T* ptr = nullptr;
    const size_t numBytes = sizeof(T) * numElts;
    switch (ptrType)
    {
        case HostPtr:
            ptr = static_cast<T*>(std::malloc(numBytes));
            break;

        case PlockedPtr:
            CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&ptr), numBytes));
            break;

        case UnifiedPtr:
            CUDA_CHECK(cudaMallocManaged(reinterpret_cast<void**>( &ptr),
                numBytes));
            break;

        case DevPtr:
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), numBytes));
            break;
    }
    return ptr;
}

//----------------------------------------------------------------------------//
// Free a pointer.
//----------------------------------------------------------------------------//
template <typename T,
          PointerType ptrType>
inline void freePtr(T* ptr)
{
    if (nullptr != ptr)
    {
        switch (ptrType)
        {
            case HostPtr:
                std::free(ptr);
                break;

            case PlockedPtr:
                CUDA_CHECK(cudaFreeHost(ptr));
                break;

            case UnifiedPtr:
                CUDA_CHECK(cudaFree(ptr));
                break;

            case DevPtr:
                CUDA_CHECK(cudaFree(ptr));
                break;
        }
    }
    ptr = nullptr;
}

// This has the aesthetics of partial specialization.
template <typename T>
inline void freePtr(T* ptr, const PointerType ptrType)
{
    switch (ptrType)
    {
        case HostPtr:
            freePtr<T, HostPtr>(ptr);
            break;

        case PlockedPtr:
            freePtr<T, PlockedPtr>(ptr);
            break;

        case UnifiedPtr:
            freePtr<T, UnifiedPtr>(ptr);
            break;

        case DevPtr:
            freePtr<T, DevPtr>(ptr);
            break;
    }
}

} // namespace qrcp
