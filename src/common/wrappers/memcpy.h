#pragma once

// includes, project
#include <helper_cuda.h>
#include <streamEvent.h>

namespace qrcp {

//----------------------------------------------------------------------------//
enum class CopyType : uint
{
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
};

constexpr CopyType H2H = CopyType::HostToHost;
constexpr CopyType H2D = CopyType::HostToDevice;
constexpr CopyType D2H = CopyType::DeviceToHost;
constexpr CopyType D2D = CopyType::DeviceToDevice;

//----------------------------------------------------------------------------//
template <typename T, CopyType cpyType>
inline void memcpy(T* dstPtr,
                   const T* srcPtr,
                   const size_t numElts)
{
    void* dst = static_cast<void*>(dstPtr);
    const void* src = static_cast<const void*>(srcPtr);
    const size_t numBytes = numElts * sizeof(T);
    switch(cpyType)
    {
        case H2H:
            std::memcpy(dst, src, numBytes);
            break;

        case H2D:
            CUDA_CHECK(cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice));
            break;

        case D2H:
            CUDA_CHECK(cudaMemcpy(dst, src, numBytes, cudaMemcpyDeviceToHost));
            break;

        case D2D:
            CUDA_CHECK(cudaMemcpy(dst, src, numBytes, cudaMemcpyDeviceToDevice));
            break;
    }
}

//----------------------------------------------------------------------------//
template <typename T, CopyType cpyType>
inline void memcpy(T* dstPtr,
                   const T* srcPtr,
                   const size_t numElts,
                   cuStream& stream)
{
    void* dst = static_cast<void*>(dstPtr);
    const void* src = static_cast<const void*>(srcPtr);
    const size_t numBytes = numElts * sizeof(T);
    switch(cpyType)
    {
        case H2D:
            CUDA_CHECK(cudaMemcpyAsync(dst, src, numBytes,
                cudaMemcpyHostToDevice, !stream));
            break;

        case D2H:
            CUDA_CHECK(cudaMemcpyAsync(dst, src, numBytes,
                cudaMemcpyDeviceToHost, !stream));
            break;

        case D2D:
            CUDA_CHECK(cudaMemcpyAsync(dst, src, numBytes,
                cudaMemcpyDeviceToDevice, !stream));
            break;
    }
}

} // namespace qrcp
