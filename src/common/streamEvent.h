#pragma once

// includes, system
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// includes, project
#include "helper_cuda.h"

namespace qrcp {

class cuEvent;
class cuStream;

//----------------------------------------------------------------------------//
class cuStream
{
    public:
        inline cuStream() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
        inline ~cuStream() { CUDA_CHECK(cudaStreamDestroy(stream_)); }
        inline cudaStream_t& getStream() { return stream_; }
        inline cudaStream_t& operator!() { return stream_; }
        inline void sync() { CUDA_CHECK(cudaStreamSynchronize(getStream())); }
        inline void wait(cuEvent &event);
        inline void wait(cudaEvent_t &event);
        inline void record(cuEvent &event);
        inline void record(cudaEvent_t &event);

    private:
        //--------------------------------------------------------------------//
        // Copy and move constructors/assignment operators are not supported
        cuStream(const cuStream &other) = delete;
        cuStream& operator=(const cuStream &other) = delete;
        cuStream(cuStream &&other) = delete;
        cuStream& operator=(cuStream &&other) = delete;
        cudaStream_t stream_;
};

//----------------------------------------------------------------------------//
class cuEvent
{
    public:
        inline cuEvent() { CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming)); }
        inline ~cuEvent() { CUDA_CHECK(cudaEventDestroy(event_)); }
        inline cudaEvent_t& getEvent() { return event_; }
        inline cudaEvent_t& operator!() { return event_; }
        inline void record() {CUDA_CHECK(cudaEventRecord(event_)); }
        inline void record(cuStream &stream) {CUDA_CHECK(cudaEventRecord(getEvent(), stream.getStream())); }
        inline void record(cudaStream_t &stream) {CUDA_CHECK(cudaEventRecord(getEvent(), stream)); }
        inline void sync() {CUDA_CHECK(cudaEventSynchronize(getEvent())); }

    private:
        //--------------------------------------------------------------------//
        // Copy and move constructors/assignment operators are not supported
        cuEvent(const cuEvent &other) = delete;
        cuEvent& operator=(const cuEvent &other) = delete;
        cuEvent(cuEvent &&other) = delete;
        cuEvent& operator=(cuEvent &&other) = delete;
        cudaEvent_t event_;
};

inline void cuStream::wait(cuEvent &event) { CUDA_CHECK(cudaStreamWaitEvent(getStream(), event.getEvent())); }
inline void cuStream::wait(cudaEvent_t &event) { CUDA_CHECK(cudaStreamWaitEvent(getStream(), event)); }
inline void cuStream::record(cuEvent &event) { CUDA_CHECK(cudaEventRecord(event.getEvent(), getStream())); }
inline void cuStream::record(cudaEvent_t &event) { CUDA_CHECK(cudaEventRecord(event, getStream())); }

using streamVector = std::vector<cuStream>;
using eventVector = std::vector<cuEvent>;

//----------------------------------------------------------------------------//
template <typename T,
          const uint stackSize>
class ringContainer
{
    public:
        inline ringContainer()
            : vec_(std::vector<T>(stackSize))
        {
            const std::vector<uint> allowedSizes {2, 4, 8, 16, 32};
            if (std::find(allowedSizes.begin(), allowedSizes.end(), stackSize) == allowedSizes.end())
            {
                std::cout << "Invalid size for ring container" << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        // Return item at current index, internally incrementing index.
        inline T& operator++()
        {
            const uint tmpCurrIdx = currIdx_;
            currIdx_ = (currIdx_ + 1) & (stackSize - 1);
            return vec_[tmpCurrIdx];
        }

        inline ~ringContainer() {}

    private:
        //--------------------------------------------------------------------//
        // Copy and move constructors/assignment operators are not supported
        ringContainer(const ringContainer &other) = delete;
        ringContainer& operator=(const ringContainer &other) = delete;
        ringContainer(ringContainer &&other) = delete;
        ringContainer& operator=(ringContainer &&other) = delete;

        std::vector<T> vec_;
        uint currIdx_ = 0;
};

} // namespace qrcp
