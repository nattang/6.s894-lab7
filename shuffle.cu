#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
using clock_type = unsigned long long;
#define clock_cycle() \
    ({ \
        unsigned long long ret; \
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(ret)); \
        ret; \
    })
using data_type = int;
constexpr unsigned int warp_size = 32;

__global__ void no_shuffle_kernel(data_type *src, clock_type *dst) {
    clock_type start_time = clock_cycle();
    clock_type end_time = clock_cycle();
    data_type val = src[threadIdx.x];
    __shared__ data_type mem[warp_size];

    // Memory fence to ensure that the reads are done.
    __threadfence();
    __syncwarp();
    start_time = clock_cycle();

    // Your code goes here

    mem[threadIdx.x] = val; // write val into shmem

    if (threadIdx.x >= 1) {
        mem[threadIdx.x] += mem[threadIdx.x - 1];
    }
    __syncthreads();
    if (threadIdx.x >= 2) {
        mem[threadIdx.x] += mem[threadIdx.x - 2];
    }
    __syncthreads();
    if (threadIdx.x >= 4) {
        mem[threadIdx.x] += mem[threadIdx.x - 4];
    }
    __syncthreads();
    if (threadIdx.x >= 8) {
        mem[threadIdx.x] += mem[threadIdx.x - 8];
    }
    __syncthreads();
    if (threadIdx.x >= 16) {
        mem[threadIdx.x] += mem[threadIdx.x - 16];
    }
    __syncthreads();

    end_time = clock_cycle();
    __threadfence();

    src[threadIdx.x] = mem[threadIdx.x];
    dst[threadIdx.x] = end_time - start_time;
}

__global__ void shuffle_kernel(data_type *src, clock_type *dst) {
    clock_type start_time = clock_cycle();
    clock_type end_time = clock_cycle();
    data_type val = src[threadIdx.x];
    // Memory fence to ensure that the reads are done.

    __threadfence();
    __syncwarp();
    start_time = clock_cycle();

    // Your code goes here
    for (int offset = 1; offset < 32; offset <<= 1) {
        int n = __shfl_up_sync(0xffffffff, val, offset);
        if (threadIdx.x >= offset)
            val += n;
    }

    end_time = clock_cycle();
    __threadfence();

    src[threadIdx.x] = val;
    dst[threadIdx.x] = end_time - start_time;
}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

// CUDA error checking macro
#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error " << static_cast<int>(err) << " (" \
                      << cudaGetErrorString(err) << ") at " << __FILE__ << ":" \
                      << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

template <typename F> void launch(F f, std::string name) {
    std::cout << "Testing " << name << " kernel:\n";

    clock_type *dst;
    data_type *src;

    clock_type *h = new unsigned long long[warp_size];
    data_type *res = new data_type[warp_size];
    data_type *expected = new data_type[warp_size];
    for (int i = 0; i < warp_size; ++i) {
        res[i] = 1;
        expected[i] = res[i];
        if (i > 0) {
            expected[i] += expected[i - 1];
        }
    }

    CUDA_CHECK(cudaMalloc(&dst, warp_size * sizeof(clock_type)));
    CUDA_CHECK(cudaMalloc(&src, warp_size * sizeof(data_type)));
    CUDA_CHECK(
        cudaMemcpy(src, res, warp_size * sizeof(data_type), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    f(src, dst);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(
        cudaMemcpy(h, dst, warp_size * sizeof(clock_type), cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(res, src, warp_size * sizeof(data_type), cudaMemcpyDeviceToHost));

    auto min_clock = h[0];
    auto max_clock = h[0];

    for (int i = 0; i < warp_size; ++i) {
        min_clock = std::min(min_clock, h[i]);
        max_clock = std::max(max_clock, h[i]);
    }

    std::cout << "clocks in (" << min_clock << ", " << max_clock << ")" << "\n"
              << std::endl;

    std::cout << "outputs:\n";
    for (int i = 0; i < warp_size; ++i)
        std::cout << res[i] << " ";

    std::cout << "\n" << std::endl;
    bool failed = false;
    for (int i = 0; i < warp_size; ++i)
        if (expected[i] != res[i]) {
            std::cout << "mismatch at index " << i << " expected " << expected[i]
                      << " got " << res[i] << std::endl;
            failed = true;
        }

    if (!failed)
        std::cout << "Output is as expected";

    std::cout << "\n\n\n\n" << std::endl;
}
int main() {
    launch(
        [](data_type *src, clock_type *dst) {
            no_shuffle_kernel<<<1, warp_size>>>(src, dst);
        },
        "shared memory cumsum");
    launch(
        [](data_type *src, clock_type *dst) {
            shuffle_kernel<<<1, warp_size>>>(src, dst);
        },
        "warpshuffle cumusm");
    return 0;
}
