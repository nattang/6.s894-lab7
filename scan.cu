#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

////////////////////////////////////////////////////////////////////////////////
// Utility Functions

void cuda_check(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(1);
    }
}

#define CUDA_CHECK(x) \
    do { \
        cuda_check((x), __FILE__, __LINE__); \
    } while (0)

#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

template <typename Op>
void print_array(
    size_t n,
    typename Op::Data const *x // allowed to be either a CPU or GPU pointer
);

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

template <typename Op>
void scan_cpu(size_t n, typename Op::Data const *x, typename Op::Data *out) {
    using Data = typename Op::Data;
    Data accumulator = Op::identity();
    for (size_t i = 0; i < n; i++) {
        accumulator = Op::combine(accumulator, x[i]);
        out[i] = accumulator;
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

#define VALS_PER_THREAD 4
#define WARPS_PER_BLOCK 32
#define SPINE_VALS_PER_THREAD 16 // hardcoded to 2^26 problem size

namespace scan_gpu {

// Reduce each block & store into workspace[blockIdx.x]
template <typename Op>
__global__ void upstream_scan(
    size_t n,
    typename Op::Data *x,        // pointer to GPU memory
    typename Op::Data *workspace // pointer to GPU memory
) {
    using Data = typename Op::Data;

    extern __shared__ __align__(16) char shmem_raw[]; // OK
    Data *shmem = reinterpret_cast<Data *>(shmem_raw);

    int threadId = threadIdx.x;
    int threads_per_block = blockDim.x;
    int block_offset = blockIdx.x * threads_per_block * VALS_PER_THREAD;

    // load from global memory & perform thread scan
    Data vals[VALS_PER_THREAD];
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        int idx = block_offset + threadId * VALS_PER_THREAD + i;
        if (idx >= n) {
            vals[i] = Op::identity();
        } else {
            vals[i] = x[idx];
        }
    }
    for (int i = 1; i < VALS_PER_THREAD; i++) {
        vals[i] = Op::combine(vals[i - 1], vals[i]);
    }

    Data thread_sum = vals[VALS_PER_THREAD - 1];
    shmem[threadId] = thread_sum;

    // scan across shmem (across all warps in the block)
    for (int i = 1; i < threads_per_block; i <<= 1) {
        __syncthreads();
        Data cur_val = shmem[threadId];
        if (threadId >= i) {
            cur_val = Op::combine(shmem[threadId - i], cur_val);
        }
        __syncthreads();
        shmem[threadId] = cur_val;
    }
    __syncthreads();

    // add prev warp reduction to each thread
    Data threadPrefix = Op::identity();
    if (threadId > 0) { // mask first warp
        threadPrefix = shmem[threadId - 1];
    }
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        vals[i] = Op::combine(threadPrefix, vals[i]);
    }

    // write back to x
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        int idx = block_offset + threadId * VALS_PER_THREAD + i;
        if (idx >= n) {
            break;
        }
        x[idx] = vals[i];
    }
    // write blockSum to workspace
    if (threadId == threads_per_block - 1) {
        workspace[blockIdx.x] = shmem[threads_per_block - 1];
    }
}

template <typename Op>
__global__ void upstream_reduction(
    size_t n,
    typename Op::Data *x,        // pointer to GPU memory
    typename Op::Data *blocksums // pointer to GPU memory
) {
    using Data = typename Op::Data;

    extern __shared__ __align__(16) char shmem_raw[]; // OK
    Data *shmem = reinterpret_cast<Data *>(shmem_raw);

    int threadId = threadIdx.x;
    int threads_per_block = blockDim.x;
    int block_offset = blockIdx.x * threads_per_block * VALS_PER_THREAD;

    // load from global memory & perform thread scan
    Data vals[VALS_PER_THREAD];
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        int idx = block_offset + threadId * VALS_PER_THREAD + i;
        if (idx >= n) {
            vals[i] = Op::identity();
        } else {
            vals[i] = x[idx];
        }
    }

    Data thread_total = Op::identity();
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        thread_total = Op::combine(thread_total, vals[i]);
    }

    shmem[threadId] = thread_total;

    // scan across shmem (across all warps in the block)
    for (int i = 1; i < threads_per_block; i <<= 1) {
        __syncthreads();
        Data cur_val = shmem[threadId];
        if (threadId >= i) {
            cur_val = Op::combine(shmem[threadId - i], cur_val);
        }
        __syncthreads();
        shmem[threadId] = cur_val;
    }
    __syncthreads();

    // add prev warp reduction to each thread
    Data threadPrefix = Op::identity();
    if (threadId > 0) { // mask first warp
        threadPrefix = shmem[threadId - 1];
    }
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        vals[i] = Op::combine(threadPrefix, vals[i]);
    }

    // write blockSum to workspace array
    if (threadId == threads_per_block - 1) {
        blocksums[blockIdx.x] = shmem[threads_per_block - 1];
    }
}

template <typename Op>
__global__ void downstream_scan(
    size_t n,
    typename Op::Data *x,        // pointer to GPU memory
    typename Op::Data *blocksums // pointer to GPU memory
) {
    using Data = typename Op::Data;
    extern __shared__ __align__(16) char shmem_raw[]; // OK
    Data *shmem = reinterpret_cast<Data *>(shmem_raw);

    int threads_per_block = blockDim.x;
    int block_offset = blockIdx.x * threads_per_block * VALS_PER_THREAD;

    int threadId = threadIdx.x;

    // load from global memory & perform thread scan
    Data vals[VALS_PER_THREAD];
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        int idx = block_offset + threadId * VALS_PER_THREAD + i;
        vals[i] = x[idx];
    }
    for (int i = 1; i < VALS_PER_THREAD; i++) {
        vals[i] = Op::combine(vals[i - 1], vals[i]);
    }

    Data thread_sum = vals[VALS_PER_THREAD - 1];
    shmem[threadId] = thread_sum;

    // scan across shmem (across all warps in the block)
    for (int i = 1; i < threads_per_block; i <<= 1) {
        __syncthreads();
        Data cur_val = shmem[threadId];
        if (threadId >= i) {
            cur_val = Op::combine(shmem[threadId - i], cur_val);
        }
        __syncthreads();
        shmem[threadId] = cur_val;
    }
    __syncthreads();

    // fix up each thread in block
    Data threadPrefix = Op::identity();
    if (threadId > 0) { // mask first warp
        threadPrefix = shmem[threadId - 1];
    }
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        vals[i] = Op::combine(threadPrefix, vals[i]);
    }

    // add block prefix
    Data block_prefix = Op::identity();
    if (blockIdx.x > 0) {
        block_prefix = blocksums[blockIdx.x - 1];
    }
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        int idx = block_offset + threadId * VALS_PER_THREAD + i;
        vals[i] = Op::combine(block_prefix, vals[i]);
    }

    // write back to x
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        int idx = block_offset + threadId * VALS_PER_THREAD + i;
        x[idx] = vals[i];
    }
}

template <typename Op>
__global__ void spine_scan(
    // size_t vals_per_thread,
    typename Op::Data *blocksums // pointer to GPU memory
) {
    using Data = typename Op::Data;

    extern __shared__ __align__(16) char shmem_raw[]; // OK
    Data *shmem = reinterpret_cast<Data *>(shmem_raw);

    int threads_per_block = blockDim.x;
    int threadId = threadIdx.x;

    Data vals[SPINE_VALS_PER_THREAD];
    for (int i = 0; i < SPINE_VALS_PER_THREAD; i++) {
        int idx = threadId * SPINE_VALS_PER_THREAD + i;
        vals[i] = blocksums[idx];
    }

    for (int i = 1; i < SPINE_VALS_PER_THREAD; i++) {
        vals[i] = Op::combine(vals[i - 1], vals[i]);
    }

    Data thread_sum = vals[SPINE_VALS_PER_THREAD - 1];
    shmem[threadId] = thread_sum;

    // scan across shmem
    for (int i = 1; i < threads_per_block; i <<= 1) {
        __syncthreads();
        Data cur_val = shmem[threadId];
        if (threadId >= i) {
            cur_val = Op::combine(shmem[threadId - i], cur_val);
        }
        __syncthreads();
        shmem[threadId] = cur_val;
    }
    __syncthreads();

    Data threadPrefix = Op::identity();
    if (threadId > 0) {
        threadPrefix = shmem[threadId - 1];
    }

    for (int i = 0; i < SPINE_VALS_PER_THREAD; i++) {
        vals[i] = Op::combine(threadPrefix, vals[i]);
    }

    for (int i = 0; i < SPINE_VALS_PER_THREAD; i++) {
        int idx = threadId * SPINE_VALS_PER_THREAD + i;
        blocksums[idx] = vals[i];
    }
}

template <typename Op>
__global__ void downstream_reduce_fix(
    typename Op::Data *x,        // pointer to GPU memory
    typename Op::Data *blocksums // pointer to GPU memory
) {
    using Data = typename Op::Data;

    int threads_per_block = blockDim.x;
    int block_offset = blockIdx.x * threads_per_block * VALS_PER_THREAD;

    int threadId = threadIdx.x;

    Data vals[VALS_PER_THREAD];
    for (int i = 0; i < VALS_PER_THREAD; i++) {
        int idx = block_offset + threadId * VALS_PER_THREAD + i;
        vals[i] = x[idx];
    }

    Data block_prefix = Op::identity();
    if (blockIdx.x > 0) {
        block_prefix = blocksums[blockIdx.x - 1];
    }

    for (int i = 0; i < VALS_PER_THREAD; i++) {
        int idx = block_offset + threadId * VALS_PER_THREAD + i;
        vals[i] = Op::combine(block_prefix, vals[i]);
    }

    for (int i = 0; i < VALS_PER_THREAD; i++) {
        int idx = block_offset + threadId * VALS_PER_THREAD + i;
        x[idx] = vals[i];
    }
}

// Returns desired size of scratch buffer in bytes.
template <typename Op> size_t get_workspace_size(size_t n) {
    using Data = typename Op::Data;
    int num_blocks = CEIL_DIV(n, WARPS_PER_BLOCK * 32);
    return num_blocks * sizeof(Data) * 2; // double buffer
    // return n * sizeof(Data); // overallocate for simplicity
}

// 'launch_scan'
//
// Input:
//
//   'n': Number of elements in the input array 'x'.
//
//   'x': Input array in GPU memory. The 'launch_scan' function is allowed to
//   overwrite the contents of this buffer.
//
//   'workspace': Scratch buffer in GPU memory. The size of the scratch buffer
//   in bytes is determined by 'get_workspace_size<Op>(n)'.
//
// Output:
//
//   Returns a pointer to GPU memory which will contain the results of the scan
//   after all launched kernels have completed. Must be either a pointer to the
//   'x' buffer or to an offset within the 'workspace' buffer.
//
//   The contents of the output array should be "partial reductions" of the
//   input; each element 'i' of the output array should be given by:
//
//     output[i] = Op::combine(x[0], x[1], ..., x[i])
//
//   where 'Op::combine(...)' of more than two arguments is defined in terms of
//   repeatedly combining pairs of arguments. Note that 'Op::combine' is
//   guaranteed to be associative, but not necessarily commutative, so
//
//        Op::combine(a, b, c)              // conceptual notation; not real C++
//     == Op::combine(a, Op::combine(b, c)) // real C++
//     == Op::combine(Op::combine(a, b), c) // real C++
//
//  but we don't necessarily have
//
//    Op::combine(a, b) == Op::combine(b, a) // not true in general!
//
template <typename Op>
typename Op::Data *launch_scan(
    size_t n,
    typename Op::Data *x, // pointer to GPU memory
    void *workspace       // pointer to GPU memory
) {
    using Data = typename Op::Data;
    int num_blocks =
        CEIL_DIV(n, VALS_PER_THREAD * WARPS_PER_BLOCK * 32); // 32 threads per warp
    // printf("Launching scan with %d blocks\n", num_blocks);

    Data *block_sums = reinterpret_cast<Data *>(workspace);
    Data *block_sums_workspace = block_sums + num_blocks;

    // scan each block, store block sums in workspace
    dim3 gridDim = dim3(num_blocks, 1, 1);
    dim3 blockDim = dim3(WARPS_PER_BLOCK * 32, 1, 1);
    uint32_t shmem_size_bytes = WARPS_PER_BLOCK * 32 * sizeof(Data);

    // printf("lauchining upstream \n");
    upstream_reduction<Op><<<gridDim, blockDim, shmem_size_bytes>>>(n, x, block_sums);

    // scan "spine" (the block sums)
    dim3 spine_gridDim(1);
    dim3 spine_blockDim(WARPS_PER_BLOCK * 32);
    uint32_t spine_shmem_size_bytes = WARPS_PER_BLOCK * 32 * sizeof(Data);
    spine_scan<Op><<<spine_gridDim, spine_blockDim, spine_shmem_size_bytes>>>(block_sums);

    // downstream fixup
    downstream_scan<Op><<<gridDim, blockDim, shmem_size_bytes>>>(n, x, block_sums);

    return x;
}

} // namespace scan_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

struct DebugRange {
    uint32_t lo;
    uint32_t hi;

    static constexpr uint32_t INVALID = 0xffffffff;

    static __host__ __device__ __forceinline__ DebugRange invalid() {
        return {INVALID, INVALID};
    }

    __host__ __device__ __forceinline__ bool operator==(const DebugRange &other) const {
        return lo == other.lo && hi == other.hi;
    }

    __host__ __device__ __forceinline__ bool operator!=(const DebugRange &other) const {
        return !(*this == other);
    }

    __host__ __device__ bool is_empty() const { return lo == hi; }

    __host__ __device__ bool is_valid() const { return lo != INVALID; }

    std::string to_string() const {
        if (lo == INVALID) {
            return "INVALID";
        } else {
            return std::to_string(lo) + ":" + std::to_string(hi);
        }
    }
};

struct DebugRangeConcatOp {
    using Data = DebugRange;

    static __host__ __device__ __forceinline__ Data identity() { return {0, 0}; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        if (a.is_empty()) {
            return b;
        } else if (b.is_empty()) {
            return a;
        } else if (a.is_valid() && b.is_valid() && a.hi == b.lo) {
            return {a.lo, b.hi};
        } else {
            return Data::invalid();
        }
    }

    static std::string to_string(Data d) { return d.to_string(); }
};

struct SumOp {
    using Data = uint32_t;

    static __host__ __device__ __forceinline__ Data identity() { return 0; }

    static __host__ __device__ __forceinline__ Data combine(Data a, Data b) {
        return a + b;
    }

    static std::string to_string(Data d) { return std::to_string(d); }
};

constexpr size_t max_print_array_output = 1025;
static thread_local size_t total_print_array_output = 0;

template <typename Op> void print_array(size_t n, typename Op::Data const *x) {
    using Data = typename Op::Data;

    // copy 'x' from device to host if necessary
    cudaPointerAttributes attr;
    CUDA_CHECK(cudaPointerGetAttributes(&attr, x));
    auto x_host_buf = std::vector<Data>();
    Data const *x_host_ptr = nullptr;
    if (attr.type == cudaMemoryTypeDevice) {
        x_host_buf.resize(n);
        x_host_ptr = x_host_buf.data();
        CUDA_CHECK(
            cudaMemcpy(x_host_buf.data(), x, n * sizeof(Data), cudaMemcpyDeviceToHost));
    } else {
        x_host_ptr = x;
    }

    if (total_print_array_output >= max_print_array_output) {
        return;
    }

    printf("[\n");
    for (size_t i = 0; i < n; i++) {
        auto s = Op::to_string(x_host_ptr[i]);
        printf("  [%zu] = %s,\n", i, s.c_str());
        total_print_array_output++;
        if (total_print_array_output > max_print_array_output) {
            printf("  ... (output truncated)\n");
            break;
        }
    }
    printf("]\n");

    if (total_print_array_output >= max_print_array_output) {
        printf(
            "(Reached maximum limit on 'print_array' output; skipping further calls "
            "to 'print_array')\n");
    }

    total_print_array_output++;
}

template <typename Reset, typename F>
double benchmark_ms(double target_time_ms, Reset &&reset, F &&f) {
    double best_time_ms = std::numeric_limits<double>::infinity();
    double elapsed_ms = 0.0;
    while (elapsed_ms < target_time_ms) {
        reset();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::high_resolution_clock::now();
        f();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double this_ms = std::chrono::duration<double, std::milli>(end - start).count();
        elapsed_ms += this_ms;
        best_time_ms = std::min(best_time_ms, this_ms);
    }
    return best_time_ms;
}

struct Results {
    double time_ms;
    double bandwidth_gb_per_sec;
};

enum class Mode {
    TEST,
    BENCHMARK,
};

template <typename Op>
Results run_config(Mode mode, std::vector<typename Op::Data> const &x) {
    // Allocate buffers
    using Data = typename Op::Data;
    size_t n = x.size();
    size_t workspace_size = scan_gpu::get_workspace_size<Op>(n);
    Data *x_gpu;
    Data *workspace_gpu;
    CUDA_CHECK(cudaMalloc(&x_gpu, n * sizeof(Data)));
    CUDA_CHECK(cudaMalloc(&workspace_gpu, workspace_size));
    CUDA_CHECK(cudaMemcpy(x_gpu, x.data(), n * sizeof(Data), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));

    // Test correctness
    auto expected = std::vector<Data>(n);
    scan_cpu<Op>(n, x.data(), expected.data());
    auto out_gpu = scan_gpu::launch_scan<Op>(n, x_gpu, workspace_gpu);
    if (out_gpu == nullptr) {
        printf("'launch_scan' function not yet implemented (returned nullptr)\n");
        exit(1);
    }
    auto actual = std::vector<Data>(n);
    CUDA_CHECK(
        cudaMemcpy(actual.data(), out_gpu, n * sizeof(Data), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        if (actual.at(i) != expected.at(i)) {
            auto actual_str = Op::to_string(actual.at(i));
            auto expected_str = Op::to_string(expected.at(i));
            printf(
                "Mismatch at position %zu: %s != %s\n",
                i,
                actual_str.c_str(),
                expected_str.c_str());
            if (n <= 128) {
                printf("Input:\n");
                print_array<Op>(n, x.data());
                printf("\nExpected:\n");
                print_array<Op>(n, expected.data());
                printf("\nActual:\n");
                print_array<Op>(n, actual.data());
            }
            exit(1);
        }
    }
    if (mode == Mode::TEST) {
        return {0.0, 0.0};
    }

    // Benchmark
    double target_time_ms = 200.0;
    double time_ms = benchmark_ms(
        target_time_ms,
        [&]() {
            CUDA_CHECK(
                cudaMemcpy(x_gpu, x.data(), n * sizeof(Data), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(workspace_gpu, 0, workspace_size));
        },
        [&]() { scan_gpu::launch_scan<Op>(n, x_gpu, workspace_gpu); });
    double bytes_processed = n * sizeof(Data) * 2;
    double bandwidth_gb_per_sec = bytes_processed / time_ms / 1e6;

    // Cleanup
    CUDA_CHECK(cudaFree(x_gpu));
    CUDA_CHECK(cudaFree(workspace_gpu));

    return {time_ms, bandwidth_gb_per_sec};
}

std::vector<DebugRange> gen_debug_ranges(uint32_t n) {
    auto ranges = std::vector<DebugRange>();
    for (uint32_t i = 0; i < n; ++i) {
        ranges.push_back({i, i + 1});
    }
    return ranges;
}

template <typename Rng> std::vector<uint32_t> gen_random_data(Rng &rng, uint32_t n) {
    auto uniform = std::uniform_int_distribution<uint32_t>(0, 100);
    auto data = std::vector<uint32_t>();
    for (uint32_t i = 0; i < n; ++i) {
        data.push_back(uniform(rng));
    }
    return data;
}

template <typename Op, typename GenData>
void run_tests(std::vector<uint32_t> const &sizes, GenData &&gen_data) {
    for (auto size : sizes) {
        auto data = gen_data(size);
        printf("  Testing size %8u\n", size);
        run_config<Op>(Mode::TEST, data);
        printf("  OK\n\n");
    }
}

int main(int argc, char const *const *argv) {
    auto correctness_sizes = std::vector<uint32_t>{
        16,
        10,
        128,
        100,
        1024,
        1000,
        1 << 20,
        1'000'000,
        16 << 20,
        64 << 20,
    };

    auto rng = std::mt19937(0xCA7CAFE);

    printf("Correctness:\n\n");
    printf("Testing scan operation: debug range concatenation\n\n");
    run_tests<DebugRangeConcatOp>(correctness_sizes, gen_debug_ranges);
    printf("Testing scan operation: integer sum\n\n");
    run_tests<SumOp>(correctness_sizes, [&](uint32_t n) {
        return gen_random_data(rng, n);
    });

    printf("Performance:\n\n");

    size_t n = 64 << 20;
    auto data = gen_random_data(rng, n);

    printf("Benchmarking scan operation: integer sum, size %zu\n\n", n);

    // Warmup
    run_config<SumOp>(Mode::BENCHMARK, data);
    // Benchmark
    auto results = run_config<SumOp>(Mode::BENCHMARK, data);
    printf("  Time: %.2f ms\n", results.time_ms);
    printf("  Throughput: %.2f GB/s\n", results.bandwidth_gb_per_sec);

    return 0;
}