#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
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

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

void rle_compress_cpu(
    uint32_t raw_count,
    char const *raw,
    std::vector<char> &compressed_data,
    std::vector<uint32_t> &compressed_lengths) {
    compressed_data.clear();
    compressed_lengths.clear();

    uint32_t i = 0;
    while (i < raw_count) {
        char c = raw[i];
        uint32_t run_length = 1;
        i++;
        while (i < raw_count && raw[i] == c) {
            run_length++;
            i++;
        }
        compressed_data.push_back(c);
        compressed_lengths.push_back(run_length);
    }
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace rle_gpu {

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
__global__ void downstream_scan_fix(
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
    int num_blocks = CEIL_DIV(n, WARPS_PER_BLOCK * 32);
    return num_blocks * sizeof(uint32_t) * 2; // double buffer
}

// Returns desired size of scratch buffer in bytes.
size_t get_workspace_size(uint32_t raw_count) {
    /* TODO: your CPU code here... */
    return 0;
}

// 'launch_rle_compress'
//
// Input:
//
//   'raw_count': Number of bytes in the input buffer 'raw'.
//
//   'raw': Uncompressed bytes in GPU memory.
//
//   'workspace': Scratch buffer in GPU memory. The size of the scratch buffer
//   in bytes is determined by 'get_workspace_size'.
//
// Output:
//
//   Returns: 'compressed_count', the number of runs in the compressed data.
//
//   'compressed_data': Output buffer of size 'raw_count' in GPU memory. The
//   function should fill the first 'compressed_count' bytes of this buffer
//   with the compressed data.
//
//   'compressed_lengths': Output buffer of size 'raw_count' in GPU memory. The
//   function should fill the first 'compressed_count' integers in this buffer
//   with the lengths of the runs in the compressed data.
//
uint32_t launch_rle_compress(
    uint32_t raw_count,
    char const *raw,             // pointer to GPU buffer
    void *workspace,             // pointer to GPU buffer
    char *compressed_data,       // pointer to GPU buffer
    uint32_t *compressed_lengths // pointer to GPU buffer
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
    upstream_scan<Op><<<gridDim, blockDim, shmem_size_bytes>>>(n, raw, block_sums);

    if (num_blocks == 1) {
        return x;
    }

    // scan "spine" (the block sums)
    dim3 spine_gridDim(1);
    dim3 spine_blockDim(WARPS_PER_BLOCK * 32);
    uint32_t spine_shmem_size_bytes = WARPS_PER_BLOCK * 32 * sizeof(Data);
    spine_scan<Op><<<spine_gridDim, spine_blockDim, spine_shmem_size_bytes>>>(block_sums);

    // downstream fixup
    downstream_scan_fix<Op><<<gridDim, blockDim>>>(raw, block_sums);

    uint32_t compressed_count = 0;
    return compressed_count;
}

} // namespace rle_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

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
};

enum class Mode {
    TEST,
    BENCHMARK,
};

Results run_config(Mode mode, std::vector<char> const &raw) {
    // Allocate buffers
    size_t workspace_size = rle_gpu::get_workspace_size(raw.size());
    char *raw_gpu;
    void *workspace;
    char *compressed_data_gpu;
    uint32_t *compressed_lengths_gpu;
    CUDA_CHECK(cudaMalloc(&raw_gpu, raw.size()));
    CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    CUDA_CHECK(cudaMalloc(&compressed_data_gpu, raw.size()));
    CUDA_CHECK(cudaMalloc(&compressed_lengths_gpu, raw.size() * sizeof(uint32_t)));

    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(raw_gpu, raw.data(), raw.size(), cudaMemcpyHostToDevice));

    auto reset = [&]() {
        CUDA_CHECK(cudaMemset(compressed_data_gpu, 0, raw.size()));
        CUDA_CHECK(cudaMemset(compressed_lengths_gpu, 0, raw.size() * sizeof(uint32_t)));
    };

    auto f = [&]() {
        rle_gpu::launch_rle_compress(
            raw.size(),
            raw_gpu,
            workspace,
            compressed_data_gpu,
            compressed_lengths_gpu);
    };

    // Test correctness
    reset();
    uint32_t compressed_count = rle_gpu::launch_rle_compress(
        raw.size(),
        raw_gpu,
        workspace,
        compressed_data_gpu,
        compressed_lengths_gpu);
    std::vector<char> compressed_data(compressed_count);
    std::vector<uint32_t> compressed_lengths(compressed_count);
    CUDA_CHECK(cudaMemcpy(
        compressed_data.data(),
        compressed_data_gpu,
        compressed_count,
        cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        compressed_lengths.data(),
        compressed_lengths_gpu,
        compressed_count * sizeof(uint32_t),
        cudaMemcpyDeviceToHost));

    std::vector<char> compressed_data_expected;
    std::vector<uint32_t> compressed_lengths_expected;
    rle_compress_cpu(
        raw.size(),
        raw.data(),
        compressed_data_expected,
        compressed_lengths_expected);

    bool correct = true;
    if (compressed_count != compressed_data_expected.size()) {
        printf("Mismatch in compressed count:\n");
        printf("  Expected: %zu\n", compressed_data_expected.size());
        printf("  Actual:   %u\n", compressed_count);
        correct = false;
    }
    if (correct) {
        for (size_t i = 0; i < compressed_data_expected.size(); i++) {
            if (compressed_data[i] != compressed_data_expected[i]) {
                printf("Mismatch in compressed data at index %zu:\n", i);
                printf(
                    "  Expected: 0x%02x\n",
                    static_cast<unsigned char>(compressed_data_expected[i]));
                printf(
                    "  Actual:   0x%02x\n",
                    static_cast<unsigned char>(compressed_data[i]));
                correct = false;
                break;
            }
            if (compressed_lengths[i] != compressed_lengths_expected[i]) {
                printf("Mismatch in compressed lengths at index %zu:\n", i);
                printf("  Expected: %u\n", compressed_lengths_expected[i]);
                printf("  Actual:   %u\n", compressed_lengths[i]);
                correct = false;
                break;
            }
        }
    }
    if (!correct) {
        if (raw.size() <= 1024) {
            printf("\nInput:\n");
            for (size_t i = 0; i < raw.size(); i++) {
                printf("  [%4zu] = 0x%02x\n", i, static_cast<unsigned char>(raw[i]));
            }
            printf("\nExpected:\n");
            for (size_t i = 0; i < compressed_data_expected.size(); i++) {
                printf(
                    "  [%4zu] = data: 0x%02x, length: %u\n",
                    i,
                    static_cast<unsigned char>(compressed_data_expected[i]),
                    compressed_lengths_expected[i]);
            }
            printf("\nActual:\n");
            if (compressed_data.size() == 0) {
                printf("  (empty)\n");
            }
            for (size_t i = 0; i < compressed_data.size(); i++) {
                printf(
                    "  [%4zu] = data: 0x%02x, length: %u\n",
                    i,
                    static_cast<unsigned char>(compressed_data[i]),
                    compressed_lengths[i]);
            }
        }
        exit(1);
    }

    if (mode == Mode::TEST) {
        return {};
    }

    // Benchmark
    double target_time_ms = 1000.0;
    double time_ms = benchmark_ms(target_time_ms, reset, f);

    // Cleanup
    CUDA_CHECK(cudaFree(raw_gpu));
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(compressed_data_gpu));
    CUDA_CHECK(cudaFree(compressed_lengths_gpu));

    return {time_ms};
}

template <typename Rng> std::vector<char> generate_test_data(uint32_t size, Rng &rng) {
    auto random_byte = std::uniform_int_distribution<int32_t>(
        std::numeric_limits<char>::min(),
        std::numeric_limits<char>::max());
    constexpr uint32_t alphabet_size = 4;
    auto alphabet = std::vector<char>();
    for (uint32_t i = 0; i < alphabet_size; i++) {
        alphabet.push_back(random_byte(rng));
    }
    auto random_symbol = std::uniform_int_distribution<uint32_t>(0, alphabet_size - 1);
    auto data = std::vector<char>();
    for (uint32_t i = 0; i < size; i++) {
        data.push_back(alphabet.at(random_symbol(rng)));
    }
    return data;
}

int main(int argc, char const *const *argv) {
    auto rng = std::mt19937(0xCA7CAFE);

    auto test_sizes = std::vector<uint32_t>{
        16,
        10,
        128,
        100,
        1 << 10,
        1000,
        1 << 20,
        1'000'000,
        16 << 20,
    };

    printf("Correctness:\n\n");
    for (auto test_size : test_sizes) {
        auto raw = generate_test_data(test_size, rng);
        printf("  Testing compression for size %u\n", test_size);
        run_config(Mode::TEST, raw);
        printf("  OK\n\n");
    }

    auto test_data_search_paths = std::vector<std::string>{".", "/"};
    std::string test_data_path;
    for (auto test_data_search_path : test_data_search_paths) {
        auto candidate_path = test_data_search_path + "/rle_raw.bmp";
        if (std::filesystem::exists(candidate_path)) {
            test_data_path = candidate_path;
            break;
        }
    }
    if (test_data_path.empty()) {
        printf("Could not find test data file.\n");
        exit(1);
    }

    auto raw = std::vector<char>();
    {
        auto file = std::ifstream(test_data_path, std::ios::binary);
        if (!file) {
            printf("Could not open test data file '%s'.\n", test_data_path.c_str());
            exit(1);
        }
        file.seekg(0, std::ios::end);
        raw.resize(file.tellg());
        file.seekg(0, std::ios::beg);
        file.read(raw.data(), raw.size());
    }

    printf("Performance:\n\n");
    printf("  Testing compression on file 'rle_raw.bmp' (size %zu)\n", raw.size());
    auto results = run_config(Mode::BENCHMARK, raw);
    printf("  Time: %.2f ms\n", results.time_ms);

    return 0;
}