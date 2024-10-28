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
// Simple Caching GPU Memory Allocator

class GpuAllocCache {
  public:
    GpuAllocCache() = default;

    ~GpuAllocCache();

    GpuAllocCache(GpuAllocCache const &) = delete;
    GpuAllocCache &operator=(GpuAllocCache const &) = delete;
    GpuAllocCache(GpuAllocCache &&) = delete;
    GpuAllocCache &operator=(GpuAllocCache &&) = delete;

    void *alloc(size_t size);
    void reset();

  private:
    void *buffer = nullptr;
    size_t capacity = 0;
    bool active = false;
};

////////////////////////////////////////////////////////////////////////////////
// CPU Reference Implementation (Already Written)

void rle_decompress_cpu(
    uint32_t compressed_count,
    char const *compressed_data,
    uint32_t const *compressed_lengths,
    std::vector<char> &raw) {
    raw.clear();
    for (uint32_t i = 0; i < compressed_count; i++) {
        char c = compressed_data[i];
        uint32_t run_length = compressed_lengths[i];
        for (uint32_t j = 0; j < run_length; j++) {
            raw.push_back(c);
        }
    }
}

struct Decompressed {
    uint32_t count;
    char const *data; // pointer to GPU memory
};

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Optimized GPU Implementation

namespace rle_gpu {

/* TODO: your GPU kernels here... */

// 'launch_rle_decompress'
//
// Input:
//
//   'compressed_count': Number of runs in the compressed data.
//
//   'compressed_data': Array of size 'compressed_count' in GPU memory,
//   containing the byte value for each run.
//
//   'compressed_lengths': Array of size 'compressed_count' in GPU memory,
//    containing the length of each run.
//
//   'workspace_alloc_1', 'workspace_alloc_2': 'GpuAllocCache' objects each of
//   which can be used to allocate a single GPU buffer of arbitrary size.
//
// Output:
//
//   Returns a 'Decompressed' struct containing the following:
//
//     'count': Number of bytes in the decompressed data.
//
//     'data': Pointer to the decompressed data in GPU memory. May point to a
//     buffer allocated using 'workspace_alloc_1' or 'workspace_alloc_2'.
//
Decompressed launch_rle_decompress(
    uint32_t compressed_count,
    char const *compressed_data,
    uint32_t const *compressed_lengths,
    GpuAllocCache &workspace_alloc_1,
    GpuAllocCache &workspace_alloc_2) {
    /* TODO: your CPU code here... */
    uint32_t decompressed_count = 0;         // replace with size of decompressed data
    char const *decompressed_data = nullptr; // replace with pointer to GPU memory
    return {decompressed_count, decompressed_data};
}

} // namespace rle_gpu

/// <--- /your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

GpuAllocCache::~GpuAllocCache() {
    if (buffer) {
        CUDA_CHECK(cudaFree(buffer));
    }
}

void *GpuAllocCache::alloc(size_t size) {
    if (active) {
        printf("Error: GpuAllocCache::alloc called while active\n");
        exit(1);
    }

    if (size > capacity) {
        if (buffer) {
            CUDA_CHECK(cudaFree(buffer));
        }
        CUDA_CHECK(cudaMalloc(&buffer, size));
        CUDA_CHECK(cudaMemset(buffer, 0, size));
        capacity = size;
    }

    return buffer;
}

void GpuAllocCache::reset() {
    if (active) {
        CUDA_CHECK(cudaMemset(buffer, 0, capacity));
    }
    active = false;
}

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

Results run_config(Mode mode, std::vector<char> const &original_raw) {
    // Compress data
    auto compressed_data = std::vector<char>();
    auto compressed_lengths = std::vector<uint32_t>();
    rle_compress_cpu(
        original_raw.size(),
        original_raw.data(),
        compressed_data,
        compressed_lengths);

    // Allocate buffers
    char *compressed_data_gpu;
    uint32_t *compressed_lengths_gpu;
    CUDA_CHECK(cudaMalloc(&compressed_data_gpu, compressed_data.size()));
    CUDA_CHECK(cudaMalloc(
        &compressed_lengths_gpu,
        compressed_lengths.size() * sizeof(uint32_t)));
    auto workspace_alloc_1 = GpuAllocCache();
    auto workspace_alloc_2 = GpuAllocCache();

    // Copy input data to GPU
    CUDA_CHECK(cudaMemcpy(
        compressed_data_gpu,
        compressed_data.data(),
        compressed_data.size(),
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        compressed_lengths_gpu,
        compressed_lengths.data(),
        compressed_lengths.size() * sizeof(uint32_t),
        cudaMemcpyHostToDevice));

    auto reset = [&]() {
        workspace_alloc_1.reset();
        workspace_alloc_2.reset();
    };

    auto f = [&]() {
        rle_gpu::launch_rle_decompress(
            compressed_data.size(),
            compressed_data_gpu,
            compressed_lengths_gpu,
            workspace_alloc_1,
            workspace_alloc_2);
    };

    // Test correctness
    auto decompressed = rle_gpu::launch_rle_decompress(
        compressed_data.size(),
        compressed_data_gpu,
        compressed_lengths_gpu,
        workspace_alloc_1,
        workspace_alloc_2);
    std::vector<char> raw(decompressed.count);
    CUDA_CHECK(cudaMemcpy(
        raw.data(),
        decompressed.data,
        decompressed.count,
        cudaMemcpyDeviceToHost));

    bool correct = true;
    if (raw.size() != original_raw.size()) {
        printf("Mismatch in decompressed size:\n");
        printf("  Expected: %zu\n", original_raw.size());
        printf("  Actual:   %zu\n", raw.size());
        correct = false;
    }
    if (correct) {
        for (size_t i = 0; i < raw.size(); i++) {
            if (raw[i] != original_raw[i]) {
                printf("Mismatch in decompressed data at index %zu:\n", i);
                printf(
                    "  Expected: 0x%02x\n",
                    static_cast<unsigned char>(original_raw[i]));
                printf("  Actual:   0x%02x\n", static_cast<unsigned char>(raw[i]));
                correct = false;
                break;
            }
        }
    }

    if (!correct) {
        if (original_raw.size() <= 1024) {
            printf("\nInput:\n");
            for (size_t i = 0; i < compressed_data.size(); i++) {
                printf(
                    "  [%4zu] = data: 0x%02x, length: %u\n",
                    i,
                    static_cast<unsigned char>(compressed_data.at(i)),
                    compressed_lengths.at(i));
            }
            printf("\nExpected:\n");
            for (size_t i = 0; i < original_raw.size(); i++) {
                printf(
                    "  [%4zu] = 0x%02x\n",
                    i,
                    static_cast<unsigned char>(original_raw[i]));
            }
            printf("\nActual:\n");
            if (raw.size() == 0) {
                printf("  (empty)\n");
            }
            for (size_t i = 0; i < raw.size(); i++) {
                printf("  [%4zu] = 0x%02x\n", i, static_cast<unsigned char>(raw[i]));
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

template <typename Rng>
std::vector<char> generate_sparse(uint32_t size, uint32_t nonzero_count, Rng &rng) {
    auto data = std::vector<char>(size, 0);
    auto random_index = std::uniform_int_distribution<uint32_t>(0, size - 1);
    auto random_byte = std::uniform_int_distribution<int32_t>(
        std::numeric_limits<char>::min(),
        std::numeric_limits<char>::max());
    for (uint32_t i = 0; i < nonzero_count; i++) {
        data.at(random_index(rng)) = random_byte(rng);
    }
    char fill = random_byte(rng);
    for (uint32_t i = 0; i < size; i++) {
        if (data.at(i) == 0) {
            data.at(i) = fill;
        } else {
            fill = random_byte(rng);
        }
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
        printf("  Testing decompression for size %u\n", test_size);
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
    printf("  Testing decompression on file 'rle_raw.bmp' (size %zu)\n", raw.size());
    auto results = run_config(Mode::BENCHMARK, raw);
    printf("  Time: %.2f ms\n", results.time_ms);

    auto raw_sparse = generate_sparse(16 << 20, 1 << 10, rng);
    printf("\n  Testing decompression on sparse data (size %u)\n", 16 << 20);
    results = run_config(Mode::BENCHMARK, raw_sparse);
    printf("  Time: %.2f ms\n", results.time_ms);

    return 0;
}