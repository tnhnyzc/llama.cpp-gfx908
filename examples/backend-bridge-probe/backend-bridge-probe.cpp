#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using register_host_buffer_fn = bool (*)(void *, size_t, uint32_t);
using unregister_host_buffer_fn = void (*)(void *);

struct backend_deleter {
    void operator()(ggml_backend_t backend) const { ggml_backend_free(backend); }
};

struct buffer_deleter {
    void operator()(ggml_backend_buffer_t buffer) const { ggml_backend_buffer_free(buffer); }
};

struct context_deleter {
    void operator()(ggml_context * ctx) const { ggml_free(ctx); }
};

using backend_ptr = std::unique_ptr<ggml_backend, backend_deleter>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, buffer_deleter>;
using context_ptr = std::unique_ptr<ggml_context, context_deleter>;

struct tensor_allocation {
    context_ptr ctx;
    buffer_ptr buffer;
    ggml_tensor * tensor;
};

static ggml_backend_dev_t find_device(const char * name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (strcmp(ggml_backend_dev_name(dev), name) == 0) {
            return dev;
        }
    }
    return nullptr;
}

static tensor_allocation allocate_tensor(ggml_backend_t backend, size_t size) {
    ggml_init_params params = {
        /* .mem_size   = */ 2 * ggml_tensor_overhead(),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    context_ptr ctx(ggml_init(params));
    GGML_ASSERT(ctx != nullptr);

    ggml_tensor * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, size);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);
    return { std::move(ctx), std::move(buffer), tensor };
}

static double elapsed_us(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
}

static bool verify_tensor(const ggml_tensor * tensor, const std::vector<uint8_t> & expected) {
    std::vector<uint8_t> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
    return actual == expected;
}

static void run_direction(
        ggml_backend_t src_backend,
        ggml_backend_t dst_backend,
        ggml_backend_dev_t src_dev,
        ggml_backend_dev_t dst_dev,
        size_t size,
        int iterations) {
    tensor_allocation src = allocate_tensor(src_backend, size);
    tensor_allocation dst = allocate_tensor(dst_backend, size);

    std::vector<uint8_t> expected(size);
    for (size_t i = 0; i < size; ++i) {
        expected[i] = (uint8_t) ((i * 131u + size) & 0xffu);
    }
    ggml_backend_tensor_set(src.tensor, expected.data(), 0, expected.size());

    ggml_backend_tensor_copy_async(src_backend, dst_backend, src.tensor, dst.tensor);
    ggml_backend_synchronize(dst_backend);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_tensor_copy_async(src_backend, dst_backend, src.tensor, dst.tensor);
        ggml_backend_synchronize(dst_backend);
    }
    const double fallback_us = elapsed_us(start) / iterations;
    const bool fallback_ok = verify_tensor(dst.tensor, expected);

    auto src_reg = ggml_backend_dev_backend_reg(src_dev);
    auto dst_reg = ggml_backend_dev_backend_reg(dst_dev);
    auto src_register = (register_host_buffer_fn) ggml_backend_reg_get_proc_address(src_reg, "ggml_backend_register_host_buffer_v2");
    auto dst_register = (register_host_buffer_fn) ggml_backend_reg_get_proc_address(dst_reg, "ggml_backend_register_host_buffer_v2");
    auto src_unregister = (unregister_host_buffer_fn) ggml_backend_reg_get_proc_address(src_reg, "ggml_backend_unregister_host_buffer_v2");
    auto dst_unregister = (unregister_host_buffer_fn) ggml_backend_reg_get_proc_address(dst_reg, "ggml_backend_unregister_host_buffer_v2");

    const size_t alignment = 4096;
    const size_t allocation_size = (size + alignment - 1) / alignment * alignment;
    void * src_staging = std::aligned_alloc(alignment, allocation_size);
    void * dst_staging = std::aligned_alloc(alignment, allocation_size);
    GGML_ASSERT(src_staging != nullptr && dst_staging != nullptr);

    // Each vendor runtime owns a distinct registration. Registering the same
    // pages with CUDA and ROCm is not a portable operation.
    const bool src_registered = src_register && src_register(src_staging, allocation_size, 0);
    const bool dst_registered = dst_register && dst_register(dst_staging, allocation_size, 0);

    ggml_backend_tensor_get_async(src_backend, src.tensor, src_staging, 0, size);
    ggml_backend_synchronize(src_backend);
    memcpy(dst_staging, src_staging, size);
    ggml_backend_tensor_set_async(dst_backend, dst.tensor, dst_staging, 0, size);
    ggml_backend_synchronize(dst_backend);

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ggml_backend_tensor_get_async(src_backend, src.tensor, src_staging, 0, size);
        ggml_backend_synchronize(src_backend);
        memcpy(dst_staging, src_staging, size);
        ggml_backend_tensor_set_async(dst_backend, dst.tensor, dst_staging, 0, size);
        ggml_backend_synchronize(dst_backend);
    }
    const double staged_us = elapsed_us(start) / iterations;
    const bool staged_ok = verify_tensor(dst.tensor, expected);

    if (dst_registered && dst_unregister) {
        dst_unregister(dst_staging);
    }
    if (src_registered && src_unregister) {
        src_unregister(src_staging);
    }
    std::free(dst_staging);
    std::free(src_staging);

    const double fallback_gbps = size / fallback_us / 1000.0;
    const double staged_gbps = size / staged_us / 1000.0;
    std::printf(
        "%s -> %s size=%zu iterations=%d fallback=%.2f us %.2f GB/s staged=%.2f us %.2f GB/s "
        "registered=%d/%d correct=%d/%d\n",
        ggml_backend_name(src_backend), ggml_backend_name(dst_backend), size, iterations,
        fallback_us, fallback_gbps, staged_us, staged_gbps,
        src_registered, dst_registered, fallback_ok, staged_ok);

    if (!fallback_ok || !staged_ok) {
        std::exit(2);
    }
}

int main(int argc, char ** argv) {
    const char * src_name = argc > 1 ? argv[1] : "CUDA0";
    const char * dst_name = argc > 2 ? argv[2] : "ROCm0";
    const size_t size = argc > 3 ? std::strtoull(argv[3], nullptr, 0) : 4 * 1024 * 1024;
    const int iterations = argc > 4 ? std::max(1, std::atoi(argv[4])) : 20;

    ggml_backend_load_all();

    ggml_backend_dev_t src_dev = find_device(src_name);
    ggml_backend_dev_t dst_dev = find_device(dst_name);
    if (!src_dev || !dst_dev) {
        std::fprintf(stderr, "required devices not found: %s, %s\n", src_name, dst_name);
        return 1;
    }

    backend_ptr src_backend(ggml_backend_dev_init(src_dev, nullptr));
    backend_ptr dst_backend(ggml_backend_dev_init(dst_dev, nullptr));
    GGML_ASSERT(src_backend != nullptr && dst_backend != nullptr);

    run_direction(src_backend.get(), dst_backend.get(), src_dev, dst_dev, size, iterations);
    return 0;
}
