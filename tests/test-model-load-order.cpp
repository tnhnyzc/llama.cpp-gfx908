#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"
#include "gguf.h"
#include "llama.h"

#include "../ggml/src/ggml-backend-impl.h"
#include "../src/llama-model-loader.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

struct temp_files {
    std::vector<std::filesystem::path> paths;
    ~temp_files() {
        for (const auto & path : paths) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }
};

void write_shard(const std::filesystem::path & path, uint16_t split_no,
        const std::vector<std::string> & names) {
    gguf_context_ptr gguf(gguf_init_empty());
    gguf_set_val_str(gguf.get(), "general.architecture", "llama");
    gguf_set_val_u16(gguf.get(), "split.no", split_no);
    gguf_set_val_u16(gguf.get(), "split.count", 2);
    gguf_set_val_i32(gguf.get(), "split.tensors.count", 4);

    ggml_context_ptr ctx(ggml_init({ names.size() * (ggml_tensor_overhead() + 64), nullptr, false }));
    for (size_t i = 0; i < names.size(); ++i) {
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_set_name(tensor, names[i].c_str());
        *static_cast<float *>(tensor->data) = 100.0f * split_no + float(i);
        gguf_add_tensor(gguf.get(), tensor);
    }
    gguf_write_to_file(gguf.get(), path.string().c_str(), false);
}

struct recorder {
    std::vector<std::string> names;
};

struct recording_buffer {
    recorder * rec;
    std::vector<uint8_t> storage;
};

const char * recording_buft_name(ggml_backend_buffer_type_t) { return "Recording"; }
size_t recording_buft_alignment(ggml_backend_buffer_type_t) { return 16; }
bool recording_buft_is_host(ggml_backend_buffer_type_t) { return true; }

void recording_buffer_free(ggml_backend_buffer_t buffer) {
    delete static_cast<recording_buffer *>(buffer->context);
}

void * recording_buffer_base(ggml_backend_buffer_t buffer) {
    return static_cast<recording_buffer *>(buffer->context)->storage.data();
}

void recording_buffer_set(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    auto * context = static_cast<recording_buffer *>(buffer->context);
    context->rec->names.emplace_back(ggml_get_name(tensor));
    std::memcpy(static_cast<uint8_t *>(tensor->data) + offset, data, size);
}

ggml_backend_buffer_t recording_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * rec = static_cast<recorder *>(buft->context);
    auto * context = new recording_buffer { rec, std::vector<uint8_t>(size) };
    ggml_backend_buffer_i iface = {};
    iface.free_buffer = recording_buffer_free;
    iface.get_base = recording_buffer_base;
    iface.set_tensor = recording_buffer_set;
    return ggml_backend_buffer_init(buft, iface, context, size);
}

ggml_backend_buffer_type make_recording_buft(recorder & rec) {
    ggml_backend_buffer_type buft = {};
    buft.iface.get_name = recording_buft_name;
    buft.iface.alloc_buffer = recording_buft_alloc;
    buft.iface.get_alignment = recording_buft_alignment;
    buft.iface.is_host = recording_buft_is_host;
    buft.context = &rec;
    return buft;
}

ggml_context_ptr make_load_context() {
    ggml_context_ptr ctx(ggml_init({ 8 * ggml_tensor_overhead(), nullptr, true }));
    // Deliberately differ from both shard and physical-offset order.
    for (const char * name : { "tensor.3", "tensor.1", "tensor.2", "tensor.0" }) {
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        ggml_set_name(tensor, name);
    }
    return ctx;
}

bool same(const std::vector<std::string> & actual, std::initializer_list<const char *> expected) {
    if (actual.size() != expected.size()) return false;
    size_t i = 0;
    for (const char * value : expected) {
        if (actual[i++] != value) return false;
    }
    return true;
}

struct cancel_state { int calls = 0; };
bool cancel_after_two(float, void * userdata) {
    auto * state = static_cast<cancel_state *>(userdata);
    return state->calls++ < 2;
}

bool run_load(const std::vector<std::string> & paths, recorder & rec, bool cancel) {
    std::vector<std::string> splits = paths;
    llama_model_loader loader(nullptr, nullptr, nullptr, paths[0], splits, nullptr,
            LLAMA_LOAD_MODE_MMAP, false, false, false, nullptr, nullptr);
    loader.init_mappings(false);

    ggml_context_ptr ctx = make_load_context();
    ggml_backend_buffer_type buft = make_recording_buft(rec);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), &buft));
    llama_buf_map bufs = { { 0, buffer.get() }, { 1, buffer.get() } };
    cancel_state state;
    return loader.load_all_data(ctx.get(), bufs, nullptr,
            cancel ? cancel_after_two : nullptr, cancel ? &state : nullptr);
}

} // namespace

int main() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    temp_files files;
    for (int i = 0; i < 2; ++i) {
        files.paths.push_back(std::filesystem::temp_directory_path() /
                ("llama-load-order-" + std::to_string(nonce) + "-" + std::to_string(i) + ".gguf"));
    }
    write_shard(files.paths[0], 0, { "tensor.0", "tensor.1" });
    write_shard(files.paths[1], 1, { "tensor.2", "tensor.3" });
    std::vector<std::string> paths = { files.paths[0].string(), files.paths[1].string() };

    recorder full;
    if (!run_load(paths, full, false) ||
            !same(full.names, { "tensor.0", "tensor.1", "tensor.2", "tensor.3" })) {
        std::fprintf(stderr, "full load did not follow physical shard/offset order\n");
        return 1;
    }

    recorder cancelled;
    if (run_load(paths, cancelled, true) ||
            !same(cancelled.names, { "tensor.0", "tensor.1" })) {
        std::fprintf(stderr, "cancellation did not stop after two physical-order uploads\n");
        return 1;
    }

    std::printf("PASS model load physical order and cancellation\n");
    return 0;
}
