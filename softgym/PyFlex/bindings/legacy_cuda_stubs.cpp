#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

struct LegacyArg {
    std::size_t offset;
    std::size_t size;
};

struct LegacyLaunchState {
    dim3 grid_dim{1, 1, 1};
    dim3 block_dim{1, 1, 1};
    std::size_t shared_mem_bytes{0};
    cudaStream_t stream{nullptr};
    std::vector<char> arg_bytes;
    std::vector<LegacyArg> args;
};

thread_local LegacyLaunchState g_state;

void ensure_capacity(LegacyLaunchState &state, std::size_t required) {
    if (state.arg_bytes.size() < required) {
        state.arg_bytes.resize(required);
    }
}

}  // namespace

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim,
                                          dim3 blockDim,
                                          std::size_t sharedMem,
                                          cudaStream_t stream) {
    auto &state = g_state;
    state.grid_dim = gridDim;
    state.block_dim = blockDim;
    state.shared_mem_bytes = sharedMem;
    state.stream = stream;
    state.arg_bytes.clear();
    state.args.clear();
    return cudaSuccess;
}

extern "C" cudaError_t cudaSetupArgument(const void *arg,
                                          std::size_t size,
                                          std::size_t offset) {
    auto &state = g_state;
    if (arg == nullptr && size > 0) {
        return cudaErrorInvalidValue;
    }

    ensure_capacity(state, offset + size);
    if (size > 0) {
        std::memcpy(state.arg_bytes.data() + offset, arg, size);
    }

    auto it = std::find_if(state.args.begin(), state.args.end(),
                           [offset](const LegacyArg &entry) { return entry.offset == offset; });
    if (it != state.args.end()) {
        it->size = size;
    } else {
        state.args.push_back(LegacyArg{offset, size});
    }

    return cudaSuccess;
}

extern "C" cudaError_t cudaLaunch(const void *func) {
    auto &state = g_state;

    std::sort(state.args.begin(), state.args.end(),
              [](const LegacyArg &lhs, const LegacyArg &rhs) { return lhs.offset < rhs.offset; });

    std::vector<void *> arg_ptrs;
    arg_ptrs.reserve(state.args.size());

    for (const auto &entry : state.args) {
        if (entry.size == 0) {
            arg_ptrs.push_back(nullptr);
            continue;
        }

        const std::size_t end = entry.offset + entry.size;
        if (state.arg_bytes.size() < end) {
            return cudaErrorInvalidValue;
        }

        arg_ptrs.push_back(state.arg_bytes.data() + entry.offset);
    }

    cudaError_t status = cudaLaunchKernel(func, state.grid_dim, state.block_dim,
                                          arg_ptrs.empty() ? nullptr : arg_ptrs.data(),
                                          state.shared_mem_bytes, state.stream);

    state.arg_bytes.clear();
    state.args.clear();

    return status;
}
