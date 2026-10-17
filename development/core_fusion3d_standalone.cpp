// Independent public C++ adapter test; all new GPU math is NineToothed AOT.
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <infinicore/context/context.hpp>
#include "core_h3_fusion.hpp"
#include <algorithm>
#include <array>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <vector>

static void check(hipError_t error) {
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}
static void *symbol(void *library, const char *name) {
    auto p = dlsym(library, name);
    if (!p) throw std::runtime_error(dlerror());
    return p;
}
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    try {
        using namespace infinicore;
        using namespace songshan3::fusion;
        context::setDevice(Device(Device::Type::HYGON, 0));
        auto a = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        auto r = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
        if (!a || !r) throw std::runtime_error(dlerror());
        songshan3_fusion_register(symbol(a, "launch_h3_adaln3d_float16_variant"), nullptr,
                                 symbol(r, "launch_h3_residual3d_float16_variant"), nullptr);
        constexpr size_t tokens = 7, width = 5376, count = tokens * width;
        constexpr uint16_t guard = 0x7bff;
        std::vector<void *> memory;
        auto upload = [&](const std::vector<uint16_t> &values) {
            void *p = nullptr;
            check(hipMalloc(&p, values.size() * 2));
            memory.push_back(p);
            check(hipMemcpy(p, values.data(), values.size() * 2, hipMemcpyHostToDevice));
            return p;
        };
        std::vector<uint16_t> state(count, 0x3e00), normalized(count, 0x3e00);
        std::vector<uint16_t> modulation(6 * count, 0);
        for (size_t t = 0; t < tokens; ++t)
            for (size_t col = 0; col < width; ++col) {
                modulation[t * 6 * width + col] = 0x3400; // shift 0.25
                modulation[t * 6 * width + width + col] = 0x3800; // scale 0.5
                modulation[t * 6 * width + 2 * width + col] = 0x3800; // gate 0.5
            }
        void *dx = upload(state), *dn = upload(normalized), *dm = upload(modulation);
        std::array<void *, 4> output{};
        for (auto &p : output) p = upload(std::vector<uint16_t>(2 * count, guard));
        {
            auto device = Device(Device::Type::HYGON, 0);
            auto wrap = [&](void *p, Strides strides) {
                return Tensor::strided_from_blob(p, {1, tokens, width}, strides, DataType::F16, device);
            };
            auto x = wrap(dx, {count, width, 1}), n = wrap(dn, {count, width, 1});
            auto shift = wrap(dm, {6 * count, 6 * width, 1});
            auto scale = wrap(static_cast<uint16_t *>(dm) + width, {6 * count, 6 * width, 1});
            auto gate = wrap(static_cast<uint16_t *>(dm) + 2 * width, {6 * count, 6 * width, 1});
            std::array<Tensor, 4> y;
            for (int i = 0; i < 4; ++i) y[i] = wrap(output[i], {2 * count, 2 * width, 2});
            auto run = [&](bool original) {
                adaln_(y[0], n, scale, shift, original);
                residual_(y[1], x, gate, y[0], original);
                adaln_(y[2], n, scale, shift, original);
                residual_(y[3], y[1], gate, y[2], original);
            };
            auto verify = [&](std::array<uint16_t, 4> expected) {
                context::syncDevice();
                for (int j = 0; j < 4; ++j) {
                    std::vector<uint16_t> values(2 * count);
                    check(hipMemcpy(values.data(), output[j], values.size() * 2, hipMemcpyDeviceToHost));
                    for (size_t i = 0; i < count; ++i)
                        if (values[2 * i] != expected[j] || values[2 * i + 1] != guard)
                            throw std::runtime_error("C++ bitwise result/output guard mismatch");
                }
            };
            std::array<std::shared_ptr<graph::Graph>, 2> graphs;
            for (int mode = 0; mode < 2; ++mode) {
                const auto before_a = songshan3_fusion_count(0), before_r = songshan3_fusion_count(1);
                const auto before_children = songshan3_fusion_count(5);
                context::startGraphRecording();
                run(mode == 0);
                graphs[mode] = context::stopGraphRecording();
                graphs[mode]->run();
                verify({0x4100, 0x4180, 0x4100, 0x4400}); // 2.5, 2.75, 2.5, 4.0
                if (songshan3_fusion_count(0) - before_a != (mode ? 2 : 0)
                    || songshan3_fusion_count(1) - before_r != (mode ? 2 : 0)
                    || songshan3_fusion_count(5) - before_children != (mode ? 0 : 10))
                    throw std::runtime_error("C++ dispatch count mismatch");
            }
            auto allocated_adaln = adaln(n, scale, shift);
            auto allocated_residual = residual(x, gate, allocated_adaln);
            context::syncDevice();
            for (auto pair : {std::make_pair(allocated_adaln, uint16_t(0x4100)),
                              std::make_pair(allocated_residual, uint16_t(0x4180))}) {
                std::vector<uint16_t> values(count);
                check(hipMemcpy(values.data(), pair.first->data(), values.size() * 2, hipMemcpyDeviceToHost));
                for (auto value : values)
                    if (value != pair.second) throw std::runtime_error("allocating C++ API mismatch");
            }
            std::fill(state.begin(), state.end(), 0x3800); // x becomes 0.5, normalized stays 1.5
            check(hipMemcpy(dx, state.data(), state.size() * 2, hipMemcpyHostToDevice));
            songshan3_fusion_clear();
            // Captured nodes own their chosen original plans/native launcher snapshot.
            for (auto &graph : graphs) {
                graph->run();
                verify({0x4100, 0x3f00, 0x4100, 0x4200}); // 2.5, 1.75, 2.5, 3.0
            }
            auto before = songshan3_fusion_count(5);
            adaln_(y[0], n, scale, shift);
            context::syncDevice();
            if (songshan3_fusion_count(5) != before + 3) throw std::runtime_error("clear fallback failed");
        }
        check(reinterpret_cast<hipError_t (*)()>(symbol(a, "h3_adaln3d_float16_triton_release"))());
        check(reinterpret_cast<hipError_t (*)()>(symbol(r, "h3_residual3d_float16_triton_release"))());
        for (auto p : memory) check(hipFree(p));
        std::cout << "{\"standalone_fusion_passed\":true,\"dtype\":\"float16\",\"shape\":[1,7,5376],"
                     "\"modulation_stride\":[225792,32256,1],\"output_stride\":[75264,10752,2],"
                     "\"original_calls_per_subgraph\":10,\"fused_calls_per_subgraph\":4,"
                     "\"changed_input_replays\":2,\"captured_launcher_survives_clear\":true,"
                     "\"allocating_api_cases\":2,"
                     "\"clear_uses_original_fallback\":true,\"output_guards\":true,\"bitwise\":true,"
                     "\"python_interpreter\":false,\"scope\":\"synthetic coupled pointwise subgraph\"}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
