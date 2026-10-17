// Standalone C++ verification. All GPU computation comes from NineToothed.
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <infinicore/context/context.hpp>
#include <infinicore/ops/rms_norm.hpp>
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(hipError_t error) {
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}
static void *open_library(const char *path) {
    auto result = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!result) throw std::runtime_error(dlerror());
    return result;
}
static void *symbol(void *library, const std::string &name) {
    auto result = dlsym(library, name.c_str());
    if (!result) throw std::runtime_error("Missing symbol: " + name);
    return result;
}
int main(int argc, char **argv) {
    if (argc != 4) return 2;
    try {
        using namespace infinicore;
        auto bridge = open_library(argv[1]), native = open_library(argv[2]);
        const std::string name = argv[3];
        auto install = reinterpret_cast<void (*)(void *, void *, void *, void *, void *)>(symbol(bridge, "songshan3_register_rms3d"));
        auto restore = reinterpret_cast<void (*)()>(symbol(bridge, "songshan3_restore_dispatch"));
        auto count = reinterpret_cast<uint64_t (*)(int)>(symbol(bridge, "songshan3_rms3d_count"));
        auto dispatch = reinterpret_cast<void *(*)(int)>(symbol(bridge, "songshan3_dispatch_address"));
        const auto original_dispatch = dispatch(1);
        auto launcher = symbol(native, "launch_" + name + "_variant");
        install(nullptr, nullptr, nullptr, nullptr, launcher);
        if (dispatch(1) == original_dispatch) return 3;
        context::setDevice(Device(Device::Type::HYGON, 0));
        constexpr size_t tokens = 7, heads = 56, width = 128;
        constexpr size_t input_stride = 3 * heads * width, output_stride = 2 * heads * 256;
        constexpr float guard = -777.0f, epsilon = 1e-5f;
        std::vector<float> input(tokens * input_stride), weight(256), output(tokens * output_stride, guard);
        for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 487) - 243) / 64.0f;
        for (size_t i = 0; i < weight.size(); ++i) weight[i] = float(int(i % 31) - 15) / 8.0f;
        void *dx = nullptr, *dw = nullptr, *dy = nullptr;
        check(hipMalloc(&dx, input.size() * sizeof(float)));
        check(hipMalloc(&dw, weight.size() * sizeof(float)));
        check(hipMalloc(&dy, output.size() * sizeof(float)));
        check(hipMemcpy(dx, input.data(), input.size() * sizeof(float), hipMemcpyHostToDevice));
        check(hipMemcpy(dw, weight.data(), weight.size() * sizeof(float), hipMemcpyHostToDevice));
        check(hipMemcpy(dy, output.data(), output.size() * sizeof(float), hipMemcpyHostToDevice));
        {
            auto device = Device(Device::Type::HYGON, 0);
            auto x = Tensor::strided_from_blob(static_cast<float *>(dx) + heads * width,
                {tokens, heads, width}, {input_stride, width, 1}, DataType::F32, device);
            auto w = Tensor::strided_from_blob(dw, {width}, {2}, DataType::F32, device);
            auto y = Tensor::strided_from_blob(dy, {tokens, heads, width},
                {output_stride, 512, 2}, DataType::F32, device);
            op::rms_norm_(y, x, w, epsilon);
            context::syncDevice();
        }
        check(hipMemcpy(output.data(), dy, output.size() * sizeof(float), hipMemcpyDeviceToHost));
        std::vector<bool> touched(output.size());
        double maximum_error = 0;
        for (size_t token = 0; token < tokens; ++token) {
            for (size_t head = 0; head < heads; ++head) {
                const size_t start = token * input_stride + heads * width + head * width;
                double mean_square = 0;
                for (size_t col = 0; col < width; ++col) mean_square += double(input[start + col]) * input[start + col] / width;
                const double scale = 1.0 / std::sqrt(mean_square + double(epsilon));
                for (size_t col = 0; col < width; ++col) {
                    const size_t offset = token * output_stride + head * 512 + col * 2;
                    const double expected = double(input[start + col]) * scale * double(weight[2 * col]);
                    const double error = std::abs(double(output[offset]) - expected);
                    if (!std::isfinite(output[offset]) || error > 1e-5 + 1e-4 * std::abs(expected)) return 4;
                    maximum_error = std::max(maximum_error, error);
                    touched[offset] = true;
                }
            }
        }
        for (size_t i = 0; i < output.size(); ++i) if (!touched[i] && output[i] != guard) return 5;
        if (count(0) != 1 || count(1) != 0) return 6;
        restore();
        if (dispatch(1) != original_dispatch) return 7;
        auto release = reinterpret_cast<hipError_t (*)()>(symbol(native, name + "_triton_release"));
        check(release());
        check(hipFree(dx)); check(hipFree(dw)); check(hipFree(dy));
        std::cout << "{\"standalone_rms3d_passed\":true,\"dtype\":\"float32\",\"shape\":[7,56,128],"
                     "\"input_stride\":[21504,128,1],\"output_stride\":[28672,512,2],\"weight_stride\":[2],"
                     "\"native_calls\":1,\"fallbacks\":0,\"output_guards\":true,\"restored\":true,"
                     "\"python_interpreter\":false,\"max_abs_error\":" << maximum_error << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
