#pragma once
#include <infinicore/tensor.hpp>
#include <cstdint>

namespace songshan3::fusion {
void adaln_(infinicore::Tensor out, const infinicore::Tensor &x,
            const infinicore::Tensor &scale, const infinicore::Tensor &shift,
            bool original = false, infinicore::Tensor scratch0 = {}, infinicore::Tensor scratch1 = {});
void residual_(infinicore::Tensor out, const infinicore::Tensor &x,
               const infinicore::Tensor &gate, const infinicore::Tensor &branch,
               bool original = false, infinicore::Tensor scratch0 = {});
infinicore::Tensor adaln(const infinicore::Tensor &x, const infinicore::Tensor &scale,
                        const infinicore::Tensor &shift, bool original = false);
infinicore::Tensor residual(const infinicore::Tensor &x, const infinicore::Tensor &gate,
                           const infinicore::Tensor &branch, bool original = false);
}

// Host descriptor for the C ABI. dtype: 0=FP16, 1=BF16, 2=FP32.
struct Songshan3FusionTensor {
    void *data;
    uint64_t ndim;
    const uint64_t *shape;
    const int64_t *strides;
    int dtype;
};
extern "C" {
void songshan3_fusion_register(void *adaln_fp16, void *adaln_bf16, void *residual_fp16, void *residual_bf16);
void songshan3_fusion_clear();
uint64_t songshan3_fusion_count(int index);
void *songshan3_fusion_stream();
const char *songshan3_fusion_error();
int songshan3_fusion_call(int op, Songshan3FusionTensor out, Songshan3FusionTensor x,
                         Songshan3FusionTensor a, Songshan3FusionTensor b,
                         Songshan3FusionTensor scratch0, Songshan3FusionTensor scratch1, int original);
}
