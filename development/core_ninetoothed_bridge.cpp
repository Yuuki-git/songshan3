// Host-only InfiniCore dispatch adapter. All device computation is generated
// from NineToothed applications and supplied through their public native ABI.
#include <infinicore/context/context.hpp>
#include <infinicore/ops/add.hpp>
#include <infinicore/ops/rms_norm.hpp>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>

using namespace infinicore;
struct NativeTensor { void *data; uint64_t *shape; int64_t *strides; };
using AddFn = int (*)(void *, NativeTensor, NativeTensor, NativeTensor);
using RmsFn = int (*)(void *, NativeTensor, NativeTensor, NativeTensor, NativeTensor);

static NativeTensor view(const Tensor &tensor) {
    static_assert(sizeof(Size) == sizeof(uint64_t) && sizeof(Stride) == sizeof(int64_t));
    return {const_cast<void *>(static_cast<const void *>(tensor->data())),
            const_cast<uint64_t *>(tensor->shape().data()),
            const_cast<int64_t *>(tensor->strides().data())};
}
static void check(int result) {
    if (result) throw std::runtime_error("NineToothed HIP launcher failed: " + std::to_string(result));
}
static std::atomic<uint64_t> add_calls{0}, rms_calls{0}, add_fallbacks{0}, rms_fallbacks{0};

namespace add_adapter {
static AddFn functions[3]{};
static op::Add::plan_schema base_plan = nullptr;
static void (*base_run)(void *) = nullptr;
static void (*base_cleanup)(void **) = nullptr;
struct Planned {
    graph::GraphTensor out, a, b;
    AddFn function;
    void *fallback;
};
void *plan(Tensor out, const Tensor &a, const Tensor &b) {
    int index = a->dtype() == DataType::F32 ? 0 : a->dtype() == DataType::F16 ? 1 : a->dtype() == DataType::BF16 ? 2 : -1;
    bool supported = index >= 0 && functions[index] && a->ndim() == 1 && a->shape() == b->shape()
        && a->shape() == out->shape() && a->dtype() == b->dtype() && a->dtype() == out->dtype();
    return new Planned{graph::GraphTensor(out), graph::GraphTensor(a), graph::GraphTensor(b),
                       supported ? functions[index] : nullptr,
                       supported ? nullptr : base_plan(out, a, b)};
}
void run(void *metadata) {
    auto *p = static_cast<Planned *>(metadata);
    if (p->function) {
        check(p->function(context::getStream(), view(p->a), view(p->b), view(p->out)));
        ++add_calls;
    } else { base_run(p->fallback); ++add_fallbacks; }
}
void cleanup(void **metadata) {
    auto *p = static_cast<Planned *>(*metadata);
    if (!p->function) base_cleanup(&p->fallback);
    delete p;
    *metadata = nullptr;
}
}

namespace rms_adapter {
static RmsFn functions[5]{};
static op::RMSNorm::plan_schema base_plan = nullptr;
static void (*base_run)(void *) = nullptr;
static void (*base_cleanup)(void **) = nullptr;
struct Planned {
    graph::GraphTensor out, x, weight;
    double epsilon;
    RmsFn function;
    void *fallback;
};
void *plan(Tensor out, const Tensor &x, const Tensor &weight, float epsilon) {
    int index = -1;
    if (x->dtype() == DataType::F16) index = weight->dtype() == DataType::F16 ? 0 : weight->dtype() == DataType::F32 ? 1 : -1;
    if (x->dtype() == DataType::BF16) index = weight->dtype() == DataType::BF16 ? 2 : weight->dtype() == DataType::F32 ? 3 : -1;
    if (x->dtype() == DataType::F32 && weight->dtype() == DataType::F32) index = 4;
    bool supported = index >= 0 && functions[index] && x->ndim() == 2 && x->size(1) == 128
        && weight->ndim() == 1 && weight->size(0) == 128 && out->shape() == x->shape()
        && out->dtype() == x->dtype() && std::isfinite(epsilon) && epsilon > 0;
    return new Planned{graph::GraphTensor(out), graph::GraphTensor(x), graph::GraphTensor(weight),
                       epsilon, supported ? functions[index] : nullptr,
                       supported ? nullptr : base_plan(out, x, weight, epsilon)};
}
void run(void *metadata) {
    auto *p = static_cast<Planned *>(metadata);
    if (p->function) {
        NativeTensor epsilon{&p->epsilon, nullptr, nullptr};
        check(p->function(context::getStream(), view(p->x), view(p->weight), epsilon, view(p->out)));
        ++rms_calls;
    } else { base_run(p->fallback); ++rms_fallbacks; }
}
void cleanup(void **metadata) {
    auto *p = static_cast<Planned *>(*metadata);
    if (!p->function) base_cleanup(&p->fallback);
    delete p;
    *metadata = nullptr;
}
}

extern "C" void songshan3_register_add(void *fp32, void *fp16, void *bf16) {
    using namespace add_adapter;
    auto device = Device::Type::HYGON;
    if (!base_plan) {
        base_plan = op::Add::plan_dispatcher().lookup(device);
        base_run = op::Add::run_dispatcher().lookup(device);
        base_cleanup = op::Add::cleanup_dispatcher().lookup(device);
    }
    functions[0] = reinterpret_cast<AddFn>(fp32);
    functions[1] = reinterpret_cast<AddFn>(fp16);
    functions[2] = reinterpret_cast<AddFn>(bf16);
    op::Add::plan_dispatcher().registerDevice(device, &plan);
    op::Add::run_dispatcher().registerDevice(device, &run);
    op::Add::cleanup_dispatcher().registerDevice(device, &cleanup);
}
extern "C" void songshan3_register_rms(void *f16_f16, void *f16_f32, void *bf16_bf16, void *bf16_f32, void *f32_f32) {
    using namespace rms_adapter;
    auto device = Device::Type::HYGON;
    if (!base_plan) {
        base_plan = op::RMSNorm::plan_dispatcher().lookup(device);
        base_run = op::RMSNorm::run_dispatcher().lookup(device);
        base_cleanup = op::RMSNorm::cleanup_dispatcher().lookup(device);
    }
    void *values[] = {f16_f16, f16_f32, bf16_bf16, bf16_f32, f32_f32};
    for (int i = 0; i < 5; ++i) functions[i] = reinterpret_cast<RmsFn>(values[i]);
    op::RMSNorm::plan_dispatcher().registerDevice(device, &plan);
    op::RMSNorm::run_dispatcher().registerDevice(device, &run);
    op::RMSNorm::cleanup_dispatcher().registerDevice(device, &cleanup);
}
extern "C" void songshan3_restore_dispatch(void) {
    auto device = Device::Type::HYGON;
    if (add_adapter::base_plan) {
        op::Add::plan_dispatcher().registerDevice(device, add_adapter::base_plan);
        op::Add::run_dispatcher().registerDevice(device, add_adapter::base_run);
        op::Add::cleanup_dispatcher().registerDevice(device, add_adapter::base_cleanup);
    }
    if (rms_adapter::base_plan) {
        op::RMSNorm::plan_dispatcher().registerDevice(device, rms_adapter::base_plan);
        op::RMSNorm::run_dispatcher().registerDevice(device, rms_adapter::base_run);
        op::RMSNorm::cleanup_dispatcher().registerDevice(device, rms_adapter::base_cleanup);
    }
}
extern "C" uint64_t songshan3_call_count(int index) {
    switch (index) {
        case 0: return add_calls.load(); case 1: return rms_calls.load();
        case 2: return add_fallbacks.load(); default: return rms_fallbacks.load();
    }
}
extern "C" void *songshan3_dispatch_address(int index) {
    auto device = Device::Type::HYGON;
    return index == 0 ? reinterpret_cast<void *>(op::Add::run_dispatcher().lookup(device))
                      : reinterpret_cast<void *>(op::RMSNorm::run_dispatcher().lookup(device));
}
extern "C" void *songshan3_current_stream(void) { return context::getStream(); }
