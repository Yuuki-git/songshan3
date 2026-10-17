// Host-only Core graph operators. Every new GPU computation is NineToothed AOT.
#include "core_h3_fusion.hpp"
#include <infinicore/context/context.hpp>
#include <infinicore/ops/add.hpp>
#include <infinicore/ops/mul.hpp>
#include <array>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace songshan3::fusion {
using namespace infinicore;
struct NativeTensor { void *data; uint64_t *shape; int64_t *strides; };
using NativeFn = int (*)(void *, NativeTensor, NativeTensor, NativeTensor, NativeTensor);
static NativeFn functions[4]{};
// native AdaLN/residual, original AdaLN/residual, empty, original child runs, nodes.
static std::array<std::atomic<uint64_t>, 7> counts{};
static NativeTensor view(const Tensor &t) {
    static_assert(sizeof(Size) == sizeof(uint64_t) && sizeof(Stride) == sizeof(int64_t));
    return {const_cast<void *>(static_cast<const void *>(t->data())),
            const_cast<uint64_t *>(t->shape().data()), const_cast<int64_t *>(t->strides().data())};
}
static bool identical(const Tensor &a, const Tensor &b) {
    return a->data() == b->data() && a->shape() == b->shape() && a->strides() == b->strides() && a->dtype() == b->dtype();
}
static std::pair<intptr_t, intptr_t> bounds(const Tensor &t) {
    intptr_t low = 0, high = 0;
    for (size_t i = 0; i < t->ndim(); ++i) {
        const intptr_t extent = static_cast<intptr_t>(t->size(i) - 1) * t->stride(i);
        low += std::min(intptr_t(0), extent);
        high += std::max(intptr_t(0), extent);
    }
    const auto base = reinterpret_cast<intptr_t>(t->data());
    return {base + low * dsize(t->dtype()), base + (high + 1) * dsize(t->dtype())};
}
static bool overlap(const Tensor &a, const Tensor &b) {
    if (!a->numel() || !b->numel()) return false;
    const auto x = bounds(a), y = bounds(b);
    return x.first < y.second && y.first < x.second;
}
static bool native_layout(const Tensor &t) {
    return t->ndim() == 3 && t->size(2) == 5376 && t->size(0) && t->size(1)
        && t->stride(2) > 0 && t->stride(1) >= 5376 * t->stride(2)
        && t->stride(0) >= static_cast<Stride>(t->size(1)) * t->stride(1);
}

class PointwiseNode final : public graph::GraphOperator {
    int kind_;
    graph::GraphTensor out_, x_, a_, b_;
    Tensor scratch0_, scratch1_;
    NativeFn function_ = nullptr;
    bool empty_;
    std::vector<std::shared_ptr<graph::GraphOperator>> children_;
public:
    PointwiseNode(int kind, Tensor out, const Tensor &x, const Tensor &a, const Tensor &b,
                  bool original, Tensor scratch0, Tensor scratch1)
        : kind_(kind), out_(out), x_(x), a_(a), b_(b), empty_(out->numel() == 0) {
        if (out->shape() != x->shape() || out->dtype() != x->dtype()
            || a->dtype() != x->dtype() || b->dtype() != x->dtype()
            || out->device() != x->device() || out->device() != a->device() || out->device() != b->device())
            throw std::invalid_argument("fusion tensors must have matching dtypes/devices and output/input shape");
        if (empty_) return;
        for (const auto &input : {x, a, b})
            if (!identical(out, input) && overlap(out, input))
                throw std::invalid_argument("partially overlapping output views are unsupported; use a separate output");
        const int dtype = x->dtype() == DataType::F16 ? 0 : x->dtype() == DataType::BF16 ? 1 : -1;
        if (!original && dtype >= 0 && x->device().getType() == Device::Type::HYGON
            && a->shape() == x->shape() && b->shape() == x->shape()
            && native_layout(out) && native_layout(x) && native_layout(a) && native_layout(b))
            function_ = functions[2 * kind + dtype];
        if (function_) return;
        auto workspace = [&](Tensor supplied) {
            auto t = supplied ? supplied : Tensor::empty(out->shape(), out->dtype(), out->device());
            if (t->shape() != out->shape() || t->dtype() != out->dtype() || t->device() != out->device())
                throw std::invalid_argument("fusion scratch shape/dtype/device mismatch");
            for (const auto &input : {out, x, a, b})
                if (overlap(t, input)) throw std::invalid_argument("fusion scratch must not overlap inputs/output");
            return t;
        };
        scratch0_ = workspace(scratch0);
        if (kind == 0) {
            scratch1_ = workspace(scratch1);
            if (overlap(scratch0_, scratch1_)) throw std::invalid_argument("fusion scratch tensors overlap");
            children_.push_back(std::make_shared<op::Mul>(scratch0_, x, a));
            children_.push_back(std::make_shared<op::Add>(scratch1_, x, scratch0_));
            children_.push_back(std::make_shared<op::Add>(out, scratch1_, b));
        } else {
            children_.push_back(std::make_shared<op::Mul>(scratch0_, a, b));
            children_.push_back(std::make_shared<op::Add>(out, x, scratch0_));
        }
    }
    void run() const override {
        if (context::getDevice() != out_->device()) context::setDevice(out_->device());
        ++counts[6];
        if (empty_) { ++counts[4]; return; }
        if (function_) {
            const int status = function_(context::getStream(), view(x_), view(a_), view(b_), view(out_));
            if (status) throw std::runtime_error("NineToothed fusion launcher failed: " + std::to_string(status));
            ++counts[kind_];
        } else {
            for (const auto &child : children_) child->run();
            ++counts[2 + kind_];
            counts[5] += children_.size();
        }
    }
};
static void execute(int kind, Tensor out, const Tensor &x, const Tensor &a, const Tensor &b,
                    bool original, Tensor scratch0, Tensor scratch1) {
    auto node = std::make_shared<PointwiseNode>(kind, out, x, a, b, original, scratch0, scratch1);
    if (context::isGraphRecording()) context::addGraphOperator(node);
    else node->run();
}
void adaln_(Tensor out, const Tensor &x, const Tensor &scale, const Tensor &shift,
            bool original, Tensor scratch0, Tensor scratch1) {
    execute(0, out, x, scale, shift, original, scratch0, scratch1);
}
void residual_(Tensor out, const Tensor &x, const Tensor &gate, const Tensor &branch,
               bool original, Tensor scratch0) {
    execute(1, out, x, gate, branch, original, scratch0, {});
}
Tensor adaln(const Tensor &x, const Tensor &scale, const Tensor &shift, bool original) {
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    adaln_(out, x, scale, shift, original);
    return out;
}
Tensor residual(const Tensor &x, const Tensor &gate, const Tensor &branch, bool original) {
    auto out = Tensor::empty(x->shape(), x->dtype(), x->device());
    residual_(out, x, gate, branch, original);
    return out;
}
}

namespace {
thread_local std::string last_error;
infinicore::Tensor tensor(Songshan3FusionTensor raw, bool optional = false) {
    using namespace infinicore;
    if (optional && !raw.shape && raw.ndim == 0) return {};
    if (raw.ndim > 8 || (raw.ndim && (!raw.shape || !raw.strides)))
        throw std::invalid_argument("invalid fusion tensor descriptor");
    const DataType dtype = raw.dtype == 0 ? DataType::F16 : raw.dtype == 1 ? DataType::BF16
        : raw.dtype == 2 ? DataType::F32 : throw std::invalid_argument("unsupported fusion dtype");
    Shape shape(raw.ndim);
    Strides strides(raw.ndim);
    for (size_t i = 0; i < raw.ndim; ++i) { shape[i] = raw.shape[i]; strides[i] = raw.strides[i]; }
    return Tensor::strided_from_blob(raw.data, shape, strides, dtype, Device(Device::Type::HYGON, 0));
}
}
extern "C" void songshan3_fusion_register(void *a16, void *abf, void *r16, void *rbf) {
    using namespace songshan3::fusion;
    void *pointers[] = {a16, abf, r16, rbf};
    for (int i = 0; i < 4; ++i) functions[i] = reinterpret_cast<NativeFn>(pointers[i]);
}
extern "C" void songshan3_fusion_clear() {
    for (auto &function : songshan3::fusion::functions) function = nullptr;
}
extern "C" uint64_t songshan3_fusion_count(int index) {
    return index >= 0 && index < 7 ? songshan3::fusion::counts[index].load() : 0;
}
extern "C" void *songshan3_fusion_stream() { return infinicore::context::getStream(); }
extern "C" const char *songshan3_fusion_error() { return last_error.c_str(); }
extern "C" int songshan3_fusion_call(int op, Songshan3FusionTensor out, Songshan3FusionTensor x,
                                    Songshan3FusionTensor a, Songshan3FusionTensor b,
                                    Songshan3FusionTensor scratch0, Songshan3FusionTensor scratch1, int original) {
    try {
        if (op < 0 || op > 1) throw std::invalid_argument("invalid fusion op");
        auto cy = tensor(out), cx = tensor(x), ca = tensor(a), cb = tensor(b);
        if (op == 0) songshan3::fusion::adaln_(cy, cx, ca, cb, original != 0, tensor(scratch0, true), tensor(scratch1, true));
        else songshan3::fusion::residual_(cy, cx, ca, cb, original != 0, tensor(scratch0, true));
        last_error.clear();
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return 1;
    }
}
