// Copyright (c) 2024-2026 Sarkar-AGI
// SPDX-License-Identifier: MIT
//
// GoTorch — Go binding for the PyTorch C++ engine (libtorch).
// PyTorch is Copyright (c) Facebook, Inc. and its affiliates,
// licensed under the BSD 3-Clause License. See LICENSE-PYTORCH.
//
// Package gotorch is the single-import entry point for GoTorch.
//
// Python:                              Go:
//   import torch                   →   import gt "github.com/Sarkar-AGI/GoTorch"
//   import torch.nn as nn          →   (included)
//   import torch.optim as optim    →   (included)
//
// Usage:
//   package main
//
//   import (
//       "fmt"
//       gt "github.com/Sarkar-AGI/GoTorch"
//   )
//
//   func main() {
//       model := gt.NewSequential(
//           gt.NewLinear(784, 256, true),
//           gt.NewReLU(),
//           gt.NewLinear(256, 10, true),
//       )
//       x    := gt.Randn(32, 784)
//       out  := model.Forward(x)
//       opt  := gt.NewAdam(model.Parameters(), gt.AdamOptions{LR: 1e-3})
//       loss := gt.CrossEntropyLoss(out, gt.Zeros(32), gt.ReduceMean)
//       loss.Backward()
//       opt.Step()
//       fmt.Println(gt.Version())
//   }
//
// Build prerequisites:
//   1. cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=OFF -DBUILD_TEST=OFF
//   2. cmake --build build --target torch torch_cpu c10 -j$(nproc)
//   3. export GOTORCH=$(pwd)
//   4. go build github.com/Sarkar-AGI/GoTorch

package gotorch

/*
#cgo CFLAGS:  -I${SRCDIR}/torch/csrc/api/include -I${SRCDIR}/csrc/go_binding
#cgo LDFLAGS: -L${SRCDIR}/build/lib -Wl,-rpath,${SRCDIR}/build/lib -ltorch -ltorch_cpu -lc10 -lstdc++
#include "csrc/go_binding/torch_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"math"
	"unsafe"
)

// ═══════════════════════════════════════════════════════════════════════════════
// TENSOR
// ═══════════════════════════════════════════════════════════════════════════════

// Tensor wraps a libtorch tensor pointer.
// Mirrors: torch.Tensor
type Tensor struct{ ptr C.Tensor }

func wrap(p C.Tensor) *Tensor { return &Tensor{ptr: p} }

// RawPtr returns the raw C handle (for advanced use).
func (t *Tensor) RawPtr() C.Tensor { return t.ptr }

// Free releases the underlying C++ tensor. Optional — GC will not call this.
func (t *Tensor) Free() { C.gotorch_tensor_free(t.ptr) }

// ─── shape helpers ────────────────────────────────────────────────────────────

func cshape(shape []int) (*C.int64_t, C.int) {
	if len(shape) == 0 {
		return nil, 0
	}
	cs := make([]C.int64_t, len(shape))
	for i, s := range shape {
		cs[i] = C.int64_t(s)
	}
	return &cs[0], C.int(len(cs))
}

// ─── dtype & device constants ─────────────────────────────────────────────────

type DType  int
type Device int

const (
	Float32 DType = C.DTYPE_FLOAT32
	Float64 DType = C.DTYPE_FLOAT64
	Int32   DType = C.DTYPE_INT32
	Int64   DType = C.DTYPE_INT64
	Bool    DType = C.DTYPE_BOOL
)

const (
	CPU  Device = C.DEVICE_CPU
	CUDA Device = C.DEVICE_CUDA
	MPS  Device = C.DEVICE_MPS
)

// Reduction mirrors torch's reduction enum.
const (
	ReduceNone = 0
	ReduceMean = 1
	ReduceSum  = 2
)

// ═══════════════════════════════════════════════════════════════════════════════
// TENSOR CONSTRUCTORS
// ═══════════════════════════════════════════════════════════════════════════════

// Zeros returns a zero tensor. Mirrors: torch.zeros(*size)
func Zeros(shape ...int) *Tensor {
	s, n := cshape(shape)
	return wrap(C.gotorch_zeros(s, n, C.int(Float32), C.int(CPU)))
}

// Ones returns an ones tensor. Mirrors: torch.ones(*size)
func Ones(shape ...int) *Tensor {
	s, n := cshape(shape)
	return wrap(C.gotorch_ones(s, n, C.int(Float32), C.int(CPU)))
}

// Randn returns standard-normal samples. Mirrors: torch.randn(*size)
func Randn(shape ...int) *Tensor {
	s, n := cshape(shape)
	return wrap(C.gotorch_randn(s, n, C.int(Float32), C.int(CPU)))
}

// Rand returns uniform [0,1) samples. Mirrors: torch.rand(*size)
func Rand(shape ...int) *Tensor {
	s, n := cshape(shape)
	return wrap(C.gotorch_rand(s, n, C.int(Float32), C.int(CPU)))
}

// Full returns a tensor filled with v. Mirrors: torch.full(size, v)
func Full(v float64, shape ...int) *Tensor {
	s, n := cshape(shape)
	return wrap(C.gotorch_full(s, n, C.double(v), C.int(Float32), C.int(CPU)))
}

// Eye returns an n×n identity matrix. Mirrors: torch.eye(n)
func Eye(n int) *Tensor {
	return wrap(C.gotorch_eye(C.int64_t(n), C.int(Float32), C.int(CPU)))
}

// Arange returns evenly spaced values. Mirrors: torch.arange(start, end, step)
func Arange(start, end, step float64) *Tensor {
	return wrap(C.gotorch_arange(C.double(start), C.double(end), C.double(step), C.int(Float32), C.int(CPU)))
}

// Linspace returns n evenly spaced points. Mirrors: torch.linspace(start, end, steps)
func Linspace(start, end float64, steps int) *Tensor {
	return wrap(C.gotorch_linspace(C.double(start), C.double(end), C.int64_t(steps), C.int(Float32), C.int(CPU)))
}

// FromData creates a tensor from a Go float64 slice. Mirrors: torch.tensor(data)
func FromData(data []float64, shape ...int) *Tensor {
	s, n := cshape(shape)
	return wrap(C.gotorch_from_data((*C.double)(&data[0]), s, n, C.int(Float32), C.int(CPU)))
}

// ZerosLike returns a zero tensor shaped like t.
func ZerosLike(t *Tensor) *Tensor { return Zeros(t.Shape()...) }

// OnesLike returns an ones tensor shaped like t.
func OnesLike(t *Tensor) *Tensor { return Ones(t.Shape()...) }

// ZerosOnDevice creates zeros on a specific device/dtype.
func ZerosOnDevice(dtype DType, device Device, shape ...int) *Tensor {
	s, n := cshape(shape)
	return wrap(C.gotorch_zeros(s, n, C.int(dtype), C.int(device)))
}

// ═══════════════════════════════════════════════════════════════════════════════
// TENSOR PROPERTIES
// ═══════════════════════════════════════════════════════════════════════════════

// Shape returns the tensor dimensions. Mirrors: tensor.shape
func (t *Tensor) Shape() []int {
	ndim := int(C.gotorch_tensor_ndim(t.ptr))
	if ndim == 0 {
		return []int{}
	}
	raw := C.gotorch_tensor_shape(t.ptr)
	defer C.free(unsafe.Pointer(raw))
	out := make([]int, ndim)
	sl := (*[1 << 30]C.int64_t)(unsafe.Pointer(raw))[:ndim:ndim]
	for i, s := range sl {
		out[i] = int(s)
	}
	return out
}

// Ndim returns the number of dimensions. Mirrors: tensor.ndim
func (t *Tensor) Ndim() int { return int(C.gotorch_tensor_ndim(t.ptr)) }

// Numel returns total number of elements. Mirrors: tensor.numel()
func (t *Tensor) Numel() int { return int(C.gotorch_tensor_numel(t.ptr)) }

// Item returns the scalar value (for 0-d tensors). Mirrors: tensor.item()
func (t *Tensor) Item() float64 { return float64(C.gotorch_tensor_item(t.ptr)) }

// Print prints the tensor to stdout. Mirrors: print(tensor)
func (t *Tensor) Print() { C.gotorch_tensor_print(t.ptr) }

// String implements fmt.Stringer.
func (t *Tensor) String() string { return fmt.Sprintf("Tensor(shape=%v)", t.Shape()) }

// RequiresGrad returns whether gradient tracking is enabled.
func (t *Tensor) RequiresGrad() bool { return C.gotorch_tensor_requires_grad(t.ptr) != 0 }

// SetRequiresGrad enables or disables gradient tracking.
func (t *Tensor) SetRequiresGrad(v bool) {
	iv := C.int(0)
	if v { iv = C.int(1) }
	C.gotorch_tensor_set_requires_grad(t.ptr, iv)
}

// Grad returns the gradient tensor (nil if none).
func (t *Tensor) Grad() *Tensor {
	g := C.gotorch_tensor_grad(t.ptr)
	if g == nil { return nil }
	return wrap(g)
}

// ZeroGrad zeroes the gradient of this tensor in-place.
func (t *Tensor) ZeroGrad() { C.gotorch_tensor_zero_grad(t.ptr) }

// To moves the tensor to the given device. Mirrors: tensor.to(device)
func (t *Tensor) To(device Device) *Tensor {
	return wrap(C.gotorch_tensor_to_device(t.ptr, C.int(device)))
}

// Cast casts the tensor to a different dtype. Mirrors: tensor.to(dtype)
func (t *Tensor) Cast(dtype DType) *Tensor {
	return wrap(C.gotorch_tensor_to_dtype(t.ptr, C.int(dtype)))
}

// ═══════════════════════════════════════════════════════════════════════════════
// SHAPE OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════════

// Reshape returns a tensor with the given shape. Mirrors: tensor.reshape(*shape)
func (t *Tensor) Reshape(shape ...int) *Tensor {
	s, n := cshape(shape)
	return wrap(C.gotorch_reshape(t.ptr, s, n))
}

// View is an alias for Reshape. Mirrors: tensor.view(*shape)
func (t *Tensor) View(shape ...int) *Tensor { return t.Reshape(shape...) }

// Flatten collapses dims [startDim, endDim]. Mirrors: tensor.flatten(start_dim, end_dim)
func (t *Tensor) Flatten(startDim, endDim int) *Tensor {
	return wrap(C.gotorch_flatten(t.ptr, C.int64_t(startDim), C.int64_t(endDim)))
}

// FlattenAll collapses all dims. Mirrors: tensor.flatten()
func (t *Tensor) FlattenAll() *Tensor { return t.Flatten(0, -1) }

// Unflatten expands one dim. Mirrors: tensor.unflatten(dim, sizes)
func (t *Tensor) Unflatten(dim int, sizes []int) *Tensor {
	s, n := cshape(sizes)
	return wrap(C.gotorch_unflatten(t.ptr, C.int64_t(dim), s, n))
}

// Transpose swaps two dims. Mirrors: tensor.transpose(dim0, dim1)
func (t *Tensor) Transpose(dim0, dim1 int) *Tensor {
	return wrap(C.gotorch_transpose(t.ptr, C.int64_t(dim0), C.int64_t(dim1)))
}

// Permute reorders all dims. Mirrors: tensor.permute(*dims)
func (t *Tensor) Permute(dims ...int) *Tensor {
	s, n := cshape(dims)
	return wrap(C.gotorch_permute(t.ptr, s, n))
}

// T transposes a 2-D tensor. Mirrors: tensor.T
func (t *Tensor) T() *Tensor { return wrap(C.gotorch_t(t.ptr)) }

// Squeeze removes size-1 dimensions. Mirrors: tensor.squeeze()
func (t *Tensor) Squeeze() *Tensor { return wrap(C.gotorch_squeeze(t.ptr)) }

// SqueezeDim removes a specific size-1 dim. Mirrors: tensor.squeeze(dim)
func (t *Tensor) SqueezeDim(dim int) *Tensor {
	return wrap(C.gotorch_squeeze_dim(t.ptr, C.int64_t(dim)))
}

// Unsqueeze inserts a new size-1 dim. Mirrors: tensor.unsqueeze(dim)
func (t *Tensor) Unsqueeze(dim int) *Tensor {
	return wrap(C.gotorch_unsqueeze(t.ptr, C.int64_t(dim)))
}

// Contiguous returns a contiguous memory layout tensor.
func (t *Tensor) Contiguous() *Tensor { return wrap(C.gotorch_contiguous(t.ptr)) }

// Detach detaches from the computation graph. Mirrors: tensor.detach()
func (t *Tensor) Detach() *Tensor { return wrap(C.gotorch_detach(t.ptr)) }

// Clone deep-copies the tensor. Mirrors: tensor.clone()
func (t *Tensor) Clone() *Tensor { return wrap(C.gotorch_clone(t.ptr)) }

// Slice slices along a dim. Mirrors: tensor[start:end:step] along dim.
func (t *Tensor) Slice(dim int, start, end, step int64) *Tensor {
	return wrap(C.gotorch_slice(t.ptr, C.int64_t(dim), C.int64_t(start), C.int64_t(end), C.int64_t(step)))
}

// IndexSelect selects along dim using index tensor. Mirrors: torch.index_select
func (t *Tensor) IndexSelect(dim int, index *Tensor) *Tensor {
	return wrap(C.gotorch_index_select(t.ptr, C.int64_t(dim), index.ptr))
}

// ─── multi-tensor ops ─────────────────────────────────────────────────────────

func rawPtrs(tensors []*Tensor) []C.Tensor {
	r := make([]C.Tensor, len(tensors))
	for i, t := range tensors { r[i] = t.ptr }
	return r
}

// Cat concatenates tensors along dim. Mirrors: torch.cat(tensors, dim)
func Cat(tensors []*Tensor, dim int) *Tensor {
	r := rawPtrs(tensors)
	return wrap(C.gotorch_cat(&r[0], C.int64_t(len(r)), C.int64_t(dim)))
}

// Stack stacks tensors along a new dim. Mirrors: torch.stack(tensors, dim)
func Stack(tensors []*Tensor, dim int) *Tensor {
	r := rawPtrs(tensors)
	return wrap(C.gotorch_stack(&r[0], C.int64_t(len(r)), C.int64_t(dim)))
}

// ═══════════════════════════════════════════════════════════════════════════════
// ARITHMETIC
// ═══════════════════════════════════════════════════════════════════════════════

// Add adds two tensors. Mirrors: torch.add(a, b) / a + b
func Add(a, b *Tensor) *Tensor { return wrap(C.gotorch_add(a.ptr, b.ptr)) }

// Sub subtracts tensors. Mirrors: torch.sub(a, b)
func Sub(a, b *Tensor) *Tensor { return wrap(C.gotorch_sub(a.ptr, b.ptr)) }

// Mul element-wise multiply. Mirrors: torch.mul(a, b)
func Mul(a, b *Tensor) *Tensor { return wrap(C.gotorch_mul(a.ptr, b.ptr)) }

// Div element-wise divide. Mirrors: torch.div(a, b)
func Div(a, b *Tensor) *Tensor { return wrap(C.gotorch_div(a.ptr, b.ptr)) }

// MatMul matrix multiply. Mirrors: torch.matmul(a, b)
func MatMul(a, b *Tensor) *Tensor { return wrap(C.gotorch_matmul(a.ptr, b.ptr)) }

// MM 2-D matrix multiply. Mirrors: torch.mm(a, b)
func MM(a, b *Tensor) *Tensor { return wrap(C.gotorch_mm(a.ptr, b.ptr)) }

// BMM batched matrix multiply. Mirrors: torch.bmm(a, b)
func BMM(a, b *Tensor) *Tensor { return wrap(C.gotorch_bmm(a.ptr, b.ptr)) }

// Dot dot product. Mirrors: torch.dot(a, b)
func Dot(a, b *Tensor) *Tensor { return wrap(C.gotorch_dot(a.ptr, b.ptr)) }

// AddScalar adds a scalar. Mirrors: tensor + v
func (t *Tensor) AddScalar(v float64) *Tensor {
	return wrap(C.gotorch_add_scalar(t.ptr, C.double(v)))
}

// MulScalar scales by v. Mirrors: tensor * v
func (t *Tensor) MulScalar(v float64) *Tensor {
	return wrap(C.gotorch_mul_scalar(t.ptr, C.double(v)))
}

// Pow raises to power. Mirrors: tensor ** exp
func (t *Tensor) Pow(exp float64) *Tensor {
	return wrap(C.gotorch_pow(t.ptr, C.double(exp)))
}

// Neg negates. Mirrors: -tensor
func (t *Tensor) Neg() *Tensor   { return wrap(C.gotorch_neg(t.ptr)) }

// Abs absolute value. Mirrors: tensor.abs()
func (t *Tensor) Abs() *Tensor   { return wrap(C.gotorch_abs(t.ptr)) }

// Exp e^x. Mirrors: tensor.exp()
func (t *Tensor) Exp() *Tensor   { return wrap(C.gotorch_exp(t.ptr)) }

// Log natural log. Mirrors: tensor.log()
func (t *Tensor) Log() *Tensor   { return wrap(C.gotorch_log(t.ptr)) }

// Sqrt square root. Mirrors: tensor.sqrt()
func (t *Tensor) Sqrt() *Tensor  { return wrap(C.gotorch_sqrt(t.ptr)) }

// Clamp clamps values. Mirrors: tensor.clamp(min, max)
func (t *Tensor) Clamp(min, max float64) *Tensor {
	return wrap(C.gotorch_clamp(t.ptr, C.double(min), C.double(max)))
}

// ─── Reduction ────────────────────────────────────────────────────────────────

// Sum reduces all elements. Mirrors: tensor.sum()
func (t *Tensor) Sum() *Tensor { return wrap(C.gotorch_sum(t.ptr)) }

// SumDim reduces along dim. Mirrors: tensor.sum(dim, keepdim)
func (t *Tensor) SumDim(dim int, keepdim bool) *Tensor {
	kd := C.int(0)
	if keepdim { kd = 1 }
	return wrap(C.gotorch_sum_dim(t.ptr, C.int64_t(dim), kd))
}

// Mean reduces all. Mirrors: tensor.mean()
func (t *Tensor) Mean() *Tensor { return wrap(C.gotorch_mean(t.ptr)) }

// MeanDim reduces along dim. Mirrors: tensor.mean(dim, keepdim)
func (t *Tensor) MeanDim(dim int, keepdim bool) *Tensor {
	kd := C.int(0)
	if keepdim { kd = 1 }
	return wrap(C.gotorch_mean_dim(t.ptr, C.int64_t(dim), kd))
}

// Max global max. Mirrors: tensor.max()
func (t *Tensor) Max() *Tensor { return wrap(C.gotorch_max(t.ptr)) }

// Min global min.
func (t *Tensor) Min() *Tensor { return wrap(C.gotorch_min(t.ptr)) }

// Std standard deviation.
func (t *Tensor) Std() *Tensor { return wrap(C.gotorch_std(t.ptr)) }

// Var variance.
func (t *Tensor) Var() *Tensor { return wrap(C.gotorch_var(t.ptr)) }

// Argmax index of max along dim.
func (t *Tensor) Argmax(dim int, keepdim bool) *Tensor {
	kd := C.int(0)
	if keepdim { kd = 1 }
	return wrap(C.gotorch_argmax(t.ptr, C.int64_t(dim), kd))
}

// Argmin index of min along dim.
func (t *Tensor) Argmin(dim int, keepdim bool) *Tensor {
	kd := C.int(0)
	if keepdim { kd = 1 }
	return wrap(C.gotorch_argmin(t.ptr, C.int64_t(dim), kd))
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVATIONS (package-level, stateless)
// ═══════════════════════════════════════════════════════════════════════════════

func ReLU(t *Tensor) *Tensor                       { return wrap(C.gotorch_relu(t.ptr)) }
func LeakyReLU(t *Tensor, negSlope float64) *Tensor { return wrap(C.gotorch_leaky_relu(t.ptr, C.double(negSlope))) }
func Sigmoid(t *Tensor) *Tensor                    { return wrap(C.gotorch_sigmoid(t.ptr)) }
func TanhF(t *Tensor) *Tensor                      { return wrap(C.gotorch_tanh(t.ptr)) }
func Softmax(t *Tensor, dim int) *Tensor           { return wrap(C.gotorch_softmax(t.ptr, C.int64_t(dim))) }
func LogSoftmax(t *Tensor, dim int) *Tensor        { return wrap(C.gotorch_log_softmax(t.ptr, C.int64_t(dim))) }
func GELU(t *Tensor) *Tensor                       { return wrap(C.gotorch_gelu(t.ptr)) }
func SiLU(t *Tensor) *Tensor                       { return wrap(C.gotorch_silu(t.ptr)) }
func ELU(t *Tensor, alpha float64) *Tensor         { return wrap(C.gotorch_elu(t.ptr, C.double(alpha))) }
func SELU(t *Tensor) *Tensor                       { return wrap(C.gotorch_selu(t.ptr)) }
func Mish(t *Tensor) *Tensor                       { return wrap(C.gotorch_mish(t.ptr)) }
func Hardswish(t *Tensor) *Tensor                  { return wrap(C.gotorch_hardswish(t.ptr)) }

// ═══════════════════════════════════════════════════════════════════════════════
// LOSS FUNCTIONS (package-level, stateless)
// ═══════════════════════════════════════════════════════════════════════════════

func MSELoss(pred, target *Tensor, reduction int) *Tensor {
	return wrap(C.gotorch_mse_loss(pred.ptr, target.ptr, C.int(reduction)))
}
func CrossEntropyLoss(pred, target *Tensor, reduction int) *Tensor {
	return wrap(C.gotorch_cross_entropy(pred.ptr, target.ptr, C.int(reduction)))
}
func BCELoss(pred, target *Tensor, reduction int) *Tensor {
	return wrap(C.gotorch_bce_loss(pred.ptr, target.ptr, C.int(reduction)))
}
func BCEWithLogitsLoss(pred, target *Tensor, reduction int) *Tensor {
	return wrap(C.gotorch_bce_with_logits(pred.ptr, target.ptr, C.int(reduction)))
}
func NLLLoss(logProbs, target *Tensor, reduction int) *Tensor {
	return wrap(C.gotorch_nll_loss(logProbs.ptr, target.ptr, C.int(reduction)))
}
func L1Loss(pred, target *Tensor, reduction int) *Tensor {
	return wrap(C.gotorch_l1_loss(pred.ptr, target.ptr, C.int(reduction)))
}
func HuberLoss(pred, target *Tensor, delta float64, reduction int) *Tensor {
	return wrap(C.gotorch_huber_loss(pred.ptr, target.ptr, C.double(delta), C.int(reduction)))
}

// ═══════════════════════════════════════════════════════════════════════════════
// AUTOGRAD
// ═══════════════════════════════════════════════════════════════════════════════

// Backward computes gradients. Mirrors: tensor.backward()
func (t *Tensor) Backward() { C.gotorch_backward(t.ptr) }

// BackwardWithGrad computes gradients with external grad tensor.
func (t *Tensor) BackwardWithGrad(grad *Tensor) { C.gotorch_backward_with_grad(t.ptr, grad.ptr) }

// NoGrad disables gradient computation globally.
// Mirrors: torch.no_grad() — use defer EnableGrad() or use WithNoGrad.
func NoGrad() { C.gotorch_set_grad_enabled(0) }

// EnableGrad re-enables gradient computation.
func EnableGrad() { C.gotorch_set_grad_enabled(1) }

// IsGradEnabled returns whether grad is currently enabled.
func IsGradEnabled() bool { return C.gotorch_is_grad_enabled() != 0 }

// WithNoGrad runs fn with gradients disabled. Mirrors: with torch.no_grad():
func WithNoGrad(fn func()) {
	was := IsGradEnabled()
	NoGrad()
	defer func() {
		if was { EnableGrad() }
	}()
	fn()
}

// ═══════════════════════════════════════════════════════════════════════════════
// CUDA
// ═══════════════════════════════════════════════════════════════════════════════

// CUDAIsAvailable returns true if CUDA is present. Mirrors: torch.cuda.is_available()
func CUDAIsAvailable() bool { return C.gotorch_cuda_is_available() != 0 }

// CUDADeviceCount returns number of GPUs. Mirrors: torch.cuda.device_count()
func CUDADeviceCount() int { return int(C.gotorch_cuda_device_count()) }

// CUDASetDevice sets the active GPU. Mirrors: torch.cuda.set_device(id)
func CUDASetDevice(id int) { C.gotorch_cuda_set_device(C.int(id)) }

// CUDASynchronize blocks until all kernels finish. Mirrors: torch.cuda.synchronize()
func CUDASynchronize() { C.gotorch_cuda_synchronize() }

// ═══════════════════════════════════════════════════════════════════════════════
// MODULE INTERFACE
// ═══════════════════════════════════════════════════════════════════════════════

// Module is the base interface for all nn layers. Mirrors: torch.nn.Module
type Module interface {
	Forward(input *Tensor) *Tensor
	Parameters() []*Tensor
	ZeroGrad()
	Train()
	Eval()
	Name() string
}

// parameters helper — converts raw C pointer array to Go slice
func cParamsToSlice(data *C.Tensor, count C.int64_t) []*Tensor {
	n := int(count)
	if n == 0 || data == nil {
		return nil
	}
	sl := (*[1 << 20]C.Tensor)(unsafe.Pointer(data))[:n:n]
	params := make([]*Tensor, n)
	for i, p := range sl {
		params[i] = wrap(p)
	}
	return params
}

func cParamPtrs(params []*Tensor) []C.Tensor {
	r := make([]C.Tensor, len(params))
	for i, p := range params {
		r[i] = p.ptr
	}
	return r
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEQUENTIAL & MODULELIST
// ═══════════════════════════════════════════════════════════════════════════════

// Sequential chains modules in order. Mirrors: nn.Sequential(*layers)
type Sequential struct {
	layers   []Module
	training bool
}

// NewSequential creates a Sequential. Mirrors: nn.Sequential(layer1, layer2, ...)
func NewSequential(layers ...Module) *Sequential {
	return &Sequential{layers: layers, training: true}
}

// Add appends a layer.
func (s *Sequential) Add(m Module) { s.layers = append(s.layers, m) }

// Forward passes input through every layer in order.
func (s *Sequential) Forward(x *Tensor) *Tensor {
	for _, l := range s.layers { x = l.Forward(x) }
	return x
}

// Parameters collects all parameters from all layers.
func (s *Sequential) Parameters() []*Tensor {
	var p []*Tensor
	for _, l := range s.layers { p = append(p, l.Parameters()...) }
	return p
}

func (s *Sequential) ZeroGrad() { for _, l := range s.layers { l.ZeroGrad() } }
func (s *Sequential) Train()    { s.training = true; for _, l := range s.layers { l.Train() } }
func (s *Sequential) Eval()     { s.training = false; for _, l := range s.layers { l.Eval() } }
func (s *Sequential) Name() string { return fmt.Sprintf("Sequential(%d layers)", len(s.layers)) }

// ModuleList holds a list of modules (no automatic forwarding). Mirrors: nn.ModuleList
type ModuleList struct{ modules []Module }

func NewModuleList(modules ...Module) *ModuleList { return &ModuleList{modules: modules} }
func (ml *ModuleList) Get(i int) Module           { return ml.modules[i] }
func (ml *ModuleList) Len() int                   { return len(ml.modules) }
func (ml *ModuleList) Append(m Module)            { ml.modules = append(ml.modules, m) }
func (ml *ModuleList) Forward(x *Tensor) *Tensor  { return x } // not auto-sequential
func (ml *ModuleList) Parameters() []*Tensor {
	var p []*Tensor
	for _, m := range ml.modules { p = append(p, m.Parameters()...) }
	return p
}
func (ml *ModuleList) ZeroGrad() { for _, m := range ml.modules { m.ZeroGrad() } }
func (ml *ModuleList) Train()    { for _, m := range ml.modules { m.Train() } }
func (ml *ModuleList) Eval()     { for _, m := range ml.modules { m.Eval() } }
func (ml *ModuleList) Name() string { return "ModuleList" }

// ═══════════════════════════════════════════════════════════════════════════════
// LINEAR
// ═══════════════════════════════════════════════════════════════════════════════

// Linear applies y = xW^T + b. Mirrors: nn.Linear(in_features, out_features, bias)
type Linear struct {
	mod      C.Module
	InF, OutF int
	HasBias  bool
	training bool
}

// NewLinear creates a Linear layer.
func NewLinear(inFeatures, outFeatures int, bias bool) *Linear {
	b := C.int(0)
	if bias { b = 1 }
	return &Linear{
		mod:      C.gotorch_nn_linear_new(C.int64_t(inFeatures), C.int64_t(outFeatures), b),
		InF:      inFeatures,
		OutF:     outFeatures,
		HasBias:  bias,
		training: true,
	}
}

func (l *Linear) Forward(x *Tensor) *Tensor { return wrap(C.gotorch_nn_linear_forward(l.mod, x.ptr)) }
func (l *Linear) Weight() *Tensor           { return wrap(C.gotorch_nn_linear_weight(l.mod)) }
func (l *Linear) Bias() *Tensor             { return wrap(C.gotorch_nn_linear_bias(l.mod)) }
func (l *Linear) Parameters() []*Tensor {
	var count C.int64_t
	data := C.gotorch_nn_linear_parameters(l.mod, &count)
	return cParamsToSlice(data, count)
}
func (l *Linear) ZeroGrad() { for _, p := range l.Parameters() { p.ZeroGrad() } }
func (l *Linear) Train()    { l.training = true; C.gotorch_nn_linear_train(l.mod, 1) }
func (l *Linear) Eval()     { l.training = false; C.gotorch_nn_linear_train(l.mod, 0) }
func (l *Linear) Free()     { C.gotorch_nn_linear_free(l.mod) }
func (l *Linear) Name() string {
	return fmt.Sprintf("Linear(in=%d, out=%d, bias=%v)", l.InF, l.OutF, l.HasBias)
}

// Identity passes input through unchanged. Mirrors: nn.Identity()
type Identity struct{}

func NewIdentity() *Identity                      { return &Identity{} }
func (i *Identity) Forward(x *Tensor) *Tensor     { return x }
func (i *Identity) Parameters() []*Tensor         { return nil }
func (i *Identity) ZeroGrad()                     {}
func (i *Identity) Train()                        {}
func (i *Identity) Eval()                         {}
func (i *Identity) Name() string                  { return "Identity()" }

// ═══════════════════════════════════════════════════════════════════════════════
// CONV
// ═══════════════════════════════════════════════════════════════════════════════

// Conv2d applies 2-D convolution. Mirrors: nn.Conv2d
type Conv2d struct {
	mod      C.Module
	InCh, OutCh, Kernel, Stride, Padding int
	training bool
}

// NewConv2d creates a Conv2d layer.
func NewConv2d(inCh, outCh, kernel, stride, padding int, bias bool) *Conv2d {
	b := C.int(0)
	if bias { b = 1 }
	return &Conv2d{
		mod:     C.gotorch_nn_conv2d_new(C.int64_t(inCh), C.int64_t(outCh), C.int64_t(kernel), C.int64_t(stride), C.int64_t(padding), b),
		InCh:    inCh, OutCh: outCh, Kernel: kernel, Stride: stride, Padding: padding,
		training: true,
	}
}

// NewConv2dFull creates a Conv2d with dilation and groups.
func NewConv2dFull(inCh, outCh, kernel, stride, padding, dilation, groups int, bias bool) *Conv2d {
	b := C.int(0)
	if bias { b = 1 }
	return &Conv2d{
		mod:     C.gotorch_nn_conv2d_new_full(C.int64_t(inCh), C.int64_t(outCh), C.int64_t(kernel), C.int64_t(stride), C.int64_t(padding), C.int64_t(dilation), C.int64_t(groups), b),
		InCh: inCh, OutCh: outCh, Kernel: kernel, Stride: stride, Padding: padding,
		training: true,
	}
}

func (c *Conv2d) Forward(x *Tensor) *Tensor { return wrap(C.gotorch_nn_conv2d_forward(c.mod, x.ptr)) }
func (c *Conv2d) Parameters() []*Tensor {
	var count C.int64_t
	data := C.gotorch_nn_conv2d_parameters(c.mod, &count)
	return cParamsToSlice(data, count)
}
func (c *Conv2d) ZeroGrad() { for _, p := range c.Parameters() { p.ZeroGrad() } }
func (c *Conv2d) Train()    { c.training = true; C.gotorch_nn_conv2d_train(c.mod, 1) }
func (c *Conv2d) Eval()     { c.training = false; C.gotorch_nn_conv2d_train(c.mod, 0) }
func (c *Conv2d) Free()     { C.gotorch_nn_conv2d_free(c.mod) }
func (c *Conv2d) Name() string {
	return fmt.Sprintf("Conv2d(%d→%d, k=%d, s=%d, p=%d)", c.InCh, c.OutCh, c.Kernel, c.Stride, c.Padding)
}

// Conv1d applies 1-D convolution. Mirrors: nn.Conv1d
type Conv1d struct {
	mod      C.Module
	training bool
}

func NewConv1d(inCh, outCh, kernel, stride, padding int, bias bool) *Conv1d {
	b := C.int(0)
	if bias { b = 1 }
	return &Conv1d{
		mod:      C.gotorch_nn_conv1d_new(C.int64_t(inCh), C.int64_t(outCh), C.int64_t(kernel), C.int64_t(stride), C.int64_t(padding), b),
		training: true,
	}
}

func (c *Conv1d) Forward(x *Tensor) *Tensor { return wrap(C.gotorch_nn_conv1d_forward(c.mod, x.ptr)) }
func (c *Conv1d) Parameters() []*Tensor     { return nil } // weights via module
func (c *Conv1d) ZeroGrad()                 {}
func (c *Conv1d) Train()                    { c.training = true }
func (c *Conv1d) Eval()                     { c.training = false }
func (c *Conv1d) Free()                     { C.gotorch_nn_conv1d_free(c.mod) }
func (c *Conv1d) Name() string              { return "Conv1d" }

// ConvTranspose2d applies transposed 2-D convolution. Mirrors: nn.ConvTranspose2d
type ConvTranspose2d struct {
	mod      C.Module
	training bool
}

func NewConvTranspose2d(inCh, outCh, kernel, stride, padding, outputPadding int, bias bool) *ConvTranspose2d {
	b := C.int(0)
	if bias { b = 1 }
	return &ConvTranspose2d{
		mod:      C.gotorch_nn_convtranspose2d_new(C.int64_t(inCh), C.int64_t(outCh), C.int64_t(kernel), C.int64_t(stride), C.int64_t(padding), C.int64_t(outputPadding), b),
		training: true,
	}
}

func (c *ConvTranspose2d) Forward(x *Tensor) *Tensor { return wrap(C.gotorch_nn_convtranspose2d_forward(c.mod, x.ptr)) }
func (c *ConvTranspose2d) Parameters() []*Tensor     { return nil }
func (c *ConvTranspose2d) ZeroGrad()                 {}
func (c *ConvTranspose2d) Train()                    { c.training = true }
func (c *ConvTranspose2d) Eval()                     { c.training = false }
func (c *ConvTranspose2d) Free()                     { C.gotorch_nn_convtranspose2d_free(c.mod) }
func (c *ConvTranspose2d) Name() string              { return "ConvTranspose2d" }

// ═══════════════════════════════════════════════════════════════════════════════
// NORMALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

// BatchNorm1d applies 1-D batch normalization. Mirrors: nn.BatchNorm1d
type BatchNorm1d struct {
	mod         C.Module
	NumFeatures int
	training    bool
}

func NewBatchNorm1d(numFeatures int) *BatchNorm1d {
	return &BatchNorm1d{mod: C.gotorch_nn_batchnorm1d_new(C.int64_t(numFeatures)), NumFeatures: numFeatures, training: true}
}
func (b *BatchNorm1d) Forward(x *Tensor) *Tensor {
	t := C.int(0)
	if b.training { t = 1 }
	return wrap(C.gotorch_nn_batchnorm1d_forward(b.mod, x.ptr, t))
}
func (b *BatchNorm1d) Parameters() []*Tensor { return nil }
func (b *BatchNorm1d) ZeroGrad()             {}
func (b *BatchNorm1d) Train()                { b.training = true; C.gotorch_nn_batchnorm1d_train(b.mod, 1) }
func (b *BatchNorm1d) Eval()                 { b.training = false; C.gotorch_nn_batchnorm1d_train(b.mod, 0) }
func (b *BatchNorm1d) Free()                 { C.gotorch_nn_batchnorm1d_free(b.mod) }
func (b *BatchNorm1d) Name() string          { return fmt.Sprintf("BatchNorm1d(%d)", b.NumFeatures) }

// BatchNorm2d applies 2-D batch normalization. Mirrors: nn.BatchNorm2d
type BatchNorm2d struct {
	mod         C.Module
	NumFeatures int
	training    bool
}

func NewBatchNorm2d(numFeatures int) *BatchNorm2d {
	return &BatchNorm2d{mod: C.gotorch_nn_batchnorm2d_new(C.int64_t(numFeatures)), NumFeatures: numFeatures, training: true}
}
func (b *BatchNorm2d) Forward(x *Tensor) *Tensor {
	t := C.int(0)
	if b.training { t = 1 }
	return wrap(C.gotorch_nn_batchnorm2d_forward(b.mod, x.ptr, t))
}
func (b *BatchNorm2d) Parameters() []*Tensor { return nil }
func (b *BatchNorm2d) ZeroGrad()             {}
func (b *BatchNorm2d) Train()                { b.training = true; C.gotorch_nn_batchnorm2d_train(b.mod, 1) }
func (b *BatchNorm2d) Eval()                 { b.training = false; C.gotorch_nn_batchnorm2d_train(b.mod, 0) }
func (b *BatchNorm2d) Free()                 { C.gotorch_nn_batchnorm2d_free(b.mod) }
func (b *BatchNorm2d) Name() string          { return fmt.Sprintf("BatchNorm2d(%d)", b.NumFeatures) }

// LayerNorm applies layer normalization. Mirrors: nn.LayerNorm
type LayerNorm struct {
	mod              C.Module
	NormalizedShape  []int
	training         bool
}

func NewLayerNorm(normalizedShape []int, eps float64) *LayerNorm {
	s, n := cshape(normalizedShape)
	return &LayerNorm{
		mod:             C.gotorch_nn_layernorm_new(s, n, C.double(eps)),
		NormalizedShape: normalizedShape,
		training:        true,
	}
}
func (l *LayerNorm) Forward(x *Tensor) *Tensor { return wrap(C.gotorch_nn_layernorm_forward(l.mod, x.ptr)) }
func (l *LayerNorm) Parameters() []*Tensor     { return nil }
func (l *LayerNorm) ZeroGrad()                 {}
func (l *LayerNorm) Train()                    { l.training = true }
func (l *LayerNorm) Eval()                     { l.training = false }
func (l *LayerNorm) Free()                     { C.gotorch_nn_layernorm_free(l.mod) }
func (l *LayerNorm) Name() string              { return fmt.Sprintf("LayerNorm(%v)", l.NormalizedShape) }

// GroupNorm applies group normalization. Mirrors: nn.GroupNorm
type GroupNorm struct {
	mod      C.Module
	training bool
}

func NewGroupNorm(numGroups, numChannels int, eps float64) *GroupNorm {
	return &GroupNorm{mod: C.gotorch_nn_groupnorm_new(C.int64_t(numGroups), C.int64_t(numChannels), C.double(eps)), training: true}
}
func (g *GroupNorm) Forward(x *Tensor) *Tensor { return wrap(C.gotorch_nn_groupnorm_forward(g.mod, x.ptr)) }
func (g *GroupNorm) Parameters() []*Tensor     { return nil }
func (g *GroupNorm) ZeroGrad()                 {}
func (g *GroupNorm) Train()                    { g.training = true }
func (g *GroupNorm) Eval()                     { g.training = false }
func (g *GroupNorm) Free()                     { C.gotorch_nn_groupnorm_free(g.mod) }
func (g *GroupNorm) Name() string              { return "GroupNorm" }

// InstanceNorm2d applies instance normalization. Mirrors: nn.InstanceNorm2d
type InstanceNorm2d struct {
	mod      C.Module
	training bool
}

func NewInstanceNorm2d(numFeatures int) *InstanceNorm2d {
	return &InstanceNorm2d{mod: C.gotorch_nn_instancenorm2d_new(C.int64_t(numFeatures)), training: true}
}
func (i *InstanceNorm2d) Forward(x *Tensor) *Tensor {
	t := C.int(0)
	if i.training { t = 1 }
	return wrap(C.gotorch_nn_instancenorm2d_forward(i.mod, x.ptr, t))
}
func (i *InstanceNorm2d) Parameters() []*Tensor { return nil }
func (i *InstanceNorm2d) ZeroGrad()             {}
func (i *InstanceNorm2d) Train()                { i.training = true }
func (i *InstanceNorm2d) Eval()                 { i.training = false }
func (i *InstanceNorm2d) Free()                 { C.gotorch_nn_instancenorm2d_free(i.mod) }
func (i *InstanceNorm2d) Name() string          { return "InstanceNorm2d" }

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVATION MODULES (stateful wrappers)
// ═══════════════════════════════════════════════════════════════════════════════

type ReLUModule    struct{ Inplace bool }
type LeakyReLUModule struct{ NegSlope float64 }
type SigmoidModule struct{}
type TanhModule    struct{}
type SoftmaxModule struct{ Dim int }
type LogSoftmaxModule struct{ Dim int }
type GELUModule    struct{}
type SiLUModule    struct{}
type ELUModule     struct{ Alpha float64 }
type SELUModule    struct{}
type MishModule    struct{}
type HardswishModule struct{}

func NewReLUModule() *ReLUModule            { return &ReLUModule{} }
func NewLeakyReLUModule(ns float64) *LeakyReLUModule { return &LeakyReLUModule{NegSlope: ns} }
func NewSigmoid() *SigmoidModule            { return &SigmoidModule{} }
func NewTanh() *TanhModule                  { return &TanhModule{} }
func NewSoftmax(dim int) *SoftmaxModule     { return &SoftmaxModule{Dim: dim} }
func NewLogSoftmax(dim int) *LogSoftmaxModule { return &LogSoftmaxModule{Dim: dim} }
func NewGELU() *GELUModule                  { return &GELUModule{} }
func NewSiLU() *SiLUModule                  { return &SiLUModule{} }
func NewELU(alpha float64) *ELUModule       { return &ELUModule{Alpha: alpha} }
func NewSELU() *SELUModule                  { return &SELUModule{} }
func NewMish() *MishModule                  { return &MishModule{} }
func NewHardswish() *HardswishModule        { return &HardswishModule{} }

func (m *ReLUModule) Forward(x *Tensor) *Tensor          { return wrap(C.gotorch_relu(x.ptr)) }
func (m *LeakyReLUModule) Forward(x *Tensor) *Tensor     { return wrap(C.gotorch_leaky_relu(x.ptr, C.double(m.NegSlope))) }
func (m *SigmoidModule) Forward(x *Tensor) *Tensor       { return wrap(C.gotorch_sigmoid(x.ptr)) }
func (m *TanhModule) Forward(x *Tensor) *Tensor          { return wrap(C.gotorch_tanh(x.ptr)) }
func (m *SoftmaxModule) Forward(x *Tensor) *Tensor       { return wrap(C.gotorch_softmax(x.ptr, C.int64_t(m.Dim))) }
func (m *LogSoftmaxModule) Forward(x *Tensor) *Tensor    { return wrap(C.gotorch_log_softmax(x.ptr, C.int64_t(m.Dim))) }
func (m *GELUModule) Forward(x *Tensor) *Tensor          { return wrap(C.gotorch_gelu(x.ptr)) }
func (m *SiLUModule) Forward(x *Tensor) *Tensor          { return wrap(C.gotorch_silu(x.ptr)) }
func (m *ELUModule) Forward(x *Tensor) *Tensor           { return wrap(C.gotorch_elu(x.ptr, C.double(m.Alpha))) }
func (m *SELUModule) Forward(x *Tensor) *Tensor          { return wrap(C.gotorch_selu(x.ptr)) }
func (m *MishModule) Forward(x *Tensor) *Tensor          { return wrap(C.gotorch_mish(x.ptr)) }
func (m *HardswishModule) Forward(x *Tensor) *Tensor     { return wrap(C.gotorch_hardswish(x.ptr)) }

// All activation modules share no-op Module interface methods
func (m *ReLUModule) Parameters() []*Tensor       { return nil }
func (m *ReLUModule) ZeroGrad()                   {}
func (m *ReLUModule) Train()                      {}
func (m *ReLUModule) Eval()                       {}
func (m *ReLUModule) Name() string                { return "ReLU" }
func (m *LeakyReLUModule) Parameters() []*Tensor  { return nil }
func (m *LeakyReLUModule) ZeroGrad()              {}
func (m *LeakyReLUModule) Train()                 {}
func (m *LeakyReLUModule) Eval()                  {}
func (m *LeakyReLUModule) Name() string           { return fmt.Sprintf("LeakyReLU(%.3f)", m.NegSlope) }
func (m *SigmoidModule) Parameters() []*Tensor    { return nil }
func (m *SigmoidModule) ZeroGrad()                {}
func (m *SigmoidModule) Train()                   {}
func (m *SigmoidModule) Eval()                    {}
func (m *SigmoidModule) Name() string             { return "Sigmoid" }
func (m *TanhModule) Parameters() []*Tensor       { return nil }
func (m *TanhModule) ZeroGrad()                   {}
func (m *TanhModule) Train()                      {}
func (m *TanhModule) Eval()                       {}
func (m *TanhModule) Name() string                { return "Tanh" }
func (m *SoftmaxModule) Parameters() []*Tensor    { return nil }
func (m *SoftmaxModule) ZeroGrad()                {}
func (m *SoftmaxModule) Train()                   {}
func (m *SoftmaxModule) Eval()                    {}
func (m *SoftmaxModule) Name() string             { return fmt.Sprintf("Softmax(dim=%d)", m.Dim) }
func (m *LogSoftmaxModule) Parameters() []*Tensor { return nil }
func (m *LogSoftmaxModule) ZeroGrad()             {}
func (m *LogSoftmaxModule) Train()                {}
func (m *LogSoftmaxModule) Eval()                 {}
func (m *LogSoftmaxModule) Name() string          { return fmt.Sprintf("LogSoftmax(dim=%d)", m.Dim) }
func (m *GELUModule) Parameters() []*Tensor       { return nil }
func (m *GELUModule) ZeroGrad()                   {}
func (m *GELUModule) Train()                      {}
func (m *GELUModule) Eval()                       {}
func (m *GELUModule) Name() string                { return "GELU" }
func (m *SiLUModule) Parameters() []*Tensor       { return nil }
func (m *SiLUModule) ZeroGrad()                   {}
func (m *SiLUModule) Train()                      {}
func (m *SiLUModule) Eval()                       {}
func (m *SiLUModule) Name() string                { return "SiLU" }
func (m *ELUModule) Parameters() []*Tensor        { return nil }
func (m *ELUModule) ZeroGrad()                    {}
func (m *ELUModule) Train()                       {}
func (m *ELUModule) Eval()                        {}
func (m *ELUModule) Name() string                 { return fmt.Sprintf("ELU(%.2f)", m.Alpha) }
func (m *SELUModule) Parameters() []*Tensor       { return nil }
func (m *SELUModule) ZeroGrad()                   {}
func (m *SELUModule) Train()                      {}
func (m *SELUModule) Eval()                       {}
func (m *SELUModule) Name() string                { return "SELU" }
func (m *MishModule) Parameters() []*Tensor       { return nil }
func (m *MishModule) ZeroGrad()                   {}
func (m *MishModule) Train()                      {}
func (m *MishModule) Eval()                       {}
func (m *MishModule) Name() string                { return "Mish" }
func (m *HardswishModule) Parameters() []*Tensor  { return nil }
func (m *HardswishModule) ZeroGrad()              {}
func (m *HardswishModule) Train()                 {}
func (m *HardswishModule) Eval()                  {}
func (m *HardswishModule) Name() string           { return "Hardswish" }

// ═══════════════════════════════════════════════════════════════════════════════
// DROPOUT
// ═══════════════════════════════════════════════════════════════════════════════

// Dropout randomly zeroes elements. Mirrors: nn.Dropout(p)
type Dropout struct{ P float64; training bool }

func NewDropout(p float64) *Dropout                { return &Dropout{P: p, training: true} }
func (d *Dropout) Forward(x *Tensor) *Tensor {
	t := C.int(0)
	if d.training { t = 1 }
	return wrap(C.gotorch_nn_dropout_forward(x.ptr, C.double(d.P), t))
}
func (d *Dropout) Parameters() []*Tensor           { return nil }
func (d *Dropout) ZeroGrad()                       {}
func (d *Dropout) Train()                          { d.training = true }
func (d *Dropout) Eval()                           { d.training = false }
func (d *Dropout) Name() string                    { return fmt.Sprintf("Dropout(p=%.2f)", d.P) }

// Dropout2d zeroes full feature-map channels. Mirrors: nn.Dropout2d(p)
type Dropout2d struct{ P float64; training bool }

func NewDropout2d(p float64) *Dropout2d { return &Dropout2d{P: p, training: true} }
func (d *Dropout2d) Forward(x *Tensor) *Tensor {
	t := C.int(0)
	if d.training { t = 1 }
	return wrap(C.gotorch_nn_dropout2d_forward(x.ptr, C.double(d.P), t))
}
func (d *Dropout2d) Parameters() []*Tensor { return nil }
func (d *Dropout2d) ZeroGrad()             {}
func (d *Dropout2d) Train()                { d.training = true }
func (d *Dropout2d) Eval()                 { d.training = false }
func (d *Dropout2d) Name() string          { return fmt.Sprintf("Dropout2d(p=%.2f)", d.P) }

// AlphaDropout preserves SELU self-normalizing property. Mirrors: nn.AlphaDropout(p)
type AlphaDropout struct{ P float64; training bool }

func NewAlphaDropout(p float64) *AlphaDropout { return &AlphaDropout{P: p, training: true} }
func (d *AlphaDropout) Forward(x *Tensor) *Tensor {
	t := C.int(0)
	if d.training { t = 1 }
	return wrap(C.gotorch_nn_alpha_dropout_forward(x.ptr, C.double(d.P), t))
}
func (d *AlphaDropout) Parameters() []*Tensor { return nil }
func (d *AlphaDropout) ZeroGrad()             {}
func (d *AlphaDropout) Train()                { d.training = true }
func (d *AlphaDropout) Eval()                 { d.training = false }
func (d *AlphaDropout) Name() string          { return fmt.Sprintf("AlphaDropout(p=%.2f)", d.P) }

// ═══════════════════════════════════════════════════════════════════════════════
// POOLING
// ═══════════════════════════════════════════════════════════════════════════════

// MaxPool2d applies 2-D max pooling. Mirrors: nn.MaxPool2d
type MaxPool2d struct {
	KernelSize, Stride, Padding, Dilation int
	CeilMode bool
}

func NewMaxPool2d(kernelSize, stride, padding, dilation int) *MaxPool2d {
	return &MaxPool2d{KernelSize: kernelSize, Stride: stride, Padding: padding, Dilation: dilation}
}
func (p *MaxPool2d) Forward(x *Tensor) *Tensor {
	s := p.Stride
	if s == 0 { s = p.KernelSize }
	cm := C.int(0)
	if p.CeilMode { cm = 1 }
	return wrap(C.gotorch_nn_max_pool2d(x.ptr, C.int64_t(p.KernelSize), C.int64_t(s), C.int64_t(p.Padding), C.int64_t(p.Dilation), cm))
}
func (p *MaxPool2d) Parameters() []*Tensor { return nil }
func (p *MaxPool2d) ZeroGrad()             {}
func (p *MaxPool2d) Train()                {}
func (p *MaxPool2d) Eval()                 {}
func (p *MaxPool2d) Name() string          { return fmt.Sprintf("MaxPool2d(k=%d)", p.KernelSize) }

// AvgPool2d applies 2-D average pooling. Mirrors: nn.AvgPool2d
type AvgPool2d struct {
	KernelSize, Stride, Padding int
	CountIncludePad             bool
}

func NewAvgPool2d(kernelSize, stride, padding int) *AvgPool2d {
	return &AvgPool2d{KernelSize: kernelSize, Stride: stride, Padding: padding, CountIncludePad: true}
}
func (p *AvgPool2d) Forward(x *Tensor) *Tensor {
	s := p.Stride
	if s == 0 { s = p.KernelSize }
	cip := C.int(1)
	if !p.CountIncludePad { cip = 0 }
	return wrap(C.gotorch_nn_avg_pool2d(x.ptr, C.int64_t(p.KernelSize), C.int64_t(s), C.int64_t(p.Padding), cip))
}
func (p *AvgPool2d) Parameters() []*Tensor { return nil }
func (p *AvgPool2d) ZeroGrad()             {}
func (p *AvgPool2d) Train()                {}
func (p *AvgPool2d) Eval()                 {}
func (p *AvgPool2d) Name() string          { return fmt.Sprintf("AvgPool2d(k=%d)", p.KernelSize) }

// AdaptiveAvgPool2d reduces to fixed output size. Mirrors: nn.AdaptiveAvgPool2d
type AdaptiveAvgPool2d struct{ OutH, OutW int }

func NewAdaptiveAvgPool2d(outH, outW int) *AdaptiveAvgPool2d { return &AdaptiveAvgPool2d{OutH: outH, OutW: outW} }
func (p *AdaptiveAvgPool2d) Forward(x *Tensor) *Tensor {
	return wrap(C.gotorch_nn_adaptive_avg_pool2d(x.ptr, C.int64_t(p.OutH), C.int64_t(p.OutW)))
}
func (p *AdaptiveAvgPool2d) Parameters() []*Tensor { return nil }
func (p *AdaptiveAvgPool2d) ZeroGrad()             {}
func (p *AdaptiveAvgPool2d) Train()                {}
func (p *AdaptiveAvgPool2d) Eval()                 {}
func (p *AdaptiveAvgPool2d) Name() string          { return fmt.Sprintf("AdaptiveAvgPool2d(%d,%d)", p.OutH, p.OutW) }

// MaxPool1d applies 1-D max pooling. Mirrors: nn.MaxPool1d
type MaxPool1d struct{ KernelSize, Stride, Padding int }

func NewMaxPool1d(kernelSize, stride, padding int) *MaxPool1d {
	return &MaxPool1d{KernelSize: kernelSize, Stride: stride, Padding: padding}
}
func (p *MaxPool1d) Forward(x *Tensor) *Tensor {
	s := p.Stride
	if s == 0 { s = p.KernelSize }
	return wrap(C.gotorch_nn_max_pool1d(x.ptr, C.int64_t(p.KernelSize), C.int64_t(s), C.int64_t(p.Padding)))
}
func (p *MaxPool1d) Parameters() []*Tensor { return nil }
func (p *MaxPool1d) ZeroGrad()             {}
func (p *MaxPool1d) Train()                {}
func (p *MaxPool1d) Eval()                 {}
func (p *MaxPool1d) Name() string          { return fmt.Sprintf("MaxPool1d(k=%d)", p.KernelSize) }

// Flatten collapses contiguous dims. Mirrors: nn.Flatten(start_dim, end_dim)
type FlattenModule struct{ StartDim, EndDim int }

func NewFlatten(startDim, endDim int) *FlattenModule { return &FlattenModule{StartDim: startDim, EndDim: endDim} }
func (f *FlattenModule) Forward(x *Tensor) *Tensor {
	return wrap(C.gotorch_flatten(x.ptr, C.int64_t(f.StartDim), C.int64_t(f.EndDim)))
}
func (f *FlattenModule) Parameters() []*Tensor { return nil }
func (f *FlattenModule) ZeroGrad()             {}
func (f *FlattenModule) Train()                {}
func (f *FlattenModule) Eval()                 {}
func (f *FlattenModule) Name() string          { return fmt.Sprintf("Flatten(%d,%d)", f.StartDim, f.EndDim) }

// ═══════════════════════════════════════════════════════════════════════════════
// EMBEDDING
// ═══════════════════════════════════════════════════════════════════════════════

// Embedding is a lookup table. Mirrors: nn.Embedding(num_embeddings, embedding_dim)
type Embedding struct {
	mod           C.Module
	NumEmbeddings int
	EmbeddingDim  int
	training      bool
}

func NewEmbedding(numEmbeddings, embeddingDim int) *Embedding {
	return &Embedding{
		mod:           C.gotorch_nn_embedding_new(C.int64_t(numEmbeddings), C.int64_t(embeddingDim), C.int64_t(-1), C.int(0)),
		NumEmbeddings: numEmbeddings,
		EmbeddingDim:  embeddingDim,
		training:      true,
	}
}

func NewEmbeddingFull(numEmbeddings, embeddingDim, paddingIdx int, sparse bool) *Embedding {
	sp := C.int(0)
	if sparse { sp = 1 }
	return &Embedding{
		mod:           C.gotorch_nn_embedding_new(C.int64_t(numEmbeddings), C.int64_t(embeddingDim), C.int64_t(paddingIdx), sp),
		NumEmbeddings: numEmbeddings,
		EmbeddingDim:  embeddingDim,
		training:      true,
	}
}

func (e *Embedding) Forward(x *Tensor) *Tensor { return wrap(C.gotorch_nn_embedding_forward(e.mod, x.ptr)) }
func (e *Embedding) Weight() *Tensor           { return wrap(C.gotorch_nn_embedding_weight(e.mod)) }
func (e *Embedding) Parameters() []*Tensor {
	var count C.int64_t
	data := C.gotorch_nn_embedding_parameters(e.mod, &count)
	return cParamsToSlice(data, count)
}
func (e *Embedding) ZeroGrad() { for _, p := range e.Parameters() { p.ZeroGrad() } }
func (e *Embedding) Train()    { e.training = true }
func (e *Embedding) Eval()     { e.training = false }
func (e *Embedding) Free()     { C.gotorch_nn_embedding_free(e.mod) }
func (e *Embedding) Name() string {
	return fmt.Sprintf("Embedding(%d, %d)", e.NumEmbeddings, e.EmbeddingDim)
}

// EmbeddingBag Embedding + reduction. Mirrors: nn.EmbeddingBag
type EmbeddingBag struct {
	mod  C.Module
	Mode string
}

func NewEmbeddingBag(numEmbeddings, embeddingDim int, mode string) *EmbeddingBag {
	m := C.int(1) // mean
	if mode == "sum" { m = 0 } else if mode == "max" { m = 2 }
	return &EmbeddingBag{mod: C.gotorch_nn_embedding_bag_new(C.int64_t(numEmbeddings), C.int64_t(embeddingDim), m), Mode: mode}
}

// Forward computes bag embedding. input: 1-D indices, offsets: bag start positions.
func (e *EmbeddingBag) ForwardBag(input, offsets *Tensor) *Tensor {
	return wrap(C.gotorch_nn_embedding_bag_forward(e.mod, input.ptr, offsets.ptr))
}
func (e *EmbeddingBag) Forward(x *Tensor) *Tensor { panic("EmbeddingBag: use ForwardBag(input, offsets)") }
func (e *EmbeddingBag) Weight() *Tensor           { return wrap(C.gotorch_nn_embedding_bag_weight(e.mod)) }
func (e *EmbeddingBag) Parameters() []*Tensor     { return []*Tensor{e.Weight()} }
func (e *EmbeddingBag) ZeroGrad()                 { e.Weight().ZeroGrad() }
func (e *EmbeddingBag) Train()                    {}
func (e *EmbeddingBag) Eval()                     {}
func (e *EmbeddingBag) Free()                     { C.gotorch_nn_embedding_bag_free(e.mod) }
func (e *EmbeddingBag) Name() string              { return fmt.Sprintf("EmbeddingBag(mode=%s)", e.Mode) }

// ═══════════════════════════════════════════════════════════════════════════════
// LSTM
// ═══════════════════════════════════════════════════════════════════════════════

// LSTMOutput bundles (output, h_n, c_n). Mirrors the Python LSTM forward tuple.
type LSTMOutput struct {
	Output *Tensor // (seq_len, batch, num_dir * hidden_size)
	Hn     *Tensor // (num_layers * num_dir, batch, hidden_size)
	Cn     *Tensor // (num_layers * num_dir, batch, hidden_size)
}

// LSTM applies multi-layer LSTM. Mirrors: nn.LSTM
type LSTM struct {
	mod      C.Module
	training bool
}

// NewLSTM creates an LSTM.
// Mirrors: nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
func NewLSTM(inputSize, hiddenSize, numLayers int, bias, batchFirst bool, dropout float64, bidirectional bool) *LSTM {
	bi, bf, bd := C.int(0), C.int(0), C.int(0)
	if bias          { bi = 1 }
	if batchFirst    { bf = 1 }
	if bidirectional { bd = 1 }
	return &LSTM{
		mod:      C.gotorch_nn_lstm_new(C.int64_t(inputSize), C.int64_t(hiddenSize), C.int64_t(numLayers), bi, bf, C.double(dropout), bd),
		training: true,
	}
}

// Forward runs the LSTM. Pass nil h0/c0 to auto-initialise to zeros.
func (l *LSTM) ForwardLSTM(input, h0, c0 *Tensor) LSTMOutput {
	var ih0, ic0 C.Tensor
	if h0 != nil { ih0 = h0.ptr }
	if c0 != nil { ic0 = c0.ptr }
	var out, hn, cn C.Tensor
	C.gotorch_nn_lstm_forward(l.mod, input.ptr, ih0, ic0, &out, &hn, &cn)
	return LSTMOutput{Output: wrap(out), Hn: wrap(hn), Cn: wrap(cn)}
}
func (l *LSTM) Forward(x *Tensor) *Tensor { return l.ForwardLSTM(x, nil, nil).Output }
func (l *LSTM) Parameters() []*Tensor {
	var count C.int64_t
	data := C.gotorch_nn_lstm_parameters(l.mod, &count)
	return cParamsToSlice(data, count)
}
func (l *LSTM) ZeroGrad() { for _, p := range l.Parameters() { p.ZeroGrad() } }
func (l *LSTM) Train()    { l.training = true; C.gotorch_nn_lstm_train(l.mod, 1) }
func (l *LSTM) Eval()     { l.training = false; C.gotorch_nn_lstm_train(l.mod, 0) }
func (l *LSTM) Free()     { C.gotorch_nn_lstm_free(l.mod) }
func (l *LSTM) Name() string { return "LSTM" }

// ═══════════════════════════════════════════════════════════════════════════════
// GRU
// ═══════════════════════════════════════════════════════════════════════════════

// GRUOutput bundles (output, h_n).
type GRUOutput struct {
	Output *Tensor
	Hn     *Tensor
}

// GRU applies multi-layer GRU. Mirrors: nn.GRU
type GRU struct {
	mod      C.Module
	training bool
}

// NewGRU creates a GRU.
func NewGRU(inputSize, hiddenSize, numLayers int, bias, batchFirst bool, dropout float64, bidirectional bool) *GRU {
	bi, bf, bd := C.int(0), C.int(0), C.int(0)
	if bias          { bi = 1 }
	if batchFirst    { bf = 1 }
	if bidirectional { bd = 1 }
	return &GRU{
		mod:      C.gotorch_nn_gru_new(C.int64_t(inputSize), C.int64_t(hiddenSize), C.int64_t(numLayers), bi, bf, C.double(dropout), bd),
		training: true,
	}
}

// ForwardGRU runs the GRU. Pass nil h0 to auto-initialise.
func (g *GRU) ForwardGRU(input, h0 *Tensor) GRUOutput {
	var ih0 C.Tensor
	if h0 != nil { ih0 = h0.ptr }
	var out, hn C.Tensor
	C.gotorch_nn_gru_forward(g.mod, input.ptr, ih0, &out, &hn)
	return GRUOutput{Output: wrap(out), Hn: wrap(hn)}
}
func (g *GRU) Forward(x *Tensor) *Tensor { return g.ForwardGRU(x, nil).Output }
func (g *GRU) Parameters() []*Tensor {
	var count C.int64_t
	data := C.gotorch_nn_gru_parameters(g.mod, &count)
	return cParamsToSlice(data, count)
}
func (g *GRU) ZeroGrad() { for _, p := range g.Parameters() { p.ZeroGrad() } }
func (g *GRU) Train()    { g.training = true; C.gotorch_nn_gru_train(g.mod, 1) }
func (g *GRU) Eval()     { g.training = false; C.gotorch_nn_gru_train(g.mod, 0) }
func (g *GRU) Free()     { C.gotorch_nn_gru_free(g.mod) }
func (g *GRU) Name() string { return "GRU" }

// ═══════════════════════════════════════════════════════════════════════════════
// MULTIHEAD ATTENTION
// ═══════════════════════════════════════════════════════════════════════════════

// MultiheadAttention implements scaled dot-product MHA. Mirrors: nn.MultiheadAttention
type MultiheadAttention struct {
	mod      C.Module
	EmbedDim int
	NumHeads int
	training bool
}

// NewMultiheadAttention creates MHA.
func NewMultiheadAttention(embedDim, numHeads int, dropout float64, bias bool) *MultiheadAttention {
	b := C.int(0)
	if bias { b = 1 }
	return &MultiheadAttention{
		mod:      C.gotorch_nn_mha_new(C.int64_t(embedDim), C.int64_t(numHeads), C.double(dropout), b),
		EmbedDim: embedDim,
		NumHeads: numHeads,
		training: true,
	}
}

// ForwardMHA computes attention. keyPaddingMask may be nil.
// Returns (attn_output, attn_weights).
func (m *MultiheadAttention) ForwardMHA(query, key, value, keyPaddingMask *Tensor) (*Tensor, *Tensor) {
	var kpm C.Tensor
	if keyPaddingMask != nil { kpm = keyPaddingMask.ptr }
	var attnOut, attnW C.Tensor
	C.gotorch_nn_mha_forward(m.mod, query.ptr, key.ptr, value.ptr, kpm, &attnOut, &attnW)
	return wrap(attnOut), wrap(attnW)
}
func (m *MultiheadAttention) Forward(x *Tensor) *Tensor {
	out, _ := m.ForwardMHA(x, x, x, nil)
	return out
}
func (m *MultiheadAttention) Parameters() []*Tensor {
	var count C.int64_t
	data := C.gotorch_nn_mha_parameters(m.mod, &count)
	return cParamsToSlice(data, count)
}
func (m *MultiheadAttention) ZeroGrad() { for _, p := range m.Parameters() { p.ZeroGrad() } }
func (m *MultiheadAttention) Train()    { m.training = true; C.gotorch_nn_mha_train(m.mod, 1) }
func (m *MultiheadAttention) Eval()     { m.training = false; C.gotorch_nn_mha_train(m.mod, 0) }
func (m *MultiheadAttention) Free()     { C.gotorch_nn_mha_free(m.mod) }
func (m *MultiheadAttention) Name() string {
	return fmt.Sprintf("MultiheadAttention(embed=%d, heads=%d)", m.EmbedDim, m.NumHeads)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRANSFORMER ENCODER
// ═══════════════════════════════════════════════════════════════════════════════

// TransformerEncoderLayer is one Pre/Post-LN block. Mirrors: nn.TransformerEncoderLayer
type TransformerEncoderLayer struct {
	mod      C.Module
	training bool
}

func NewTransformerEncoderLayer(dModel, nHead, dimFeedforward int, dropout float64) *TransformerEncoderLayer {
	return &TransformerEncoderLayer{
		mod:      C.gotorch_nn_transformer_enc_layer_new(C.int64_t(dModel), C.int64_t(nHead), C.int64_t(dimFeedforward), C.double(dropout)),
		training: true,
	}
}
func (t *TransformerEncoderLayer) ForwardWithMask(src, srcMask *Tensor) *Tensor {
	var mask C.Tensor
	if srcMask != nil { mask = srcMask.ptr }
	return wrap(C.gotorch_nn_transformer_enc_layer_forward(t.mod, src.ptr, mask))
}
func (t *TransformerEncoderLayer) Forward(x *Tensor) *Tensor { return t.ForwardWithMask(x, nil) }
func (t *TransformerEncoderLayer) Parameters() []*Tensor {
	var count C.int64_t
	data := C.gotorch_nn_transformer_enc_layer_parameters(t.mod, &count)
	return cParamsToSlice(data, count)
}
func (t *TransformerEncoderLayer) ZeroGrad() { for _, p := range t.Parameters() { p.ZeroGrad() } }
func (t *TransformerEncoderLayer) Train()    { t.training = true; C.gotorch_nn_transformer_enc_layer_train(t.mod, 1) }
func (t *TransformerEncoderLayer) Eval()     { t.training = false; C.gotorch_nn_transformer_enc_layer_train(t.mod, 0) }
func (t *TransformerEncoderLayer) Free()     { C.gotorch_nn_transformer_enc_layer_free(t.mod) }
func (t *TransformerEncoderLayer) Name() string { return "TransformerEncoderLayer" }

// TransformerEncoder stacks N encoder layers. Mirrors: nn.TransformerEncoder
type TransformerEncoder struct {
	mod      C.Module
	training bool
}

func NewTransformerEncoder(layer *TransformerEncoderLayer, numLayers int) *TransformerEncoder {
	return &TransformerEncoder{
		mod:      C.gotorch_nn_transformer_encoder_new(layer.mod, C.int64_t(numLayers)),
		training: true,
	}
}
func (te *TransformerEncoder) ForwardWithMask(src, srcMask *Tensor) *Tensor {
	var mask C.Tensor
	if srcMask != nil { mask = srcMask.ptr }
	return wrap(C.gotorch_nn_transformer_encoder_forward(te.mod, src.ptr, mask))
}
func (te *TransformerEncoder) Forward(x *Tensor) *Tensor { return te.ForwardWithMask(x, nil) }
func (te *TransformerEncoder) Parameters() []*Tensor {
	var count C.int64_t
	data := C.gotorch_nn_transformer_encoder_parameters(te.mod, &count)
	return cParamsToSlice(data, count)
}
func (te *TransformerEncoder) ZeroGrad() { for _, p := range te.Parameters() { p.ZeroGrad() } }
func (te *TransformerEncoder) Train()    { te.training = true; C.gotorch_nn_transformer_encoder_train(te.mod, 1) }
func (te *TransformerEncoder) Eval()     { te.training = false; C.gotorch_nn_transformer_encoder_train(te.mod, 0) }
func (te *TransformerEncoder) Free()     { C.gotorch_nn_transformer_encoder_free(te.mod) }
func (te *TransformerEncoder) Name() string { return "TransformerEncoder" }

// ═══════════════════════════════════════════════════════════════════════════════
// OPTIMIZERS
// ═══════════════════════════════════════════════════════════════════════════════

// Optimizer is the base interface. Mirrors: torch.optim.Optimizer
type Optimizer interface {
	Step()
	ZeroGrad()
	GetLR() float64
	SetLR(lr float64)
}

func paramRawPtrs(params []*Tensor) ([]C.Tensor, C.int64_t) {
	r := make([]C.Tensor, len(params))
	for i, p := range params { r[i] = p.ptr }
	return r, C.int64_t(len(r))
}

// SGDOptions configures SGD.
type SGDOptions struct {
	LR           float64
	Momentum     float64
	WeightDecay  float64
	Nesterov     bool
}

// SGD stochastic gradient descent. Mirrors: optim.SGD
type SGD struct{ ptr C.Optimizer }

// NewSGD creates an SGD optimizer.
func NewSGD(params []*Tensor, opts SGDOptions) *SGD {
	r, n := paramRawPtrs(params)
	nest := C.int(0)
	if opts.Nesterov { nest = 1 }
	return &SGD{ptr: C.gotorch_optim_sgd_new(&r[0], n, C.double(opts.LR), C.double(opts.Momentum), C.double(opts.WeightDecay), nest)}
}
func (o *SGD) Step()          { C.gotorch_optim_step(o.ptr) }
func (o *SGD) ZeroGrad()      { C.gotorch_optim_zero_grad(o.ptr) }
func (o *SGD) GetLR() float64 { return float64(C.gotorch_optim_get_lr(o.ptr)) }
func (o *SGD) SetLR(lr float64) { C.gotorch_optim_set_lr(o.ptr, C.double(lr)) }
func (o *SGD) Free()          { C.gotorch_optim_free(o.ptr) }

// AdamOptions configures Adam and AdamW.
type AdamOptions struct {
	LR          float64
	Beta1       float64
	Beta2       float64
	Eps         float64
	WeightDecay float64
}

func (o *AdamOptions) defaults() {
	if o.LR == 0    { o.LR = 1e-3 }
	if o.Beta1 == 0 { o.Beta1 = 0.9 }
	if o.Beta2 == 0 { o.Beta2 = 0.999 }
	if o.Eps == 0   { o.Eps = 1e-8 }
}

// Adam optimizer. Mirrors: optim.Adam
type Adam struct{ ptr C.Optimizer }

// NewAdam creates an Adam optimizer.
func NewAdam(params []*Tensor, opts AdamOptions) *Adam {
	opts.defaults()
	r, n := paramRawPtrs(params)
	return &Adam{ptr: C.gotorch_optim_adam_new(&r[0], n, C.double(opts.LR), C.double(opts.Beta1), C.double(opts.Beta2), C.double(opts.Eps), C.double(opts.WeightDecay))}
}
func (o *Adam) Step()           { C.gotorch_optim_step(o.ptr) }
func (o *Adam) ZeroGrad()       { C.gotorch_optim_zero_grad(o.ptr) }
func (o *Adam) GetLR() float64  { return float64(C.gotorch_optim_get_lr(o.ptr)) }
func (o *Adam) SetLR(lr float64) { C.gotorch_optim_set_lr(o.ptr, C.double(lr)) }
func (o *Adam) Free()           { C.gotorch_optim_free(o.ptr) }

// AdamW Adam with decoupled weight decay. Mirrors: optim.AdamW
type AdamW struct{ ptr C.Optimizer }

func NewAdamW(params []*Tensor, opts AdamOptions) *AdamW {
	opts.defaults()
	r, n := paramRawPtrs(params)
	return &AdamW{ptr: C.gotorch_optim_adamw_new(&r[0], n, C.double(opts.LR), C.double(opts.Beta1), C.double(opts.Beta2), C.double(opts.Eps), C.double(opts.WeightDecay))}
}
func (o *AdamW) Step()           { C.gotorch_optim_step(o.ptr) }
func (o *AdamW) ZeroGrad()       { C.gotorch_optim_zero_grad(o.ptr) }
func (o *AdamW) GetLR() float64  { return float64(C.gotorch_optim_get_lr(o.ptr)) }
func (o *AdamW) SetLR(lr float64) { C.gotorch_optim_set_lr(o.ptr, C.double(lr)) }
func (o *AdamW) Free()           { C.gotorch_optim_free(o.ptr) }

// RMSpropOptions configures RMSprop.
type RMSpropOptions struct {
	LR          float64
	Alpha       float64
	Eps         float64
	WeightDecay float64
}

// RMSprop optimizer. Mirrors: optim.RMSprop
type RMSprop struct{ ptr C.Optimizer }

func NewRMSprop(params []*Tensor, opts RMSpropOptions) *RMSprop {
	if opts.LR == 0    { opts.LR = 1e-2 }
	if opts.Alpha == 0 { opts.Alpha = 0.99 }
	if opts.Eps == 0   { opts.Eps = 1e-8 }
	r, n := paramRawPtrs(params)
	return &RMSprop{ptr: C.gotorch_optim_rmsprop_new(&r[0], n, C.double(opts.LR), C.double(opts.Alpha), C.double(opts.Eps), C.double(opts.WeightDecay))}
}
func (o *RMSprop) Step()           { C.gotorch_optim_step(o.ptr) }
func (o *RMSprop) ZeroGrad()       { C.gotorch_optim_zero_grad(o.ptr) }
func (o *RMSprop) GetLR() float64  { return float64(C.gotorch_optim_get_lr(o.ptr)) }
func (o *RMSprop) SetLR(lr float64) { C.gotorch_optim_set_lr(o.ptr, C.double(lr)) }
func (o *RMSprop) Free()           { C.gotorch_optim_free(o.ptr) }

// ═══════════════════════════════════════════════════════════════════════════════
// LR SCHEDULERS (pure Go — wraps any Optimizer)
// ═══════════════════════════════════════════════════════════════════════════════

// StepLR decays LR by gamma every stepSize epochs. Mirrors: lr_scheduler.StepLR
type StepLR struct {
	opt      Optimizer
	StepSize int
	Gamma    float64
	epoch    int
}

func NewStepLR(opt Optimizer, stepSize int, gamma float64) *StepLR {
	return &StepLR{opt: opt, StepSize: stepSize, Gamma: gamma}
}
func (s *StepLR) Step() {
	s.epoch++
	if s.epoch%s.StepSize == 0 {
		s.opt.SetLR(s.opt.GetLR() * s.Gamma)
	}
}

// CosineAnnealingLR anneals LR on a cosine schedule. Mirrors: lr_scheduler.CosineAnnealingLR
type CosineAnnealingLR struct {
	opt    Optimizer
	TMax   int
	EtaMin float64
	baseLR float64
	step   int
}

func NewCosineAnnealingLR(opt Optimizer, tMax int, etaMin float64) *CosineAnnealingLR {
	return &CosineAnnealingLR{opt: opt, TMax: tMax, EtaMin: etaMin, baseLR: opt.GetLR()}
}
func (c *CosineAnnealingLR) Step() {
	c.step++
	t := c.step % (2 * c.TMax)
	if t > c.TMax { t = 2*c.TMax - t }
	lr := c.EtaMin + 0.5*(c.baseLR-c.EtaMin)*(1+math.Cos(math.Pi*float64(t)/float64(c.TMax)))
	c.opt.SetLR(lr)
}

// ReduceLROnPlateau reduces LR when metric stagnates. Mirrors: lr_scheduler.ReduceLROnPlateau
type ReduceLROnPlateau struct {
	opt        Optimizer
	Mode       string
	Factor     float64
	Patience   int
	MinLR      float64
	Threshold  float64
	bestVal    float64
	noImpCount int
}

func NewReduceLROnPlateau(opt Optimizer, mode string, factor float64, patience int, minLR float64) *ReduceLROnPlateau {
	best := math.Inf(1)
	if mode == "max" { best = math.Inf(-1) }
	return &ReduceLROnPlateau{opt: opt, Mode: mode, Factor: factor, Patience: patience, MinLR: minLR, Threshold: 1e-4, bestVal: best}
}

// StepWithMetric updates the scheduler given the current metric value.
func (r *ReduceLROnPlateau) StepWithMetric(metric float64) {
	improved := false
	if r.Mode == "min" { improved = metric < r.bestVal*(1-r.Threshold) } else { improved = metric > r.bestVal*(1+r.Threshold) }
	if improved { r.bestVal = metric; r.noImpCount = 0 } else { r.noImpCount++ }
	if r.noImpCount >= r.Patience {
		newLR := math.Max(r.opt.GetLR()*r.Factor, r.MinLR)
		r.opt.SetLR(newLR)
		r.noImpCount = 0
	}
}

// LinearLR linearly warms up / anneals LR. Mirrors: lr_scheduler.LinearLR
type LinearLR struct {
	opt         Optimizer
	StartFactor float64
	EndFactor   float64
	TotalIters  int
	baseLR      float64
	iter        int
}

func NewLinearLR(opt Optimizer, startFactor, endFactor float64, totalIters int) *LinearLR {
	return &LinearLR{opt: opt, StartFactor: startFactor, EndFactor: endFactor, TotalIters: totalIters, baseLR: opt.GetLR()}
}
func (l *LinearLR) Step() {
	if l.iter >= l.TotalIters { return }
	l.iter++
	pct := float64(l.iter) / float64(l.TotalIters)
	factor := l.StartFactor + (l.EndFactor-l.StartFactor)*pct
	l.opt.SetLR(l.baseLR * factor)
}

// ═══════════════════════════════════════════════════════════════════════════════
// DATA UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

// Dataset interface. Mirrors: torch.utils.data.Dataset
type Dataset interface {
	Len() int
	GetItem(i int) (*Tensor, *Tensor)
}

// Batch holds one mini-batch.
type Batch struct {
	Inputs  *Tensor
	Targets *Tensor
}

// DataLoader iterates a Dataset in mini-batches. Mirrors: torch.utils.data.DataLoader
type DataLoader struct {
	dataset   Dataset
	BatchSize int
	Shuffle   bool
	indices   []int
	pos       int
}

func NewDataLoader(dataset Dataset, batchSize int, shuffle bool) *DataLoader {
	dl := &DataLoader{dataset: dataset, BatchSize: batchSize, Shuffle: shuffle}
	dl.Reset()
	return dl
}

func (dl *DataLoader) Reset() {
	n := dl.dataset.Len()
	dl.indices = make([]int, n)
	for i := range dl.indices { dl.indices[i] = i }
	if dl.Shuffle {
		// Fisher-Yates using Go's math/rand would need import;
		// simple deterministic shuffle for now (users can seed)
		for i := n - 1; i > 0; i-- {
			j := (i * 6700417) % (i + 1) // cheap pseudo-shuffle
			dl.indices[i], dl.indices[j] = dl.indices[j], dl.indices[i]
		}
	}
	dl.pos = 0
}

func (dl *DataLoader) HasNext() bool { return dl.pos < len(dl.indices) }
func (dl *DataLoader) Len() int      { return (dl.dataset.Len() + dl.BatchSize - 1) / dl.BatchSize }

func (dl *DataLoader) Next() *Batch {
	end := dl.pos + dl.BatchSize
	if end > len(dl.indices) { end = len(dl.indices) }
	idxs := dl.indices[dl.pos:end]
	dl.pos = end

	inputs  := make([]*Tensor, len(idxs))
	targets := make([]*Tensor, len(idxs))
	for i, idx := range idxs {
		inputs[i], targets[i] = dl.dataset.GetItem(idx)
	}
	return &Batch{
		Inputs:  Stack(inputs, 0),
		Targets: Stack(targets, 0),
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════════

// SaveTensor saves a tensor to disk. Mirrors: torch.save(tensor, path)
func SaveTensor(t *Tensor, path string) {
	cs := C.CString(path)
	defer C.free(unsafe.Pointer(cs))
	C.gotorch_save_tensor(t.ptr, cs)
}

// LoadTensor loads a tensor from disk. Mirrors: torch.load(path)
func LoadTensor(path string) *Tensor {
	cs := C.CString(path)
	defer C.free(unsafe.Pointer(cs))
	return wrap(C.gotorch_load_tensor(cs))
}

// ═══════════════════════════════════════════════════════════════════════════════
// VERSION
// ═══════════════════════════════════════════════════════════════════════════════

// Version returns the GoTorch version string.
func Version() string { return "GoTorch 2.0 (single-import, libtorch C++ backend)" }
