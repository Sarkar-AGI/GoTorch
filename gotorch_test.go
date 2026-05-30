here// Copyright (c) 2024-2026 Sarkar-AGI
// SPDX-License-Identifier: MIT
//
// gotorch_test.go — Go test runner for GoTorch.
//
// 
//   1. Go-level unit tests  → `go test ./...` 
//   2. PyTorch C++ tests    → build
//
// Run all:
//   go test -v -timeout 300s .
//
// Run only Go tests:
//   go test -v -run TestGo .
//
// Run only PyTorch C++ tests:
//   go test -v -run TestPyTorch .

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
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// ═══════════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

// repoRoot returns the absolute path to the GoTorch repo root.
func repoRoot() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Dir(file)
}

// almostEqual checks two float64 values are within tolerance.
func almostEqual(a, b, tol float64) bool {
	return math.Abs(a-b) < tol
}

// ═══════════════════════════════════════════════════════════════════════════════
// GO-LEVEL UNIT TESTS
// These test the Go API layer (gotorch.go → torch_api.cpp → libtorch).
// ═══════════════════════════════════════════════════════════════════════════════

// TestGoTensorCreation tests basic tensor constructors.
func TestGoTensorCreation(t *testing.T) {
	tests := []struct {
		name  string
		shape []int
	}{
		{"zeros_1d", []int{10}},
		{"zeros_2d", []int{4, 8}},
		{"zeros_3d", []int{2, 3, 4}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			x := Zeros(tc.shape...)
			if x == nil {
				t.Fatal("Zeros returned nil")
			}
			got := x.Shape()
			if len(got) != len(tc.shape) {
				t.Fatalf("ndim: want %d got %d", len(tc.shape), len(got))
			}
			for i, s := range tc.shape {
				if got[i] != s {
					t.Fatalf("shape[%d]: want %d got %d", i, s, got[i])
				}
			}
		})
	}
}

// TestGoRandn checks randn shape and numel.
func TestGoRandn(t *testing.T) {
	x := Randn(3, 4, 5)
	if x.Numel() != 60 {
		t.Fatalf("numel: want 60 got %d", x.Numel())
	}
}

// TestGoOnes checks ones tensor values via item on a scalar.
func TestGoOnesItem(t *testing.T) {
	x := Ones(1)
	if !almostEqual(x.Item(), 1.0, 1e-6) {
		t.Fatalf("ones item: want 1.0 got %.6f", x.Item())
	}
}

// TestGoAdd checks element-wise addition.
func TestGoAdd(t *testing.T) {
	a := Ones(1)
	b := Ones(1)
	c := Add(a, b)
	if !almostEqual(c.Item(), 2.0, 1e-6) {
		t.Fatalf("add: want 2.0 got %.6f", c.Item())
	}
}

// TestGoMatMul checks matrix multiplication output shape.
func TestGoMatMul(t *testing.T) {
	a := Randn(3, 4)
	b := Randn(4, 5)
	c := MatMul(a, b)
	shape := c.Shape()
	if shape[0] != 3 || shape[1] != 5 {
		t.Fatalf("matmul shape: want [3 5] got %v", shape)
	}
}

// TestGoReshape checks reshape.
func TestGoReshape(t *testing.T) {
	x := Randn(2, 3, 4)
	y := x.Reshape(6, 4)
	shape := y.Shape()
	if shape[0] != 6 || shape[1] != 4 {
		t.Fatalf("reshape: want [6 4] got %v", shape)
	}
}

// TestGoFlatten checks flatten.
func TestGoFlatten(t *testing.T) {
	x := Randn(4, 3, 8, 8)
	y := x.Flatten(1, -1)
	shape := y.Shape()
	if shape[0] != 4 || shape[1] != 192 {
		t.Fatalf("flatten: want [4 192] got %v", shape)
	}
}

// TestGoTranspose checks transpose.
func TestGoTranspose(t *testing.T) {
	x := Randn(3, 5)
	y := x.Transpose(0, 1)
	shape := y.Shape()
	if shape[0] != 5 || shape[1] != 3 {
		t.Fatalf("transpose: want [5 3] got %v", shape)
	}
}

// TestGoActivations checks activation functions return correct shape.
func TestGoActivations(t *testing.T) {
	x := Randn(4, 8)
	fns := []struct {
		name string
		fn   func(*Tensor) *Tensor
	}{
		{"relu", ReLU},
		{"sigmoid", Sigmoid},
		{"tanh", TanhF},
		{"gelu", GELU},
		{"silu", SiLU},
		{"selu", SELU},
		{"mish", Mish},
		{"hardswish", Hardswish},
	}
	for _, f := range fns {
		t.Run(f.name, func(t *testing.T) {
			y := f.fn(x)
			if fmt.Sprint(y.Shape()) != fmt.Sprint(x.Shape()) {
				t.Fatalf("%s shape mismatch: want %v got %v", f.name, x.Shape(), y.Shape())
			}
		})
	}
}

// TestGoLinear checks nn.Linear forward shape.
func TestGoLinear(t *testing.T) {
	layer := NewLinear(16, 32, true)
	x := Randn(8, 16)
	y := layer.Forward(x)
	shape := y.Shape()
	if shape[0] != 8 || shape[1] != 32 {
		t.Fatalf("linear output: want [8 32] got %v", shape)
	}
}

// TestGoSequential checks Sequential forward.
func TestGoSequential(t *testing.T) {
	model := NewSequential(
		NewLinear(32, 64, true),
		NewReLUModule(),
		NewLinear(64, 10, true),
	)
	x := Randn(4, 32)
	y := model.Forward(x)
	shape := y.Shape()
	if shape[0] != 4 || shape[1] != 10 {
		t.Fatalf("sequential output: want [4 10] got %v", shape)
	}
}

// TestGoConv2d checks Conv2d forward shape.
func TestGoConv2d(t *testing.T) {
	conv := NewConv2d(3, 16, 3, 1, 1, true)
	x := Randn(2, 3, 28, 28)
	y := conv.Forward(x)
	shape := y.Shape()
	// output: (2, 16, 28, 28) — same spatial due to padding=1
	if shape[0] != 2 || shape[1] != 16 || shape[2] != 28 || shape[3] != 28 {
		t.Fatalf("conv2d output: want [2 16 28 28] got %v", shape)
	}
}

// TestGoBatchNorm2d checks BatchNorm2d forward shape.
func TestGoBatchNorm2d(t *testing.T) {
	bn := NewBatchNorm2d(16)
	bn.Train()
	x := Randn(2, 16, 8, 8)
	y := bn.Forward(x)
	if fmt.Sprint(y.Shape()) != fmt.Sprint(x.Shape()) {
		t.Fatalf("batchnorm2d shape mismatch")
	}
}

// TestGoDropout checks Dropout output shape (training mode).
func TestGoDropout(t *testing.T) {
	d := NewDropout(0.5)
	d.Train()
	x := Randn(4, 8)
	y := d.Forward(x)
	if fmt.Sprint(y.Shape()) != fmt.Sprint(x.Shape()) {
		t.Fatalf("dropout shape mismatch")
	}
}

// TestGoMaxPool2d checks MaxPool2d output shape.
func TestGoMaxPool2d(t *testing.T) {
	pool := NewMaxPool2d(2, 2, 0, 1)
	x := Randn(2, 8, 16, 16)
	y := pool.Forward(x)
	shape := y.Shape()
	if shape[2] != 8 || shape[3] != 8 {
		t.Fatalf("maxpool2d output spatial: want [8 8] got [%d %d]", shape[2], shape[3])
	}
}

// TestGoAdaptiveAvgPool2d checks AdaptiveAvgPool2d output shape.
func TestGoAdaptiveAvgPool2d(t *testing.T) {
	pool := NewAdaptiveAvgPool2d(1, 1)
	x := Randn(4, 32, 7, 7)
	y := pool.Forward(x)
	shape := y.Shape()
	if shape[2] != 1 || shape[3] != 1 {
		t.Fatalf("adaptive avg pool output: want [1 1] got [%d %d]", shape[2], shape[3])
	}
}

// TestGoEmbedding checks Embedding forward shape.
func TestGoEmbedding(t *testing.T) {
	emb := NewEmbedding(1000, 64)
	idx := Zeros(4, 10).Cast(Int64)
	y := emb.Forward(idx)
	shape := y.Shape()
	if shape[0] != 4 || shape[1] != 10 || shape[2] != 64 {
		t.Fatalf("embedding output: want [4 10 64] got %v", shape)
	}
}

// TestGoLSTM checks LSTM forward output shape.
func TestGoLSTM(t *testing.T) {
	lstm := NewLSTM(32, 64, 2, true, true, 0.0, false)
	lstm.Train()
	x := Randn(8, 20, 32) // (batch, seq, input)
	out := lstm.ForwardLSTM(x, nil, nil)
	shape := out.Output.Shape()
	// batch_first=true → (batch, seq, hidden)
	if shape[0] != 8 || shape[1] != 20 || shape[2] != 64 {
		t.Fatalf("lstm output: want [8 20 64] got %v", shape)
	}
}

// TestGoGRU checks GRU forward output shape.
func TestGoGRU(t *testing.T) {
	gru := NewGRU(32, 64, 1, true, true, 0.0, false)
	gru.Train()
	x := Randn(4, 15, 32)
	out := gru.ForwardGRU(x, nil)
	shape := out.Output.Shape()
	if shape[0] != 4 || shape[1] != 15 || shape[2] != 64 {
		t.Fatalf("gru output: want [4 15 64] got %v", shape)
	}
}

// TestGoTransformerEncoder checks TransformerEncoder output shape.
func TestGoTransformerEncoder(t *testing.T) {
	layer := NewTransformerEncoderLayer(128, 8, 512, 0.1)
	enc := NewTransformerEncoder(layer, 3)
	enc.Train()
	// (seq_len, batch, d_model)
	src := Randn(10, 4, 128)
	out := enc.Forward(src)
	if fmt.Sprint(out.Shape()) != fmt.Sprint(src.Shape()) {
		t.Fatalf("transformer encoder shape mismatch: %v != %v", out.Shape(), src.Shape())
	}
}

// TestGoLossFunctions checks loss functions return scalar.
func TestGoLossFunctions(t *testing.T) {
	pred := Randn(8, 10)
	tgt := Zeros(8).Cast(Int64)

	loss := CrossEntropyLoss(pred, tgt, ReduceMean)
	if loss.Numel() != 1 {
		t.Fatalf("cross_entropy not scalar: numel=%d", loss.Numel())
	}

	predReg := Randn(8, 1)
	tgtReg := Randn(8, 1)
	mseLoss := MSELoss(predReg, tgtReg, ReduceMean)
	if mseLoss.Numel() != 1 {
		t.Fatalf("mse_loss not scalar: numel=%d", mseLoss.Numel())
	}
}

// TestGoBackward checks that .Backward() populates gradients.
func TestGoBackward(t *testing.T) {
	layer := NewLinear(4, 2, true)
	x := Randn(3, 4)
	y := layer.Forward(x)
	loss := y.Sum()
	loss.Backward()

	w := layer.Weight()
	g := w.Grad()
	if g == nil {
		t.Fatal("weight gradient is nil after backward")
	}
}

// TestGoOptimizer checks that optimizer.Step() changes parameters.
func TestGoOptimizer(t *testing.T) {
	layer := NewLinear(4, 2, true)
	x := Randn(3, 4)

	// Record initial weight value
	w0 := layer.Weight().Clone().Item()

	opt := NewAdam(layer.Parameters(), AdamOptions{LR: 1e-1})
	opt.ZeroGrad()
	loss := layer.Forward(x).Sum()
	loss.Backward()
	opt.Step()

	w1 := layer.Weight().Item()
	if w0 == w1 {
		t.Fatal("weight did not change after optimizer step")
	}
}

// TestGoNoGrad checks WithNoGrad disables gradient tracking.
func TestGoNoGrad(t *testing.T) {
	if !IsGradEnabled() {
		t.Fatal("grad should be enabled initially")
	}
	WithNoGrad(func() {
		if IsGradEnabled() {
			t.Error("grad should be disabled inside WithNoGrad")
		}
	})
	if !IsGradEnabled() {
		t.Fatal("grad should be re-enabled after WithNoGrad")
	}
}

// TestGoStepLR checks StepLR reduces learning rate.
func TestGoStepLR(t *testing.T) {
	layer := NewLinear(2, 2, false)
	opt := NewSGD(layer.Parameters(), SGDOptions{LR: 0.1, Momentum: 0.9})
	sched := NewStepLR(opt, 3, 0.5)

	initialLR := opt.GetLR()
	for i := 0; i < 3; i++ {
		sched.Step()
	}
	newLR := opt.GetLR()
	if newLR >= initialLR {
		t.Fatalf("StepLR: LR should have decreased, got %.6f → %.6f", initialLR, newLR)
	}
}

// TestGoDataLoader checks DataLoader iteration.
func TestGoDataLoader(t *testing.T) {
	ds := &testDataset{n: 100, featDim: 8}
	dl := NewDataLoader(ds, 16, false)

	batches := 0
	dl.Reset()
	for dl.HasNext() {
		b := dl.Next()
		if b == nil {
			t.Fatal("batch is nil")
		}
		batches++
	}
	// 100 samples / 16 batch_size = 7 batches (last batch may be smaller)
	if batches != 7 {
		t.Fatalf("expected 7 batches, got %d", batches)
	}
}

// testDataset is a minimal Dataset for testing.
type testDataset struct {
	n       int
	featDim int
}

func (d *testDataset) Len() int { return d.n }
func (d *testDataset) GetItem(i int) (*Tensor, *Tensor) {
	return Randn(d.featDim), Zeros(1).Cast(Int64)
}

// ═══════════════════════════════════════════════════════════════════════════════
// PYTORCH C++ TEST RUNNER
// Calls the compiled PyTorch C++ test binaries from build/.
// Mirrors: python -m pytest test/ (but via compiled C++ test executables)
// ═══════════════════════════════════════════════════════════════════════════════

// pytorchTestBinary maps a test name to its compiled binary path under build/.
var pytorchTestBinaries = []struct {
	name   string // Go test sub-name
	binary string // path relative to repo root
	args   []string
}{
	{
		name:   "test_tensor_basic",
		binary: "build/bin/test_tensor_basic",
		args:   []string{"--gtest_filter=*"},
	},
	{
		name:   "test_tensor_api",
		binary: "build/bin/test_tensor_api",
		args:   []string{"--gtest_filter=*"},
	},
	{
		name:   "test_nn_modules",
		binary: "build/bin/test_nn_modules",
		args:   []string{"--gtest_filter=*"},
	},
	{
		name:   "test_autograd",
		binary: "build/bin/test_autograd",
		args:   []string{"--gtest_filter=*"},
	},
	{
		name:   "test_dispatch_key",
		binary: "build/bin/test_dispatch_key",
		args:   []string{"--gtest_filter=*"},
	},
	{
		name:   "test_cpu_ops",
		binary: "build/bin/test_cpu_ops",
		args:   []string{"--gtest_filter=*"},
	},
}

// TestPyTorchCppTests runs compiled PyTorch C++ GTest binaries.
//
// These are the same tests PyTorch CI runs, now callable from `go test`.
//
// Prerequisites:
//
//	cmake --build build --target test_tensor_basic test_nn_modules ... -j$(nproc)
//
// Skip individual tests by setting GOTORCH_SKIP_CPP_TESTS=1.
func TestPyTorchCppTests(t *testing.T) {
	if os.Getenv("GOTORCH_SKIP_CPP_TESTS") == "1" {
		t.Skip("GOTORCH_SKIP_CPP_TESTS=1 — skipping PyTorch C++ tests")
	}

	root := repoRoot()

	for _, tc := range pytorchTestBinaries {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			binPath := filepath.Join(root, tc.binary)

			// Skip if binary not built yet
			if _, err := os.Stat(binPath); os.IsNotExist(err) {
				t.Skipf("binary not found: %s (run cmake --build build --target %s)",
					binPath, tc.name)
				return
			}

			cmd := exec.Command(binPath, tc.args...)
			cmd.Dir = root

			// Set LD_LIBRARY_PATH so libtorch shared libs are found
			libPath := filepath.Join(root, "build", "lib")
			cmd.Env = append(os.Environ(),
				fmt.Sprintf("LD_LIBRARY_PATH=%s:%s", libPath, os.Getenv("LD_LIBRARY_PATH")),
				fmt.Sprintf("DYLD_LIBRARY_PATH=%s:%s", libPath, os.Getenv("DYLD_LIBRARY_PATH")), // macOS
			)

			out, err := cmd.CombinedOutput()
			output := string(out)

			// Print GTest output to Go test log
			for _, line := range strings.Split(output, "\n") {
				t.Log(line)
			}

			if err != nil {
				t.Fatalf("PyTorch C++ test failed: %s\nerror: %v", tc.name, err)
			}

			// Check GTest summary
			if strings.Contains(output, "FAILED") {
				t.Fatalf("PyTorch C++ test reported failures: %s", tc.name)
			}
		})
	}
}

// TestPyTorchCppTestsCustom lets you run any C++ test binary by path.
//
// Usage:
//
//	GOTORCH_CPP_TEST=build/bin/test_tensor_api go test -v -run TestPyTorchCppTestsCustom .
func TestPyTorchCppTestsCustom(t *testing.T) {
	binPath := os.Getenv("GOTORCH_CPP_TEST")
	if binPath == "" {
		t.Skip("set GOTORCH_CPP_TEST=<path> to run a custom C++ test binary")
	}

	root := repoRoot()
	fullPath := filepath.Join(root, binPath)

	if _, err := os.Stat(fullPath); os.IsNotExist(err) {
		t.Fatalf("binary not found: %s", fullPath)
	}

	filter := os.Getenv("GOTORCH_CPP_TEST_FILTER")
	if filter == "" {
		filter = "*"
	}

	cmd := exec.Command(fullPath, "--gtest_filter="+filter)
	cmd.Dir = root
	libPath := filepath.Join(root, "build", "lib")
	cmd.Env = append(os.Environ(),
		fmt.Sprintf("LD_LIBRARY_PATH=%s:%s", libPath, os.Getenv("LD_LIBRARY_PATH")),
		fmt.Sprintf("DYLD_LIBRARY_PATH=%s:%s", libPath, os.Getenv("DYLD_LIBRARY_PATH")),
	)

	out, err := cmd.CombinedOutput()
	for _, line := range strings.Split(string(out), "\n") {
		t.Log(line)
	}
	if err != nil {
		t.Fatalf("test failed: %v", err)
	}
}

// TestPyTorchPythonTests runs PyTorch's Python test suite via subprocess.
// This mirrors CI: python test/run_tests.py
//
// Usage:
//
//	GOTORCH_RUN_PY_TESTS=1 go test -v -run TestPyTorchPythonTests -timeout 600s .
func TestPyTorchPythonTests(t *testing.T) {
	if os.Getenv("GOTORCH_RUN_PY_TESTS") != "1" {
		t.Skip("set GOTORCH_RUN_PY_TESTS=1 to run PyTorch Python tests")
	}

	root := repoRoot()

	// Choose specific test files or all
	testFiles := strings.Split(os.Getenv("GOTORCH_PY_TEST_FILES"), ",")
	if len(testFiles) == 0 || testFiles[0] == "" {
		testFiles = []string{
			"test/test_torch.py",
			"test/test_nn.py",
			"test/test_autograd.py",
			"test/test_ops.py",
		}
	}

	for _, tf := range testFiles {
		tf := tf
		t.Run(filepath.Base(tf), func(t *testing.T) {
			t.Parallel()

			testPath := filepath.Join(root, tf)
			if _, err := os.Stat(testPath); os.IsNotExist(err) {
				t.Skipf("test file not found: %s", testPath)
				return
			}

			cmd := exec.Command("python3", testPath, "-v")
			cmd.Dir = root
			cmd.Env = append(os.Environ(),
				fmt.Sprintf("PYTHONPATH=%s", root),
			)

			out, err := cmd.CombinedOutput()
			for _, line := range strings.Split(string(out), "\n") {
				t.Log(line)
			}
			if err != nil {
				t.Fatalf("Python test failed: %s\n%v", tf, err)
			}
		})
	}
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARK
// ═══════════════════════════════════════════════════════════════════════════════

// BenchmarkLinearForward benchmarks a Linear forward pass.
// Run: go test -bench=BenchmarkLinear -benchmem .
func BenchmarkLinearForward(b *testing.B) {
	layer := NewLinear(512, 512, true)
	x := Randn(64, 512)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = layer.Forward(x)
	}
}

// BenchmarkConv2dForward benchmarks a Conv2d forward pass.
func BenchmarkConv2dForward(b *testing.B) {
	conv := NewConv2d(64, 64, 3, 1, 1, true)
	x := Randn(16, 64, 32, 32)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = conv.Forward(x)
	}
}

// BenchmarkMatMul benchmarks matrix multiplication.
func BenchmarkMatMul(b *testing.B) {
	a := Randn(256, 256)
	bTensor := Randn(256, 256)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = MatMul(a, bTensor)
	}
}
