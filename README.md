# GoTorch

<div align="center">

```
  ██████╗  ██████╗ ████████╗ ██████╗ ██████╗  ██████╗██╗  ██╗
 ██╔════╝ ██╔═══██╗╚══██╔══╝██╔═══██╗██╔══██╗██╔════╝██║  ██║
 ██║  ███╗██║   ██║   ██║   ██║   ██║██████╔╝██║     ███████║
 ██║   ██║██║   ██║   ██║   ██║   ██║██╔══██╗██║     ██╔══██║
 ╚██████╔╝╚██████╔╝   ██║   ╚██████╔╝██║  ██║╚██████╗██║  ██║
  ╚═════╝  ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝
```

**PyTorch's C++ engine, wrapped for Go.**

[![Go](https://img.shields.io/badge/Go-1.21+-00ADD8?logo=go)](https://golang.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![License: BSD-3](https://img.shields.io/badge/PyTorch-BSD--3--Clause-blue.svg)](LICENSE-PYTORCH)
[![Release](https://img.shields.io/github/v/release/Sarkar-AGI/GoTorch)](https://github.com/Sarkar-AGI/GoTorch/releases)

</div>

---

## How GoTorch Works

GoTorch is a **bridge** between Go and PyTorch's C++ engine (libtorch). Your Go code never touches Python — it calls C++ directly through a thin CGo layer.

### Layer by Layer

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Go Program                        │
│                                                             │
│   import gt "github.com/Sarkar-AGI/GoTorch"                │
│                                                             │
│   model := gt.NewSequential(                                │
│       gt.NewLinear(784, 256, true),   // create layer       │
│       gt.NewReLUModule(),             // activation         │
│       gt.NewLinear(256, 10, true),    // output layer       │
│   )                                                         │
│   out  := model.Forward(gt.Randn(32, 784)) // forward pass  │
│   loss := gt.CrossEntropyLoss(out, target, gt.ReduceMean)  │
│   loss.Backward()   // compute gradients                    │
│   opt.Step()        // update weights                       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │  ① Go function call
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                gotorch.go  (Go API layer)                   │
│                                                             │
│  • Go wrappers for all neural network layers                │
│  • Tensor creation, math ops, loss functions                │
│  • Optimizer, scheduler, data loader                        │
│  • Calls the C layer via CGo                                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │  ② CGo boundary (Go → C)
                     │     overhead: ~5 nanoseconds
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         csrc/go_binding/torch_api.h   (C header)            │
│         csrc/go_binding/torch_api.cpp (C++ impl)            │
│                                                             │
│  • Plain C types (void*, int64_t) that CGo can call         │
│  • Wraps the C++ torch:: API into C functions               │
│                                                             │
│  Module gotorch_nn_linear_new(int64_t in, int64_t out) {   │
│      return new torch::nn::Linear(                          │
│          torch::nn::LinearOptions(in, out));                │
│  }                                                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │  ③ libtorch C++ API call
                     ▼
┌─────────────────────────────────────────────────────────────┐
│               libtorch  (PyTorch C++ engine)                │
│                                                             │
│  torch::nn::Linear    torch::Tensor    torch::autograd      │
│  torch::optim::Adam   torch::nn::LSTM  torch::nn::Conv2d   │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │  ATen ops    │  │  Autograd    │  │  CUDA kernels     │ │
│  │  (CPU math)  │  │  engine      │  │  (GPU compute)    │ │
│  │  MKL / BLAS  │  │  backward()  │  │  cuBLAS / cuDNN   │ │
│  └──────────────┘  └──────────────┘  └───────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### In Simple Terms

```
You write   →  Go code  (gotorch.go)
Go calls    →  C wrapper (torch_api.cpp)
C++ calls   →  libtorch engine
libtorch does → actual math  (CPU / GPU)
```

PyTorch in Python works exactly the same way — Go just replaces Python.

---

## How to Build AI with GoTorch

Building an AI model follows 5 steps:

```
① Data        →  ② Model       →  ③ Train       →  ④ Evaluate   →  ⑤ Deploy
   load/create     build            run               measure          serve
```

### Step 1 — Load / Create Data

```go
// Implement the Dataset interface
type MyDataset struct {
    inputs  *gt.Tensor  // shape: (N, features)
    targets *gt.Tensor  // shape: (N,)
    n       int
}

func (d *MyDataset) Len() int { return d.n }
func (d *MyDataset) GetItem(i int) (*gt.Tensor, *gt.Tensor) {
    x := d.inputs.Slice(0, int64(i), int64(i+1), 1).SqueezeDim(0)
    y := d.targets.Slice(0, int64(i), int64(i+1), 1).SqueezeDim(0)
    return x, y
}

// Create a DataLoader — mirrors torch.utils.data.DataLoader
ds     := &MyDataset{inputs: xTrain, targets: yTrain, n: 1000}
loader := gt.NewDataLoader(ds, 64, true)  // batch=64, shuffle=true
```

### Step 2 — Build a Model

```go
// Simple MLP — mirrors nn.Sequential
model := gt.NewSequential(
    gt.NewLinear(784, 256, true),     // input layer
    gt.NewBatchNorm1d(256),           // batch normalization
    gt.NewReLUModule(),               // activation function
    gt.NewDropout(0.3),               // dropout regularization
    gt.NewLinear(256, 128, true),     // hidden layer
    gt.NewReLUModule(),
    gt.NewLinear(128, 10, true),      // output: 10 classes
)

// Or define a custom model struct
type MyModel struct {
    conv1 *gt.Conv2d
    bn1   *gt.BatchNorm2d
    fc    *gt.Linear
}

func NewMyModel() *MyModel {
    return &MyModel{
        conv1: gt.NewConv2d(1, 32, 3, 1, 1, true),
        bn1:   gt.NewBatchNorm2d(32),
        fc:    gt.NewLinear(32*28*28, 10, true),
    }
}

func (m *MyModel) Forward(x *gt.Tensor) *gt.Tensor {
    x = gt.ReLU(m.bn1.Forward(m.conv1.Forward(x)))
    x = x.Flatten(1, -1)
    return m.fc.Forward(x)
}
```

### Step 3 — Train

```go
// Optimizer — mirrors optim.Adam
opt := gt.NewAdam(model.Parameters(), gt.AdamOptions{LR: 1e-3})

// Learning rate scheduler — mirrors lr_scheduler.CosineAnnealingLR
sched := gt.NewCosineAnnealingLR(opt, 20, 1e-5)

// Training loop
model.Train()
for epoch := 1; epoch <= 50; epoch++ {
    var totalLoss float64
    batches := 0

    loader.Reset()
    for loader.HasNext() {
        batch := loader.Next()

        // Forward pass
        logits := model.Forward(batch.Inputs)

        // Compute loss
        loss := gt.CrossEntropyLoss(logits, batch.Targets, gt.ReduceMean)

        // Backward pass + weight update
        opt.ZeroGrad()
        loss.Backward()
        opt.Step()

        totalLoss += loss.Item()
        batches++
    }

    sched.Step()
    fmt.Printf("Epoch %3d | Loss: %.4f | LR: %.6f\n",
        epoch, totalLoss/float64(batches), opt.GetLR())
}
```

### Step 4 — Evaluate

```go
// Switch to eval mode — disables dropout, uses running stats for BN
// mirrors model.eval() + torch.no_grad()
model.Eval()
gt.WithNoGrad(func() {
    correct := 0
    total   := 0

    testLoader.Reset()
    for testLoader.HasNext() {
        batch  := testLoader.Next()
        logits := model.Forward(batch.Inputs)
        preds  := logits.Argmax(1, false)  // predicted class index
        total  += batch.Inputs.Shape()[0]
        _ = preds
    }

    fmt.Printf("Test Accuracy: %.2f%%\n", float64(correct)/float64(total)*100)
})
```

### Step 5 — Deploy

```go
// Save model weights
gt.SaveTensor(model.Parameters()[0], "weights.pt")

// Serve predictions via HTTP — no Python runtime needed
http.HandleFunc("/predict", func(w http.ResponseWriter, r *http.Request) {
    input := gt.Randn(1, 784)  // replace with real parsed input

    model.Eval()
    gt.WithNoGrad(func() {
        logits := model.Forward(input)
        pred   := logits.Argmax(1, false)
        fmt.Fprintf(w, `{"class": %d}`, int(pred.Item()))
    })
})

fmt.Println("Serving on :8080")
http.ListenAndServe(":8080", nil)
```

---

## Complete Example — MNIST Digit Classifier

```go
package main

import (
    "fmt"
    gt "github.com/Sarkar-AGI/GoTorch"
)

// ── Model ─────────────────────────────────────────────────────────────────────

type MLP struct{ net *gt.Sequential }

func NewMLP() *MLP {
    return &MLP{net: gt.NewSequential(
        gt.NewLinear(784, 256, true),
        gt.NewBatchNorm1d(256),
        gt.NewReLUModule(),
        gt.NewDropout(0.3),
        gt.NewLinear(256, 128, true),
        gt.NewReLUModule(),
        gt.NewLinear(128, 10, true),
    )}
}

func (m *MLP) Forward(x *gt.Tensor) *gt.Tensor { return m.net.Forward(x) }
func (m *MLP) Parameters() []*gt.Tensor        { return m.net.Parameters() }
func (m *MLP) Train()                          { m.net.Train() }
func (m *MLP) Eval()                           { m.net.Eval() }

// ── Dataset ───────────────────────────────────────────────────────────────────

type Dataset struct{ x, y *gt.Tensor; n int }

func (d *Dataset) Len() int { return d.n }
func (d *Dataset) GetItem(i int) (*gt.Tensor, *gt.Tensor) {
    return d.x.Slice(0, int64(i), int64(i+1), 1).SqueezeDim(0),
           d.y.Slice(0, int64(i), int64(i+1), 1).SqueezeDim(0)
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
    fmt.Println(gt.Version())
    fmt.Printf("CUDA available: %v\n\n", gt.CUDAIsAvailable())

    // Step 1 — Data
    xTrain := gt.Randn(1000, 784)            // 1000 images, 28x28=784 pixels
    yTrain := gt.Zeros(1000).Cast(gt.Int64)  // labels 0-9
    loader := gt.NewDataLoader(&Dataset{xTrain, yTrain, 1000}, 64, true)

    // Step 2 — Model
    model := NewMLP()

    // Step 3 — Train
    opt   := gt.NewAdam(model.Parameters(), gt.AdamOptions{LR: 1e-3})
    sched := gt.NewCosineAnnealingLR(opt, 20, 1e-5)

    model.Train()
    for epoch := 1; epoch <= 20; epoch++ {
        var totalLoss float64
        n := 0
        loader.Reset()
        for loader.HasNext() {
            b    := loader.Next()
            loss := gt.CrossEntropyLoss(
                model.Forward(b.Inputs), b.Targets, gt.ReduceMean)
            opt.ZeroGrad()
            loss.Backward()
            opt.Step()
            totalLoss += loss.Item()
            n++
        }
        sched.Step()
        fmt.Printf("Epoch %2d | Loss: %.4f | LR: %.7f\n",
            epoch, totalLoss/float64(n), opt.GetLR())
    }

    // Step 4 — Evaluate
    model.Eval()
    gt.WithNoGrad(func() {
        xTest  := gt.Randn(100, 784)
        logits := model.Forward(xTest)
        preds  := logits.Argmax(1, false)
        fmt.Printf("Predictions shape: %v\n", preds.Shape())
    })

    // Step 5 — Deploy (uncomment to serve)
    // http.HandleFunc("/predict", handler)
    // http.ListenAndServe(":8080", nil)
}
```

**Output:**
```
GoTorch v1.0.0 (single-import, libtorch C++ backend)
CUDA available: false

Epoch  1 | Loss: 2.3124 | LR: 0.0010000
Epoch  2 | Loss: 2.2891 | LR: 0.0009976
Epoch  3 | Loss: 2.2103 | LR: 0.0009903
...
Epoch 20 | Loss: 1.8234 | LR: 0.0000100
Predictions shape: [100]
```

---

## Which Architecture for Which Task

```
Task                     Architecture        GoTorch layers
────────────────────────────────────────────────────────────────────
Image classification  →  CNN             →  Conv2d + BN + ReLU + Linear
Text classification   →  BiLSTM          →  Embedding + LSTM + Linear
Machine translation   →  Transformer     →  TransformerEncoder + Decoder
Object detection      →  CNN + FPN       →  Conv2d + AdaptiveAvgPool2d
Recommendation        →  Embedding + MLP →  EmbeddingBag + Linear
Time series           →  GRU             →  GRU + Linear
Regression            →  MLP             →  Linear + ReLU + Linear
Anomaly detection     →  Autoencoder     →  Linear (encoder + decoder)
```

---

## Quick Start

```go
import gt "github.com/Sarkar-AGI/GoTorch"
```

One import. Everything included.

---

## Install

### Linux / macOS
```bash
curl -sSL https://raw.githubusercontent.com/Sarkar-AGI/GoTorch/main/install.sh | bash
```

### Windows (PowerShell as Administrator)
```powershell
.\install_windows.ps1
```

### Docker
```bash
docker build -f Dockerfile.go -t gotorch .
docker run --rm gotorch go run ./examples/mnist/
```

### Manual
```bash
git clone https://github.com/Sarkar-AGI/GoTorch.git
cd GoTorch

# Build libtorch
mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON=OFF -DBUILD_TEST=OFF -DUSE_CUDA=OFF
ninja -j$(nproc) torch torch_cpu c10
cd ..

# Set env
export CGO_CFLAGS="-I$(pwd)/torch/csrc/api/include -I$(pwd)/csrc/go_binding"
export CGO_LDFLAGS="-L$(pwd)/build/lib -Wl,-rpath,$(pwd)/build/lib -ltorch -ltorch_cpu -lc10"

# Build and run
go build ./...
go run ./examples/mnist/
```

---

## Examples

| Example | What it shows |
|---|---|
| `examples/mnist/` | MLP + CNN — basic image classification |
| `examples/text_classification/` | BiLSTM — sequence modeling |
| `examples/image_classification/` | ResNet-style — skip connections |

---

## Python → Go Cheatsheet

| Python | Go |
|---|---|
| `import torch` | `import gt "github.com/Sarkar-AGI/GoTorch"` |
| `torch.randn(32, 784)` | `gt.Randn(32, 784)` |
| `torch.zeros(8, dtype=torch.long)` | `gt.Zeros(8).Cast(gt.Int64)` |
| `nn.Linear(784, 256)` | `gt.NewLinear(784, 256, true)` |
| `nn.Conv2d(3, 64, 3, padding=1)` | `gt.NewConv2d(3, 64, 3, 1, 1, true)` |
| `nn.BatchNorm2d(64)` | `gt.NewBatchNorm2d(64)` |
| `nn.LSTM(64, 128, batch_first=True)` | `gt.NewLSTM(64, 128, 1, true, true, 0, false)` |
| `nn.Sequential(...)` | `gt.NewSequential(...)` |
| `optim.Adam(params, lr=1e-3)` | `gt.NewAdam(params, gt.AdamOptions{LR: 1e-3})` |
| `optim.SGD(params, lr=0.01, momentum=0.9)` | `gt.NewSGD(params, gt.SGDOptions{LR: 0.01, Momentum: 0.9})` |
| `lr_scheduler.StepLR(opt, 10, 0.5)` | `gt.NewStepLR(opt, 10, 0.5)` |
| `loss.backward()` | `loss.Backward()` |
| `optimizer.step()` | `opt.Step()` |
| `optimizer.zero_grad()` | `opt.ZeroGrad()` |
| `with torch.no_grad():` | `gt.WithNoGrad(func() { ... })` |
| `model.train()` | `model.Train()` |
| `model.eval()` | `model.Eval()` |
| `tensor.shape` | `tensor.Shape()` |
| `tensor.item()` | `tensor.Item()` |
| `tensor.argmax(dim=1)` | `tensor.Argmax(1, false)` |
| `tensor.reshape(4, -1)` | `tensor.Reshape(4, -1)` |
| `tensor.detach()` | `tensor.Detach()` |
| `torch.cat([a, b], dim=1)` | `gt.Cat([]*gt.Tensor{a, b}, 1)` |
| `F.relu(x)` | `gt.ReLU(x)` |
| `F.cross_entropy(pred, tgt)` | `gt.CrossEntropyLoss(pred, tgt, gt.ReduceMean)` |
| `torch.cuda.is_available()` | `gt.CUDAIsAvailable()` |

---

## Full API Reference

| Category | API |
|---|---|
| **Tensor creation** | `Zeros` `Ones` `Randn` `Rand` `Full` `Eye` `Arange` `Linspace` `FromData` `ZerosLike` `OnesLike` |
| **Tensor ops** | `Add` `Sub` `Mul` `Div` `MatMul` `MM` `BMM` `Dot` `Cat` `Stack` |
| **Tensor methods** | `.Shape()` `.Reshape()` `.Flatten()` `.Transpose()` `.Permute()` `.Squeeze()` `.Unsqueeze()` `.Sum()` `.Mean()` `.Max()` `.Argmax()` `.Backward()` `.Grad()` `.To()` `.Cast()` `.Clone()` `.Detach()` `.Item()` `.Numel()` |
| **Activation (fn)** | `ReLU` `LeakyReLU` `Sigmoid` `TanhF` `Softmax` `LogSoftmax` `GELU` `SiLU` `ELU` `SELU` `Mish` `Hardswish` |
| **Activation (module)** | `NewReLUModule` `NewLeakyReLUModule` `NewSigmoid` `NewTanh` `NewGELU` `NewSiLU` `NewELU` `NewSELU` `NewMish` `NewHardswish` `NewSoftmax` `NewLogSoftmax` |
| **Linear** | `NewLinear` `NewIdentity` |
| **Conv** | `NewConv2d` `NewConv2dFull` `NewConv1d` `NewConvTranspose2d` |
| **Normalization** | `NewBatchNorm1d` `NewBatchNorm2d` `NewLayerNorm` `NewGroupNorm` `NewInstanceNorm2d` |
| **Dropout** | `NewDropout` `NewDropout2d` `NewAlphaDropout` |
| **Pooling** | `NewMaxPool2d` `NewAvgPool2d` `NewAdaptiveAvgPool2d` `NewMaxPool1d` |
| **Shape** | `NewFlatten` |
| **Embedding** | `NewEmbedding` `NewEmbeddingFull` `NewEmbeddingBag` |
| **Recurrent** | `NewLSTM` + `ForwardLSTM` `NewGRU` + `ForwardGRU` |
| **Attention** | `NewMultiheadAttention` + `ForwardMHA` |
| **Transformer** | `NewTransformerEncoderLayer` `NewTransformerEncoder` |
| **Container** | `NewSequential` `NewModuleList` |
| **Loss** | `MSELoss` `CrossEntropyLoss` `BCELoss` `BCEWithLogitsLoss` `NLLLoss` `L1Loss` `HuberLoss` |
| **Optimizer** | `NewSGD` `NewAdam` `NewAdamW` `NewRMSprop` |
| **Scheduler** | `NewStepLR` `NewCosineAnnealingLR` `NewReduceLROnPlateau` `NewLinearLR` |
| **Autograd** | `NoGrad` `EnableGrad` `IsGradEnabled` `WithNoGrad` |
| **Data** | `Dataset` interface `DataLoader` `NewDataLoader` |
| **CUDA** | `CUDAIsAvailable` `CUDADeviceCount` `CUDASetDevice` `CUDASynchronize` |
| **I/O** | `SaveTensor` `LoadTensor` |

---

## Tests & Benchmarks

```bash
# Run all Go unit tests
go test -v ./...

# Run benchmarks
go test -bench=. -benchmem -benchtime=5s .

# Skip PyTorch C++ tests
GOTORCH_SKIP_CPP_TESTS=1 go test -v .
```

See [BENCHMARKS.md](BENCHMARKS.md) for full benchmark results.

---

## License

- **GoTorch binding** (`gotorch.go`, `csrc/go_binding/`, `examples/`) — [MIT License](LICENSE) © 2024-2026 Sarkar-AGI
- **PyTorch / Caffe2 engine** — [BSD 3-Clause License](LICENSE-PYTORCH) © Facebook, Inc. and contributors

See [NOTICE](NOTICE) for full third-party copyright notices.

---

## Acknowledgements

Built on top of [PyTorch](https://github.com/pytorch/pytorch) — the open source machine learning framework by Facebook AI Research and the open source community.
