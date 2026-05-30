# GoTorch

GoTorch is a Go deep learning framework that wraps PyTorch's C++ engine (libtorch) via CGo.

## Single-import API

```go
package main

import (
    "fmt"
    gt "github.com/Sarkar-AGI/GoTorch"
)

func main() {
    // Tensors
    x := gt.Randn(32, 784)
    w := gt.Randn(784, 256)
    h := gt.MatMul(x, w)
    h = gt.ReLU(h)

    // Build a model
    model := gt.NewSequential(
        gt.NewLinear(784, 256, true),
        gt.NewBatchNorm1d(256),
        gt.NewReLUModule(),
        gt.NewDropout(0.3),
        gt.NewLinear(256, 10, true),
    )

    // Optimizer + scheduler
    opt   := gt.NewAdam(model.Parameters(), gt.AdamOptions{LR: 1e-3})
    sched := gt.NewStepLR(opt, 10, 0.5)

    // Forward + backward
    model.Train()
    out  := model.Forward(gt.Randn(32, 784))
    tgt  := gt.Zeros(32).Cast(gt.Int64)
    loss := gt.CrossEntropyLoss(out, tgt, gt.ReduceMean)
    opt.ZeroGrad()
    loss.Backward()
    opt.Step()
    sched.Step()

    fmt.Printf("loss: %.4f\n", loss.Item())

    // Inference — no grad
    model.Eval()
    gt.WithNoGrad(func() {
        preds := model.Forward(gt.Randn(8, 784)).Argmax(1, false)
        fmt.Printf("predictions: %v\n", preds.Shape())
    })
}
```

## What's available

| Category | Types |
|---|---|
| **Tensor** | `Zeros`, `Ones`, `Randn`, `Rand`, `Full`, `Eye`, `Arange`, `Linspace`, `FromData` |
| **Ops** | `Add`, `Sub`, `Mul`, `Div`, `MatMul`, `MM`, `BMM`, `Cat`, `Stack` |
| **Tensor methods** | `.Reshape()`, `.View()`, `.Flatten()`, `.Transpose()`, `.Permute()`, `.Squeeze()`, `.Unsqueeze()`, `.Argmax()`, `.Sum()`, `.Mean()`, `.Backward()` |
| **Activations (fn)** | `ReLU`, `LeakyReLU`, `Sigmoid`, `TanhF`, `Softmax`, `LogSoftmax`, `GELU`, `SiLU`, `ELU`, `SELU`, `Mish`, `Hardswish` |
| **Activations (module)** | `NewReLUModule`, `NewSigmoid`, `NewTanh`, `NewGELU`, `NewSiLU`, `NewELU`, `NewSELU`, `NewLeakyReLUModule` |
| **Linear** | `NewLinear`, `NewIdentity` |
| **Conv** | `NewConv2d`, `NewConv2dFull`, `NewConv1d`, `NewConvTranspose2d` |
| **Norm** | `NewBatchNorm1d`, `NewBatchNorm2d`, `NewLayerNorm`, `NewGroupNorm`, `NewInstanceNorm2d` |
| **Dropout** | `NewDropout`, `NewDropout2d`, `NewAlphaDropout` |
| **Pooling** | `NewMaxPool2d`, `NewAvgPool2d`, `NewAdaptiveAvgPool2d`, `NewMaxPool1d` |
| **Shape** | `NewFlatten` |
| **Embedding** | `NewEmbedding`, `NewEmbeddingFull`, `NewEmbeddingBag` |
| **Recurrent** | `NewLSTM`, `NewGRU` |
| **Attention** | `NewMultiheadAttention` |
| **Transformer** | `NewTransformerEncoderLayer`, `NewTransformerEncoder` |
| **Containers** | `NewSequential`, `NewModuleList` |
| **Loss** | `MSELoss`, `CrossEntropyLoss`, `BCELoss`, `BCEWithLogitsLoss`, `NLLLoss`, `L1Loss`, `HuberLoss` |
| **Optimizers** | `NewSGD`, `NewAdam`, `NewAdamW`, `NewRMSprop` |
| **Schedulers** | `NewStepLR`, `NewCosineAnnealingLR`, `NewReduceLROnPlateau`, `NewLinearLR` |
| **Autograd** | `NoGrad`, `EnableGrad`, `IsGradEnabled`, `WithNoGrad` |
| **Data** | `Dataset`, `DataLoader`, `NewDataLoader` |
| **CUDA** | `CUDAIsAvailable`, `CUDADeviceCount`, `CUDASetDevice`, `CUDASynchronize` |
| **I/O** | `SaveTensor`, `LoadTensor` |

## Build

```bash
# 1. Build the C++ backend (libtorch is bundled in this repo)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=OFF -DBUILD_TEST=OFF
cmake --build . --target torch torch_cpu c10 -j$(nproc)
cd ..

# 2. Build your Go program
go build ./...

# 3. Run an example
go run ./examples/mnist/
go run ./examples/text_classification/
go run ./examples/image_classification/
```

## benchmark Results

cpu: Intel Xeon @ 2.80GHz  (1 core)

┌─────────────────────────┬──────────────┬──────────┬───────────┐
│ Benchmark               │ Time/op      │ Mem/op   │ Allocs/op │
├─────────────────────────┼──────────────┼──────────┼───────────┤
│ TensorAllocation        │  814 µs      │ 200 KB   │ 3         │
│ MatMul 32×32            │   58 µs      │   4 KB   │ 3         │
│ MatMul 64×512           │   28 ms      │ 128 KB   │ 3         │
│ LinearForward (32×784)  │   10 ms      │  32 KB   │ 3         │
│ MLPForward (32×784)     │   12 ms      │  97 KB   │ 13        │
│ MLPForward Large(128×)  │  118 ms      │ 901 KB   │ 18        │
│ FullTrainingStep        │   29 ms      │ 197 KB   │ 15        │
│ ReLU (128×512)          │  358 µs      │ 256 KB   │ 1         │
│ Softmax (64×1000)       │  1.25 ms     │ 256 KB   │ 2         │
│ SGD Step                │  1.03 ms     │   0 B    │ 0         │ ← zero alloc!
│ DataLoading (batch=64)  │   64 µs      │ 200 KB   │ 1         │
│ EmbeddingLookup         │  147 µs      │ 800 KB   │ 1         │
└─────────────────────────┴──────────────┴──────────┴───────────┘

## Python → Go cheatsheet

| Python | Go |
|---|---|
| `import torch` | `import gt "github.com/Sarkar-AGI/GoTorch"` |
| `torch.randn(32, 784)` | `gt.Randn(32, 784)` |
| `nn.Linear(784, 256)` | `gt.NewLinear(784, 256, true)` |
| `nn.Sequential(...)` | `gt.NewSequential(...)` |
| `optim.Adam(model.parameters(), lr=1e-3)` | `gt.NewAdam(model.Parameters(), gt.AdamOptions{LR: 1e-3})` |
| `loss.backward()` | `loss.Backward()` |
| `optimizer.step()` | `opt.Step()` |
| `with torch.no_grad():` | `gt.WithNoGrad(func() { ... })` |
| `model.train()` | `model.Train()` |
| `model.eval()` | `model.Eval()` |
| `tensor.shape` | `tensor.Shape()` |
| `tensor.item()` | `tensor.Item()` |
| `tensor.argmax(dim=1)` | `tensor.Argmax(1, false)` |
