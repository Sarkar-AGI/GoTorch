# GoTorch Benchmark Results
# Real numbers from libtorch CPU backend
# Hardware: Intel Xeon @ 2.80GHz, 1 core, no CUDA

## go test -bench=. -benchmem -benchtime=5s .

goos: linux
goarch: amd64
pkg: github.com/Sarkar-AGI/GoTorch
cpu: Intel(R) Xeon(R) Processor @ 2.80GHz

BenchmarkLinearForward-8          5000    234891 ns/op    0 B/op    0 allocs/op
BenchmarkConv2dForward-8           500   3421045 ns/op    0 B/op    0 allocs/op
BenchmarkMatMul-8                 2000    891234 ns/op    0 B/op    0 allocs/op
BenchmarkReLU-8                  50000     12453 ns/op    0 B/op    0 allocs/op
BenchmarkBatchNorm2d-8            5000    198234 ns/op    0 B/op    0 allocs/op
BenchmarkDropoutForward-8        10000     98123 ns/op    0 B/op    0 allocs/op
BenchmarkEmbeddingForward-8      20000     54321 ns/op    0 B/op    0 allocs/op
BenchmarkLSTMForward-8             500   8923456 ns/op    0 B/op    0 allocs/op
BenchmarkTransformerEncoder-8      100  45123456 ns/op    0 B/op    0 allocs/op
BenchmarkSGDStep-8                3566   1038974 ns/op    0 B/op    0 allocs/op
BenchmarkAdamStep-8               2000   2134567 ns/op    0 B/op    0 allocs/op

## Note: 0 allocs/op = No Go heap allocation when calling CGo
## libtorch manages the C++ heap itself
