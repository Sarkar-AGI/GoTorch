here# GoTorch Makefile
# Copyright (c) 2024-2026 Sarkar-AGI. MIT License.
#
# Usage:
#   make build        — build libtorch C++ backend
#   make go-build     — build Go binding
#   make test         — run all tests
#   make bench        — run benchmarks
#   make docker       — build Docker image
#   make docker-run   — run example in Docker
#   make all          — build + test + bench

GOTORCH     := $(shell pwd)
BUILD_DIR   := $(GOTORCH)/build
NPROC       := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
GO          := go
DOCKER      := docker
IMAGE_NAME  := gotorch
IMAGE_TAG   := latest

# CGo flags
export CGO_CFLAGS  := -I$(GOTORCH)/torch/csrc/api/include -I$(GOTORCH)/csrc/go_binding -I$(BUILD_DIR)/include
export CGO_LDFLAGS := -L$(BUILD_DIR)/lib -Wl,-rpath,$(BUILD_DIR)/lib -ltorch -ltorch_cpu -lc10 -lstdc++
export LD_LIBRARY_PATH := $(BUILD_DIR)/lib:$(LD_LIBRARY_PATH)

.PHONY: all build go-build test bench bench-cpu docker docker-run docker-bench clean help

## all: Build everything then test and benchmark
all: build go-build test bench

## help: Show this help
help:
	@echo ""
	@echo "GoTorch — PyTorch C++ engine for Go"
	@echo ""
	@grep -E '^##' Makefile | sed 's/## /  /'
	@echo ""

# ─── C++ Build ────────────────────────────────────────────────────────────────

## build: Configure and build libtorch C++ backend (CPU)
build:
	@echo "── Configuring libtorch (CPU) ──────────────────────────────────"
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake .. \
		-GNinja \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_PYTHON=OFF \
		-DBUILD_TEST=OFF \
		-DBUILD_CAFFE2=OFF \
		-DBUILD_SHARED_LIBS=ON \
		-DUSE_CUDA=OFF \
		-DUSE_DISTRIBUTED=OFF \
		-DUSE_MKLDNN=ON \
		-DUSE_QNNPACK=OFF \
		-DUSE_PYTORCH_QNNPACK=OFF \
		-DUSE_XNNPACK=OFF \
		-DUSE_NNPACK=OFF \
		-DUSE_OPENMP=ON \
		-DBUILD_BINARY=OFF
	@echo "── Building libtorch ($(NPROC) cores) ──────────────────────────"
	cd $(BUILD_DIR) && ninja -j$(NPROC) torch torch_cpu c10
	@echo "── Build complete ──────────────────────────────────────────────"
	@ls -lh $(BUILD_DIR)/lib/libtorch*.so 2>/dev/null || echo "Check build/lib/"

## build-cuda: Build with CUDA support
build-cuda:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && cmake .. \
		-GNinja \
		-DCMAKE_BUILD_TYPE=Release \
		-DBUILD_PYTHON=OFF \
		-DBUILD_TEST=OFF \
		-DBUILD_SHARED_LIBS=ON \
		-DUSE_CUDA=ON \
		-DUSE_MKLDNN=ON \
		-DUSE_DISTRIBUTED=OFF \
		-DBUILD_BINARY=OFF
	cd $(BUILD_DIR) && ninja -j$(NPROC) torch torch_cpu torch_cuda c10

# ─── Go Build ─────────────────────────────────────────────────────────────────

## go-build: Build the Go binding
go-build:
	@echo "── Building Go binding ─────────────────────────────────────────"
	$(GO) build ./...
	@echo "── Go build complete ───────────────────────────────────────────"

## go-install: Install GoTorch to GOPATH
go-install:
	$(GO) install ./...

# ─── Test ─────────────────────────────────────────────────────────────────────

## test: Run all Go unit tests
test:
	@echo "── Running Go tests ────────────────────────────────────────────"
	$(GO) test -v -timeout 120s -run "^TestGo" . 2>&1 | tee test_results.txt
	@echo ""
	@echo "── Test summary ────────────────────────────────────────────────"
	@grep -E "^(ok|FAIL|---)" test_results.txt || true

## test-all: Run Go tests + PyTorch C++ tests
test-all:
	$(GO) test -v -timeout 300s . 2>&1 | tee test_results_all.txt

## test-cpp: Run only PyTorch C++ tests
test-cpp:
	$(GO) test -v -timeout 300s -run "^TestPyTorchCppTests" .

# ─── Benchmark ────────────────────────────────────────────────────────────────

## bench: Run Go benchmarks
bench:
	@echo "── Running benchmarks ──────────────────────────────────────────"
	@echo "   cpu: $$(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
	@echo ""
	$(GO) test -bench=. -benchmem -benchtime=5s -run=^$$ . 2>&1 | tee bench_results.txt
	@echo ""
	@echo "── Saved to bench_results.txt ──────────────────────────────────"

## bench-cpu: Benchmark CPU ops only
bench-cpu:
	$(GO) test -bench="BenchmarkLinear|BenchmarkMatMul|BenchmarkConv" \
		-benchmem -benchtime=5s -run=^$$ .

## bench-compare: Compare with previous benchmark results
bench-compare:
	@if [ -f bench_results_prev.txt ]; then \
		benchstat bench_results_prev.txt bench_results.txt; \
	else \
		echo "No previous results. Run 'make bench' first, then rename to bench_results_prev.txt"; \
	fi

# ─── Examples ─────────────────────────────────────────────────────────────────

## run-mnist: Run MNIST example
run-mnist:
	$(GO) run ./examples/mnist/

## run-text: Run text classification example
run-text:
	$(GO) run ./examples/text_classification/

## run-image: Run image classification example
run-image:
	$(GO) run ./examples/image_classification/

# ─── Docker ───────────────────────────────────────────────────────────────────

## docker: Build GoTorch Docker image
docker:
	@echo "── Building Docker image: $(IMAGE_NAME):$(IMAGE_TAG) ──────────"
	$(DOCKER) build -f Dockerfile.gotorch -t $(IMAGE_NAME):$(IMAGE_TAG) .
	@echo "── Docker image ready ──────────────────────────────────────────"
	@$(DOCKER) images $(IMAGE_NAME):$(IMAGE_TAG)

## docker-cuda: Build GoTorch Docker image with CUDA
docker-cuda:
	$(DOCKER) build -f Dockerfile.gotorch \
		--build-arg BASE=nvidia/cuda:12.1.0-devel-ubuntu22.04 \
		-t $(IMAGE_NAME):cuda .

## docker-run: Run MNIST example in Docker
docker-run:
	$(DOCKER) run --rm $(IMAGE_NAME):$(IMAGE_TAG) go run ./examples/mnist/

## docker-test: Run tests in Docker
docker-test:
	$(DOCKER) run --rm $(IMAGE_NAME):$(IMAGE_TAG) \
		go test -v -timeout 120s -run "^TestGo" .

## docker-bench: Run benchmarks in Docker and show results
docker-bench:
	@echo "── Running benchmarks in Docker ────────────────────────────────"
	$(DOCKER) run --rm $(IMAGE_NAME):$(IMAGE_TAG) \
		go test -bench=. -benchmem -benchtime=5s -run=^$$ .

## docker-shell: Open shell in Docker container
docker-shell:
	$(DOCKER) run --rm -it $(IMAGE_NAME):$(IMAGE_TAG) /bin/bash

## docker-push: Push to Docker Hub (set DOCKER_USER)
docker-push:
	$(DOCKER) tag $(IMAGE_NAME):$(IMAGE_TAG) $(DOCKER_USER)/$(IMAGE_NAME):$(IMAGE_TAG)
	$(DOCKER) push $(DOCKER_USER)/$(IMAGE_NAME):$(IMAGE_TAG)

# ─── go get support ───────────────────────────────────────────────────────────

## goget-check: Verify go get works
goget-check:
	@echo "── Checking go get compatibility ───────────────────────────────"
	$(GO) mod tidy
	$(GO) mod verify
	@echo "── go.mod is valid ─────────────────────────────────────────────"
	@cat go.mod

## release: Tag and push a new release
release:
	@if [ -z "$(VERSION)" ]; then echo "Usage: make release VERSION=v1.0.1"; exit 1; fi
	git tag $(VERSION)
	git push origin $(VERSION)
	@echo "Released $(VERSION)"

# ─── Clean ────────────────────────────────────────────────────────────────────

## clean: Remove build artifacts
clean:
	rm -rf $(BUILD_DIR)
	$(GO) clean -cache
	rm -f test_results*.txt bench_results*.txt

## clean-docker: Remove Docker images
clean-docker:
	$(DOCKER) rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	$(DOCKER) rmi $(IMAGE_NAME):cuda 2>/dev/null || true
