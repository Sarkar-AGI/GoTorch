# GoTorch Dockerfile
# Builds libtorch C++ backend + Go binding in one image
#
# Usage:
#   docker build -f Dockerfile.gotorch -t gotorch:latest .
#   docker run --rm gotorch:latest go test -v ./...
#   docker run --rm gotorch:latest go run ./examples/mnist/
#
# With CUDA:
#   docker build -f Dockerfile.gotorch --build-arg BASE=nvidia/cuda:12.1.0-devel-ubuntu22.04 -t gotorch:cuda .

ARG BASE=ubuntu:22.04
FROM ${BASE} AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        git \
        wget \
        curl \
        ca-certificates \
        python3 \
        python3-pip \
        libblas-dev \
        liblapack-dev \
        libopenblas-dev \
        libssl-dev \
        zlib1g-dev \
        && rm -rf /var/lib/apt/lists/*

# ─── Install Go ───────────────────────────────────────────────────────────────
ARG GO_VERSION=1.22.2
RUN wget -q https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go${GO_VERSION}.linux-amd64.tar.gz \
    && rm go${GO_VERSION}.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/go"
ENV GOMODCACHE="/go/pkg/mod"

# ─── Build Stage: libtorch ────────────────────────────────────────────────────
FROM base AS libtorch-builder

WORKDIR /gotorch

# Copy source
COPY . .

# Initialize submodules (third_party)
RUN git submodule update --init --recursive --depth 1 \
        third_party/eigen \
        third_party/fmt \
        third_party/glog \
        third_party/googletest \
        third_party/onnx \
        third_party/protobuf \
        third_party/pybind11 \
        third_party/sleef \
    || true

# Install Python build deps
RUN pip3 install --no-cache-dir pyyaml numpy typing_extensions

# Build libtorch (CPU only, minimal)
RUN mkdir -p build && cd build && cmake .. \
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
        -DBUILD_BINARY=OFF \
        -DCMAKE_INSTALL_PREFIX=/gotorch/build/install \
    && ninja -j$(nproc) torch torch_cpu c10 \
    && ninja install

# ─── Build Stage: Go ──────────────────────────────────────────────────────────
FROM libtorch-builder AS go-builder

WORKDIR /gotorch

# Set CGo flags
ENV CGO_CFLAGS="-I/gotorch/torch/csrc/api/include -I/gotorch/csrc/go_binding -I/gotorch/build/install/include"
ENV CGO_LDFLAGS="-L/gotorch/build/lib -Wl,-rpath,/gotorch/build/lib -ltorch -ltorch_cpu -lc10 -lstdc++"
ENV LD_LIBRARY_PATH="/gotorch/build/lib:${LD_LIBRARY_PATH}"

# Build Go binding
RUN go build ./...

# ─── Test Stage ───────────────────────────────────────────────────────────────
FROM go-builder AS tester

WORKDIR /gotorch

ENV GOTORCH_SKIP_CPP_TESTS=1
RUN go test -v -timeout 120s . 2>&1 | tee /test_results.txt
RUN go test -bench=. -benchmem -benchtime=3s . 2>&1 | tee /bench_results.txt

# ─── Final Runtime Image ──────────────────────────────────────────────────────
FROM base AS runtime

# Copy only what's needed to run
COPY --from=go-builder /gotorch/build/lib/*.so* /usr/local/lib/gotorch/
COPY --from=go-builder /gotorch /gotorch
COPY --from=tester /test_results.txt /test_results.txt
COPY --from=tester /bench_results.txt /bench_results.txt

# Runtime library path
ENV LD_LIBRARY_PATH="/usr/local/lib/gotorch:${LD_LIBRARY_PATH}"
ENV CGO_CFLAGS="-I/gotorch/torch/csrc/api/include -I/gotorch/csrc/go_binding"
ENV CGO_LDFLAGS="-L/usr/local/lib/gotorch -Wl,-rpath,/usr/local/lib/gotorch -ltorch -ltorch_cpu -lc10 -lstdc++"

WORKDIR /gotorch

# Default: run MNIST example
CMD ["go", "run", "./examples/mnist/"]
