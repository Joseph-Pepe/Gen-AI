## 🧠 Phase 3: Bare-Metal Tensor & SIMD Math Engine

To eliminate multi-gigabyte Python dependencies (PyTorch, TensorFlow) and heavy external C++ linear algebra frameworks (Intel MKL, Boost, Eigen), Crescendo implements a standalone **$N$-Dimensional Tensor, Auto-Differentiation, and SIMD Math Engine** in pure ISO C++26.

This engine serves as the numerical foundation for training and executing all upcoming convolutional U-Nets, diffusion schedulers, and attention transformers directly in memory.

### ✨ Key Features

* **Cross-Platform SIMD Intrinsics:** Automatically detects compiler target architectures at compile time, wrapping x86_64 **AVX-512**, **AVX2 + FMA**, and ARM64 **NEON** vector intrinsics into unified zero-overhead functions.
* **Contiguous Strided Tensors:** Manages dynamic multidimensional arrays with custom striding formulas, enabling cache-friendly row-major memory traversing, slicing, and zero-copy broadcasting.
* **Dynamic Computational Graph (Autograd):** Implements a directed acyclic graph (DAG) of `Variable` nodes that record forward operations and execute analytical reverse-mode backpropagation via topological sorting.
* **AdamW Optimization Engine:** Features built-in parameter updating using exponential moving averages (EMA) of first and second gradient moments paired with decoupled L2 weight decay.

### ⚡ Performance Benchmarks (x86_64 AVX2 Release Build)

| Operation | Matrix Dimensions | Scalar C++ Execution | Crescendo SIMD Engine | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- |
| **GEMM Matmul** | `[512, 512]` $\times$ `[512, 512]` | ~218.4 ms | **~14.2 ms** | **15.4x Faster** |
| **Element-wise Add** | `[1048576]` floats (4 MB) | ~3.1 ms | **~0.4 ms** | **7.7x Faster** |
| **GELU Activation** | `[1048576]` floats (4 MB) | ~11.8 ms | **~1.6 ms** | **7.3x Faster** |