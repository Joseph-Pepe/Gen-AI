import std;
import crescendo.gpu.types;
import crescendo.gpu.buffer;
import crescendo.gpu.backend;
import crescendo.gpu.gemm;

using namespace crescendo::gpu;

int main() {
    std::println("========================================================================");
    std::println("🎛️  Crescendo Engine: Phase 8 GPU Compute HAL & Dispatch Suite");
    std::println("========================================================================\n");

    // 1. Discover target GPU compute hardware
    ComputeBackend active_backend = discover_optimal_backend();
    std::println("✔ Initialized Hardware Abstraction Layer | Active Backend: {}\n", 
                 backend_to_string(active_backend));

    constexpr std::uint32_t M = 512, N = 512, K = 512;

    // 2. Allocate RAII GPU buffers with HostVisible staging flags
    std::println("--- Allocating Staging & VRAM Compute Buffers ([{}x{}] Matrices) ---", M, N);
    GPUBuffer<float> buf_A(M * K, active_backend, MemoryProperty::HostVisible | MemoryProperty::DeviceLocal);
    GPUBuffer<float> buf_B(K * N, active_backend, MemoryProperty::HostVisible | MemoryProperty::DeviceLocal);
    GPUBuffer<float> buf_C(M * N, active_backend, MemoryProperty::HostVisible | MemoryProperty::DeviceLocal);

    std::println("✔ Allocated 3x GPU Buffers | Total VRAM Footprint: {:.2f} MB\n",
                 static_cast<float>(buf_A.byte_size() + buf_B.byte_size() + buf_C.byte_size()) / (1024.0f * 1024.0f));

    // 3. Populate host staging data
    std::vector<float> host_A(M * K, 1.5f);
    std::vector<float> host_B(K * N, 2.0f);
    std::vector<float> host_C(M * N, 0.0f);

    std::println("--- Uploading Host Staging Arrays to GPU Compute Buffers ---");
    buf_A.upload(host_A);
    buf_B.upload(host_B);
    std::println("✔ Upload complete | Asynchronous DMA transfers synchronized.\n");

    // 4. Instantiate and execute GPU GEMM Pipeline
    std::println("--- Dispatching Tiled GEMM Compute Pipeline ---");
    GPUGEMMPipeline<float> gemm_engine(active_backend);

    auto start_time = std::chrono::high_resolution_clock::now();
    gemm_engine.execute(buf_A, buf_B, buf_C, M, N, K);
    auto end_time = std::chrono::high_resolution_clock::now();

    auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::println("✔ Execution complete | Latency: {} ms | Pipeline Backend: {}\n", 
                 dur_ms, backend_to_string(gemm_engine.backend()));

    // 5. Download results back to host memory and verify numerical precision
    std::println("--- Downloading Results from VRAM & Verifying Math ---");
    buf_C.download(host_C);

    float expected_val = 1.5f * 2.0f * static_cast<float>(K); // 1.5 * 2.0 * 512 = 1536.0
    float sample_out = host_C[0];
    float error = std::abs(sample_out - expected_val);

    std::println("✔ Sample Output [0,0]: {:.2f} | Expected: {:.2f} | Error: {:.6f}", 
                 sample_out, expected_val, error);

    if (error < 1e-3f) {
        std::println("\n🏆 PHASE 8 PASSED: Unified GPU Compute HAL & Dispatch Engine operational!");
    } else {
        std::println("\n❌ PHASE 8 FAILED: Numerical divergence detected during compute dispatch.");
    }

    return 0;
}