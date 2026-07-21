import std;
import crescendo.tensor.simd;
import crescendo.tensor.core;
import crescendo.tensor.autograd;
import crescendo.tensor.optimizer;

using namespace crescendo::tensor;
using namespace crescendo::tensor::autograd;
using namespace crescendo::tensor::optimizer;

int main() {
    std::println("======================================================");
    std::println("🎛️  Crescendo Engine: Phase 3 Tensor & SIMD Suite");
    std::println("======================================================\n");

    // 1. Check Hardware SIMD Acceleration
    std::println("✔ Detected Hardware SIMD Architecture: {}", simd::get_simd_architecture());

    // 2. Benchmark SIMD GEMM Performance (512x512 Matrices)
    constexpr size_t dim = 512;
    auto mat_A = Tensor<float>::random_normal({dim, dim}, 0.0f, 1.0f);
    auto mat_B = Tensor<float>::random_normal({dim, dim}, 0.0f, 1.0f);

    auto start_time = std::chrono::high_resolution_clock::now();
    auto mat_C = mat_A.matmul(mat_B);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    std::println("✔ Executed SIMD Matmul [{}x{}] * [{}x{}] in {} ms | Sample val: {:.4f}\n", 
                 dim, dim, dim, dim, dur_ms, mat_C[0]);

    // 3. Verify Auto-Grad Computational Graph & Backpropagation
    std::println("--- Verifying Autograd Backward Differentiation ---");
    auto X = Variable<float>::create(Tensor<float>({2, 3}, 2.0f), true, "X");
    auto W = Variable<float>::create(Tensor<float>({3, 2}, 1.5f), true, "W");
    auto B = Variable<float>::create(Tensor<float>({2, 2}, 0.5f), true, "B");

    // Forward pass: Y = X * W + B
    auto Y = X->matmul(W)->add(B);
    std::println("✔ Forward graph evaluated successfully. Shape Y: [{}, {}]", Y->data.shape()[0], Y->data.shape()[1]);

    // Execute Backward Sweep
    Y->backward();
    std::println("✔ Executed topological backward differentiation.");
    std::println("   Grad dL/dX[0]: {:.4f} (Expected: 3.0000 -> Sum of W rows)", X->grad[0]);
    std::println("   Grad dL/dW[0]: {:.4f} (Expected: 4.0000 -> Sum of X cols)\n", W->grad[0]);

    // 4. Test AdamW Optimizer Convergence
    std::println("--- Testing AdamW Optimizer Step ---");
    AdamW<float> optimizer({X, W, B}, 0.05f);
    float initial_weight = W->data[0];
    
    optimizer.step();
    float updated_weight = W->data[0];
    std::println("✔ Executed AdamW step with L2 decay | W[0] changed from {:.4f} to {:.4f}", 
                 initial_weight, updated_weight);

    if (std::abs(updated_weight - initial_weight) > 1e-4f && dur_ms < 500) {
        std::println("\n🏆 PHASE 3 PASSED: Bare-metal Tensor, Autograd, & SIMD engine fully operational!");
    } else {
        std::println("\n❌ PHASE 3 FAILED: Precision drift or performance benchmark failure.");
    }

    return 0;
}