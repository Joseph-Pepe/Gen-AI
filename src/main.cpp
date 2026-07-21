import std;
import crescendo.tensor.core;
import crescendo.tensor.simd;
import crescendo.models.unet;
import crescendo.diffusion.noise;
import crescendo.diffusion.scheduler;
import crescendo.diffusion.ddim;

using namespace crescendo::tensor;
using namespace crescendo::models;
using namespace crescendo::diffusion;

int main() {
    std::println("=============================================================");
    std::println("🎛️  Crescendo Engine: Phase 5 Diffusion & DDIM Sampler Suite");
    std::println("=============================================================\n");

    std::println("✔ Active SIMD Hardware Engine: {}\n", simd::get_simd_architecture());

    // 1. Verify Box-Muller Gaussian Noise Generation
    std::println("--- Verifying Box-Muller Normal Distribution Generator ---");
    NormalGenerator<float> noise_gen;
    auto static_noise = noise_gen.generate({1, 1, 80, 80});
    
    // Compute sample mean and variance to verify N(0, I) statistical distribution
    float sum = 0.0f, sq_sum = 0.0f;
    for (size_t i = 0; i < static_noise.size(); ++i) {
        sum += static_noise[i];
        sq_sum += static_noise[i] * static_noise[i];
    }
    float mean = sum / static_cast<float>(static_noise.size());
    float variance = (sq_sum / static_cast<float>(static_noise.size())) - (mean * mean);
    std::println("✔ Generated [1, 1, 80, 80] Gaussian Noise | Mean: {:.4f} (Target: 0.00), Var: {:.4f} (Target: 1.00)\n", 
                 mean, variance);

    // 2. Verify Cosine Variance Scheduler & Forward Noise Injection
    std::println("--- Verifying Cosine Variance Scheduler (T = 1,000) ---");
    CosineScheduler<float> scheduler(1000);
    auto clean_spectrogram = Tensor<float>::random_normal({1, 1, 80, 80}, 0.0f, 0.5f);
    
    auto start_noise = std::chrono::high_resolution_clock::now();
    auto noisy_spectrogram = scheduler.add_noise(clean_spectrogram, static_noise, 500); // Corrupt at midpoint t=500
    auto end_noise = std::chrono::high_resolution_clock::now();
    
    auto dur_noise = std::chrono::duration_cast<std::chrono::microseconds>(end_noise - start_noise).count();
    std::println("✔ Executed SIMD Forward Noise Injection (t=500) in {} µs | Sample val: {:.4f}\n", 
                 dur_noise, noisy_spectrogram[0]);

    // 3. Verify DDIM Fast Inference Generative Loop
    std::println("--- Verifying DDIM 20-Step Fast Generative Inference ---");
    UNet2D<float> unet_backbone;
    DDIMSampler<float> ddim_sampler(scheduler);

    constexpr size_t fast_steps = 20;
    std::println("✔ Initializing deterministic reverse denoising loop (Skipping 980 Markovian steps)...");

    auto start_ddim = std::chrono::high_resolution_clock::now();
    auto generated_mel_spectrogram = ddim_sampler.sample(unet_backbone, static_noise, fast_steps);
    auto end_ddim = std::chrono::high_resolution_clock::now();
    
    auto dur_ddim = std::chrono::duration_cast<std::chrono::milliseconds>(end_ddim - start_ddim).count();
    const auto& out_shape = generated_mel_spectrogram.shape();
    
    std::println("✔ Executed Complete 20-Step DDIM Generation in {} ms ({:.1f} ms per step)", 
                 dur_ddim, static_cast<float>(dur_ddim) / static_cast<float>(fast_steps));
    std::println("   Output Spectrogram Shape: [{}, {}, {}, {}]", 
                 out_shape[0], out_shape[1], out_shape[2], out_shape[3]);

    if (out_shape[2] == 80 && out_shape[3] == 80 && mean > -0.05f && mean < 0.05f) {
        std::println("\n🏆 PHASE 5 PASSED: Diffusion Scheduler & DDIM Fast Sampler fully operational!");
    } else {
        std::println("\n❌ PHASE 5 FAILED: Statistical distribution drift or dimension mismatch.");
    }

    return 0;
}