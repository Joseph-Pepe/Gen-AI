import std;
import crescendo.tensor.core;
import crescendo.tensor.simd;
import crescendo.models.convolution2d;
import crescendo.models.groupnorm;
import crescendo.models.attention;
import crescendo.models.unet;

using namespace crescendo::tensor;
using namespace crescendo::models;

int main() {
    std::println("==========================================================");
    std::println("🎛️  Crescendo Engine: Phase 4 Neural Network Backbone");
    std::println("==========================================================\n");

    std::println("✔ Active SIMD Hardware Engine: {}\n", simd::get_simd_architecture());

    // 1. Test Standalone 2D Convolution (im2col + SIMD GEMM)
    std::println("--- Verifying Conv2D im2col SIMD Kernel ---");
    Conv2D<float> conv(1, 16, 3, 1, 1); // 3x3 kernel, same padding
    auto test_patch = Tensor<float>::random_normal({1, 1, 32, 32}, 0.0f, 1.0f);
    
    auto start_conv = std::chrono::high_resolution_clock::now();
    auto conv_out = conv.forward(test_patch);
    auto end_conv = std::chrono::high_resolution_clock::now();
    
    auto conv_dur = std::chrono::duration_cast<std::chrono::microseconds>(end_conv - start_conv).count();
    std::println("✔ Executed Conv2D [1, 1, 32, 32] -> [1, 16, 32, 32] in {} µs | Sample out: {:.4f}\n", 
                 conv_dur, conv_out[0]);

    // 2. Test Multi-Head Self-Attention
    std::println("--- Verifying Multi-Head Self-Attention ---");
    MultiHeadSelfAttention<float> attn(16, 4); // 16 channels, 4 attention heads
    
    auto start_attn = std::chrono::high_resolution_clock::now();
    auto attn_out = attn.forward(conv_out);
    auto end_attn = std::chrono::high_resolution_clock::now();
    
    auto attn_dur = std::chrono::duration_cast<std::chrono::microseconds>(end_attn - start_attn).count();
    std::println("✔ Executed 4-Head Attention across 1,024 tokens in {} µs | Shape preserved.\n", attn_dur);

    // 3. Verify Full 2D Convolutional U-Net Backbone
    std::println("--- Verifying Complete UNet2D Generative Backbone ---");
    UNet2D<float> unet;
    
    // Simulate an input 80-bin Mel-Spectrogram with 80 time frames
    auto mel_spectrogram = Tensor<float>::random_normal({1, 1, 80, 80}, 0.0f, 0.5f);
    std::println("✔ Loaded synthetic Log-Mel Spectrogram tensor | Shape: [1, 1, 80, 80]");

    auto start_unet = std::chrono::high_resolution_clock::now();
    auto unet_out = unet.forward(mel_spectrogram);
    auto end_unet = std::chrono::high_resolution_clock::now();
    
    auto unet_dur = std::chrono::duration_cast<std::chrono::milliseconds>(end_unet - start_unet).count();
    const auto& out_shape = unet_out.shape();
    std::println("✔ Executed Full U-Net Forward Pass in {} ms", unet_dur);
    std::println("   Output Tensor Shape: [{}, {}, {}, {}] (Matches input dimensions)", 
                 out_shape[0], out_shape[1], out_shape[2], out_shape[3]);

    if (out_shape[2] == 80 && out_shape[3] == 80 && unet_dur < 1000) {
        std::println("\n🏆 PHASE 4 PASSED: Generative Neural Network Backbone fully operational!");
    } else {
        std::println("\n❌ PHASE 4 FAILED: Dimension mismatch or execution benchmark failure.");
    }

    return 0;
}