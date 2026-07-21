import std;
import crescendo.tensor.core;
import crescendo.tensor.simd;
import crescendo.models.stem_unet;
import crescendo.dsp.wiener_filter;

using namespace crescendo::tensor;
using namespace crescendo::models;
using namespace crescendo::dsp;

int main() {
    std::println("====================================================================");
    std::println("🎛️  Crescendo Engine: Phase 6 Contd.. 4-Stem Wiener Separation Suite");
    std::println("=====================================================================\n");

    std::println("✔ Active SIMD Hardware Acceleration: {}\n", simd::get_simd_architecture());

    constexpr size_t bins = 80;
    constexpr size_t frames = 80;

    // 1. Create synthetic complex mixture STFT spectrogram [1, 1, 80, 80]
    std::println("--- Generating Synthetic Complex Audio Mixture ---");
    Tensor<std::complex<float>> complex_mix({1, 1, bins, frames});
    Tensor<float> mixture_magnitude({1, 1, bins, frames});

    for (size_t i = 0; i < complex_mix.size(); ++i) {
        float real_part = std::sin(static_cast<float>(i) * 0.1f) * 10.0f;
        float imag_part = std::cos(static_cast<float>(i) * 0.1f) * 10.0f;
        complex_mix.data()[i] = std::complex<float>(real_part, imag_part);
        mixture_magnitude.data()[i] = std::abs(complex_mix.data()[i]);
    }
    std::println("✔ Complex mixture STFT instantiated | Total Spectrogram Bins: {}\n", complex_mix.size());

    // 2. Predict 4 Stem Masks using UNet4StemSeparation
    std::println("--- Executing 4-Stem U-Net Mask Prediction ---");
    UNet4StemSeparation<float> sep_unet;
    
    auto start_unet = std::chrono::high_resolution_clock::now();
    std::array<Tensor<float>, 4> stem_masks = sep_unet.forward_masks(mixture_magnitude);
    auto end_unet = std::chrono::high_resolution_clock::now();
    
    std::println("✔ Predicted 4 independent Soft-Masks in {} ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(end_unet - start_unet).count());
    std::println("   Stem 0 (Vocals) Mask Sample: {:.4f}", stem_masks[0].data()[0]);
    std::println("   Stem 1 (Drums)  Mask Sample: {:.4f}", stem_masks[1].data()[0]);
    std::println("   Stem 2 (Bass)   Mask Sample: {:.4f}", stem_masks[2].data()[0]);
    std::println("   Stem 3 (Other)  Mask Sample: {:.4f}\n", stem_masks[3].data()[0]);

    // 3. Execute SIMD Multi-Channel Wiener Filtering
    std::println("--- Applying SIMD Multi-Channel Wiener Filtering ---");
    WienerFilter4Stem<float> wiener;

    auto start_wiener = std::chrono::high_resolution_clock::now();
    auto separated_stems = wiener.separate(complex_mix, stem_masks);
    auto end_wiener = std::chrono::high_resolution_clock::now();

    std::println("✔ Executed SIMD 4-Stem Wiener Separation in {} µs\n",
                 std::chrono::duration_cast<std::chrono::microseconds>(end_wiener - start_wiener).count());

    // 4. Verify Energy Conservation: Sum(Stems) == Mixture
    std::println("--- Verifying Wiener Energy Conservation & Phase Coherence ---");
    float max_reconstruction_error = 0.0f;
    for (size_t i = 0; i < complex_mix.size(); ++i) {
        std::complex<float> reconstructed = separated_stems[0].data()[i] + 
                                            separated_stems[1].data()[i] + 
                                            separated_stems[2].data()[i] + 
                                            separated_stems[3].data()[i];
        float err = std::abs(complex_mix.data()[i] - reconstructed);
        if (err > max_reconstruction_error) max_reconstruction_error = err;
    }

    std::println("✔ Maximum Complex Reconstruction Error across all stems: {:.9f}", max_reconstruction_error);

    if (max_reconstruction_error < 1e-4f) {
        std::println("\n🏆 PHASE 6 PASSED: 4-Stem Wiener Audio Separation Engine fully operational!");
    } else {
        std::println("\n❌ PHASE 6 FAILED: Energy loss or phase distortion exceeded threshold.");
    }

    return 0;
}