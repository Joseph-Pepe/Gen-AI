import std;
import crescendo.tensor.core;
import crescendo.models.unet;
import crescendo.diffusion.scheduler;
import crescendo.diffusion.ddim;
import crescendo.dsp.griffin_lim;
import crescendo.audio.io.wav_writer;

using namespace crescendo::tensor;
using namespace crescendo::models;
using namespace crescendo::diffusion;
using namespace crescendo::dsp;
using namespace crescendo::io;

int main() {
    std::println("=============================================================");
    std::println("🎛️  Crescendo Engine: Phase 6 Vocoder & Audio Export Suite");
    std::println("=============================================================\n");

    // 1. Instantiate the generative modules (Phases 4 & 5)
    UNet2D<float> unet_backbone;
    CosineScheduler<float> scheduler(1000);
    DDIMSampler<float> ddim_sampler(scheduler);

    // Generate a mock synthetic Mel filterbank reference grid [80 x 1025]
    Tensor<float> mock_filterbank({80, 1025});
    mock_filterbank.zero_();
    for (size_t m = 0; m < 80; ++m) {
        mock_filterbank[{m, m * 12}] = 1.0f; // Diagonal identity mapping distribution
    }

    std::println("--- Running Generative Audio Diffusion Model ---");
    Tensor<float> latent_seed = Tensor<float>::random_normal({1, 1, 80, 60}, 0.0f, 1.0f);
    
    auto start_gen = std::chrono::high_resolution_clock::now();
    auto generated_mel = ddim_sampler.sample(unet_backbone, latent_seed, 25);
    auto end_gen = std::chrono::high_resolution_clock::now();
    
    std::println("✔ Generated Mel Spectrogram Matrix in {} ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(end_gen - start_gen).count());

    // 2. Execute Phase 6 Phase Reconstruction Vocoder
    std::println("\n--- Initializing Phase 6 Griffin-Lim Vocoder Loop ---");
    GriffinLimVocoder<float> vocoder(2048, 512);

    auto start_vocode = std::chrono::high_resolution_clock::now();
    auto linear_magnitude = vocoder.invert_mel_filterbank(generated_mel, mock_filterbank);
    std::println("✔ Linear frequency conversion completed. Grid size: [1, 1, 1025, 60]");
    
    std::println("✔ Reconstructing phase maps (60 GL Iterations)...");
    auto audio_waveform = vocoder.reconstruct(linear_magnitude, 60);
    auto end_vocode = std::chrono::high_resolution_clock::now();

    std::println("✔ Vocoder synthesis loop finished in {} ms",
                 std::chrono::duration_cast<std::chrono::milliseconds>(end_vocode - start_vocode).count());

    // 3. Export to Wave Container
    std::println("\n--- Serializing Raw PCM Floating Buffers to Hard Drive ---");
    std::string output_path = "ai_generated_track.wav";
    
    bool export_success = crescendo::io::WavExporter::write_wav(output_path, audio_waveform, 44100);

    if (export_success) {
        std::println("✔ File successfully compiled: {}", output_path);
        std::println("   Total samples: {} | Duration: {:.2f} seconds", 
                     audio_waveform.size(), static_cast<double>(audio_waveform.size()) / 44100.0);
        std::println("\n🏆 PHASE 6 PASSED: End-To-End Generative Audio Synthesis Complete!");
    } else {
        std::println("\n❌ PHASE 6 FAILED: Unable to allocate binary IO file stream handle.");
    }

    return 0;
}