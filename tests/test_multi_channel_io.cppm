import std;
import crescendo.tensor.simd;
import crescendo.audio.io.wav_writer;

using namespace crescendo::tensor;
using namespace crescendo::io;

int main() {
    std::println("==================================================================");
    std::println("🎛️  Crescendo Engine: Phase 7 Multi-Channel & Batch WAV I/O Suite");
    std::println("==================================================================\n");

    std::println("✔ Active SIMD Hardware Acceleration: {}\n", simd::get_simd_architecture());

    constexpr std::uint32_t sample_rate = 44100;
    constexpr size_t duration_sec = 2;
    constexpr size_t total_samples = sample_rate * duration_sec;

    // 1. Synthesize 4 distinct audio tracks simulating isolated stems from Phase 6
    std::println("--- Synthesizing 4 Isolated Stem Waveforms (2 Seconds at 44.1 kHz) ---");
    std::array<std::vector<float>, 4> stems = {
        std::vector<float>(total_samples), // Vocals (440 Hz Sine)
        std::vector<float>(total_samples), // Drums  (White Noise Burst Simulation)
        std::vector<float>(total_samples), // Bass   (110 Hz Low Sine)
        std::vector<float>(total_samples)  // Other  (330 Hz Chord Tone)
    };

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> noise_dist(-0.8f, 0.8f);

    for (size_t i = 0; i < total_samples; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        stems[0][i] = 0.7f * std::sin(2.0f * std::numbers::pi_v<float> * 440.0f * t); // Vocals
        stems[1][i] = (i % 22050 < 2000) ? noise_dist(rng) * 0.9f : 0.0f;             // Snare drum hits
        stems[2][i] = 0.8f * std::sin(2.0f * std::numbers::pi_v<float> * 110.0f * t); // Sub Bass
        stems[3][i] = 0.5f * std::sin(2.0f * std::numbers::pi_v<float> * 330.0f * t); // Synth Other
    }
    std::println("✔ Synthesized 4 independent stem buffers | Samples per track: {}\n", total_samples);

    // 2. Execute Batch Export of Isolated Stems (16-Bit Signed PCM)
    std::println("--- Executing Batch Stem Export (16-Bit PCM with SIMD Normalization) ---");
    auto start_batch = std::chrono::high_resolution_clock::now();
    bool batch_success = WavExporter::write_stems_batch("isolated_track", stems, sample_rate, WavBitDepth::PCM_16);
    auto end_batch = std::chrono::high_resolution_clock::now();

    std::println("✔ Batch export completed in {} ms | Success: {}", 
                 std::chrono::duration_cast<std::chrono::milliseconds>(end_batch - start_batch).count(),
                 batch_success ? "YES" : "NO");
    std::println("   Compiled: isolated_track_vocals.wav, isolated_track_drums.wav, etc.\n");

    // 3. Interleave Stems into a Synchronized 4-Channel Master File
    std::println("--- Interleaving 4 Stems into a Single 4-Channel Master Track ---");
    std::vector<std::vector<float>> channel_matrix = {stems[0], stems[1], stems[2], stems[3]};
    
    auto start_interleave = std::chrono::high_resolution_clock::now();
    std::vector<float> master_4ch = WavExporter::interleave_channels(channel_matrix);
    bool master_success = WavExporter::write_wav("master_4ch_interleaved.wav", master_4ch, 4, sample_rate, WavBitDepth::PCM_16);
    auto end_interleave = std::chrono::high_resolution_clock::now();

    std::println("✔ 4-Channel interleave and export completed in {} ms | File: master_4ch_interleaved.wav", 
                 std::chrono::duration_cast<std::chrono::milliseconds>(end_interleave - start_interleave).count());
    std::println("   Total Interleaved Samples: {} | Bytes written: {} KB\n", 
                 master_4ch.size(), (master_4ch.size() * sizeof(std::int16_t)) / 1024);

    // 4. Export Studio 32-Bit IEEE Floating-Point Stereo Mix
    std::println("--- Exporting 32-Bit IEEE Floating-Point Stereo Mix ---");
    std::vector<std::vector<float>> stereo_mix = {
        std::vector<float>(total_samples), // Left Channel: Vocals + Bass
        std::vector<float>(total_samples)  // Right Channel: Drums + Other
    };

    for (size_t i = 0; i < total_samples; ++i) {
        stereo_mix[0][i] = stems[0][i] + stems[2][i];
        stereo_mix[1][i] = stems[1][i] + stems[3][i];
    }

    std::vector<float> stereo_interleaved = WavExporter::interleave_channels(stereo_mix);
    bool float_success = WavExporter::write_wav("studio_master_32bit_float.wav", stereo_interleaved, 2, sample_rate, WavBitDepth::FLOAT_32);

    std::println("✔ 32-Bit Float Stereo export completed | File: studio_master_32bit_float.wav");

    if (batch_success && master_success && float_success) {
        std::println("\n🏆 PHASE 7 PASSED: Multi-Channel Interleaved & Batch WAV I/O Engine operational!");
    } else {
        std::println("\n❌ PHASE 7 FAILED: Binary stream write error encountered.");
    }

    return 0;
}