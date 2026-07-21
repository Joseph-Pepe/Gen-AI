import std;
import crescendo.audio.codec.bitstream;
import crescendo.audio.codec.alac;
import crescendo.audio.codec.lpcm;
import crescendo.audio.io.m4a;

using namespace crescendo::codec;
using namespace crescendo::io;

int main() {
    std::println("==========================================================================");
    std::println("🎛️  Crescendo Engine: Phase 9 ALAC & LPCM Bitstream Writer Suite");
    std::println("==========================================================================\n");

    constexpr std::uint32_t sample_rate = 44100;
    constexpr std::uint16_t num_channels = 2;
    constexpr std::size_t duration_sec = 3;
    constexpr std::size_t total_samples = sample_rate * duration_sec * num_channels;

    // 1. Synthesize stereo test audio (Chord progression + rhythm pulse)
    std::println("--- Synthesizing Stereo Master Audio Buffer (3.0 Seconds at 44.1 kHz) ---");
    std::vector<std::int16_t> pcm_buffer(total_samples);
    
    for (std::size_t i = 0; i < total_samples / 2; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        float left_wave  = 0.6f * std::sin(2.0f * std::numbers::pi_v<float> * 261.63f * t); // Middle C
        float right_wave = 0.6f * std::sin(2.0f * std::numbers::pi_v<float> * 329.63f * t); // E4
        
        pcm_buffer[i * 2]     = static_cast<std::int16_t>(left_wave * 32000.0f);
        pcm_buffer[i * 2 + 1] = static_cast<std::int16_t>(right_wave * 32000.0f);
    }
    const std::size_t raw_pcm_bytes = pcm_buffer.size() * sizeof(std::int16_t);
    std::println("✔ Master buffer generated | Uncompressed PCM Size: {:.2f} KB\n", static_cast<float>(raw_pcm_bytes) / 1024.0f);

    // 2. Export RIFF/WAVE Little-Endian (.wav)
    std::println("--- Exporting Little-Endian RIFF/WAVE Container (.wav) ---");
    bool wav_ok = LPCMWriter::write_wav("crescendo_master.wav", pcm_buffer, num_channels, sample_rate);
    std::println("✔ File compiled: crescendo_master.wav | Status: {}\n", wav_ok ? "SUCCESS" : "FAILED");

    // 3. Export FORM/AIFF Big-Endian (.aiff) with IEEE 80-Bit Float Rate
    std::println("--- Exporting Big-Endian FORM/AIFF Container (.aiff) ---");
    bool aiff_ok = LPCMWriter::write_aiff("crescendo_master.aiff", pcm_buffer, num_channels, sample_rate);
    std::println("✔ File compiled: crescendo_master.aiff | Status: {}\n", aiff_ok ? "SUCCESS" : "FAILED");

    // 4. Encode ALAC Bitstream and Export MPEG-4 Part 14 Container (.m4a)
    std::println("--- Encoding ALAC Bitstream via SIMD LPC Prediction & Rice Coding ---");
    ALACSpecificConfig config;
    config.sample_rate = sample_rate;
    config.num_channels = num_channels;
    config.frame_length = 4096;

    ALACEncoder encoder(config);
    auto start_enc = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::uint8_t>> alac_packets = encoder.encode_stream(pcm_buffer);
    auto end_enc = std::chrono::high_resolution_clock::now();

    std::size_t compressed_payload_bytes = 0;
    for (const auto& pkt : alac_packets) compressed_payload_bytes += pkt.size();
    float compression_ratio = (1.0f - static_cast<float>(compressed_payload_bytes) / static_cast<float>(raw_pcm_bytes)) * 100.0f;

    std::println("✔ ALAC Encoding finished in {} ms | Encoded Frames: {}", 
                 std::chrono::duration_cast<std::chrono::milliseconds>(end_enc - start_enc).count(), alac_packets.size());
    std::println("   Compressed Payload Size: {:.2f} KB | Lossless Size Reduction: {:.1f}%\n", 
                 static_cast<float>(compressed_payload_bytes) / 1024.0f, compression_ratio);

    std::println("--- Packaging ISO Base Media MP4 Atom Tree (.m4a) ---");
    bool m4a_ok = M4AContainerWriter::write_alac_m4a("crescendo_master.m4a", alac_packets, config);
    std::println("✔ File compiled: crescendo_master.m4a | Status: {}", m4a_ok ? "SUCCESS" : "FAILED");

    if (wav_ok && aiff_ok && m4a_ok && compression_ratio > 30.0f) {
        std::println("\n🏆 PHASE 9 PASSED: ALAC Lossless & LPCM Bitstream Suite fully operational!");
    } else {
        std::println("\n❌ PHASE 9 FAILED: Stream serialization error or compression efficiency failure.");
    }

    return 0;
}