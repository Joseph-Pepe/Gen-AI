import std;
import crescendo.dsp.fast_fourier_transform;

using namespace crescendo::dsp;

int main() {
    constexpr size_t sample_rate = 44100;
    constexpr size_t fft_size = 1024; // Power of 2 window size

    std::println("==================================================");
    std::println("🎛️  Crescendo Engine: Radix-2 FFT C++26 Test Suite");
    std::println("==================================================\n");

    // 1. Instantiate the FFT engine (precomputes bit-reversal & twiddle tables)
    auto fft_engine = Radix2FFT<float>(fft_size);
    std::println("✔ Initialized Radix2FFT engine with window size N = {}", fft_engine.size());

    // 2. Generate a synthetic audio signal: 440 Hz + 1000 Hz sine waves
    std::vector<std::complex<float>> audio_buffer(fft_size);
    constexpr float freq1 = 440.0f;
    constexpr float freq2 = 1000.0f;

    for (size_t i = 0; i < fft_size; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(sample_rate);
        float sample = 0.6f * std::sin(2.0f * std::numbers::pi_v<float> * freq1 * t) +
                       0.4f * std::sin(2.0f * std::numbers::pi_v<float> * freq2 * t);
        audio_buffer[i] = std::complex<float>(sample, 0.0f);
    }

    // Keep a copy of the original time-domain signal for IFFT verification
    std::vector<std::complex<float>> original_signal = audio_buffer;

    // 3. Execute Forward FFT (In-place transformation)
    auto start_time = std::chrono::high_resolution_clock::now();
    fft_engine.forward(audio_buffer);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    std::println("✔ Executed Forward FFT in {} µs\n", duration);

    // 4. Analyze Frequency Spectrum (Find peak bins)
    std::println("--- Frequency Spectrum Peaks ---");
    for (size_t i = 0; i < fft_size / 2; ++i) {
        float magnitude = std::abs(audio_buffer[i]);
        // Normalize magnitude for display
        if (magnitude > 100.0f) {
            float bin_frequency = static_cast<float>(i) * static_cast<float>(sample_rate) / static_cast<float>(fft_size);
            std::println("Bin {:4d} | Freq: {:7.1f} Hz | Magnitude: {:6.2f}", i, bin_frequency, magnitude);
        }
    }

    // 5. Execute Inverse FFT (IFFT) to reconstruct time-domain waveform
    fft_engine.inverse(audio_buffer);
    std::println("\n✔ Executed Inverse FFT (IFFT) with 1/N normalization.");

    // 6. Verify mathematically that Reconstructed == Original
    float max_error = 0.0f;
    for (size_t i = 0; i < fft_size; ++i) {
        float error = std::abs(audio_buffer[i].real() - original_signal[i].real());
        if (error > max_error) {
            max_error = error;
        }
    }

    std::println("✔ Reconstruction Max Absolute Error: {:.9f}", max_error);
    if (max_error < 1e-5f) {
        std::println("\n🏆 TEST PASSED: Perfect signal reconstruction achieved!");
    } else {
        std::println("\n❌ TEST FAILED: Precision drift exceeded threshold.");
    }

    return 0;
}