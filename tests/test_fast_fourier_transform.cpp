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

// import std;
// import crescendo.dsp.fast_fourier_transform;
// import crescendo.dsp.short_term_fourier_transform;
// import crescendo.dsp.mel_filterbank;

// using namespace crescendo::dsp;

// int main() {
//     constexpr size_t sample_rate = 44100;
//     constexpr size_t fft_size = 1024;
//     constexpr size_t hop_size = 256;    // 75% window overlap
//     constexpr size_t num_mel_bins = 80; // Standard AI audio resolution

//     std::println("==================================================");
//     std::println("🎛️  Crescendo Engine: STFT & Mel-Filterbank Suite");
//     std::println("==================================================\n");

//     // 1. Initialize DSP Engines
//     auto stft = ShortTimeFourierTransform<float>(fft_size, hop_size);
//     auto mel_bank = MelFilterbank<float>(num_mel_bins, fft_size, sample_rate, 20.0f, 8000.0f);

//     std::println("✔ Initialized STFT (N={}, Hop={}) | Linear Bins: {}", fft_size, hop_size, stft.num_bins());
//     std::println("✔ Initialized Mel-Filterbank | Target Mel Bins: {}\n", mel_bank.num_mel_bins());

//     // 2. Generate 1 second of synthetic audio: Frequency sweep from 200 Hz -> 3000 Hz
//     const size_t total_samples = sample_rate;
//     std::vector<float> audio_buffer(total_samples);
//     for (size_t i = 0; i < total_samples; ++i) {
//         float t = static_cast<float>(i) / static_cast<float>(sample_rate);
//         float instant_freq = 200.0f + 2800.0f * t; // Linear frequency sweep
//         audio_buffer[i] = 0.8f * std::sin(2.0f * std::numbers::pi_v<float> * instant_freq * t);
//     }

//     // 3. Execute Forward STFT
//     auto start_stft = std::chrono::high_resolution_clock::now();
//     auto complex_spectrogram = stft.forward(audio_buffer);
//     auto end_stft = std::chrono::high_resolution_clock::now();
    
//     const size_t num_frames = complex_spectrogram.size();
//     auto stft_dur = std::chrono::duration_cast<std::chrono::microseconds>(end_stft - start_stft).count();
//     std::println("✔ Executed Forward STFT in {} µs | Extracted {} frames.", stft_dur, num_frames);

//     // 4. Compute Linear Magnitude Spectrogram (|X|^2)
//     std::vector<std::vector<float>> linear_magnitudes(num_frames, std::vector<float>(stft.num_bins()));
//     for (size_t f = 0; f < num_frames; ++f) {
//         for (size_t b = 0; b < stft.num_bins(); ++b) {
//             linear_magnitudes[f][b] = std::abs(complex_spectrogram[f][b]);
//         }
//     }

//     // 5. Execute Log-Mel Transformation
//     auto start_mel = std::chrono::high_resolution_clock::now();
//     auto log_mel_spectrogram = mel_bank.to_log_mel(linear_magnitudes);
//     auto end_mel = std::chrono::high_resolution_clock::now();
//     auto mel_dur = std::chrono::duration_cast<std::chrono::microseconds>(end_mel - start_mel).count();

//     std::println("✔ Converted to Log-Mel Spectrogram in {} µs | Shape: [{}, {}]", 
//                  mel_dur, log_mel_spectrogram.size(), log_mel_spectrogram[0].size());

//     // Display a slice of the generated Log-Mel matrix (first 5 bins of frame 50)
//     std::print("   Sample Log-Mel tokens (Frame 50, Bins 0-4): [ ");
//     for (size_t m = 0; m < 5; ++m) {
//         std::print("{:6.3f} ", log_mel_spectrogram[50][m]);
//     }
//     std::println("]\n");

//     // 6. Verify Overlap-Add Inverse STFT Reconstruction
//     auto reconstructed_audio = stft.inverse(complex_spectrogram, total_samples);
    
//     float max_error = 0.0f;
//     // Skip boundary frames where window overlap is incomplete at the very edges
//     for (size_t i = fft_size; i < total_samples - fft_size; ++i) {
//         float err = std::abs(audio_buffer[i] - reconstructed_audio[i]);
//         if (err > max_error) max_error = err;
//     }

//     std::println("✔ WOLA Reconstruction Max Absolute Error: {:.9f}", max_error);
//     if (max_error < 1e-4f) {
//         std::println("\n🏆 PHASE 2 PASSED: STFT & Mel-Filterbank pipeline ready for Neural Networks!");
//     } else {
//         std::println("\n❌ PHASE 2 FAILED: Overlap-Add amplitude distortion exceeded threshold.");
//     }

//     return 0;
// }