export module crescendo.dsp.short_term_fourier_transform;

import std;
import crescendo.dsp.fast_fourier_transform;

export namespace crescendo::dsp {

    /**
     * @brief High-performance Short-Time Fourier Transform (STFT) and Inverse STFT.
     * Operates on overlapping time-domain frames using a precomputed Hanning window.
     */
    template <std::floating_point T = float>
    class ShortTimeFourierTransform {
    public:
        ShortTimeFourierTransform(size_t fft_size, size_t hop_size)
            : fft_size_(fft_size),
              hop_size_(hop_size),
              num_bins_(fft_size / 2 + 1),
              fft_(fft_size) {
            if (hop_size_ == 0 || hop_size_ > fft_size_) {
                throw std::invalid_argument("Hop size must be > 0 and <= FFT size.");
            }
            precompute_hanning_window();
        }

        /**
         * @brief Slices 1D PCM audio into a 2D complex spectrogram.
         * Output Shape: [num_frames][num_bins] (Positive frequencies only)
         */
        [[nodiscard]] std::vector<std::vector<std::complex<T>>> 
        forward(std::span<const T> pcm_audio) const {
            if (pcm_audio.size() < fft_size_) {
                return {};
            }

            const size_t num_frames = (pcm_audio.size() - fft_size_) / hop_size_ + 1;
            std::vector<std::vector<std::complex<T>>> spectrogram(
                num_frames, std::vector<std::complex<T>>(num_bins_)
            );

            std::vector<std::complex<T>> fft_buffer(fft_size_);

            for (size_t frame = 0; frame < num_frames; ++frame) {
                const size_t start_idx = frame * hop_size_;

                // Apply Hanning window and load into complex FFT buffer
                for (size_t i = 0; i < fft_size_; ++i) {
                    fft_buffer[i] = std::complex<T>(pcm_audio[start_idx + i] * window_[i], T{0.0});
                }

                // Execute in-place FFT
                fft_.forward(fft_buffer);

                // Store only positive frequency bins (0 to N/2 inclusive)
                for (size_t bin = 0; bin < num_bins_; ++bin) {
                    spectrogram[frame][bin] = fft_buffer[bin];
                }
            }

            return spectrogram;
        }

        /**
         * @brief Reconstructs 1D PCM audio from a 2D complex spectrogram via Overlap-Add.
         */
        [[nodiscard]] std::vector<T> 
        inverse(const std::vector<std::vector<std::complex<T>>>& spectrogram, size_t output_length) const {
            if (spectrogram.empty()) return {};

            const size_t num_frames = spectrogram.size();
            std::vector<T> pcm_audio(output_length, T{0.0});
            std::vector<T> window_sum(output_length, T{0.0});
            std::vector<std::complex<T>> ifft_buffer(fft_size_);

            for (size_t frame = 0; frame < num_frames; ++frame) {
                const size_t start_idx = frame * hop_size_;
                if (start_idx + fft_size_ > output_length) break;

                // Reconstruct full Hermitian symmetric spectrum from positive bins
                for (size_t bin = 0; bin < num_bins_; ++bin) {
                    ifft_buffer[bin] = spectrogram[frame][bin];
                }
                for (size_t bin = num_bins_; bin < fft_size_; ++bin) {
                    ifft_buffer[bin] = std::conj(spectrogram[frame][fft_size_ - bin]);
                }

                // Execute in-place IFFT
                fft_.inverse(ifft_buffer);

                // Overlap-add windowed time-domain samples
                for (size_t i = 0; i < fft_size_; ++i) {
                    pcm_audio[start_idx + i] += ifft_buffer[i].real() * window_[i];
                    window_sum[start_idx + i] += window_[i] * window_[i];
                }
            }

            // Normalize by window overlap energy to prevent amplitude distortion
            for (size_t i = 0; i < output_length; ++i) {
                if (window_sum[i] > T{1e-8}) {
                    pcm_audio[i] /= window_sum[i];
                }
            }

            return pcm_audio;
        }

        [[nodiscard]] size_t num_bins() const noexcept { return num_bins_; }

    private:
        size_t fft_size_;
        size_t hop_size_;
        size_t num_bins_;
        Radix2FFT<T> fft_;
        std::vector<T> window_;

        /**
         * @brief Precomputes Hanning window: w[n] = 0.5 * (1 - cos(2*pi*n / (N - 1)))
         */
        void precompute_hanning_window() {
            window_.resize(fft_size_);
            const T denom = static_cast<T>(fft_size_ - 1);
            for (size_t i = 0; i < fft_size_; ++i) {
                window_[i] = T{0.5} * (T{1.0} - std::cos(T{2.0} * std::numbers::pi_v<T> * static_cast<T>(i) / denom));
            }
        }
    };
}