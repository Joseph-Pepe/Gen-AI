export module crescendo.dsp.mel_filterbank;

import std;

export namespace crescendo::dsp {

    /**
     * @brief High-performance Log-Mel Filterbank matrix transformation.
     * Maps linear FFT magnitude bins onto perceptual logarithmic Mel frequency bands.
     */
    template <std::floating_point T = float>
    class MelFilterbank {
    public:
        MelFilterbank(size_t num_mel_bins, size_t fft_size, size_t sample_rate, T f_min = T{0.0}, T f_max = T{0.0})
            : num_mel_bins_(num_mel_bins),
              num_linear_bins_(fft_size / 2 + 1),
              sample_rate_(sample_rate),
              f_min_(f_min),
              f_max_(f_max <= T{0.0} ? static_cast<T>(sample_rate) / T{2.0} : f_max) {
            
            if (f_max_ <= f_min_ || f_max_ > static_cast<T>(sample_rate) / T{2.0}) {
                throw std::invalid_argument("Invalid frequency range for Mel filterbank.");
            }
            build_filterbank_matrix(fft_size);
        }

        /**
         * @brief Converts a 2D linear magnitude spectrogram into a Log-Mel spectrogram normalized to [-1, 1].
         * Input Shape:  [num_frames][num_linear_bins]
         * Output Shape: [num_frames][num_mel_bins]
         */
        [[nodiscard]] std::vector<std::vector<T>> 
        to_log_mel(const std::vector<std::vector<T>>& linear_magnitudes) const {
            if (linear_magnitudes.empty()) return {};

            const size_t num_frames = linear_magnitudes.size();
            std::vector<std::vector<T>> log_mel(num_frames, std::vector<T>(num_mel_bins_, T{0.0}));

            constexpr T epsilon = T{1e-5}; // Prevent log(0)
            T max_val = -std::numeric_limits<T>::infinity();
            T min_val = std::numeric_limits<T>::infinity();

            // 1. Matrix Multiplication: Mel = Filterbank * Magnitude
            for (size_t frame = 0; frame < num_frames; ++frame) {
                for (size_t m = 0; m < num_mel_bins_; ++m) {
                    T mel_energy = T{0.0};
                    const auto& filter_weights = filterbank_[m];
                    
                    // Dot product across active linear bins
                    for (size_t k = 0; k < num_linear_bins_; ++k) {
                        if (filter_weights[k] > T{0.0}) {
                            mel_energy += filter_weights[k] * linear_magnitudes[frame][k];
                        }
                    }

                    // Apply natural logarithm compression
                    T log_val = std::log(mel_energy + epsilon);
                    log_mel[frame][m] = log_val;

                    if (log_val > max_val) max_val = log_val;
                    if (log_val < min_val) min_val = log_val;
                }
            }

            // 2. Linear Min-Max Normalization to [-1.0, 1.0] for neural network training
            const T range = std::max(max_val - min_val, T{1e-8});
            for (size_t frame = 0; frame < num_frames; ++frame) {
                for (size_t m = 0; m < num_mel_bins_; ++m) {
                    log_mel[frame][m] = T{2.0} * ((log_mel[frame][m] - min_val) / range) - T{1.0};
                }
            }

            return log_mel;
        }

        [[nodiscard]] size_t num_mel_bins() const noexcept { return num_mel_bins_; }

    private:
        size_t num_mel_bins_;
        size_t num_linear_bins_;
        size_t sample_rate_;
        T f_min_;
        T f_max_;
        std::vector<std::vector<T>> filterbank_; // Shape: [num_mel_bins][num_linear_bins]

        static inline T hz_to_mel(T hz) noexcept {
            return T{2595.0} * std::log10(T{1.0} + hz / T{700.0});
        }

        static inline T mel_to_hz(T mel) noexcept {
            return T{700.0} * (std::pow(T{10.0}, mel / T{2595.0}) - T{1.0});
        }

        /**
         * @brief Constructs triangular overlapping filters spaced uniformly on the Mel scale.
         */
        void build_filterbank_matrix(size_t fft_size) {
            filterbank_.assign(num_mel_bins_, std::vector<T>(num_linear_bins_, T{0.0}));

            const T mel_min = hz_to_mel(f_min_);
            const T mel_max = hz_to_mel(f_max_);
            
            // We need M + 2 center points to build M overlapping triangular filters
            std::vector<T> mel_points(num_mel_bins_ + 2);
            for (size_t i = 0; i < mel_points.size(); ++i) {
                mel_points[i] = mel_min + static_cast<T>(i) * (mel_max - mel_min) / static_cast<T>(num_mel_bins_ + 1);
            }

            // Convert Mel points back to linear FFT bin indices
            std::vector<size_t> bin_indices(mel_points.size());
            for (size_t i = 0; i < mel_points.size(); ++i) {
                const T hz = mel_to_hz(mel_points[i]);
                bin_indices[i] = static_cast<size_t>(std::floor((static_cast<T>(fft_size) + T{1.0}) * hz / static_cast<T>(sample_rate_)));
            }

            // Construct triangular slopes
            for (size_t m = 0; m < num_mel_bins_; ++m) {
                const size_t left   = bin_indices[m];
                const size_t center = bin_indices[m + 1];
                const size_t right  = bin_indices[m + 2];

                for (size_t k = left; k < center; ++k) {
                    if (center != left) {
                        filterbank_[m][k] = static_cast<T>(k - left) / static_cast<T>(center - left);
                    }
                }
                for (size_t k = center; k < right; ++k) {
                    if (right != center) {
                        filterbank_[m][k] = static_cast<T>(right - k) / static_cast<T>(right - center);
                    }
                }
            }
        }
    };
}