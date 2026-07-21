export module crescendo.dsp.fast_fourier_transform;

import std;

export namespace crescendo::dsp {

    /**
     * @brief High-performance, zero-dependency Radix-2 Cooley-Tukey FFT.
     * Operates in-place on std::span<std::complex<T>> with precomputed lookup tables.
     */
    template <std::floating_point T = float>
    class Radix2FFT {
    public:
        explicit Radix2FFT(size_t fft_size) : n_(fft_size) {
            // Radix-2 FFT strictly requires a power of 2
            if (!std::has_single_bit(n_)) {
                throw std::invalid_argument("FFT size must be a power of 2.");
            }
            log2_n_ = std::countr_zero(n_);
            precompute_bit_reversal();
            precompute_twiddles();
        }

        /**
         * @brief Executes an in-place Forward FFT.
         * Time-Domain (PCM samples) -> Frequency-Domain (Complex Spectrum)
         */
        void forward(std::span<std::complex<T>> buffer) const {
            validate_buffer(buffer);
            bit_reverse_permute(buffer);
            compute_butterflies(buffer, false);
        }

        /**
         * @brief Executes an in-place Inverse FFT (IFFT) with 1/N normalization.
         * Frequency-Domain (Complex Spectrum) -> Time-Domain (PCM samples)
         */
        void inverse(std::span<std::complex<T>> buffer) const {
            validate_buffer(buffer);
            bit_reverse_permute(buffer);
            compute_butterflies(buffer, true);

            // Apply 1/N scaling factor for inverse transformation
            const T scale = T{1.0} / static_cast<T>(n_);
            for (auto& sample : buffer) {
                sample *= scale;
            }
        }

        [[nodiscard]] size_t size() const noexcept { return n_; }

    private:
        size_t n_;
        size_t log2_n_;
        std::vector<size_t> bit_rev_indices_;
        std::vector<std::complex<T>> forward_twiddles_;
        std::vector<std::complex<T>> inverse_twiddles_;

        void validate_buffer(std::span<std::complex<T>> buffer) const {
            if (buffer.size() != n_) {
                throw std::invalid_argument("Buffer size does not match precomputed FFT size.");
            }
        }

        /**
         * @brief Precomputes bit-reversal swap indices to avoid runtime bit-twiddling.
         */
        void precompute_bit_reversal() {
            bit_rev_indices_.resize(n_);
            for (size_t i = 0; i < n_; ++i) {
                size_t rev = 0;
                size_t val = i;
                for (size_t bit = 0; bit < log2_n_; ++bit) {
                    rev = (rev << 1) | (val & 1);
                    val >>= 1;
                }
                bit_rev_indices_[i] = rev;
            }
        }

        /**
         * @brief Precomputes forward and inverse twiddle factor tables: W_N^k = exp(-i * 2*pi*k / N)
         * Storing half the FFT size (N/2) is sufficient for all butterfly stages.
         */
        void precompute_twiddles() {
            const size_t half_n = n_ / 2;
            forward_twiddles_.resize(half_n);
            inverse_twiddles_.resize(half_n);

            for (size_t k = 0; k < half_n; ++k) {
                const T angle = -T{2.0} * std::numbers::pi_v<T> * static_cast<T>(k) / static_cast<T>(n_);
                forward_twiddles_[k]  = std::polar(T{1.0}, angle);
                inverse_twiddles_[k]  = std::polar(T{1.0}, -angle); // Conjugate for inverse
            }
        }

        /**
         * @brief In-place array permutation using precomputed lookup table.
         * Complexity: O(N) memory swaps.
         */
        void bit_reverse_permute(std::span<std::complex<T>> buffer) const {
            for (size_t i = 0; i < n_; ++i) {
                const size_t rev_i = bit_rev_indices_[i];
                if (i < rev_i) {
                    std::swap(buffer[i], buffer[rev_i]);
                }
            }
        }

        /**
         * @brief Iterative Cooley-Tukey butterfly computation.
         * Complexity: O(N log N) arithmetic operations with sequential memory access.
         */
        void compute_butterflies(std::span<std::complex<T>> buffer, bool is_inverse) const {
            const auto& twiddles = is_inverse ? inverse_twiddles_ : forward_twiddles_;

            // Outer loop: Iterate over tree stages (len = 2, 4, 8, ..., N)
            for (size_t len = 2; len <= n_; len <<= 1) {
                const size_t half_len = len >> 1;
                const size_t step = n_ / len; // Stride through our precomputed N/2 twiddle table

                // Middle loop: Iterate over independent butterfly blocks
                for (size_t i = 0; i < n_; i += len) {
                    // Inner loop: Execute butterfly pairs within the block
                    for (size_t j = 0; j < half_len; ++j) {
                        const std::complex<T> w = twiddles[j * step];
                        const std::complex<T> u = buffer[i + j];
                        const std::complex<T> v = buffer[i + j + half_len] * w;

                        buffer[i + j]            = u + v;
                        buffer[i + j + half_len] = u - v;
                    }
                }
            }
        }
    };
}