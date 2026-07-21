export module crescendo.dsp.wiener_filter;

import std;
import crescendo.tensor.core;
import crescendo.tensor.simd;

export namespace crescendo::dsp {

    using namespace crescendo::tensor;

    // Takes the original complex STFT mixture spectrogram and applies Wiener normalization across the 4 neural network mask predictions to clealy isolate each stem.
    template <std::floating_point T = float>
    class WienerFilter4Stem {
    public:
        static constexpr size_t NUM_STEMS = 4; // 0: Vocals, 1: Drums, 2: Bass, 3: Other
        T eps_;

        explicit WienerFilter4Stem(T epsilon = T{1e-8}) : eps_(epsilon) {}

        /**
         * @brief Applies 4-stem Wiener filtering to a complex mixture spectrogram.
         * @param complex_mix Original mixture STFT spectrogram of shape [1, 1, num_bins, frames]
         * @param masks 4 magnitude mask tensors predicted by U-Net, each of shape [1, 1, num_bins, frames]
         * @return std::array of 4 separated complex spectrogram tensors ready for Inverse STFT.
         */
        [[nodiscard]] std::array<Tensor<std::complex<T>>, NUM_STEMS> 
        separate(const Tensor<std::complex<T>>& complex_mix, 
                 const std::array<Tensor<T>, NUM_STEMS>& masks) const {
            
            const auto& shape = complex_mix.shape();
            const size_t total_elements = complex_mix.size();

            for (size_t k = 0; k < NUM_STEMS; ++k) {
                if (masks[k].shape() != shape) {
                    throw std::invalid_argument("All stem masks must match the dimensions of the complex mixture.");
                }
            }

            std::array<Tensor<std::complex<T>>, NUM_STEMS> separated_stems = {
                Tensor<std::complex<T>>(shape), Tensor<std::complex<T>>(shape),
                Tensor<std::complex<T>>(shape), Tensor<std::complex<T>>(shape)
            };

            const std::complex<T>* __restrict mix_ptr = complex_mix.data();
            const T* __restrict m0_ptr = masks[0].data();
            const T* __restrict m1_ptr = masks[1].data();
            const T* __restrict m2_ptr = masks[2].data();
            const T* __restrict m3_ptr = masks[3].data();

            std::complex<T>* __restrict out0_ptr = separated_stems[0].data();
            std::complex<T>* __restrict out1_ptr = separated_stems[1].data();
            std::complex<T>* __restrict out2_ptr = separated_stems[2].data();
            std::complex<T>* __restrict out3_ptr = separated_stems[3].data();

            size_t i = 0;
            #if defined(CRESCENDO_SIMD_AVX512)
                if constexpr (std::is_same_v<T, float>) {
                    const __m512 v_eps = _mm512_set1_ps(eps_);
                    for (; i + 16 <= total_elements; i += 16) {
                        __m512 v0 = _mm512_loadu_ps(&m0_ptr[i]);
                        __m512 v1 = _mm512_loadu_ps(&m1_ptr[i]);
                        __m512 v2 = _mm512_loadu_ps(&m2_ptr[i]);
                        __m512 v3 = _mm512_loadu_ps(&m3_ptr[i]);

                        // Square masks: M_k^2
                        __m512 sq0 = _mm512_mul_ps(v0, v0);
                        __m512 sq1 = _mm512_mul_ps(v1, v1);
                        __m512 sq2 = _mm512_mul_ps(v2, v2);
                        __m512 sq3 = _mm512_mul_ps(v3, v3);

                        // Denominator: sum(M_j^2) + eps
                        __m512 denom = _mm512_add_ps(_mm512_add_ps(sq0, sq1), _mm512_add_ps(sq2, sq3));
                        denom = _mm512_add_ps(denom, v_eps);

                        // Wiener Ratios: W_k = M_k^2 / denom
                        __m512 w0 = _mm512_div_ps(sq0, denom);
                        __m512 w1 = _mm512_div_ps(sq1, denom);
                        __m512 w2 = _mm512_div_ps(sq2, denom);
                        __m512 w3 = _mm512_div_ps(sq3, denom);

                        // Apply ratio to complex mixture (scale real and imaginary parts identical to scalar mul)
                        for (size_t offset = 0; offset < 16; ++offset) {
                            out0_ptr[i + offset] = mix_ptr[i + offset] * reinterpret_cast<float*>(&w0)[offset];
                            out1_ptr[i + offset] = mix_ptr[i + offset] * reinterpret_cast<float*>(&w1)[offset];
                            out2_ptr[i + offset] = mix_ptr[i + offset] * reinterpret_cast<float*>(&w2)[offset];
                            out3_ptr[i + offset] = mix_ptr[i + offset] * reinterpret_cast<float*>(&w3)[offset];
                        }
                    }
                }
            #elif defined(CRESCENDO_SIMD_AVX2)
                if constexpr (std::is_same_v<T, float>) {
                    const __m256 v_eps = _mm256_set1_ps(eps_);
                    for (; i + 8 <= total_elements; i += 8) {
                        __m256 v0 = _mm256_loadu_ps(&m0_ptr[i]);
                        __m256 v1 = _mm256_loadu_ps(&m1_ptr[i]);
                        __m256 v2 = _mm256_loadu_ps(&m2_ptr[i]);
                        __m256 v3 = _mm256_loadu_ps(&m3_ptr[i]);

                        __m256 sq0 = _mm256_mul_ps(v0, v0);
                        __m256 sq1 = _mm256_mul_ps(v1, v1);
                        __m256 sq2 = _mm256_mul_ps(v2, v2);
                        __m256 sq3 = _mm256_mul_ps(v3, v3);

                        __m256 denom = _mm256_add_ps(_mm256_add_ps(sq0, sq1), _mm256_add_ps(sq2, sq3));
                        denom = _mm256_add_ps(denom, v_eps);

                        __m256 w0 = _mm256_div_ps(sq0, denom);
                        __m256 w1 = _mm256_div_ps(sq1, denom);
                        __m256 w2 = _mm256_div_ps(sq2, denom);
                        __m256 w3 = _mm256_div_ps(sq3, denom);

                        for (size_t offset = 0; offset < 8; ++offset) {
                            out0_ptr[i + offset] = mix_ptr[i + offset] * reinterpret_cast<float*>(&w0)[offset];
                            out1_ptr[i + offset] = mix_ptr[i + offset] * reinterpret_cast<float*>(&w1)[offset];
                            out2_ptr[i + offset] = mix_ptr[i + offset] * reinterpret_cast<float*>(&w2)[offset];
                            out3_ptr[i + offset] = mix_ptr[i + offset] * reinterpret_cast<float*>(&w3)[offset];
                        }
                    }
                }
            #elif defined(CRESCENDO_SIMD_NEON)
                if constexpr (std::is_same_v<T, float>) {
                    const float32x4_t v_eps = vdupq_n_f32(eps_);
                    for (; i + 4 <= total_elements; i += 4) {
                        float32x4_t v0 = vld1q_f32(&m0_ptr[i]);
                        float32x4_t v1 = vld1q_f32(&m1_ptr[i]);
                        float32x4_t v2 = vld1q_f32(&m2_ptr[i]);
                        float32x4_t v3 = vld1q_f32(&m3_ptr[i]);

                        float32x4_t sq0 = vmulq_f32(v0, v0);
                        float32x4_t sq1 = vmulq_f32(v1, v1);
                        float32x4_t sq2 = vmulq_f32(v2, v2);
                        float32x4_t sq3 = vmulq_f32(v3, v3);

                        float32x4_t denom = vaddq_f32(vaddq_f32(sq0, sq1), vaddq_f32(sq2, sq3));
                        denom = vaddq_f32(denom, v_eps);

                        float32x4_t w0 = vmulq_f32(sq0, vrecpeq_f32(denom)); // Fast NEON reciprocal
                        float32x4_t w1 = vmulq_f32(sq1, vrecpeq_f32(denom));
                        float32x4_t w2 = vmulq_f32(sq2, vrecpeq_f32(denom));
                        float32x4_t w3 = vmulq_f32(sq3, vrecpeq_f32(denom));

                        for (size_t offset = 0; offset < 4; ++offset) {
                            out0_ptr[i + offset] = mix_ptr[i + offset] * reinterpret_cast<float*>(&w0)[offset];
                            out1_ptr[i + offset] = mix_ptr[i + offset] * reinterpret_cast<float*>(&w1)[offset];
                            out2_ptr[i + offset] = mix_ptr[i + offset] * reinterpret_cast<float*>(&w2)[offset];
                            out3_ptr[i + offset] = mix_ptr[i + offset] * reinterpret_cast<float*>(&w3)[offset];
                        }
                    }
                }
            #endif
            // Scalar fallback loop for remaining elements
            for (; i < total_elements; ++i) {
                T sq0 = m0_ptr[i] * m0_ptr[i];
                T sq1 = m1_ptr[i] * m1_ptr[i];
                T sq2 = m2_ptr[i] * m2_ptr[i];
                T sq3 = m3_ptr[i] * m3_ptr[i];

                T denom = sq0 + sq1 + sq2 + sq3 + eps_;
                out0_ptr[i] = mix_ptr[i] * (sq0 / denom);
                out1_ptr[i] = mix_ptr[i] * (sq1 / denom);
                out2_ptr[i] = mix_ptr[i] * (sq2 / denom);
                out3_ptr[i] = mix_ptr[i] * (sq3 / denom);
            }

            return separated_stems;
        }
    };
}