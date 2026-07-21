export module crescendo.diffusion.scheduler;

import std;
import crescendo.tensor.core;
import crescendo.tensor.simd;

export namespace crescendo::diffusion {

    using namespace crescendo::tensor;

    // Precomputes the cumulative alpha coefficients using a cosine schedule (which prevents spectrogram corruption from destroying low-frequency information too rapidly in early timesteps) and executes hardware SIMD Fused-Multiply-Add instruction to add noise in a single step.
    template <std::floating_point T = float>
    class CosineScheduler {
    public:
        size_t total_steps_;
        std::vector<T> alphas_cumprod_;      // \bar{\alpha}_t
        std::vector<T> sqrt_alphas_cumprod_; // \sqrt{\bar{\alpha}_t}
        std::vector<T> sqrt_one_minus_alphas_cumprod_; // \sqrt{1 - \bar{\alpha}_t}

        explicit CosineScheduler(size_t total_timesteps = 1000, T s = T{0.008})
            : total_steps_(total_timesteps) {
            precompute_cosine_schedule(s);
        }

        /**
         * @brief Corrupts a clean Mel-Spectrogram x_0 with Gaussian noise \epsilon at arbitrary timestep t.
         * Executes via SIMD vector scaling: Out = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
         */
        [[nodiscard]] Tensor<T> add_noise(const Tensor<T>& x_0, const Tensor<T>& noise, size_t t) const {
            if (t >= total_steps_) {
                throw std::out_of_range("Timestep exceeds maximum training scheduler steps.");
            }
            if (x_0.shape() != noise.shape()) {
                throw std::invalid_argument("Clean spectrogram and noise tensors must have identical dimensions.");
            }

            const T sqrt_alpha = sqrt_alphas_cumprod_[t];
            const T sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod_[t];
            const size_t total_size = x_0.size();

            Tensor<T> noisy_x(x_0.shape());
            const T* __restrict x_ptr = x_0.data();
            const T* __restrict n_ptr = noise.data();
            T* __restrict out_ptr = noisy_x.data();

            // Execute hardware SIMD vector loop
            size_t i = 0;
            #if defined(CRESCENDO_SIMD_AVX2)
                if constexpr (std::is_same_v<T, float>) {
                    const __m256 v_sqrt_alpha = _mm256_set1_ps(sqrt_alpha);
                    const __m256 v_sqrt_one_minus = _mm256_set1_ps(sqrt_one_minus_alpha);
                    for (; i + 8 <= total_size; i += 8) {
                        __m256 vx = _mm256_loadu_ps(&x_ptr[i]);
                        __m256 vn = _mm256_loadu_ps(&n_ptr[i]);
                        __m256 scaled_x = _mm256_mul_ps(vx, v_sqrt_alpha);
                        __m256 out_vec  = _mm256_fmadd_ps(vn, v_sqrt_one_minus, scaled_x);
                        _mm256_storeu_ps(&out_ptr[i], out_vec);
                    }
                }
            #elif defined(CRESCENDO_SIMD_NEON)
                if constexpr (std::is_same_v<T, float>) {
                    const float32x4_t v_sqrt_alpha = vdupq_n_f32(sqrt_alpha);
                    const float32x4_t v_sqrt_one_minus = vdupq_n_f32(sqrt_one_minus_alpha);
                    for (; i + 4 <= total_size; i += 4) {
                        float32x4_t vx = vld1q_f32(&x_ptr[i]);
                        float32x4_t vn = vld1q_f32(&n_ptr[i]);
                        float32x4_t scaled_x = vmulq_f32(vx, v_sqrt_alpha);
                        float32x4_t out_vec  = vmlaq_f32(scaled_x, vn, v_sqrt_one_minus);
                        vst1q_f32(&out_ptr[i], out_vec);
                    }
                }
            #endif
            // Scalar fallback loop for remaining trailing elements
            for (; i < total_size; ++i) {
                out_ptr[i] = sqrt_alpha * x_ptr[i] + sqrt_one_minus_alpha * n_ptr[i];
            }

            return noisy_x;
        }

    private:
        void precompute_cosine_schedule(T s) {
            alphas_cumprod_.resize(total_steps_);
            sqrt_alphas_cumprod_.resize(total_steps_);
            sqrt_one_minus_alphas_cumprod_.resize(total_steps_);

            auto f = [s](T t_val) -> T {
                T angle = (t_val + s) / (T{1.0} + s) * (std::numbers::pi_v<T> / T{2.0});
                T cos_val = std::cos(angle);
                return cos_val * cos_val;
            };

            const T f_0 = f(T{0.0});
            for (size_t t = 0; t < total_steps_; ++t) {
                T t_ratio = static_cast<T>(t + 1) / static_cast<T>(total_steps_);
                T alpha_bar = f(t_ratio) / f_0;
                
                // Clamp to prevent singularities at extreme ends of the schedule
                if (alpha_bar > T{0.9999}) alpha_bar = T{0.9999};
                if (alpha_bar < T{0.0001}) alpha_bar = T{0.0001};

                alphas_cumprod_[t] = alpha_bar;
                sqrt_alphas_cumprod_[t] = std::sqrt(alpha_bar);
                sqrt_one_minus_alphas_cumprod_[t] = std::sqrt(T{1.0} - alpha_bar);
            }
        }
    };
}