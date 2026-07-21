export module crescendo.diffusion.ddim;

import std;
import crescendo.tensor.core;
import crescendo.tensor.simd;
import crescendo.models.unet;
import crescendo.diffusion.scheduler;

export namespace crescendo::diffusion {

    using namespace crescendo::tensor;
    using namespace crescendo::models;

    // Takes the trained Unet2D model, generates a deterministic sub-sequence of timesteps (T1, ..., Ts) and iteratively predicts clean spectrograms without random stochastic noise injection.
    template <std::floating_point T = float>
    class DDIMSampler {
    public:
        explicit DDIMSampler(const CosineScheduler<T>& scheduler)
            : scheduler_(scheduler) {}

        /**
         * @brief Executes deterministic DDIM fast inference to generate a clean Mel-Spectrogram.
         * @param model Trained UNet2D generative backbone from Phase 4
         * @param initial_noise Pure Gaussian static N(0, I) tensor of shape [1, 1, mel_bins, time_frames]
         * @param inference_steps Number of fast sampling steps (e.g., 20 to 50)
         */
        [[nodiscard]] Tensor<T> sample(const UNet2D<T>& model, const Tensor<T>& initial_noise, 
                                       size_t inference_steps = 30) const {
            if (inference_steps == 0 || inference_steps > scheduler_.total_steps_) {
                throw std::invalid_argument("Inference steps must be strictly between 1 and total scheduler steps.");
            }

            // 1. Generate uniform timestep sub-sequence: e.g., [999, 965, 931, ..., 0]
            std::vector<size_t> timesteps;
            const size_t step_ratio = scheduler_.total_steps_ / inference_steps;
            for (ptrdiff_t i = static_cast<ptrdiff_t>(inference_steps) - 1; i >= 0; --i) {
                timesteps.push_back(static_cast<size_t>(i) * step_ratio);
            }

            Tensor<T> x_curr = initial_noise;
            const size_t total_elements = x_curr.size();

            // 2. Execute Reverse Denoising Loop
            for (size_t step_idx = 0; step_idx < timesteps.size(); ++step_idx) {
                const size_t t = timesteps[step_idx];
                const size_t t_prev = (step_idx + 1 < timesteps.size()) ? timesteps[step_idx + 1] : 0;

                // Predict noise matrix \epsilon_\theta using U-Net backbone
                Tensor<T> predicted_noise = model.forward(x_curr);

                // Extract precomputed square root schedule parameters for current step
                const T sqrt_alpha_t     = scheduler_.sqrt_alphas_cumprod_[t];  // const T alpha_bar_t = scheduler_.alphas_cumprod_[t];
                const T sqrt_one_minus_t = scheduler_.sqrt_one_minus_alphas_cumprod_[t];

                const T alpha_bar_prev      = (step_idx + 1 < timesteps.size()) ? scheduler_.alphas_cumprod_[t_prev] : T{1.0};
                const T sqrt_alpha_prev     = std::sqrt(alpha_bar_prev);
                const T sqrt_one_minus_prev = std::sqrt(T{1.0} - alpha_bar_prev);

                const T* __restrict x_ptr = x_curr.data();
                const T* __restrict eps_ptr = predicted_noise.data();
                T* __restrict next_x_ptr = x_curr.data(); // In-place update

                // 3. Compute deterministic DDIM step via SIMD vectorization
                size_t i = 0;
                #if defined(CRESCENDO_SIMD_AVX2)
                    if constexpr (std::is_same_v<T, float>) {
                        const __m256 v_sqrt_alpha_t = _mm256_set1_ps(sqrt_alpha_t);
                        const __m256 v_sqrt_one_minus_t = _mm256_set1_ps(sqrt_one_minus_t);
                        const __m256 v_sqrt_alpha_prev = _mm256_set1_ps(sqrt_alpha_prev);
                        const __m256 v_sqrt_one_minus_prev = _mm256_set1_ps(sqrt_one_minus_prev);

                        for (; i + 8 <= total_elements; i += 8) {
                            __m256 vx = _mm256_loadu_ps(&x_ptr[i]);
                            __m256 veps = _mm256_loadu_ps(&eps_ptr[i]);

                            // Compute predicted clean spectrogram x_0_pred = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
                            // v_sqrt_one_minus_t = sqrt(1 - alpha_bar_t)
                            // v_sqrt_alpha_t = sqrt(alpha_bar_t)
                            __m256 noise_part = _mm256_mul_ps(veps, v_sqrt_one_minus_t);
                            __m256 x0_pred    = _mm256_div_ps(_mm256_sub_ps(vx, noise_part), v_sqrt_alpha_t);

                            // Compute direction pointing to x_t = sqrt(1 - alpha_bar_prev) * eps
                            __m256 dir_xt = _mm256_mul_ps(veps, v_sqrt_one_minus_prev);

                            // Compute x_{t_prev} = sqrt(alpha_bar_prev) * x0_pred + dir_xt
                            __m256 next_vec = _mm256_fmadd_ps(v_sqrt_alpha_prev, x0_pred, dir_xt);
                            _mm256_storeu_ps(&next_x_ptr[i], next_vec);
                        }
                    }
                #elif defined(CRESCENDO_SIMD_NEON)
                    if constexpr (std::is_same_v<T, float>) {
                        const float32x4_t v_sqrt_alpha_t = vdupq_n_f32(sqrt_alpha_t);
                        const float32x4_t v_sqrt_one_minus_t = vdupq_n_f32(sqrt_one_minus_t);
                        const float32x4_t v_sqrt_alpha_prev = vdupq_n_f32(sqrt_alpha_prev);
                        const float32x4_t v_sqrt_one_minus_prev = vdupq_n_f32(sqrt_one_minus_prev);

                        for (; i + 4 <= total_elements; i += 4) {
                            float32x4_t vx = vld1q_f32(&x_ptr[i]);
                            float32x4_t veps = vld1q_f32(&eps_ptr[i]);

                            float32x4_t noise_part = vmulq_f32(veps, v_sqrt_one_minus_t);
                            float32x4_t x0_pred    = vmulq_f32(vsubq_f32(vx, noise_part), vdivq_f32(vdupq_n_f32(1.0f), v_sqrt_alpha_t));
                            float32x4_t dir_xt     = vmulq_f32(veps, v_sqrt_one_minus_prev);
                            float32x4_t next_vec   = vmlaq_f32(dir_xt, v_sqrt_alpha_prev, x0_pred);
                            vst1q_f32(&next_x_ptr[i], next_vec);
                        }
                    }
                #endif
                // Scalar fallback loop for remaining trailing elements
                for (; i < total_elements; ++i) {
                    T x0_pred = (x_ptr[i] - sqrt_one_minus_t * eps_ptr[i]) / sqrt_alpha_t;
                    T dir_xt  = sqrt_one_minus_prev * eps_ptr[i];
                    next_x_ptr[i] = sqrt_alpha_prev * x0_pred + dir_xt;
                }
            }

            return x_curr;
        }

    private:
        const CosineScheduler<T>& scheduler_;
    };
}