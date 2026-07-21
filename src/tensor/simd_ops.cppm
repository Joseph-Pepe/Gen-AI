// 1. GLOBAL MODULE FRAGMENT
module; 

#if defined(__AVX512F__)
  #include <immintrin.h>
  #define CRESCENDO_SIMD_AVX512 1
#elif defined(__AVX2__) || defined(CRESCENDO_USE_AVX2)
  #include <immintrin.h>
  #define CRESCENDO_SIMD_AVX2 1
#elif defined(__aarch64__) || defined(_M_ARM64) || defined(CRESCENDO_USE_NEON)
  #include <arm_neon.h>
  #define CRESCENDO_SIMD_NEON 1
#else
  #define CRESCENDO_SIMD_SCALAR 1
#endif

// 2. MODULE DECLARATION & IMPORTS
export module crescendo.tensor.simd;

import std;

// 3. MODULE PURVIEW (Exported Code)
export namespace crescendo::tensor::simd {

    [[nodiscard]] constexpr std::string_view get_simd_architecture() noexcept {
        #if defined(CRESCENDO_SIMD_AVX512)
            return "x86_64 / AVX-512 (512-bit Vector Registers)";
        #elif defined(CRESCENDO_SIMD_AVX2)
            return "x86_64 / AVX2 + FMA (256-bit Vector Registers)";
        #elif defined(CRESCENDO_SIMD_NEON)
            return "ARM64 / NEON (128-bit Vector Registers)";
        #else
            return "Scalar Fallback (No Hardware SIMD Detected)";
        #endif
    }

    /**
     * @brief High-performance SIMD General Matrix Multiplication (GEMM): C = A * B
     * A: [M x K], B: [K x N], C: [M x N]
     */
    void gemm_f32(const float* __restrict A, const float* __restrict B, float* __restrict C,
                  size_t M, size_t N, size_t K) noexcept {
        // Zero-initialize output buffer
        std::fill_n(C, M * N, 0.0f);

        #if defined(CRESCENDO_SIMD_AVX512)
            // AVX-512 processes 16 floats per vector instruction
            for (size_t i = 0; i < M; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    const __m512 a_val = _mm512_set1_ps(A[i * K + k]);
                    size_t j = 0;
                    for (; j + 16 <= N; j += 16) {
                        __m512 b_vec = _mm512_loadu_ps(&B[k * N + j]);
                        __m512 c_vec = _mm512_loadu_ps(&C[i * N + j]);
                        c_vec = _mm512_fmadd_ps(a_val, b_vec, c_vec); // C += A * B
                        _mm512_storeu_ps(&C[i * N + j], c_vec);
                    }
                    for (; j < N; ++j) {
                        C[i * N + j] += A[i * K + k] * B[k * N + j];
                    }
                }
            }
        #elif defined(CRESCENDO_SIMD_AVX2)
            // AVX2 processes 8 floats per vector instruction
            for (size_t i = 0; i < M; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    const __m256 a_val = _mm256_set1_ps(A[i * K + k]);
                    size_t j = 0;
                    for (; j + 8 <= N; j += 8) {
                        __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                        __m256 c_vec = _mm256_loadu_ps(&C[i * N + j]);
                        c_vec = _mm256_fmadd_ps(a_val, b_vec, c_vec);
                        _mm256_storeu_ps(&C[i * N + j], c_vec);
                    }
                    for (; j < N; ++j) {
                        C[i * N + j] += A[i * K + k] * B[k * N + j];
                    }
                }
            }
        #elif defined(CRESCENDO_SIMD_NEON)
            // ARM NEON processes 4 floats per vector instruction
            for (size_t i = 0; i < M; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    const float32x4_t a_val = vdupq_n_f32(A[i * K + k]);
                    size_t j = 0;
                    for (; j + 4 <= N; j += 4) {
                        float32x4_t b_vec = vld1q_f32(&B[k * N + j]);
                        float32x4_t c_vec = vld1q_f32(&C[i * N + j]);
                        c_vec = vmlaq_f32(c_vec, a_val, b_vec);
                        vst1q_f32(&C[i * N + j], c_vec);
                    }
                    for (; j < N; ++j) {
                        C[i * N + j] += A[i * K + k] * B[k * N + j];
                    }
                }
            }
        #else
            // Scalar Cache-Friendly Tiled Fallback
            for (size_t i = 0; i < M; ++i) {
                for (size_t k = 0; k < K; ++k) {
                    const float a_ik = A[i * K + k];
                    for (size_t j = 0; j < N; ++j) {
                        C[i * N + j] += a_ik * B[k * N + j];
                    }
                }
            }
        #endif
    }

    /**
     * @brief SIMD Element-wise Vector Addition: Out = A + B
     */
    void add_f32(const float* __restrict A, const float* __restrict B, float* __restrict Out, size_t size) noexcept {
        size_t i = 0;
        #if defined(CRESCENDO_SIMD_AVX2)
            for (; i + 8 <= size; i += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i]);
                __m256 b_vec = _mm256_loadu_ps(&B[i]);
                _mm256_storeu_ps(&Out[i], _mm256_add_ps(a_vec, b_vec));
            }
        #elif defined(CRESCENDO_SIMD_NEON)
            for (; i + 4 <= size; i += 4) {
                float32x4_t a_vec = vld1q_f32(&A[i]);
                float32x4_t b_vec = vld1q_f32(&B[i]);
                vst1q_f32(&Out[i], vaddq_f32(a_vec, b_vec));
            }
        #endif
            for (; i < size; ++i) { Out[i] = A[i] + B[i]; }
    }

    /**
     * @brief SIMD GELU Activation Function Approximation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
     */
    void gelu_f32(const float* __restrict In, float* __restrict Out, size_t size) noexcept {
        constexpr float sqrt_2_pi = 0.7978845608028654f;
        constexpr float coef = 0.044715f;
        for (size_t i = 0; i < size; ++i) {
            float x = In[i];
            float x3 = x * x * x;
            Out[i] = 0.5f * x * (1.0f + std::tanh(sqrt_2_pi * (x + coef * x3)));
        }
    }
}