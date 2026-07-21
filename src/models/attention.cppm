export module crescendo.models.attention;

import std;
import crescendo.tensor.core;
import crescendo.tensor.simd;

export namespace crescendo::models {

    using namespace crescendo::tensor;

    // It flattens 2D spectrogram feature maps [ B, C, H, W ] into token sequences [ B, L, C ]. Computes projection matrices for Queries, Keys, and Values, evaluates scaled dot-product attention via SIMD matrix multiplication, and reshapes back into 4D spatial tensors.
    template <std::floating_point T = float>
    class MultiHeadSelfAttention {
    public:
        size_t channels_;
        size_t num_heads_;
        size_t head_dim_;

        // Projection weights: Shape [channels, channels]
        Tensor<T> w_q_, w_k_, w_v_, w_out_;

        MultiHeadSelfAttention(size_t channels, size_t num_heads)
            : channels_(channels), num_heads_(num_heads), head_dim_(channels / num_heads) {
            if (channels_ % num_heads_ != 0) {
                throw std::invalid_argument("Channels must be divisible by number of attention heads.");
            }
            T stddev = std::sqrt(T{2.0} / static_cast<T>(channels_));
            w_q_   = Tensor<T>::random_normal({channels_, channels_}, T{0.0}, stddev);
            w_k_   = Tensor<T>::random_normal({channels_, channels_}, T{0.0}, stddev);
            w_v_   = Tensor<T>::random_normal({channels_, channels_}, T{0.0}, stddev);
            w_out_ = Tensor<T>::random_normal({channels_, channels_}, T{0.0}, stddev);
        }

        /**
         * @brief Evaluates Scaled Dot-Product Attention: Softmax(Q * K^T / sqrt(head_dim)) * V
         * Input Shape:  [batch, channels, height, width]
         * Output Shape: [batch, channels, height, width] (Residual connection applied)
         */
        [[nodiscard]] Tensor<T> forward(const Tensor<T>& input) const {
            const auto& shape = input.shape();
            const size_t batch = shape[0];
            const size_t C = shape[1];
            const size_t H = shape[2];
            const size_t W = shape[3];
            const size_t L = H * W; // Token sequence length

            Tensor<T> output(shape);
            const T scale = T{1.0} / std::sqrt(static_cast<T>(head_dim_));

            // Working buffers for flattened sequence matrix multiplication
            std::vector<T> X_flat(L * C);
            std::vector<T> Q(L * C), K(L * C), V(L * C), AttnOut(L * C);
            std::vector<T> scores(L * L);

            for (size_t b = 0; b < batch; ++b) {
                // 1. Flatten spatial grid [C, H, W] into sequence [L, C] (Transpose spatial to row-major)
                const T* __restrict in_ptr = input.data() + b * (C * L);
                for (size_t c = 0; c < C; ++c) {
                    for (size_t l = 0; l < L; ++l) {
                        X_flat[l * C + c] = in_ptr[c * L + l];
                    }
                }

                // 2. SIMD Linear Projections: Q = X * W_Q, K = X * W_K, V = X * W_V
                if constexpr (std::is_same_v<T, float>) {
                    simd::gemm_f32(X_flat.data(), w_q_.data(), Q.data(), L, C, C);
                    simd::gemm_f32(X_flat.data(), w_k_.data(), K.data(), L, C, C);
                    simd::gemm_f32(X_flat.data(), w_v_.data(), V.data(), L, C, C);
                }

                // 3. Evaluate Attention per head
                std::fill(AttnOut.begin(), AttnOut.end(), T{0.0});
                for (size_t h = 0; h < num_heads_; ++h) {
                    const size_t head_offset = h * head_dim_;

                    // Compute Q_h * K_h^T scaled dot-product scores [L, L]
                    for (size_t i = 0; i < L; ++i) {
                        for (size_t j = 0; j < L; ++j) {
                            T dot = T{0.0};
                            for (size_t d = 0; d < head_dim_; ++d) {
                                dot += Q[i * C + head_offset + d] * K[j * C + head_offset + d];
                            }
                            scores[i * L + j] = dot * scale;
                        }
                    }

                    // Apply row-wise Softmax to scores matrix
                    for (size_t i = 0; i < L; ++i) {
                        T max_score = -std::numeric_limits<T>::infinity();
                        for (size_t j = 0; j < L; ++j) {
                            if (scores[i * L + j] > max_score) max_score = scores[i * L + j];
                        }
                        T exp_sum = T{0.0};
                        for (size_t j = 0; j < L; ++j) {
                            scores[i * L + j] = std::exp(scores[i * L + j] - max_score);
                            exp_sum += scores[i * L + j];
                        }
                        for (size_t j = 0; j < L; ++j) {
                            scores[i * L + j] /= exp_sum;
                        }
                    }

                    // Multiply attention weights by Value matrix V_h -> [L, head_dim]
                    for (size_t i = 0; i < L; ++i) {
                        for (size_t d = 0; d < head_dim_; ++d) {
                            T val = T{0.0};
                            for (size_t j = 0; j < L; ++j) {
                                val += scores[i * L + j] * V[j * C + head_offset + d];
                            }
                            AttnOut[i * C + head_offset + d] = val;
                        }
                    }
                }

                // 4. Final Output Projection: Out = AttnOut * W_Out
                std::vector<T> projected(L * C, T{0.0});
                if constexpr (std::is_same_v<T, float>) {
                    simd::gemm_f32(AttnOut.data(), w_out_.data(), projected.data(), L, C, C);
                }

                // 5. Reshape sequence back to [C, H, W] and add residual skip connection
                T* __restrict out_ptr = output.data() + b * (C * L);
                for (size_t c = 0; c < C; ++c) {
                    for (size_t l = 0; l < L; ++l) {
                        out_ptr[c * L + l] = in_ptr[c * L + l] + projected[l * C + c]; // Residual Add
                    }
                }
            }

            return output;
        }
    };
}