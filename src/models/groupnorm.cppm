export module crescendo.models.groupnorm;

import std;
import crescendo.tensor.core;

export namespace crescendo::models {

    using namespace crescendo::tensor;

    // It divides channel feature maps into discrete groups and normalizes spatial activations independently of batch size.
    template <std::floating_point T = float>
    class GroupNorm {
    public:
        size_t num_groups_;
        size_t num_channels_;
        T eps_;
        Tensor<T> gamma_; // Scale parameter
        Tensor<T> beta_;  // Shift parameter

        GroupNorm(size_t num_groups, size_t num_channels, T eps = T{1e-5})
            : num_groups_(num_groups), num_channels_(num_channels), eps_(eps) {
            if (num_channels_ % num_groups_ != 0) {
                throw std::invalid_argument("Channels must be evenly divisible by number of groups.");
            }
            gamma_ = Tensor<T>({num_channels_}, T{1.0});
            beta_  = Tensor<T>({num_channels_}, T{0.0});
        }

        [[nodiscard]] Tensor<T> forward(const Tensor<T>& input) const {
            const auto& shape = input.shape();
            const size_t batch = shape[0];
            const size_t C = shape[1];
            const size_t spatial_size = shape[2] * shape[3];
            const size_t channels_per_group = C / num_groups_;
            const size_t group_size = channels_per_group * spatial_size;

            Tensor<T> output(shape);
            const T* __restrict in_ptr = input.data();
            T* __restrict out_ptr = output.data();

            for (size_t b = 0; b < batch; ++b) {
                for (size_t g = 0; g < num_groups_; ++g) {
                    const size_t offset = b * (C * spatial_size) + g * group_size;
                    
                    // 1. Compute Mean within channel group
                    T sum = T{0.0};
                    for (size_t i = 0; i < group_size; ++i) {
                        sum += in_ptr[offset + i];
                    }
                    const T mean = sum / static_cast<T>(group_size);

                    // 2. Compute Variance within channel group
                    T var_sum = T{0.0};
                    for (size_t i = 0; i < group_size; ++i) {
                        T diff = in_ptr[offset + i] - mean;
                        var_sum += diff * diff;
                    }
                    const T inv_std = T{1.0} / std::sqrt(var_sum / static_cast<T>(group_size) + eps_);

                    // 3. Normalize and apply affine transformation (gamma * x_hat + beta)
                    for (size_t c = 0; c < channels_per_group; ++c) {
                        const size_t ch_idx = g * channels_per_group + c;
                        const T g_val = gamma_[ch_idx];
                        const T b_val = beta_[ch_idx];

                        for (size_t s = 0; s < spatial_size; ++s) {
                            const size_t idx = offset + c * spatial_size + s;
                            out_ptr[idx] = (in_ptr[idx] - mean) * inv_std * g_val + b_val;
                        }
                    }
                }
            }
            return output;
        }
    };
}