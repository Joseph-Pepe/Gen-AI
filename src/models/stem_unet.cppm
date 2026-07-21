export module crescendo.models.stem_unet;

import std;
import crescendo.tensor.core;
import crescendo.tensor.simd;
import crescendo.models.convolution2d;
import crescendo.models.groupnorm;
import crescendo.models.attention;

export namespace crescendo::models {

    using namespace crescendo::tensor;

    // Expands the generative U-Net architecture to output 4 distinct feature channels instead of 1, applying vectorized Sigmoid activation at the output layer to strictly bound mask predictions in [0, 1] range.
    template <std::floating_point T = float>
    class UNet4StemSeparation {
    public:
        // Encoder Block 1: [B, 1, H, W] -> [B, 32, H/2, W/2]
        Conv2D<T> enc_conv1_{1, 32, 3, 2, 1};
        GroupNorm<T> enc_gn1_{4, 32};

        // Encoder Block 2: [B, 32, H/2, W/2] -> [B, 64, H/4, W/4]
        Conv2D<T> enc_conv2_{32, 64, 3, 2, 1};
        GroupNorm<T> enc_gn2_{8, 64};

        // Attention Bottleneck: [B, 64, H/4, W/4]
        Conv2D<T> bot_conv_{64, 64, 3, 1, 1};
        GroupNorm<T> bot_gn_{8, 64};
        MultiHeadSelfAttention<T> bot_attn_{64, 4};

        // Decoder Block 1 (Upsample + Skip Concat): [B, 96, H/2, W/2] -> [B, 32, H/2, W/2]
        Conv2D<T> dec_conv1_{96, 32, 3, 1, 1};
        GroupNorm<T> dec_gn1_{4, 32};

        // Output Mask Projection: [B, 32, H, W] -> [B, 4, H, W] (4 Stems!)
        Conv2D<T> out_mask_conv_{32, 4, 3, 1, 1};

        UNet4StemSeparation() = default;

        /**
         * @brief Evaluates mixture magnitude spectrogram and outputs 4 soft-mask tensors.
         * Input Shape:  [1, 1, num_bins, frames]
         * Output Shape: std::array of 4 tensors, each [1, 1, num_bins, frames]
         */
        [[nodiscard]] std::array<Tensor<T>, 4> forward_masks(const Tensor<T>& mixture_mag) const {
            auto h1 = enc_conv1_.forward(mixture_mag);
            h1 = enc_gn1_.forward(h1);
            h1.apply_gelu_inplace();

            auto h2 = enc_conv2_.forward(h1);
            h2 = enc_gn2_.forward(h2);
            h2.apply_gelu_inplace();

            auto bot = bot_conv_.forward(h2);
            bot = bot_gn_.forward(bot);
            bot.apply_gelu_inplace();
            bot = bot_attn_.forward(bot);

            auto bot_upsampled = upsample_2x(bot);
            auto dec1_in = concat_channels(bot_upsampled, h1);
            
            auto dec1 = dec_conv1_.forward(dec1_in);
            dec1 = dec_gn1_.forward(dec1);
            dec1.apply_gelu_inplace();

            auto dec1_upsampled = upsample_2x(dec1);
            auto raw_masks = out_mask_conv_.forward(dec1_upsampled); // Shape: [1, 4, H, W]

            // Apply in-place Sigmoid activation to bound masks in [0.0, 1.0]
            apply_sigmoid_inplace(raw_masks);

            // Split 4-channel tensor into 4 independent stem tensors [1, 1, H, W]
            return split_stems(raw_masks);
        }

    private:
        static void apply_sigmoid_inplace(Tensor<T>& tensor) noexcept {
            T* __restrict ptr = tensor.data();
            const size_t size = tensor.size();
            for (size_t i = 0; i < size; ++i) {
                ptr[i] = T{1.0} / (T{1.0} + std::exp(-ptr[i]));
            }
        }

        static std::array<Tensor<T>, 4> split_stems(const Tensor<T>& multi_channel) {
            const auto& shape = multi_channel.shape();
            const size_t H = shape[2], W = shape[3];
            const size_t spatial = H * W;

            std::array<Tensor<T>, 4> stems = {
                Tensor<T>({1, 1, H, W}), Tensor<T>({1, 1, H, W}),
                Tensor<T>({1, 1, H, W}), Tensor<T>({1, 1, H, W})
            };

            const T* __restrict in_ptr = multi_channel.data();
            for (size_t k = 0; k < 4; ++k) {
                std::copy_n(in_ptr + k * spatial, spatial, stems[k].data());
            }
            return stems;
        }

        static Tensor<T> upsample_2x(const Tensor<T>& input) {
            const auto& shape = input.shape();
            const size_t B = shape[0], C = shape[1], H = shape[2], W = shape[3];
            Tensor<T> out({B, C, H * 2, W * 2}, T{0.0});
            const T* __restrict in_ptr = input.data();
            T* __restrict out_ptr = out.data();

            for (size_t b = 0; b < B; ++b) {
                for (size_t c = 0; c < C; ++c) {
                    for (size_t h = 0; h < H; ++h) {
                        for (size_t w = 0; w < W; ++w) {
                            T val = in_ptr[b * (C * H * W) + c * (H * W) + h * W + w];
                            size_t offset = b * (C * H * 4 * W * 4) + c * (H * 2 * W * 2);
                            out_ptr[offset + (h * 2) * (W * 2) + (w * 2)]     = val;
                            out_ptr[offset + (h * 2) * (W * 2) + (w * 2 + 1)] = val;
                            out_ptr[offset + (h * 2 + 1) * (W * 2) + (w * 2)] = val;
                            out_ptr[offset + (h * 2 + 1) * (W * 2) + (w * 2 + 1)] = val;
                        }
                    }
                }
            }
            return out;
        }

        static Tensor<T> concat_channels(const Tensor<T>& a, const Tensor<T>& b) {
            const auto& sa = a.shape(); const auto& sb = b.shape();
            const size_t B = sa[0], Ca = sa[1], Cb = sb[1], H = sa[2], W = sa[3];
            Tensor<T> out({B, Ca + Cb, H, W}, T{0.0});
            const size_t spatial = H * W;
            for (size_t b_idx = 0; b_idx < B; ++b_idx) {
                std::copy_n(a.data() + b_idx * (Ca * spatial), Ca * spatial, out.data() + b_idx * ((Ca + Cb) * spatial));
                std::copy_n(b.data() + b_idx * (Cb * spatial), Cb * spatial, out.data() + b_idx * ((Ca + Cb) * spatial) + (Ca * spatial));
            }
            return out;
        }
    };
}