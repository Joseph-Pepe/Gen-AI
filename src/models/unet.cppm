export module crescendo.models.unet;

import std;
import crescendo.tensor.core;
import crescendo.models.convolution2d;
import crescendo.models.groupnorm;
import crescendo.models.attention;

export namespace crescendo::models {

    using namespace crescendo::tensor;

    // It combines downsampling convolution blocks, spatial GroupNorm, GELU activations, attention bottlenecks, and skip-connection memory concatenations.
    template <std::floating_point T = float>
    class UNet2D {
    public:
        // Encoder Block 1: [B, 1, H, W] -> [B, 32, H/2, W/2]
        Conv2D<T> enc_conv1_{1, 32, 3, 2, 1};
        GroupNorm<T> enc_gn1_{4, 32};

        // Encoder Block 2: [B, 32, H/2, W/2] -> [B, 64, H/4, W/4]
        Conv2D<T> enc_conv2_{32, 64, 3, 2, 1};
        GroupNorm<T> enc_gn2_{8, 64};

        // Bottleneck with Multi-Head Self-Attention: [B, 64, H/4, W/4]
        Conv2D<T> bot_conv_{64, 64, 3, 1, 1};
        GroupNorm<T> bot_gn_{8, 64};
        MultiHeadSelfAttention<T> bot_attn_{64, 4}; // 4 attention heads

        // Decoder Block 1 (Upsample + Skip Concat): [B, 64+32, H/2, W/2] -> [B, 32, H/2, W/2]
        Conv2D<T> dec_conv1_{96, 32, 3, 1, 1}; // 64 bottleneck + 32 skip
        GroupNorm<T> dec_gn1_{4, 32};

        // Output Projection: [B, 32, H, W] -> [B, 1, H, W]
        Conv2D<T> out_conv_{32, 1, 3, 1, 1};

        UNet2D() = default;

        /**
         * @brief Evaluates U-Net forward pass with encoder skip connections.
         * Input/Output Shape: [batch, 1, mel_bins, time_frames]
         */
        [[nodiscard]] Tensor<T> forward(const Tensor<T>& x) const {
            // --- Encoder Stage ---
            auto h1 = enc_conv1_.forward(x);
            h1 = enc_gn1_.forward(h1);
            h1.apply_gelu_inplace(); // Skip tensor 1 saved for decoder

            auto h2 = enc_conv2_.forward(h1);
            h2 = enc_gn2_.forward(h2);
            h2.apply_gelu_inplace();

            // --- Bottleneck Stage with Attention ---
            auto bot = bot_conv_.forward(h2);
            bot = bot_gn_.forward(bot);
            bot.apply_gelu_inplace();
            bot = bot_attn_.forward(bot); // Capture global musical rhythms

            // --- Decoder Stage (Bilinear Upsample + Skip Concatenation) ---
            auto bot_upsampled = upsample_2x(bot);
            auto dec1_in = concat_channels(bot_upsampled, h1); // Concatenate skip connection
            
            auto dec1 = dec_conv1_.forward(dec1_in);
            dec1 = dec_gn1_.forward(dec1);
            dec1.apply_gelu_inplace();

            auto dec1_upsampled = upsample_2x(dec1);
            auto out = out_conv_.forward(dec1_upsampled);

            return out;
        }

    private:
        /**
         * @brief Nearest-Neighbor 2x Spatial Upsampling kernel.
         * Doubles height and width dimensions: [B, C, H, W] -> [B, C, H*2, W*2]
         */
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
                            size_t out_offset = b * (C * H * 4 * W * 4) + c * (H * 2 * W * 2);
                            out_ptr[out_offset + (h * 2) * (W * 2) + (w * 2)]     = val;
                            out_ptr[out_offset + (h * 2) * (W * 2) + (w * 2 + 1)] = val;
                            out_ptr[out_offset + (h * 2 + 1) * (W * 2) + (w * 2)] = val;
                            out_ptr[out_offset + (h * 2 + 1) * (W * 2) + (w * 2 + 1)] = val;
                        }
                    }
                }
            }
            return out;
        }

        /**
         * @brief Concatenates two 4D tensors along the channel dimension (Dim 1).
         */
        static Tensor<T> concat_channels(const Tensor<T>& a, const Tensor<T>& b) {
            const auto& sa = a.shape();
            const auto& sb = b.shape();
            if (sa[0] != sb[0] || sa[2] != sb[2] || sa[3] != sb[3]) {
                throw std::invalid_argument("Spatial and batch dimensions must match for channel concatenation.");
            }
            const size_t B = sa[0], Ca = sa[1], Cb = sb[1], H = sa[2], W = sa[3];
            Tensor<T> out({B, Ca + Cb, H, W}, T{0.0});

            const size_t spatial = H * W;
            for (size_t b_idx = 0; b_idx < B; ++b_idx) {
                // Copy tensor A channels
                std::copy_n(a.data() + b_idx * (Ca * spatial), Ca * spatial,
                            out.data() + b_idx * ((Ca + Cb) * spatial));
                // Copy tensor B channels right after A
                std::copy_n(b.data() + b_idx * (Cb * spatial), Cb * spatial,
                            out.data() + b_idx * ((Ca + Cb) * spatial) + (Ca * spatial));
            }
            return out;
        }
    };
}