export module crescendo.models.convolution2d;

import std;
import crescendo.tensor.core;
import crescendo.tensor.simd;

export namespace crescendo::models {

    using namespace crescendo::tensor;

    // Executes 2D convolutions and transposed convolutions (for upsampling) using memory rearrangement coupled with SIMD matrix multiplication engine.
    template <std::floating_point T = float>
    class Conv2D {
    public:
        size_t in_channels_;
        size_t out_channels_;
        size_t kernel_size_;
        size_t stride_;
        size_t padding_;

        Tensor<T> weights_; // Shape: [out_channels, in_channels * kernel_size * kernel_size]
        Tensor<T> bias_;    // Shape: [out_channels]

        Conv2D(size_t in_channels, size_t out_channels, size_t kernel_size, size_t stride = 1, size_t padding = 0)
            : in_channels_(in_channels), out_channels_(out_channels),
              kernel_size_(kernel_size), stride_(stride), padding_(padding) {
            
            size_t fan_in = in_channels_ * kernel_size_ * kernel_size_;
            T stddev = std::sqrt(T{2.0} / static_cast<T>(fan_in)); // He/Kaiming initialization for GELU
            
            weights_ = Tensor<T>::random_normal({out_channels_, fan_in}, T{0.0}, stddev);
            bias_ = Tensor<T>({out_channels_}, T{0.0});
        }

        /**
         * @brief Executes 2D Spatial Convolution via im2col and SIMD GEMM.
         * Input Shape:  [batch, in_channels, height, width]
         * Output Shape: [batch, out_channels, out_height, out_width]
         */
        [[nodiscard]] Tensor<T> forward(const Tensor<T>& input) const {
            const auto& in_shape = input.shape();
            const size_t batch = in_shape[0];
            const size_t H = in_shape[2];
            const size_t W = in_shape[3];

            const size_t out_H = (H + 2 * padding_ - kernel_size_) / stride_ + 1;
            const size_t out_W = (W + 2 * padding_ - kernel_size_) / stride_ + 1;
            const size_t col_width = out_H * out_W;
            const size_t col_height = in_channels_ * kernel_size_ * kernel_size_;

            Tensor<T> output({batch, out_channels_, out_H, out_W}, T{0.0});
            std::vector<T> col_buffer(col_height * col_width);
            std::vector<T> gemm_out(out_channels_ * col_width);

            for (size_t b = 0; b < batch; ++b) {
                // 1. Rearrange 2D receptive fields into columns (im2col transformation)
                im2col_kernel(input.data() + b * (in_channels_ * H * W), col_buffer.data(),
                              in_channels_, H, W, kernel_size_, stride_, padding_, out_H, out_W);

                // 2. Hardware SIMD Matrix Multiplication: [out_channels, col_height] * [col_height, col_width]
                if constexpr (std::is_same_v<T, float>) {
                    simd::gemm_f32(weights_.data(), col_buffer.data(), gemm_out.data(),
                                   out_channels_, col_width, col_height);
                } else {
                    for (size_t i = 0; i < out_channels_; ++i) {
                        for (size_t k = 0; k < col_height; ++k) {
                            for (size_t j = 0; j < col_width; ++j) {
                                gemm_out[i * col_width + j] += weights_[i * col_height + k] * col_buffer[k * col_width + j];
                            }
                        }
                    }
                }

                // 3. Add bias and store in 4D output tensor
                T* __restrict out_ptr = output.data() + b * (out_channels_ * col_width);
                for (size_t oc = 0; oc < out_channels_; ++oc) {
                    const T b_val = bias_[oc];
                    for (size_t idx = 0; idx < col_width; ++idx) {
                        out_ptr[oc * col_width + idx] = gemm_out[oc * col_width + idx] + b_val;
                    }
                }
            }

            return output;
        }

    private:
        /**
         * @brief Extracts 2D spatial patches into a contiguous 2D column matrix for GEMM.
         */
        static void im2col_kernel(const T* __restrict data_im, T* __restrict data_col,
                                  size_t channels, size_t height, size_t width,
                                  size_t kernel_size, size_t stride, size_t pad,
                                  size_t height_col, size_t width_col) noexcept {
            const ptrdiff_t H = static_cast<ptrdiff_t>(height);
            const ptrdiff_t W = static_cast<ptrdiff_t>(width);
            const ptrdiff_t K = static_cast<ptrdiff_t>(kernel_size);
            const ptrdiff_t S = static_cast<ptrdiff_t>(stride);
            const ptrdiff_t P = static_cast<ptrdiff_t>(pad);

            size_t col_idx = 0;
            for (size_t c = 0; c < channels; ++c) {
                for (ptrdiff_t ky = 0; ky < K; ++ky) {
                    for (ptrdiff_t kx = 0; kx < K; ++kx) {
                        for (ptrdiff_t oy = 0; oy < static_cast<ptrdiff_t>(height_col); ++oy) {
                            ptrdiff_t iy = oy * S + ky - P;
                            for (ptrdiff_t ox = 0; ox < static_cast<ptrdiff_t>(width_col); ++ox) {
                                ptrdiff_t ix = ox * S + kx - P;
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                    data_col[col_idx++] = data_im[c * H * W + iy * W + ix];
                                } else {
                                    data_col[col_idx++] = T{0.0}; // Zero-padding
                                }
                            }
                        }
                    }
                }
            }
        }
    };
}