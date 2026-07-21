export module crescendo.tensor.core;

import std;
import crescendo.tensor.simd;

export namespace crescendo::tensor {

    // Manages contiguous memory storage, shape dimensions, striding vectors, matrix broadcasting.
    template <std::floating_point T = float>
    class Tensor {
    public:
        Tensor() = default;

        explicit Tensor(std::vector<size_t> shape, T initial_val = T{0.0})
            : shape_(std::move(shape)) {
            compute_strides();
            data_.assign(total_elements_, initial_val);
        }

        static Tensor random_normal(const std::vector<size_t>& shape, T mean = T{0.0}, T stddev = T{0.02}) {
            Tensor t(shape);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dist(mean, stddev);
            for (auto& val : t.data_) {
                val = dist(gen);
            }
            return t;
        }

        /**
         * @brief In-place zero initialization of all tensor elements.
         */
        void zero_() noexcept {
            std::fill(data_.begin(), data_.end(), T{0.0});
        }

        // Element access via N-dimensional coordinate index
        [[nodiscard]] T& at(const std::vector<size_t>& indices) {
            return data_[offset(indices)];
        }

        [[nodiscard]] const T& at(const std::vector<size_t>& indices) const {
            return data_[offset(indices)];
        }

        // Flat 1D index access: tensor[idx]
        [[nodiscard]] T& operator[](size_t idx) { return data_[idx]; }
        [[nodiscard]] const T& operator[](size_t idx) const { return data_[idx]; }

        /**
         * @brief Multi-dimensional coordinate indexing: tensor[{i, j, k}]
         * Resolves MSVC error C2679 by accepting braced initializer lists.
         */
        [[nodiscard]] T& operator[](std::initializer_list<size_t> indices) {
            return data_[offset(std::span<const size_t>(indices.begin(), indices.size()))];
        }

        [[nodiscard]] const T& operator[](std::initializer_list<size_t> indices) const {
            return data_[offset(std::span<const size_t>(indices.begin(), indices.size()))];
        }

        // Matrix Multiplication (2D tensors only) accelerated by hardware SIMD
        [[nodiscard]] Tensor matmul(const Tensor& other) const {
            if (shape_.size() != 2 || other.shape_.size() != 2 || shape_[1] != other.shape_[0]) {
                throw std::invalid_argument("Invalid shapes for 2D Tensor Matmul.");
            }
            const size_t M = shape_[0];
            const size_t K = shape_[1];
            const size_t N = other.shape_[1];

            Tensor result({M, N});
            if constexpr (std::is_same_v<T, float>) {
                simd::gemm_f32(data_.data(), other.data_.data(), result.data_.data(), M, N, K);
            } else {
                for (size_t i = 0; i < M; ++i) {
                    for (size_t k = 0; k < K; ++k) {
                        for (size_t j = 0; j < N; ++j) {
                            result.data_[i * N + j] += data_[i * K + k] * other.data_[k * N + j];
                        }
                    }
                }
            }
            return result;
        }

        // Element-wise addition
        [[nodiscard]] Tensor operator+(const Tensor& other) const {
            if (shape_ != other.shape_) {
                throw std::invalid_argument("Tensor shapes must match for addition.");
            }
            Tensor result(shape_);
            if constexpr (std::is_same_v<T, float>) {
                simd::add_f32(data_.data(), other.data_.data(), result.data_.data(), total_elements_);
            } else {
                for (size_t i = 0; i < total_elements_; ++i) {
                    result.data_[i] = data_[i] + other.data_[i];
                }
            }
            return result;
        }

        // GELU activation
        void apply_gelu_inplace() noexcept {
            if constexpr (std::is_same_v<T, float>) {
                simd::gelu_f32(data_.data(), data_.data(), total_elements_);
            }
        }

        [[nodiscard]] const std::vector<size_t>& shape() const noexcept { return shape_; }
        [[nodiscard]] const std::vector<size_t>& strides() const noexcept { return strides_; }
        [[nodiscard]] size_t size() const noexcept { return total_elements_; }
        [[nodiscard]] T* data() noexcept { return data_.data(); }
        [[nodiscard]] const T* data() const noexcept { return data_.data(); }

    private:
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        std::vector<T> data_;
        size_t total_elements_ = 0;

        void compute_strides() {
            strides_.resize(shape_.size());
            total_elements_ = 1;
            for (ptrdiff_t i = static_cast<ptrdiff_t>(shape_.size()) - 1; i >= 0; --i) {
                strides_[i] = total_elements_;
                total_elements_ *= shape_[i];
            }
        }

        // Optimized span-based offset calculation (prevents dynamic vector allocations during indexing)
        [[nodiscard]] size_t offset(std::span<const size_t> indices) const {
            if (indices.size() != shape_.size()) throw std::out_of_range("Dimension mismatch.");
            size_t idx = 0;
            for (size_t i = 0; i < indices.size(); ++i) {
                if (indices[i] >= shape_[i]) throw std::out_of_range("Index out of bounds.");
                idx += indices[i] * strides_[i];
            }
            return idx;
        }

        [[nodiscard]] size_t offset(const std::vector<size_t>& indices) const {
            return offset(std::span<const size_t>(indices.begin(), indices.size()));
        }
    };
}