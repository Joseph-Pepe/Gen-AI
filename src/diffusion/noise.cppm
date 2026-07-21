export module crescendo.diffusion.noise;

import std;
import crescendo.tensor.core;
import crescendo.tensor.simd;

export namespace crescendo::diffusion {

    using namespace crescendo::tensor;

    // This implements a Box Muller transform to generate strictly standard normal distributions N(0,1) directly in contiguous memory, avoiding the OS dependent runtime call stack overhead of standard library distribution wrappers.
    template <std::floating_point T = float>
    class NormalGenerator {
    public:
        explicit NormalGenerator(std::uint64_t seed = std::random_device{}())
            : rng_(seed) {}

        /**
         * @brief Generates a contiguous tensor of independent Gaussian random numbers N(0, I).
         * Uses the Box-Muller transformation: Z0 = sqrt(-2*ln(U1)) * cos(2*pi*U2)
         */
        [[nodiscard]] Tensor<T> generate(const std::vector<size_t>& shape) {
            Tensor<T> noise(shape);
            T* __restrict data_ptr = noise.data();
            const size_t total_elements = noise.size();

            // Generate pairs of standard uniform random numbers (0, 1]
            std::uniform_real_distribution<T> dist(std::numeric_limits<T>::min(), T{1.0});

            size_t i = 0;
            for (; i + 1 < total_elements; i += 2) {
                const T u1 = dist(rng_);
                const T u2 = dist(rng_);

                const T radius = std::sqrt(-T{2.0} * std::log(u1));
                const T theta  = T{2.0} * std::numbers::pi_v<T> * u2;

                data_ptr[i]     = radius * std::cos(theta);
                data_ptr[i + 1] = radius * std::sin(theta);
            }

            // Handle odd trailing element if present
            if (i < total_elements) {
                const T u1 = dist(rng_);
                const T u2 = dist(rng_);
                data_ptr[i] = std::sqrt(-T{2.0} * std::log(u1)) * std::cos(T{2.0} * std::numbers::pi_v<T> * u2);
            }

            return noise;
        }

    private:
        std::mt19937_64 rng_;
    };
}