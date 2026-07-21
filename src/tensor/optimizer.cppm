export module crescendo.tensor.optimizer;

import std;
import crescendo.tensor.core;
import crescendo.tensor.autograd;

export namespace crescendo::tensor::optimizer {

    // Manages model parameter updates using exponential moving averages of first (m1) and secod (vt) gradient moments combined with L2 weight decay.
    template <std::floating_point T = float>
    class AdamW {
    public:
        using VarPtr = std::shared_ptr<autograd::Variable<T>>;

        AdamW(std::vector<VarPtr> parameters, T lr = T{0.001}, T beta1 = T{0.9}, T beta2 = T{0.999},
              T eps = T{1e-8}, T weight_decay = T{0.01})
            : params_(std::move(parameters)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps), wd_(weight_decay) {
            
            for (const auto& p : params_) {
                m_.push_back(Tensor<T>(p->data.shape(), T{0.0}));
                v_.push_back(Tensor<T>(p->data.shape(), T{0.0}));
            }
        }

        void zero_grad() noexcept {
            for (auto& p : params_) {
                std::fill_n(p->grad.data(), p->grad.size(), T{0.0});
            }
        }

        /**
         * @brief Executes one AdamW optimization step across all registered tensor variables.
         */
        void step() noexcept {
            step_count_++;
            const T bias_correction1 = T{1.0} - std::pow(beta1_, static_cast<T>(step_count_));
            const T bias_correction2 = T{1.0} - std::pow(beta2_, static_cast<T>(step_count_));

            for (size_t idx = 0; idx < params_.size(); ++idx) {
                auto& p = params_[idx];
                if (!p->requires_grad) continue;

                const size_t n = p->data.size();
                T* __restrict w = p->data.data();
                const T* __restrict g = p->grad.data();
                T* __restrict m_ptr = m_[idx].data();
                T* __restrict v_ptr = v_[idx].data();

                for (size_t i = 0; i < n; ++i) {
                    // Decoupled L2 Weight Decay: w = w * (1 - lr * weight_decay)
                    w[i] -= lr_ * wd_ * w[i];

                    // Update biased first moment estimate: m = beta1 * m + (1 - beta1) * g
                    m_ptr[i] = beta1_ * m_ptr[i] + (T{1.0} - beta1_) * g[i];

                    // Update biased second raw moment estimate: v = beta2 * v + (1 - beta2) * g^2
                    v_ptr[i] = beta2_ * v_ptr[i] + (T{1.0} - beta2_) * g[i] * g[i];

                    // Compute bias-corrected estimates
                    T m_hat = m_ptr[i] / bias_correction1;
                    T v_hat = v_ptr[i] / bias_correction2;

                    // Apply parameter update: w = w - lr * m_hat / (sqrt(v_hat) + eps)
                    w[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
                }
            }
        }

    private:
        std::vector<VarPtr> params_;
        std::vector<Tensor<T>> m_;
        std::vector<Tensor<T>> v_;
        T lr_, beta1_, beta2_, eps_, wd_;
        size_t step_count_ = 0;
    };
}