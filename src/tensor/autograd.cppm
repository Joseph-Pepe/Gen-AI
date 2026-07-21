export module crescendo.tensor.autograd;

import std;
import crescendo.tensor.core;

export namespace crescendo::tensor::autograd {

    // Builds a directed acyclic graph (DAG) of tensor operations, records forward dependencies, and executes analytical back-pass differentiation using topological sorting. 
    template <std::floating_point T = float>
    class Variable : public std::enable_shared_from_this<Variable<T>> {
    public:
        using TensorType = Tensor<T>;
        using VarPtr = std::shared_ptr<Variable<T>>;

        TensorType data;
        TensorType grad;
        std::vector<VarPtr> parents;
        std::function<void()> backward_op;
        std::string name;
        bool requires_grad = true;

        explicit Variable(TensorType tensor, bool req_grad = true, std::string label = "")
            : data(std::move(tensor)),
              grad(TensorType(data.shape(), T{0.0})),
              name(std::move(label)),
              requires_grad(req_grad) {}

        static VarPtr create(TensorType tensor, bool req_grad = true, std::string label = "") {
            return std::make_shared<Variable<T>>(std::move(tensor), req_grad, std::move(label));
        }

        // Matrix Multiplication Node with analytical backward pass
        VarPtr matmul(const VarPtr& other) {
            auto out_data = this->data.matmul(other->data);
            auto out = Variable<T>::create(out_data, this->requires_grad || other->requires_grad, "matmul");
            out->parents = {this->shared_from_this(), other};

            out->backward_op = [this_var = this->shared_from_this(), other_var = other, out_var = out.get()]() {
                // dL/dA = dL/dOut * B^T
                if (this_var->requires_grad) {
                    const size_t M = out_var->grad.shape()[0];
                    const size_t N = out_var->grad.shape()[1];
                    const size_t K = other_var->data.shape()[0];
                    for (size_t i = 0; i < M; ++i) {
                        for (size_t k = 0; k < K; ++k) {
                            for (size_t j = 0; j < N; ++j) {
                                this_var->grad[i * K + k] += out_var->grad[i * N + j] * other_var->data[k * N + j];
                            }
                        }
                    }
                }
                // dL/dB = A^T * dL/dOut
                if (other_var->requires_grad) {
                    const size_t M = this_var->data.shape()[0];
                    const size_t K = this_var->data.shape()[1];
                    const size_t N = out_var->grad.shape()[1];
                    for (size_t k = 0; k < K; ++k) {
                        for (size_t j = 0; j < N; ++j) {
                            for (size_t i = 0; i < M; ++i) {
                                other_var->grad[k * N + j] += this_var->data[i * K + k] * out_var->grad[i * N + j];
                            }
                        }
                    }
                }
            };
            return out;
        }

        // Element-wise addition node
        VarPtr add(const VarPtr& other) {
            auto out_data = this->data + other->data;
            auto out = Variable<T>::create(out_data, this->requires_grad || other->requires_grad, "add");
            out->parents = {this->shared_from_this(), other};

            out->backward_op = [this_var = this->shared_from_this(), other_var = other, out_var = out.get()]() {
                if (this_var->requires_grad) {
                    for (size_t i = 0; i < this_var->grad.size(); ++i) {
                        this_var->grad[i] += out_var->grad[i];
                    }
                }
                if (other_var->requires_grad) {
                    for (size_t i = 0; i < other_var->grad.size(); ++i) {
                        other_var->grad[i] += out_var->grad[i];
                    }
                }
            };
            return out;
        }

        /**
         * @brief Executes backpropagation by topologically sorting graph nodes and running reverse derivative sweeps.
         */
        void backward() {
            std::vector<VarPtr> topo_order;
            std::unordered_set<Variable<T>*> visited;

            auto build_topo = [&](auto& self, const VarPtr& node) -> void {
                if (!visited.insert(node.get()).second) return;
                for (const auto& parent : node->parents) {
                    self(self, parent);
                }
                topo_order.push_back(node);
            };

            build_topo(build_topo, this->shared_from_this());

            // Initialize seed gradient dL/dL = 1.0
            std::fill_n(this->grad.data(), this->grad.size(), T{1.0});

            // Execute backward operations in reverse topological order
            for (auto it = topo_order.rbegin(); it != topo_order.rend(); ++it) {
                if ((*it)->backward_op) {
                    (*it)->backward_op();
                }
            }
        }
    };
}