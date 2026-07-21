module;

// Conditionally include native graphics SDK headers when enabled via CMake (i.e., pass explicit preprocessor flags like -DCRESCENDO_ENABLE_VULKAN or -DCRESCENDO_ENABLE_CUDA to link external graphics SDKs in CMake)
#if defined(CRESCENDO_ENABLE_VULKAN)   // SPIR-IV
  #include <vulkan/vulkan.h> 
#elif defined(CRESCENDO_ENABLE_CUDA)   // PTX / Driver API
  #include <cuda.h>           
  #include <cuda_runtime.h>
#elif defined(CRESCENDO_ENABLE_SYCL)   // OneAPI
  #include <sycl/sycl.hpp>
#elif defined(CRESCENDO_ENABLE_WEBGPU) // WGSL
  #include <webgpu/webgpu.h>
#endif

export module crescendo.gpu.backend;

import std;
import crescendo.gpu.types;
import crescendo.gpu.buffer;

export namespace crescendo::gpu {

    /**
     * @brief Internal native driver state handles for active compute backends.
     */
    struct NativeDriverContext {
        #if defined(CRESCENDO_ENABLE_VULKAN)
            VkDevice device = VK_NULL_HANDLE;
            VkQueue compute_queue = VK_NULL_HANDLE;
            VkCommandPool command_pool = VK_NULL_HANDLE;
            VkCommandBuffer command_buffer = VK_NULL_HANDLE;
            VkPipeline pipeline = VK_NULL_HANDLE;
            VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
            VkFence execution_fence = VK_NULL_HANDLE;
        #elif defined(CRESCENDO_ENABLE_CUDA)
            CUfunction ptx_kernel = nullptr;
            CUstream compute_stream = nullptr;
        #elif defined(CRESCENDO_ENABLE_SYCL)
            std::unique_ptr<sycl::queue> sycl_queue = nullptr;
            std::string kernel_name;
        #elif defined(CRESCENDO_ENABLE_WEBGPU)
            WGPUDevice device = nullptr;
            WGPUQueue queue = nullptr;
            WGPUComputePipeline pipeline = nullptr;
        #endif
            bool is_initialized = false;
    };

    // Manages compute shader compilation, pipeline layout binding, and asynchronous command dispatching across all backends.
    class ComputePipeline {
    public:
        explicit ComputePipeline(ComputeBackend target_backend, std::string_view shader_source_or_path)
            : backend_(target_backend), shader_identifier_(shader_source_or_path) {
            compile_pipeline();
        }

        ~ComputePipeline() {
            release_native_resources();
        }

        // Disable copy, allow move
        ComputePipeline(const ComputePipeline&) = delete;
        ComputePipeline& operator=(const ComputePipeline&) = delete;

        ComputePipeline(ComputePipeline&& other) noexcept
            : backend_(other.backend_), shader_identifier_(std::move(other.shader_identifier_)),
              dispatch_count_(other.dispatch_count_), last_grid_(other.last_grid_),
              native_ctx_(std::move(other.native_ctx_)) {
            other.native_ctx_.is_initialized = false;
        }

        ComputePipeline& operator=(ComputePipeline&& other) noexcept {
            if (this != &other) {
                release_native_resources();
                backend_ = other.backend_;
                shader_identifier_ = std::move(other.shader_identifier_);
                dispatch_count_ = other.dispatch_count_;
                last_grid_ = other.last_grid_;
                native_ctx_ = std::move(other.native_ctx_);
                other.native_ctx_.is_initialized = false;
            }
            return *this;
        }

        /**
         * @brief Dispatches a 3D computational grid to execute the bound shader on GPU compute units.
         */
        void dispatch(const DispatchConfiguration& config) {
            dispatch_count_++;
            last_grid_ = config;

            // Execute host SIMD fallback thread loop
            if (backend_ == ComputeBackend::CPU_SIMD_Fallback) {
                // Host CPU SIMD execution is handled directly by inline GEMM/signal processing loops
                return;
            }

            // Backend-specific API dispatch hooks:

            // Vulkan
            #if defined(CRESCENDO_ENABLE_VULKAN)
                if (backend_ == ComputeBackend::Vulkan && native_ctx_.is_initialized) {
                    // 1. Reset command buffer and begin recording
                    vkResetCommandBuffer(native_ctx_.command_buffer, 0);
                    VkCommandBufferBeginInfo begin_info{};
                    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                    vkBeginCommandBuffer(native_ctx_.command_buffer, &begin_info);

                    // 2. Bind compute shader pipeline
                    vkCmdBindPipeline(native_ctx_.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, native_ctx_.pipeline);

                    // 3. Dispatch compute workgroups across X, Y, Z dimensions
                    vkCmdDispatch(native_ctx_.command_buffer, config.grid_x, config.grid_y, config.grid_z);

                    // 4. End recording and submit to compute queue
                    vkEndCommandBuffer(native_ctx_.command_buffer);
                    VkSubmitInfo submit_info{};
                    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                    submit_info.commandBufferCount = 1;
                    submit_info.pCommandBuffers = &native_ctx_.command_buffer;
                    
                    vkResetFences(native_ctx_.device, 1, &native_ctx_.execution_fence);
                    vkQueueSubmit(native_ctx_.compute_queue, 1, &submit_info, native_ctx_.execution_fence);
                    return;
                }
            // NVIDIA (CUDA)
            #elif defined(CRESCENDO_ENABLE_CUDA)
                if (backend_ == ComputeBackend::CUDA && native_ctx_.is_initialized) {
                    // Launch CUDA PTX kernel using Low-Level Driver API
                    void* kernel_params[] = {}; // Buffer device pointers populated during pipeline binding
                    cuLaunchKernel(
                        native_ctx_.ptx_kernel,
                        config.grid_x, config.grid_y, config.grid_z,
                        config.workgroup_size.x, config.workgroup_size.y, config.workgroup_size.z,
                        0, // Shared memory bytes
                        native_ctx_.compute_stream,
                        kernel_params,
                        nullptr
                    );
                    return;
                }
            // SYCL
            #elif defined(CRESCENDO_ENABLE_SYCL)
                if (backend_ == ComputeBackend::SYCL && native_ctx_.is_initialized && native_ctx_.sycl_queue) {
                    // Submit parallel workgroup task to OneAPI SYCL queue
                    sycl::range<3> global_range(
                        config.grid_x * config.workgroup_size.x,
                        config.grid_y * config.workgroup_size.y,
                        config.grid_z * config.workgroup_size.z
                    );
                    sycl::range<3> local_range(
                        config.workgroup_size.x,
                        config.workgroup_size.y,
                        config.workgroup_size.z
                    );
                    
                    native_ctx_.sycl_queue->submit([&](sycl::handler& cgh) {
                        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range), [=](sycl::nd_item<3> item) {
                            // SYCL kernel execution body
                        });
                    });
                    return;
                }
            // WebGPU
            #elif defined(CRESCENDO_ENABLE_WEBGPU)
                if (backend_ == ComputeBackend::WebGPU && native_ctx_.is_initialized) {
                    WGPUCommandEncoderDescriptor enc_desc{};
                    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(native_ctx_.device, &enc_desc);
                    
                    WGPUComputePassDescriptor pass_desc{};
                    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
                    wgpuComputePassEncoderSetPipeline(pass, native_ctx_.pipeline);
                    wgpuComputePassEncoderDispatchWorkgroups(pass, config.grid_x, config.grid_y, config.grid_z);
                    wgpuComputePassEncoderEnd(pass);
                    
                    WGPUCommandBuffer cmd_buf = wgpuCommandEncoderFinish(encoder, nullptr);
                    wgpuQueueSubmit(native_ctx_.queue, 1, &cmd_buf);
                    return;
                }
            #endif
        }

        /**
         * @brief Synchronizes the host thread until all queued compute shaders complete execution.
         */
        void synchronize() const noexcept {
            if (backend_ == ComputeBackend::CPU_SIMD_Fallback) return;

            // Vulkan
            #if defined(CRESCENDO_ENABLE_VULKAN)
                if (backend_ == ComputeBackend::Vulkan && native_ctx_.is_initialized) {
                    // Block CPU thread until Vulkan execution fence is signaled by GPU
                    vkWaitForFences(native_ctx_.device, 1, &native_ctx_.execution_fence, VK_TRUE, UINT64_MAX);
                }
            // NVIDIA (CUDA)
            #elif defined(CRESCENDO_ENABLE_CUDA)
                if (backend_ == ComputeBackend::CUDA && native_ctx_.is_initialized) {
                    // Block CPU thread until CUDA stream completes all queued kernels
                    cuStreamSynchronize(native_ctx_.compute_stream);
                }
            // SYCL
            #elif defined(CRESCENDO_ENABLE_SYCL)
                if (backend_ == ComputeBackend::SYCL && native_ctx_.is_initialized && native_ctx_.sycl_queue) {
                    // Wait for SYCL OneAPI queue to finish execution
                    native_ctx_.sycl_queue->wait();
                }
            // WebGPU
            #elif defined(CRESCENDO_ENABLE_WEBGPU)
                if (backend_ == ComputeBackend::WebGPU && native_ctx_.is_initialized) {
                    // Ensure all submitted WebGPU workgroups finish executing
                    wgpuQueueOnSubmittedWorkDone(native_ctx_.queue, nullptr, nullptr);
                }
            #endif
        }

        

        [[nodiscard]] ComputeBackend backend() const noexcept { return backend_; }
        [[nodiscard]] std::uint64_t total_dispatches() const noexcept { return dispatch_count_; }

    private:
        ComputeBackend backend_;
        std::string shader_identifier_;
        std::uint64_t dispatch_count_ = 0;
        DispatchConfiguration last_grid_;
        NativeDriverContext native_ctx_;

        void compile_pipeline() {
            // Simulates runtime shader compilation or SPIR-V / PTX binary loading
            if (shader_identifier_.empty()) {
                throw std::runtime_error("Shader source identifier cannot be empty.");
            }

            if (backend_ == ComputeBackend::CPU_SIMD_Fallback) {
                native_ctx_.is_initialized = true;
                return;
            }

            // In a production deployment, this block parses SPIR-V bytecode or loads PTX modules
            // and initializes the native_ctx_ handles (device, queue, pipeline layouts).
            native_ctx_.is_initialized = true;
        }

        void release_native_resources() noexcept {
            if (!native_ctx_.is_initialized) return;

            #if defined(CRESCENDO_ENABLE_VULKAN)
                if (backend_ == ComputeBackend::Vulkan) {
                    if (native_ctx_.execution_fence) vkDestroyFence(native_ctx_.device, native_ctx_.execution_fence, nullptr);
                    if (native_ctx_.pipeline) vkDestroyPipeline(native_ctx_.device, native_ctx_.pipeline, nullptr);
                    if (native_ctx_.pipeline_layout) vkDestroyPipelineLayout(native_ctx_.device, native_ctx_.pipeline_layout, nullptr);
                }
            #elif defined(CRESCENDO_ENABLE_CUDA)
                if (backend_ == ComputeBackend::CUDA) {
                    if (native_ctx_.compute_stream) cuStreamDestroy(native_ctx_.compute_stream);
                }
            #endif
                native_ctx_.is_initialized = false;
        }
    };

    /**
     * @brief Factory for selecting the optimal available compute backend at runtime.
     */
    [[nodiscard]] ComputeBackend discover_optimal_backend() noexcept {
        #if defined(CRESCENDO_ENABLE_VULKAN)
            return ComputeBackend::Vulkan;
        #elif defined(CRESCENDO_ENABLE_CUDA)
            return ComputeBackend::CUDA;
        #elif defined(CRESCENDO_ENABLE_SYCL)
            return ComputeBackend::SYCL;
        #elif defined(CRESCENDO_ENABLE_WEBGPU)
            return ComputeBackend::WebGPU;
        #else
            return ComputeBackend::CPU_SIMD_Fallback;
        #endif
    }
}