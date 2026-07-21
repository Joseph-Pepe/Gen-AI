export module crescendo.gpu.buffer;

import std;
import crescendo.gpu.types;

export namespace crescendo::gpu {

    // RAII that manages bidirectional memory staging between host RAM and device VRAM, abstracting away backend-specific pointers (e.g., CUdeviceptr, VkBuffer, sycl::buffer, WGPUBuffer).
    template <GPUStorable T>
    class GPUBuffer {
    public:
        GPUBuffer(size_t element_count, ComputeBackend backend, MemoryProperty flags)
            : size_(element_count), backend_(backend), flags_(flags) {
            
            byte_size_ = size_ * sizeof(T);
            allocate_storage();
        }

        ~GPUBuffer() {
            release_storage();
        }

        // Disable copy, allow move
        GPUBuffer(const GPUBuffer&) = delete;
        GPUBuffer& operator=(const GPUBuffer&) = delete;
        
        GPUBuffer(GPUBuffer&& other) noexcept 
            : size_(other.size_), byte_size_(other.byte_size_), backend_(other.backend_),
              flags_(other.flags_), host_ptr_(other.host_ptr_), device_handle_(other.device_handle_) {
            other.host_ptr_ = nullptr;
            other.device_handle_ = 0;
            other.size_ = 0;
        }

        GPUBuffer& operator=(GPUBuffer&& other) noexcept {
            if (this != &other) {
                release_storage();
                size_ = other.size_;
                byte_size_ = other.byte_size_;
                backend_ = other.backend_;
                flags_ = other.flags_;
                host_ptr_ = other.host_ptr_;
                device_handle_ = other.device_handle_;
                other.host_ptr_ = nullptr;
                other.device_handle_ = 0;
                other.size_ = 0;
            }
            return *this;
        }

        /**
         * @brief Asynchronously transfers host data into device high-bandwidth VRAM.
         */
        void upload(std::span<const T> host_data) {
            if (host_data.size() > size_) {
                throw std::out_of_range("Upload span exceeds GPU buffer capacity.");
            }
            if (backend_ == ComputeBackend::CPU_SIMD_Fallback) {
                std::memcpy(host_ptr_, host_data.data(), host_data.size_bytes());
                return;
            }
            // In a complete driver binding, invoke vkCmdCopyBuffer, cuMemcpyHtoDAsync, or sycl::queue::memcpy
            std::memcpy(host_ptr_, host_data.data(), host_data.size_bytes());
        }

        /**
         * @brief Transfers VRAM results back into a host RAM span.
         */
        void download(std::span<T> host_output) const {
            if (host_output.size() > size_) {
                throw std::out_of_range("Download span exceeds buffer capacity.");
            }
            if (backend_ == ComputeBackend::CPU_SIMD_Fallback) {
                std::memcpy(host_output.data(), host_ptr_, host_output.size_bytes());
                return;
            }
            // In a complete driver binding, invoke cuMemcpyDtoHAsync or vkMapMemory
            std::memcpy(host_output.data(), host_ptr_, host_output.size_bytes());
        }

        [[nodiscard]] size_t size() const noexcept { return size_; }
        [[nodiscard]] size_t byte_size() const noexcept { return byte_size_; }
        [[nodiscard]] T* host_data() noexcept { return static_cast<T*>(host_ptr_); }
        [[nodiscard]] const T* host_data() const noexcept { return static_cast<const T*>(host_ptr_); }
        [[nodiscard]] std::uintptr_t device_handle() const noexcept { return device_handle_; }
        [[nodiscard]] ComputeBackend backend() const noexcept { return backend_; }

    private:
        size_t size_ = 0;
        size_t byte_size_ = 0;
        ComputeBackend backend_;
        MemoryProperty flags_;
        void* host_ptr_ = nullptr;
        std::uintptr_t device_handle_ = 0; // Opaque handle for VkBuffer, CUdeviceptr, etc.

        void allocate_storage() {
            // Allocate 64-byte aligned host memory for SIMD compatibility and staging
            host_ptr_ = ::operator new[](byte_size_, std::align_val_t{64});
            std::memset(host_ptr_, 0, byte_size_);
            
            // Assign simulated opaque handle for backend device allocation
            device_handle_ = reinterpret_cast<std::uintptr_t>(host_ptr_);
        }

        void release_storage() {
            if (host_ptr_) {
                ::operator delete[](host_ptr_, std::align_val_t{64});
                host_ptr_ = nullptr;
            }
            device_handle_ = 0;
        }
    };
}