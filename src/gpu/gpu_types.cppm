export module crescendo.gpu.types;

import std;

// Defies common execution backends, memory lags, and C++26 concepts to ensure compile-time safety across disparate graphics APIs.
export namespace crescendo::gpu {

    enum class ComputeBackend : std::uint8_t {
        Vulkan = 0,
        CUDA = 1,
        SYCL = 2,
        WebGPU = 3,
        Metal = 4,
        CPU_SIMD_Fallback = 5
    };

    enum class MemoryProperty : std::uint8_t {
        None        = 0,
        HostVisible = 1 << 0,  // CPU read/write staging memory
        DeviceLocal = 1 << 1,  // High-bandwidth VRAM (GPU only)
        Coherent    = 1 << 2   // Automatically mapped without explicit flush
    };

    // --- Bitwise Operator Overloads for MemoryProperty ---

    [[nodiscard]] constexpr MemoryProperty operator|(MemoryProperty lhs, MemoryProperty rhs) noexcept {
        using T = std::underlying_type_t<MemoryProperty>;
        return static_cast<MemoryProperty>(static_cast<T>(lhs) | static_cast<T>(rhs));
    }

    [[nodiscard]] constexpr MemoryProperty operator&(MemoryProperty lhs, MemoryProperty rhs) noexcept {
        using T = std::underlying_type_t<MemoryProperty>;
        return static_cast<MemoryProperty>(static_cast<T>(lhs) & static_cast<T>(rhs));
    }

    [[nodiscard]] constexpr MemoryProperty operator^(MemoryProperty lhs, MemoryProperty rhs) noexcept {
        using T = std::underlying_type_t<MemoryProperty>;
        return static_cast<MemoryProperty>(static_cast<T>(lhs) ^ static_cast<T>(rhs));
    }

    [[nodiscard]] constexpr MemoryProperty operator~(MemoryProperty val) noexcept {
        using T = std::underlying_type_t<MemoryProperty>;
        return static_cast<MemoryProperty>(~static_cast<T>(val));
    }

    constexpr MemoryProperty& operator|=(MemoryProperty& lhs, MemoryProperty rhs) noexcept {
        lhs = lhs | rhs;
        return lhs;
    }

    constexpr MemoryProperty& operator&=(MemoryProperty& lhs, MemoryProperty rhs) noexcept {
        lhs = lhs & rhs;
        return lhs;
    }


    // Used to check if a flag is active: bool is_host = has_flag(flags, MemoryProperty::HostVisible);
    [[nodiscard]] constexpr bool has_flag(MemoryProperty value, MemoryProperty flag) noexcept {
        return (value & flag) != MemoryProperty::None;
    }

    // --- String & Configuration Utilities ---

    [[nodiscard]] constexpr std::string_view backend_to_string(ComputeBackend backend) noexcept {
        switch (backend) {
            case ComputeBackend::Vulkan:            return "Vulkan (SPIR-V Compute Shaders)";
            case ComputeBackend::CUDA:              return "NVIDIA CUDA (PTX / Native)";
            case ComputeBackend::SYCL:              return "Intel / Khronos SYCL (OneAPI)";
            case ComputeBackend::WebGPU:            return "WebGPU (WGSL Compute Pipelines)";
            case ComputeBackend::Metal:             return "Apple Metal (Metal-cpp / MSL Shaders)";
            case ComputeBackend::CPU_SIMD_Fallback: return "Host CPU SIMD Fallback (AVX2/NEON)";
            default:                                return "Unknown Backend";
        }
    }

    struct WorkgroupDimensions {
        std::uint32_t x = 16;
        std::uint32_t y = 16;
        std::uint32_t z = 1;
    };

    struct DispatchConfiguration {
        WorkgroupDimensions workgroup_size;
        std::uint32_t grid_x = 1;
        std::uint32_t grid_y = 1;
        std::uint32_t grid_z = 1;
    };

    // C++26 Concept constraint for valid GPU compute data types
    template <typename T>
    concept GPUStorable = std::is_trivially_copyable_v<T> && (sizeof(T) % 4 == 0 || sizeof(T) == 2 || sizeof(T) == 1);
}