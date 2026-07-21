// 1. GLOBAL MODULE FRAGMENT
module;

#if defined(__AVX512F__)
  #include <immintrin.h>
#elif defined(__AVX2__)
  #include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
  #include <arm_neon.h>
#endif

export module crescendo.audio.io.wav_writer;

import std;
import crescendo.tensor.simd;

export namespace crescendo::io {

    enum class WavBitDepth : std::uint16_t {
        PCM_16 = 16, // Standard 16-bit signed integer (audio_format = 1)
        FLOAT_32 = 32 // Studio 32-bit IEEE floating point (audio_format = 3)
    };

    // Force 1-byte alignment so struct fields map 1:1 to binary disk headers
    #pragma pack(push, 1)
    struct WavHeader {
        char     riff_header[4] = {'R', 'I', 'F', 'F'};
        std::uint32_t wav_size       = 0; 
        char     wave_header[4] = {'W', 'A', 'V', 'E'};
        char     fmt_header[4]  = {'f', 'm', 't', ' '};
        std::uint32_t fmt_chunk_size = 16;
        std::uint16_t audio_format   = 1; // 1 for PCM integer, 3 for IEEE float
        std::uint16_t num_channels   = 1;
        std::uint32_t sample_rate    = 44100;
        std::uint32_t byte_rate      = 44100 * 2;
        std::uint16_t sample_alignment = 2;
        std::uint16_t bits_per_sample  = 16;
        char     data_header[4] = {'d', 'a', 't', 'a'};
        std::uint32_t data_chunk_size = 0;
    };
    #pragma pack(pop) // Restore default compiler memory alignment

    // RIFF/WAVE serializer supporting both 16-bit integer PCM and 32-bit floating-point bitstreams, multi-channel memory interleaving, and SIMD peak normalization.
    class WavExporter {
    public:
        /**
         * @brief Interleaves multiple independent mono channel arrays into a single sequential buffer.
         * Example for 2 channels (Stereo): [L0, R0, L1, R1, L2, R2...]
         */
        template <std::floating_point T>
        static std::vector<T> interleave_channels(const std::vector<std::vector<T>>& channels) {
            if (channels.empty()) return {};
            const size_t num_channels = channels.size();
            const size_t num_samples = channels[0].size();

            for (size_t c = 1; c < num_channels; ++c) {
                if (channels[c].size() != num_samples) {
                    throw std::invalid_argument("All channel buffers must have identical sample lengths for interleaving.");
                }
            }

            std::vector<T> interleaved(num_channels * num_samples);
            T* __restrict out_ptr = interleaved.data();

            for (size_t i = 0; i < num_samples; ++i) {
                for (size_t c = 0; c < num_channels; ++c) {
                    out_ptr[i * num_channels + c] = channels[c][i];
                }
            }
            return interleaved;
        }

        /**
         * @brief Writes a multi-channel interleaved floating-point buffer to a WAV file.
         */
        template <std::floating_point T>
        static bool write_wav(const std::string& filepath, 
                              const std::vector<T>& interleaved_buffer, 
                              std::uint16_t num_channels = 1,
                              std::uint32_t sample_rate = 44100,
                              WavBitDepth bit_depth = WavBitDepth::PCM_16) {
            
            if (interleaved_buffer.empty() || num_channels == 0) return false;
            std::ofstream file(filepath, std::ios::binary);
            if (!file.is_open()) return false;

            const std::uint16_t bits = static_cast<std::uint16_t>(bit_depth);
            const std::uint16_t bytes_per_sample = bits / 8;
            const std::uint32_t data_bytes = static_cast<std::uint32_t>(interleaved_buffer.size() * bytes_per_sample);

            WavHeader header;
            header.audio_format     = (bit_depth == WavBitDepth::FLOAT_32) ? 3 : 1;
            header.num_channels     = num_channels;
            header.sample_rate      = sample_rate;
            header.bits_per_sample  = bits;
            header.sample_alignment = num_channels * bytes_per_sample;
            header.byte_rate        = sample_rate * header.sample_alignment;
            header.data_chunk_size  = data_bytes;
            header.wav_size         = data_bytes + sizeof(WavHeader) - 8;

            file.write(reinterpret_cast<const char*>(&header), sizeof(WavHeader));

            if (bit_depth == WavBitDepth::FLOAT_32) {
                // Directly write 32-bit floating point stream (no scaling required for format 3)
                if constexpr (std::is_same_v<T, float>) {
                    file.write(reinterpret_cast<const char*>(interleaved_buffer.data()), data_bytes);
                } else {
                    std::vector<float> float_buffer(interleaved_buffer.begin(), interleaved_buffer.end());
                    file.write(reinterpret_cast<const char*>(float_buffer.data()), data_bytes);
                }
            } else {
                // Quantize to 16-bit PCM integer with SIMD headroom normalization (-0.44 dBFS)
                const T norm_factor = compute_simd_normalization_factor(interleaved_buffer);
                std::vector<std::int16_t> pcm_buffer(interleaved_buffer.size());
                
                quantize_simd_pcm16(interleaved_buffer.data(), pcm_buffer.data(), interleaved_buffer.size(), norm_factor);
                file.write(reinterpret_cast<const char*>(pcm_buffer.data()), data_bytes);
            }

            return true;
        }

        /**
         * @brief Batch exports the 4 separated Wiener stems simultaneously to separate WAV files.
         */
        template <std::floating_point T>
        static bool write_stems_batch(const std::string& directory_prefix,
                                      const std::array<std::vector<T>, 4>& stems,
                                      std::uint32_t sample_rate = 44100,
                                      WavBitDepth bit_depth = WavBitDepth::PCM_16) {
            const std::array<std::string, 4> names = {"_vocals.wav", "_drums.wav", "_bass.wav", "_other.wav"};
            bool all_success = true;

            for (size_t k = 0; k < 4; ++k) {
                std::string full_path = directory_prefix + names[k];
                bool success = write_wav(full_path, stems[k], 1, sample_rate, bit_depth);
                if (!success) all_success = false;
            }
            return all_success;
        }

    private:
        template <std::floating_point T>
        static T compute_simd_normalization_factor(const std::vector<T>& buffer) noexcept {
            T max_peak = T{0.0};
            const T* __restrict ptr = buffer.data();
            const size_t size = buffer.size();
            size_t i = 0;

            #if defined(CRESCENDO_SIMD_AVX512)
                if constexpr (std::is_same_v<T, float>) {
                    __m512 v_max = _mm512_setzero_ps();
                    for (; i + 16 <= size; i += 16) {
                        __m512 val = _mm512_abs_ps(_mm512_loadu_ps(&ptr[i]));
                        v_max = _mm512_max_ps(v_max, val);
                    }
                    float tmp[16];
                    _mm512_storeu_ps(tmp, v_max);
                    for (float v : tmp) if (v > max_peak) max_peak = v;
                }
            #elif defined(CRESCENDO_SIMD_AVX2)
                if constexpr (std::is_same_v<T, float>) {
                    __m256 v_max = _mm256_setzero_ps();
                    const __m256 sign_mask = _mm256_set1_ps(-0.0f); // Mask to strip sign bit
                    for (; i + 8 <= size; i += 8) {
                        __m256 val = _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(&ptr[i]));
                        v_max = _mm256_max_ps(v_max, val);
                    }
                    float tmp[8];
                    _mm256_storeu_ps(tmp, v_max);
                    for (float v : tmp) if (v > max_peak) max_peak = v;
                }
            #elif defined(CRESCENDO_SIMD_NEON)
                if constexpr (std::is_same_v<T, float>) {
                    float32x4_t v_max = vdupq_n_f32(0.0f);
                    for (; i + 4 <= size; i += 4) {
                        float32x4_t val = vabsq_f32(vld1q_f32(&ptr[i]));
                        v_max = vmaxq_f32(v_max, val);
                    }
                    float tmp[4];
                    vst1q_f32(tmp, v_max);
                    for (float v : tmp) if (v > max_peak) max_peak = v;
                }
            #endif
            for (; i < size; ++i) {
                T abs_val = std::abs(ptr[i]);
                if (abs_val > max_peak) max_peak = abs_val;
            }

            // Scale to -0.44 dBFS (0.95 linear headroom) if peak exceeds threshold
            return (max_peak > T{1e-6}) ? (T{0.95} / max_peak) : T{1.0};
        }

        template <std::floating_point T>
        static void quantize_simd_pcm16(const T* __restrict in_ptr, std::int16_t* __restrict out_ptr, 
                                        size_t size, T norm_factor) noexcept {
            const T scale = norm_factor * T{32767.0};
            size_t i = 0;

            #if defined(CRESCENDO_SIMD_AVX2)
                if constexpr (std::is_same_v<T, float>) {
                    const __m256 v_scale = _mm256_set1_ps(scale);
                    const __m256 v_min = _mm256_set1_ps(-32768.0f);
                    const __m256 v_max = _mm256_set1_ps(32767.0f);

                    for (; i + 8 <= size; i += 8) {
                        __m256 val = _mm256_mul_ps(_mm256_loadu_ps(&in_ptr[i]), v_scale);
                        val = _mm256_max_ps(v_min, _mm256_min_ps(v_max, val)); // Clamp to int16 range
                        
                        // Convert float32 to int32, then pack into int16
                        __m256i int32_vec = _mm256_cvtps_epi32(val);
                        __m128i low_128  = _mm256_castsi256_si128(int32_vec);
                        __m128i high_128 = _mm256_extracti128_si256(int32_vec, 1);
                        __m128i packed   = _mm_packs_epi32(low_128, high_128);
                        
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(&out_ptr[i]), packed);
                    }
                }
            #elif defined(CRESCENDO_SIMD_NEON)
                if constexpr (std::is_same_v<T, float>) {
                    const float32x4_t v_scale = vdupq_n_f32(scale);
                    const float32x4_t v_min = vdupq_n_f32(-32768.0f);
                    const float32x4_t v_max = vdupq_n_f32(32767.0f);

                    for (; i + 4 <= size; i += 4) {
                        float32x4_t val = vmulq_f32(vld1q_f32(&in_ptr[i]), v_scale);
                        val = vmaxq_f32(v_min, vminq_f32(v_max, val));
                        
                        int32x4_t int32_vec = vcvtq_s32_f32(val);
                        int16x4_t int16_vec = vqmovn_s32(int32_vec); // Saturating narrow to int16
                        vst1_s16(&out_ptr[i], int16_vec);
                    }
                }
            #endif
            for (; i < size; ++i) {
                T scaled = std::clamp(in_ptr[i] * scale, T{-32768.0}, T{32767.0});
                out_ptr[i] = static_cast<std::int16_t>(scaled);
            }
        }
    };
}