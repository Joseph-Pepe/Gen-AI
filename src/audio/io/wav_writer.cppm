export module crescendo.audio.io.wav_writer;

import std;

export namespace crescendo::io {

    // Use pragma packing to ensure 1-byte alignment across MSVC, GCC, and Clang
    #pragma pack(push, 1)
    struct WavHeader {
        char     riff_header[4] = {'R', 'I', 'F', 'F'};
        std::uint32_t wav_size       = 0; 
        char     wave_header[4] = {'W', 'A', 'V', 'E'};
        char     fmt_header[4]  = {'f', 'm', 't', ' '};
        std::uint32_t fmt_chunk_size = 16;
        std::uint16_t audio_format   = 1; // PCM Linear format
        std::uint16_t num_channels   = 1; // Mono channel config
        std::uint32_t sample_rate    = 44100;
        std::uint32_t byte_rate      = 44100 * 2;
        std::uint16_t sample_alignment = 2;
        std::uint16_t bits_per_sample  = 16;
        char     data_header[4] = {'d', 'a', 't', 'a'};
        std::uint32_t data_chunk_size = 0;
    };
    #pragma pack(pop)

    // Normalizes float arrays, clips bounds to avoid structural integer wrapping distortion, scales to 16-bit PCM targets, and serializes the 44-byte standard RIFF header directly to disk.
    // Uncrompressed, studio-grade 16-bit PCM (.wav) file.
    class WavExporter {
    public:
        /**
         * @brief Normalizes floating point audio streams and saves them as 16-bit signed WAV tracks.
         */
        template <std::floating_point T>
        static bool write_wav(const std::string& filepath, const std::vector<T>& audio_buffer, std::uint32_t sample_rate = 44100) {
            std::ofstream file(filepath, std::ios::binary);
            if (!file.is_open()) return false;

            // 1. Locate peak element to implement peak amplitude normalization
            T max_peak = 0.0;
            for (const auto& sample : audio_buffer) {
                T abs_val = std::abs(sample);
                if (abs_val > max_peak) max_peak = abs_val;
            }
            const T norm_factor = (max_peak > 1e-6) ? (T{0.95} / max_peak) : T{1.0};

            // 2. Configure sizes within the binary header packet
            std::uint32_t data_bytes = static_cast<std::uint32_t>(audio_buffer.size() * sizeof(std::int16_t));
            WavHeader header;
            header.sample_rate     = sample_rate;
            header.byte_rate       = sample_rate * 2;
            header.data_chunk_size = data_bytes;
            header.wav_size        = data_bytes + sizeof(WavHeader) - 8;

            // 3. Output structural binary header layout
            file.write(reinterpret_cast<const char*>(&header), sizeof(WavHeader));

            // 4. Transform and serialize floating array blocks to int16_t arrays
            std::vector<std::int16_t> pcm_buffer(audio_buffer.size());
            for (size_t i = 0; i < audio_buffer.size(); ++i) {
                T scaled = audio_buffer[i] * norm_factor * T{32767.0};
                // Symmetric guard clipping limits
                scaled = std::clamp(scaled, T{-32768.0}, T{32767.0});
                pcm_buffer[i] = static_cast<std::int16_t>(scaled);
            }

            file.write(reinterpret_cast<const char*>(pcm_buffer.data()), data_bytes);
            return true;
        }
    };
}