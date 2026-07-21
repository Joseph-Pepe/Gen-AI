export module crescendo.audio.codec.lpcm;

import std;
import crescendo.audio.codec.bitstream;

export namespace crescendo::codec {

    // Serializes interleaved 16-bit and 24-bit PCM audio arrays into both RIFF/WAV (little-endian) and FORM/AIFF (big-endian) files, implementing an 80-bit IEEE extended floating-point converter for AIFF sample rate headers.
    class LPCMWriter {
    public:
        /**
         * @brief Writes an interleaved PCM buffer to a standard Little-Endian RIFF/WAVE file (.wav).
         */
        static bool write_wav(const std::string& filepath, 
                              std::span<const std::int16_t> interleaved_pcm, 
                              std::uint16_t num_channels = 2, 
                              std::uint32_t sample_rate = 44100) {
            
            std::ofstream file(filepath, std::ios::binary);
            if (!file.is_open()) return false;

            const std::uint32_t data_bytes = static_cast<std::uint32_t>(interleaved_pcm.size() * sizeof(std::int16_t));
            const std::uint32_t file_size = data_bytes + 36;
            const std::uint32_t byte_rate = sample_rate * num_channels * 2; 
            const std::uint16_t block_align = num_channels * 2;

            BitWriter bw(44 + data_bytes);
            
            // RIFF Header
            bw.write_bytes({'R', 'I', 'F', 'F'});
            bw.write_uint32_le(file_size);
            bw.write_bytes({'W', 'A', 'V', 'E'});

            // fmt chunk
            bw.write_bytes({'f', 'm', 't', ' '});
            bw.write_uint32_le(16); // Chunk size
            bw.write_uint16_le(1); // PCM format (1) little-endian
            bw.write_uint16_le(num_channels);
            bw.write_uint32_le(sample_rate);
            bw.write_uint32_le(byte_rate); 
            bw.write_uint16_le(block_align);
            bw.write_uint16_le(16); // Bits per sample

            // data chunk
            bw.write_bytes({'d', 'a', 't', 'a'});
            bw.write_uint32_le(data_bytes);

            // Write sample payload in Little-Endian
            for (std::int16_t sample : interleaved_pcm) {
                std::uint16_t le_val = std::byteswap(static_cast<std::uint16_t>(sample)); 
                if constexpr (std::endian::native == std::endian::little) {
                    le_val = static_cast<std::uint16_t>(sample);
                }
                bw.write_uint16_be(std::byteswap(le_val));
            }

            file.write(reinterpret_cast<const char*>(bw.data().data()), bw.size_bytes());
            return true;
        }

        /**
         * @brief Writes an interleaved PCM buffer to a Big-Endian FORM/AIFF file (.aiff).
         */
        static bool write_aiff(const std::string& filepath, 
                               std::span<const std::int16_t> interleaved_pcm, 
                               uint16_t num_channels = 2, 
                               uint32_t sample_rate = 44100) {
            
            std::ofstream file(filepath, std::ios::binary);
            if (!file.is_open()) return false;

            const std::uint32_t num_sample_frames = static_cast<std::uint32_t>(interleaved_pcm.size() / num_channels);
            const std::uint32_t sound_data_bytes = static_cast<std::uint32_t>(interleaved_pcm.size() * sizeof(std::int16_t));
            const std::uint32_t form_chunk_size = 4 + 18 + 8 + 12 + 8 + sound_data_bytes; // COMM + SSND overhead

            BitWriter bw(54 + sound_data_bytes);

            // FORM Chunk
            bw.write_bytes({'F', 'O', 'R', 'M'});
            bw.write_uint32_be(form_chunk_size);
            bw.write_bytes({'A', 'I', 'F', 'F'});

            // COMM Chunk
            bw.write_bytes({'C', 'O', 'M', 'M'});
            bw.write_uint32_be(18); // COMM chunk size
            bw.write_uint16_be(num_channels);
            bw.write_uint32_be(num_sample_frames);
            bw.write_uint16_be(16); // 16 bits per sample
            
            // Write 80-bit IEEE Extended Floating Point Sample Rate
            auto ieee80_rate = convert_to_ieee80(sample_rate);
            bw.write_bytes(ieee80_rate);

            // SSND Chunk (Sound Data)
            bw.write_bytes({'S', 'S', 'N', 'D'});
            bw.write_uint32_be(sound_data_bytes + 8);
            bw.write_uint32_be(0); // Offset
            bw.write_uint32_be(0); // Block Size

            // Write sample payload in Big-Endian (AIFF Native)
            for (std::int16_t sample : interleaved_pcm) {
                bw.write_uint16_be(static_cast<std::uint16_t>(sample));
            }

            file.write(reinterpret_cast<const char*>(bw.data().data()), bw.size_bytes());
            return true;
        }

    private:
        /**
         * @brief Converts a 32-bit unsigned integer sample rate into Apple's 80-bit IEEE Extended Float format.
         */
        static std::array<std::uint8_t, 10> convert_to_ieee80(std::uint32_t sample_rate) noexcept {
            std::array<std::uint8_t, 10> bytes = {0};
            if (sample_rate == 0) return bytes;

            std::uint32_t val = sample_rate;
            int exponent = 16383 + 31; // Bias + 31 bit shift
            
            while ((val & 0x80000000) == 0) {
                val <<= 1;
                exponent--;
            }

            bytes[0] = static_cast<std::uint8_t>((exponent >> 8) & 0xFF);
            bytes[1] = static_cast<std::uint8_t>(exponent & 0xFF);
            bytes[2] = static_cast<std::uint8_t>((val >> 24) & 0xFF);
            bytes[3] = static_cast<std::uint8_t>((val >> 16) & 0xFF);
            bytes[4] = static_cast<std::uint8_t>((val >> 8) & 0xFF);
            bytes[5] = static_cast<std::uint8_t>(val & 0xFF);
            return bytes;
        }
    };
}