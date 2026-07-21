export module crescendo.audio.codec.bitstream;

import std;

export namespace crescendo::codec {

    // Provides an endian aware bi writer capable of packing arbitrary bit-length integers, big-endian byte swapping, and Golden-Rice unary prefix coding. 
    class BitWriter {
    public:
        explicit BitWriter(std::size_t initial_capacity_bytes = 65536) {
            buffer_.reserve(initial_capacity_bytes);
        }

        /**
         * @brief Writes up to 32 bits into the bitstream in big-endian order (standard for ALAC/M4A).
         */
        void write_bits(std::uint32_t value, std::uint8_t num_bits) {
            if (num_bits == 0 || num_bits > 32) return;
            
            // Mask out unused upper bits
            if (num_bits < 32) {
                value &= (1U << num_bits) - 1;
            }

            while (num_bits > 0) {
                std::uint8_t bits_to_write = std::min<std::uint8_t>(num_bits, 8 - bit_offset_);
                std::uint8_t shift = num_bits - bits_to_write;
                std::uint8_t chunk = (value >> shift) & ((1U << bits_to_write) - 1);

                current_byte_ |= (chunk << (8 - bit_offset_ - bits_to_write));
                bit_offset_ += bits_to_write;
                num_bits -= bits_to_write;

                if (bit_offset_ == 8) {
                    buffer_.push_back(current_byte_);
                    current_byte_ = 0;
                    bit_offset_ = 0;
                }
            }
        }

        /**
         * @brief Writes a Golomb-Rice encoded integer residual using parameter k.
         * Rice coding represents a number as a unary quotient followed by a k-bit binary remainder.
         */
        void write_rice_signed(std::int32_t value, std::uint8_t k) {
            // Map signed integer to unsigned integer: 0->0, -1->1, 1->2, -2->3...
            std::uint32_t uval = (value < 0) ? (static_cast<std::uint32_t>(-value - 1) << 1) | 1 : static_cast<std::uint32_t>(value) << 1;
            
            std::uint32_t quotient = uval >> k;
            std::uint32_t remainder = uval & ((1U << k) - 1);

            // Write unary quotient (quotient 1s followed by a terminating 0)
            for (std::uint32_t i = 0; i < quotient; ++i) {
                write_bits(1, 1);
            }
            write_bits(0, 1);

            // Write k-bit remainder
            if (k > 0) {
                write_bits(remainder, k);
            }
        }

        /**
         * @brief Flushes remaining partial bits (padded with 0s) and returns the raw byte vector.
         */
        void flush() {
            if (bit_offset_ > 0) {
                buffer_.push_back(current_byte_);
                current_byte_ = 0;
                bit_offset_ = 0;
            }
        }

        /**
         * @brief Appends a span of raw bytes directly to the buffer (must be byte-aligned).
         */
        void write_bytes(std::span<const std::uint8_t> data) {
            flush();
            buffer_.insert(buffer_.end(), data.begin(), data.end());
        }

        /**
         * @brief Overload to support braced initializer lists: bw.write_bytes({'a', 'l', 'a', 'c'})
         */
        void write_bytes(std::initializer_list<uint8_t> data) {
            flush();
            buffer_.insert(buffer_.end(), data.begin(), data.end());
        }

        /**
         * @brief Writes a 16-bit integer in Big-Endian order.
         */
        void write_uint16_be(std::uint16_t value) {
            flush();
            buffer_.push_back(static_cast<std::uint8_t>(value >> 8) & 0xFF);
            buffer_.push_back(static_cast<std::uint8_t>(value & 0xFF));
        }

        void write_uint16_le(uint16_t value) {
            flush();
            buffer_.push_back(static_cast<uint8_t>(value & 0xFF));
            buffer_.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
        }

        /**
         * @brief Writes a 32-bit integer in Big-Endian order.
         */
        void write_uint32_be(std::uint32_t value) {
            flush();
            buffer_.push_back(static_cast<std::uint8_t>((value >> 24) & 0xFF));
            buffer_.push_back(static_cast<std::uint8_t>((value >> 16) & 0xFF));
            buffer_.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFF));
            buffer_.push_back(static_cast<std::uint8_t>(value & 0xFF));
        }

        /**
         * @brief Writes a 32-bit integer in Little-Endian order (for WAV containers).
         */
        void write_uint32_le(std::uint32_t value) {
            flush();
            buffer_.push_back(static_cast<std::uint8_t>(value & 0xFF));
            buffer_.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFF));
            buffer_.push_back(static_cast<std::uint8_t>((value >> 16) & 0xFF));
            buffer_.push_back(static_cast<std::uint8_t>((value >> 24) & 0xFF));
        }

        [[nodiscard]] const std::vector<std::uint8_t>& data() const noexcept { return buffer_; }
        [[nodiscard]] size_t size_bytes() const noexcept { return buffer_.size(); }

    private:
        std::vector<std::uint8_t> buffer_;
        std::uint8_t current_byte_ = 0;
        std::uint8_t bit_offset_ = 0;
    };
}