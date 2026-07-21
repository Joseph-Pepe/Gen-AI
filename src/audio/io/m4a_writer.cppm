export module crescendo.audio.io.m4a;

import std;
import crescendo.audio.codec.bitstream;
import crescendo.audio.codec.alac;

using namespace crescendo::codec;

export namespace crescendo::io {

    // Builds the hierarchical atom tree required by QuickTime and MPEG-4 players (ftyp, moov, trak, mdia, stbl, alac, mdat), back patching 32-bit big-endian atom size headers dynamically as media chunks are serialized.
    class M4AContainerWriter {
    public:
        static bool write_alac_m4a(const std::string& filepath,
                                   const std::vector<std::vector<std::uint8_t>>& alac_packets,
                                   const ALACSpecificConfig& config) {
            
            std::ofstream file(filepath, std::ios::binary);
            if (!file.is_open()) return false;

            BitWriter bw(1048576);

            // 1. ftyp box (File Type compatibility)
            write_box(bw, "ftyp", [&](BitWriter& b) {
                b.write_bytes({'M', '4', 'A', ' '}); // Major brand
                b.write_uint32_be(0);                // Minor version
                b.write_bytes({'M', '4', 'A', ' '}); // Compatible brands
                b.write_bytes({'m', 'p', '4', '2'});
                b.write_bytes({'i', 's', 'o', 'm'});
            });

            // Calculate total audio samples and duration
            const std::uint32_t num_frames = static_cast<std::uint32_t>(alac_packets.size());
            const std::uint32_t total_samples = num_frames * config.frame_length;
            std::uint32_t mdat_payload_bytes = 0;
            std::vector<uint32_t> sample_sizes;
            for (const auto& pkt : alac_packets) {
                mdat_payload_bytes += static_cast<std::uint32_t>(pkt.size());
                sample_sizes.push_back(static_cast<std::uint32_t>(pkt.size()));
            }

            // 2. moov box (Movie metadata header)
            write_box(bw, "moov", [&](BitWriter& moov) {
                // mvhd (Movie Header)
                write_box(moov, "mvhd", [&](BitWriter& b) {
                    b.write_uint32_be(0); // Version & flags
                    b.write_uint32_be(0); // Creation time
                    b.write_uint32_be(0); // Modification time
                    b.write_uint32_be(config.sample_rate); // Timescale
                    b.write_uint32_be(total_samples);      // Duration
                    b.write_uint32_be(0x00010000);         // Rate (1.0)
                    b.write_uint16_be(0x0100);             // Volume (Full)
                    b.write_bytes({0,0, 0,0, 0,0, 0,0, 0,0}); // Reserved
                    // Identity Matrix (3x3 fixed point)
                    b.write_uint32_be(0x00010000); b.write_uint32_be(0); b.write_uint32_be(0);
                    b.write_uint32_be(0); b.write_uint32_be(0x00010000); b.write_uint32_be(0);
                    b.write_uint32_be(0); b.write_uint32_be(0); b.write_uint32_be(0x40000000);
                    b.write_bytes({0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0}); // Pre-defined
                    b.write_uint32_be(2); // Next Track ID
                });

                // trak box (Track Header)
                write_box(moov, "trak", [&](BitWriter& trak) {
                    write_box(trak, "tkhd", [&](BitWriter& b) {
                        b.write_uint32_be(0x00000007); // Version 0, flags: enabled, in movie, in preview
                        b.write_uint32_be(0); // Creation time
                        b.write_uint32_be(0); // Modification time
                        b.write_uint32_be(1); // Track ID
                        b.write_uint32_be(0); // Reserved
                        b.write_uint32_be(total_samples); // Duration
                        b.write_bytes({0,0,0,0, 0,0,0,0}); // Reserved & Layer/AltGroup
                        b.write_uint16_be(0x0100); // Audio Volume
                        b.write_uint16_be(0); // Reserved
                        // Identity Matrix
                        b.write_uint32_be(0x00010000); b.write_uint32_be(0); b.write_uint32_be(0);
                        b.write_uint32_be(0); b.write_uint32_be(0x00010000); b.write_uint32_be(0);
                        b.write_uint32_be(0); b.write_uint32_be(0); b.write_uint32_be(0x40000000);
                        b.write_uint32_be(0); b.write_uint32_be(0); // Width / Height (0 for audio)
                    });

                    // mdia box (Media structure)
                    write_box(trak, "mdia", [&](BitWriter& mdia) {
                        write_box(mdia, "mdhd", [&](BitWriter& b) {
                            b.write_uint32_be(0); // Version & flags
                            b.write_uint32_be(0); // Creation time
                            b.write_uint32_be(0); // Modification time
                            b.write_uint32_be(config.sample_rate); // Timescale
                            b.write_uint32_be(total_samples);      // Duration
                            b.write_uint16_be(0x55C4); // ISO Language Code: 'und' (undetermined)
                            b.write_uint16_be(0);      // Pre-defined
                        });

                        write_box(mdia, "hdlr", [&](BitWriter& b) {
                            b.write_uint32_be(0); // Version & flags
                            b.write_uint32_be(0); // Pre-defined
                            b.write_bytes({'s', 'o', 'u', 'n'}); // Handler type: Sound
                            b.write_bytes({0,0,0,0, 0,0,0,0, 0,0,0,0}); // Reserved
                            b.write_bytes({'C', 'r', 'e', 's', 'c', 'e', 'n', 'd', 'o', ' ', 'A', 'L', 'A', 'C', 0}); // Name string
                        });

                        // minf box (Media Information)
                        write_box(mdia, "minf", [&](BitWriter& minf) {
                            write_box(minf, "smhd", [&](BitWriter& b) {
                                b.write_uint32_be(0); // Version & flags
                                b.write_uint16_be(0); // Balance
                                b.write_uint16_be(0); // Reserved
                            });

                            write_box(minf, "dinf", [&](BitWriter& dinf) {
                                write_box(dinf, "dref", [&](BitWriter& b) {
                                    b.write_uint32_be(0); // Version & flags
                                    b.write_uint32_be(1); // Entry count
                                    // url box (Self-referencing flag 0x000001)
                                    b.write_uint32_be(12);
                                    b.write_bytes({'u', 'r', 'l', ' '});
                                    b.write_uint32_be(0x00000001);
                                });
                            });

                            // stbl box (Sample Table Box - Core ALAC metadata)
                            write_box(minf, "stbl", [&](BitWriter& stbl) {
                                // stsd (Sample Description)
                                write_box(stbl, "stsd", [&](BitWriter& b) {
                                    b.write_uint32_be(0); // Version & flags
                                    b.write_uint32_be(1); // Entry count
                                    
                                    // alac audio sample entry box
                                    ALACEncoder encoder(config);
                                    auto cookie = encoder.get_magic_cookie();
                                    std::uint32_t alac_box_size = 36 + static_cast<std::uint32_t>(cookie.size());
                                    
                                    b.write_uint32_be(alac_box_size);
                                    b.write_bytes({'a', 'l', 'a', 'c'});
                                    b.write_bytes({0,0,0,0, 0,0}); // Reserved
                                    b.write_uint16_be(1); // Data reference index
                                    b.write_bytes({0,0,0,0, 0,0,0,0}); // Reserved
                                    b.write_uint16_be(config.num_channels);
                                    b.write_uint16_be(config.bit_depth);
                                    b.write_uint16_be(0); // Pre-defined
                                    b.write_uint16_be(0); // Packet size
                                    b.write_uint32_be(config.sample_rate << 16); // 16.16 fixed point rate
                                    
                                    b.write_bytes(cookie); // Embed ALAC Magic Cookie
                                });

                                // stts (Time-to-Sample)
                                write_box(stbl, "stts", [&](BitWriter& b) {
                                    b.write_uint32_be(0); // Version & flags
                                    b.write_uint32_be(1); // Entry count
                                    b.write_uint32_be(num_frames);
                                    b.write_uint32_be(config.frame_length);
                                });

                                // stsc (Sample-to-Chunk) - 1 chunk containing all frames
                                write_box(stbl, "stsc", [&](BitWriter& b) {
                                    b.write_uint32_be(0);
                                    b.write_uint32_be(1);
                                    b.write_uint32_be(1); // First chunk
                                    b.write_uint32_be(num_frames); // Samples per chunk
                                    b.write_uint32_be(1); // Sample description index
                                });

                                // stsz (Sample Sizes)
                                write_box(stbl, "stsz", [&](BitWriter& b) {
                                    b.write_uint32_be(0); // Version & flags
                                    b.write_uint32_be(0); // 0 = variable sizes
                                    b.write_uint32_be(num_frames);
                                    for (std::uint32_t sz : sample_sizes) {
                                        b.write_uint32_be(sz);
                                    }
                                });

                                // stco (Chunk Offsets)
                                write_box(stbl, "stco", [&](BitWriter& b) {
                                    b.write_uint32_be(0);
                                    b.write_uint32_be(1);
                                    // Chunk offset will point to mdat payload after ftyp + moov bytes
                                    // We back-patch this exact offset once moov size is determined!
                                    b.write_uint32_be(0xDEADBEEF); 
                                });
                            });
                        });
                    });
                });
            });

            // Back-patch exact stco chunk offset (ftyp size [32] + moov size + mdat header [8])
            const std::uint32_t mdat_offset = static_cast<std::uint32_t>(bw.size_bytes()) + 8;
            auto& buf = const_cast<std::vector<std::uint8_t>&>(bw.data());
            for (size_t i = 0; i < buf.size() - 4; ++i) {
                if (buf[i] == 0xDE && buf[i+1] == 0xAD && buf[i+2] == 0xBE && buf[i+3] == 0xEF) {
                    buf[i]   = static_cast<std::uint8_t>((mdat_offset >> 24) & 0xFF);
                    buf[i+1] = static_cast<std::uint8_t>((mdat_offset >> 16) & 0xFF);
                    buf[i+2] = static_cast<std::uint8_t>((mdat_offset >> 8) & 0xFF);
                    buf[i+3] = static_cast<std::uint8_t>(mdat_offset & 0xFF);
                    break;
                }
            }

            // 3. mdat box (Media Data Payload)
            bw.write_uint32_be(mdat_payload_bytes + 8);
            bw.write_bytes({'m', 'd', 'a', 't'});
            for (const auto& pkt : alac_packets) {
                bw.write_bytes(pkt);
            }

            file.write(reinterpret_cast<const char*>(bw.data().data()), bw.size_bytes());
            return true;
        }

    private:
        template <typename Func>
        static void write_box(BitWriter& bw, const char tag[4], Func&& lambda) {
            size_t start_offset = bw.size_bytes();
            bw.write_uint32_be(0); // Placeholder for 32-bit box length
            bw.write_bytes({static_cast<std::uint8_t>(tag[0]), static_cast<std::uint8_t>(tag[1]),
                            static_cast<std::uint8_t>(tag[2]), static_cast<std::uint8_t>(tag[3])});
            
            lambda(bw); // Execute nested box writing

            size_t end_offset = bw.size_bytes();
            uint32_t box_size = static_cast<std::uint32_t>(end_offset - start_offset);

            // Back-patch exact box size at start_offset in big-endian
            auto& buf = const_cast<std::vector<std::uint8_t>&>(bw.data());
            buf[start_offset]     = static_cast<std::uint8_t>((box_size >> 24) & 0xFF);
            buf[start_offset + 1] = static_cast<std::uint8_t>((box_size >> 16) & 0xFF);
            buf[start_offset + 2] = static_cast<std::uint8_t>((box_size >> 8) & 0xFF);
            buf[start_offset + 3] = static_cast<std::uint8_t>(box_size & 0xFF);
        }
    };
}