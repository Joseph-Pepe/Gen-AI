---

## 📦 Phase 9: Native ALAC (.m4a) Lossless Compression & LPCM Bitstream Suite

The `crescendo::codec` module implements a zero-dependency multimedia serialization engine capable of encoding and packaging studio audio streams into industry-standard container formats without external multimedia libraries (such as FFmpeg, libsndfile, or Apple's ALAC reference SDK).

To support cross-platform professional audio workflows, the suite writes little-endian RIFF/WAVE (`.wav`), big-endian FORM/AIFF (`.aiff`), and MPEG-4 Part 14 (`.m4a`) QuickTime containers. The integrated Apple Lossless Audio Codec (ALAC) core achieves significant file size reduction while preserving 100% bit-exact acoustic fidelity through SIMD linear prediction and adaptive entropy coding.

### ⚡ Performance Benchmarks (MSVC x64 Debug vs Release)

Note: Benchmarks recorded encoding 3.0 seconds of stereo audio at 44,100 Hz (264,600 samples / 516.80 KB uncompressed PCM) on x86_64 architecture.

| Export Format | Container TypeCodec | Payload Size | Debug Execution Time | Release Execution Time (Estimated) | 
| :--- | :--- | :--- | :--- | :--- |
| RIFF/WAVE (.wav) | Little-Endian | Uncompressed LPCM | 516.84 KB | ~4.2 ms | ~0.3 ms | 
| FORM/AIFF (.aiff) | Big-Endian |  Uncompressed LPCM | 516.86 KB | ~5.1 ms | ~0.4 ms | 
| MPEG-4 (.m4a) | ISO Base Media | ALAC Lossless | 356.12 KB (31.2% smaller) | ~44.0 ms | ~3.2 ms

### ✨ Key Features

* **Endian-Aware Bitstream Writer (`bitstream.cppm`):** Manages sub-byte bit packing, big-endian byte swapping, and Golomb-Rice unary prefix coding with dynamic buffer growth and zero-padding alignment.
* **Dual LPCM Container Serialization (`lpcm_writer.cppm`):** Exports interleaved 16-bit and 24-bit PCM audio directly to RIFF/WAVE containers and FORM/AIFF containers, implementing an 80-bit IEEE 754 extended floating-point encoder for Apple AIFF sample rate headers.
* **SIMD ALAC Residual Prediction (`alac_encoder.cppm`):** Leverages AVX-512, AVX2, and ARM NEON intrinsics to calculate Linear Prediction Coding (LPC) residuals via delta-delta FIR estimation, packing error distributions using adaptive Golomb-Rice entropy coding.
* **ISO Base Media Atom Tree Packaging (`m4a_writer.cppm`):** Constructs hierarchical MPEG-4 atom structures (`ftyp`, `moov`, `trak`, `mdia`, `minf`, `stbl`, `mdat`), generating the 36-byte ALAC Magic Cookie and dynamically back-patching sample chunk offsets (`stco`) during streaming serialization.

### 🧮 Architectural Foundation

#### 1. Order-2 LPC Residual Estimation
For each audio frame of length $N$, spatial redundancy is removed by subtracting a linear combination of previous samples from the current sample $x[n]$, generating a low-variance residual error signal $e[n]$:

$$e[n] = x[n] - 2x[n-1] + x[n-2]$$

#### 2. Golomb-Rice Unary Residual Packing
Residual integers $e[n]$ are mapped to non-negative integers $u[n]$ using zigzag encoding, then divided by $2^k$ (where $k$ is the adaptive Rice tuning parameter) into a unary quotient $q$ and a $k$-bit binary remainder $r$:

$$q = \lfloor u[n] \cdot 2^{-k} \rfloor, \quad r = u[n] \bmod 2^k$$

### 🚀 Usage Example

```cpp
import std;
import crescendo.codec.bitstream;
import crescendo.codec.lpcm;
import crescendo.codec.alac;
import crescendo.codec.m4a;

using namespace crescendo::codec;

// 1. Ingest an interleaved 16-bit PCM master audio buffer (e.g., from Phase 7)
std::vector<int16_t> pcm_master = /* ... 44.1 kHz Stereo PCM Data ... */;
uint32_t sample_rate = 44100;
uint16_t num_channels = 2;

// 2. Export standard uncompressed LPCM containers
LPCMWriter::write_wav("master_track.wav", pcm_master, num_channels, sample_rate);
LPCMWriter::write_aiff("master_track.aiff", pcm_master, num_channels, sample_rate);

// 3. Configure and execute ALAC Lossless Encoding
ALACSpecificConfig config;
config.sample_rate = sample_rate;
config.num_channels = num_channels;
config.frame_length = 4096;

ALACEncoder encoder(config);
std::vector<std::vector<uint8_t>> alac_packets = encoder.encode_stream(pcm_master);

// 4. Package compressed bitstream frames into an MPEG-4 media container (.m4a)
M4AContainerWriter::write_alac_m4a("master_track.m4a", alac_packets, config);
```

```json?chameleon
{"component":"LlmGeneratedComponent","props":{"height":"750px","prompt":"Create an interactive ALAC Lossless Codec & MPEG-4 Container Explorer using HTML5 Canvas or SVG with interactive DOM elements. Objective: Allow the user to explore how Linear Prediction Coding (LPC) and Golomb-Rice entropy coding compress PCM audio without data loss, and inspect the hierarchical box/atom structure of an .m4a Apple Lossless media container. Data State: Initialize with values matching the user's C++ test output: Audio Duration = 3.0s (44100 Hz Stereo), Uncompressed PCM Size = 516.80 KB, Frame Length = 4096 samples, Encoded Frames = 33, LPC Filter Order = 2, Rice Parameter k = 10, Achieved Compression Reduction = 31.2%. Strategy: Form Layout with interactive encoder controls at the top and real-time visualizers below. Inputs: Slider for LPC Filter Order (0 to 8), Slider for Rice Parameter k (4 to 16), Dropdown for Audio Waveform Complexity ('Simple Pure Sine Chord', 'Dense Acoustic Drum Loop', 'White Noise Burst'), and clickable tabs to switch between 'LPC Residual & Rice Bitstream View' and 'MPEG-4 (.m4a) Atom Tree View'. Behavior: In LPC Residual View, render the input PCM waveform overlaid with the LPC predicted waveform, alongside a residual error chart showing how higher filter orders flatten residual energy variance. Demonstrate how Rice coding splits residuals into unary quotients and k-bit remainders, updating a live compression efficiency gauge. In MP4 Atom Tree View, render an interactive hierarchical expandable tree diagram showing ftyp, moov -> mvhd, trak -> mdia -> minf -> stbl -> stsd (highlighting the 36-byte ALAC Magic Cookie alac atom), and mdat. When the user adjusts the LPC Order or Complexity sliders, dynamically recalculate and animate: Predicted ALAC File Size (KB), Lossless Compression Ratio (%), SIMD Encoder Latency (ms), and Rice Bitstream Entropy.","id":"im_001b4e704f6eb379"}}
```