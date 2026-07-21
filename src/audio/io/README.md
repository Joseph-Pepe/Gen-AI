---

## 🔊 Phase 6: Griffin-Lim Phase Vocoder & Native WAV Exporter

The `crescendo::dsp::griffin_lim` and `crescendo::io::wav_writer` modules form the final operational stage of the Crescendo AI Audio Engine. They bridge the gap between AI-generated Log-Mel Spectrogram matrices and physical, playable time-domain audio waveforms.

Because diffusion models and convolutional U-Nets generate *magnitude* spectrograms (discarding phase angle timing during perceptual Mel compression), direct Inverse Fourier Transformation produces severe phase-cancellation distortion. This module implements an iterative **Griffin-Lim Phase Estimation Vocoder** coupled with a bare-metal **RIFF/WAVE PCM Exporter** to compile clean audio binaries directly to disk.

### ⚡ Performance Benchmarks (x86_64 AVX2 Release Build)

| Pipeline Stage | Debug Mode Time | Release Mode Time | Hardware Speedup | 
| :--- | :--- | :--- | :--- |
| **DDIM Diffusion Generator** (25 steps) | 10,523 ms | **1,706 ms** | **6.1x Faster** | 
| **Griffin-Lim Phase Vocoder** (60 iterations) | 3,692 ms | **198 ms** | **18.6x Faster** |
| **Total Pipeline Execution** | ~14.2 seconds | **~1.9 seconds** | **~7.5x Faster** |

### ✨ Key Features

* **Pseudo-Inverse Mel Filterbank Inversion:** Projections apply transposed matrix approximation to expand compact 80-bin or 128-bin Mel scale matrices back into full-resolution linear magnitude spectra (e.g., 1,025 FFT bins).
* **Iterative Griffin-Lim Phase Reconstruction:** Executes alternating Forward STFT and Overlap-Add Inverse STFT cycles (typically 30 to 60 iterations), mathematically converging on an acoustically consistent phase alignment from a uniform random initial phase seed.
* **SIMD Window Energy Normalization:** Leverages AVX-512, AVX2, or NEON vector intrinsics to perform division across overlapping Hanning synthesis windows, eliminating amplitude modulation artifacts at frame boundaries.
* **Peak-Preserving WAV Serialization:** Automatically scans raw floating-point time-domain arrays for peak amplitude, normalizes headroom to -0.44 dBFS (0.95 linear), applies symmetric integer guard clipping, and writes standard 44-byte RIFF/PCM headers directly to disk.


### 🧮 Mathematical Foundation

#### 1. Linear Magnitude Approximation
Given an AI-generated Mel matrix $M \in \mathbb{R}^{M_{bins} \times T}$, linear magnitude representations $S \in \mathbb{R}^{K \times T}$ (where $K = \frac{N}{2} + 1$) are approximated via the filterbank transpose $B^T$:

$$S \approx B^T \cdot (\exp(M) - \epsilon)$$

#### 2. Griffin-Lim Phase Optimization Loop
At each iteration $i$, the time-domain signal $x^{(i)}$ is synthesized using the current phase estimate $\angle \hat{S}^{(i)}$ and the fixed target magnitude $|S_{target}|$:

$$x^{(i)} = \text{ISTFT}\left( |S_{target}| \odot \exp\left(j \angle \hat{S}^{(i-1)}\right) \right)$$

$$\hat{S}^{(i)} = \text{STFT}\left( x^{(i)} \right)$$

$$\angle \hat{S}^{(i)} = \text{atan2}\left( \Im(\hat{S}^{(i)}), \Re(\hat{S}^{(i)}) \right)$$

### 🚀 Usage Example

```cpp
import std;
import crescendo.tensor.core;
import crescendo.dsp.griffin_lim;
import crescendo.io.wav_writer;

using namespace crescendo::tensor;
using namespace crescendo::dsp;
using namespace crescendo::io;

// 1. Initialize Vocoder with standard studio windowing (N=2048, Hop=512)
GriffinLimVocoder<float> vocoder(2048, 512);

// 2. Ingest AI-generated Log-Mel Spectrogram from Phase 5 [Batch, Channels, Mel-Bins, Frames]
Tensor<float> generated_mel = /* ... from DDIM Sampler ... */;
Tensor<float> filterbank    = /* ... from Phase 2 Mel Filterbank ... */;

// 3. Unfold Mel scale back to linear FFT bins
auto linear_mag = vocoder.invert_mel_filterbank(generated_mel, filterbank);

// 4. Reconstruct playable time-domain audio via 60 Griffin-Lim iterations
std::vector<float> pcm_audio = vocoder.reconstruct(linear_mag, 60);

// 5. Serialize floating array to a 16-bit PCM WAV file
std::string filepath = "output_track.wav";
bool success = WavExporter::write_wav(filepath, pcm_audio, 44100);

if (success) {
    std::println("✔ Audio successfully exported to disk!");
}
```