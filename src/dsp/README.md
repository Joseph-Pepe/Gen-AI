# ⚡ Crescendo DSP: High-Performance Radix-2 FFT Engine

[![Standard](https://img.shields.io/badge/C%2B%2B-26-00599C?style=for-the-badge&logo=c%2b%2b)](https://en.cppreference.com/w/cpp/26)
[![Architecture](https://img.shields.io/badge/Algorithm-Cooley--Tukey%20Radix--2-purple?style=for-the-badge)](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)
[![Dependencies](https://img.shields.io/badge/Dependencies-ZERO-brightgreen?style=for-the-badge)]()

> **A bare-metal, zero-dependency Fast Fourier Transform (FFT) and Inverse Fast Fourier Transform (IFFT) module written in pure ISO C++26.**

---

## 📖 Overview

The `crescendo::dsp::Radix2FFT` module provides ultra-fast time-frequency domain transformations for real-time audio processing, synthesizer vocoding, and AI spectrogram generation. 

By avoiding deep recursive call stacks in favor of an **iterative, in-place Cooley-Tukey butterfly architecture**, and by precomputing bit-reversal permutations and trigonometric twiddle factors during instantiation, this engine achieves deterministic $O(N \log N)$ execution speeds suitable for mission-critical DSP pipelines.

---

## ✨ Key Features

* **C++26 Module Architecture:** Built using modern ISO C++ modules (`export module crescendo.dsp.fast_fourier_transform;`) for instantaneous header compilation and clean symbol encapsulation.
* **Zero-Dependency & Bare-Metal:** Relies exclusively on the standard library (`import std;`), requiring zero external math or DSP frameworks (no FFTW, no Intel MKL, no Boost).
* **Iterative Bit-Reversal:** Eliminates recursive function-call overhead by precalculating bit-reversed memory swap indices.
* **Precomputed Twiddle Factors:** Caches complex roots of unity ($W_N^k$) in contiguous memory vectors, preventing floating-point trigonometric calculation bottlenecks inside the processing loop.
* **In-Place Transformation:** Operates directly on `std::span<std::complex<T>>` memory buffers, requiring zero dynamic memory allocations during real-time execution.
* **Floating-Point Agnostic:** Templated using C++20/26 concepts (`std::floating_point`), allowing seamless switching between high-speed `float` (32-bit) and studio-precision `double` (64-bit) pipelines.

---

## 🧮 Mathematical Foundation

### The Discrete Fourier Transform (DFT)
The engine transforms a sequence of $N$ complex time-domain samples $x_0, x_1, \dots, x_{N-1}$ into frequency-domain bins $X_0, X_1, \dots, X_{N-1}$ according to the formula:

$$X_k = \sum_{n=0}^{N-1} x_n \cdot e^{-i \frac{2\pi}{N} k n} \quad \text{for } k = 0, 1, \dots, N-1$$

### Frequency Bin Resolution ($\Delta f$)
When processing PCM audio, the physical frequency (in Hertz) represented by bin index $k$ depends on the audio sample rate ($f_s$) and the window size ($N$):

$$\text{Bin Frequency}(k) = k \cdot \left( \frac{f_s}{N} \right)$$

*Example: At $f_s = 44,100\text{ Hz}$ with $N = 1024$, each bin represents a frequency step of $\approx 43.07\text{ Hz}$.*

---

## 🚀 Quick Start Guide

### 1. Importing the Module
Ensure your build system (CMake 3.28+) supports C++26 module scanning, then import the engine directly into your source code:

```cpp
import std;
import crescendo.dsp.fast_fourier_transform;

using namespace crescendo::dsp;
```

---

## 🌊 Phase 2: Short-Time Fourier Transform (STFT) & Log-Mel Filterbank

The `crescendo::dsp::stft` and `crescendo::dsp::mel_filterbank` modules form the data preprocessing bridge between raw 1D time-domain PCM audio and 2D neural-network-ready representations.

Because human hearing perceives pitch logarithmically—with high sensitivity to low-frequency intervals and lower sensitivity at high frequencies—feeding raw linear FFT bins into an AI model wastes computational capacity. This module slices continuous audio into overlapping windowed frames, extracts frequency spectras, and projects linear magnitudes onto a perceptual **Log-Mel Spectrogram matrix**.

### ✨ Key Features

* **Zero-Dependency 2D Spectrogram Engine:** Built entirely from scratch in ISO C++26 using contiguous vectors and custom matrix multiplication kernels.
* **Hanning Window Precomputation:** Caches von Hann trigonometric curve weights during class instantiation to eliminate spectral leakage without runtime calculation overhead.
* **Weighted Overlap-Add (WOLA) Reconstruction:** Reverses 2D complex spectrograms back into 1D playable PCM audio arrays with energy normalization, preventing amplitude distortion across overlapping frame boundaries.
* **Sparse Triangular Filterbank Matrix:** Dynamically constructs mapping matrices $B \in \mathbb{R}^{M \times K}$ to compress 513 linear frequency bins into compact 80-bin or 128-bin Mel representations.
* **Neural Network Normalization:** Automatically applies natural logarithm compression $L = \log(S + \epsilon)$ and linearly scales output tokens to strictly sit within the $[-1.0, 1.0]$ range required for diffusion and transformer training.

### 🧮 Mathematical Foundation

#### 1. Mel-Scale Frequency Conversion
To map linear frequencies ($f$ in Hz) onto the perceptual Mel scale ($m$), the engine applies the standard acoustic conversion formula:

$$m = 2595 \cdot \log_{10}\left(1 + \frac{f}{700}\right)$$

#### 2. Log-Mel Filterbank Matrix Multiplication
Given a linear magnitude spectrogram frame $S \in \mathbb{R}^{K}$ (where $K = \frac{N}{2} + 1$), the Mel energy vector $M \in \mathbb{R}^{M}$ is computed via matrix dot product against the triangular filterbank matrix $B$:

$$M = B \cdot S$$

The dynamic range is then logarithmically compressed to prevent variance dominance:

$$L = \log(M + \epsilon)$$

### 🚀 Usage Example

```cpp
import std;
import crescendo.dsp.stft;
import crescendo.dsp.mel_filterbank;

using namespace crescendo::dsp;

// 1. Configure audio parameters
constexpr size_t sample_rate = 44100;
constexpr size_t fft_size = 1024;
constexpr size_t hop_size = 256;    // 75% window overlap
constexpr size_t mel_bins = 80;     // Standard AI resolution

// 2. Instantiate Phase 2 engines
auto stft = ShortTimeFourierTransform<float>(fft_size, hop_size);
auto mel_bank = MelFilterbank<float>(mel_bins, fft_size, sample_rate, 20.0f, 8000.0f);

// 3. Convert 1D PCM audio to a 2D Log-Mel Spectrogram
std::vector<float> pcm_audio = /* ... load audio samples ... */;
auto complex_spec = stft.forward(pcm_audio);

// Extract linear magnitudes and convert to neural network tokens [-1.0, 1.0]
std::vector<std::vector<float>> linear_mag(complex_spec.size(), std::vector<float>(stft.num_bins()));
for (size_t f = 0; f < complex_spec.size(); ++f) {
    for (size_t b = 0; b < stft.num_bins(); ++b) {
        linear_mag[f][b] = std::abs(complex_spec[f][b]);
    }
}
auto log_mel_tokens = mel_bank.to_log_mel(linear_mag);

// 4. Reconstruct clean 1D PCM audio from spectrogram frames
auto reconstructed_pcm = stft.inverse(complex_spec, pcm_audio.size());
```

---

## ✂️ Phase 6: Frequency-Domain 4-Stem Audio Source Separation

The `crescendo::dsp::wiener_filter` and `crescendo::models::stem_unet` modules implement the audio demixing engine responsible for isolating individual instruments from mixed master recordings.

Because naive direct magnitude masking ($M_k \odot X$) causes severe acoustic phase cancellation, metallic artifacts, and energy loss when instruments overlap in the same frequency bin, this engine deploys a **Multi-Channel Wiener Filter (MWF)**. By squaring the neural network's predicted soft-mask ratios, the Wiener filter calculates the statistically optimal energy distribution across uncorrelated acoustic sources while preserving 100% of the original phase alignment.

### ✨ Key Features

* **4-Stem Convolutional U-Net (`stem_unet.cppm`):** Expands the Phase 4 generative architecture to project four independent output channels corresponding to **Vocals, Drums, Bass, and Other (Accompaniment)**, applying an in-place Sigmoid activation to strictly bound predictions within the $[0.0, 1.0]$ range.
* **SIMD Multi-Channel Wiener Filter (`wiener_filter.cppm`):** Leverages AVX-512, AVX2, and ARM NEON intrinsics to calculate squared energy ratios across overlapping stems in microsecond intervals, preventing inter-stem bleeding.
* **Strict Energy Conservation:** Guaranteed mathematical energy preservation. When the complex tensors of the four isolated stems are summed together ($\sum_{k=0}^{3} \hat{S}_k$), the resulting spectrogram equals the original input mixture with near-zero floating-point drift ($\approx 10^{-6}$ max error).
* **Zero-Copy Stem Splitting:** Manages multi-channel tensor slicing using contiguous memory offsets, allowing independent stem processing without unnecessary heap allocations.

### 🧮 Mathematical Foundation

#### 1. Multi-Channel Wiener Masking Ratio
Given a complex mixture spectrogram $X_{\text{mix}}(t, f)$ and four neural network magnitude mask predictions $M_k(t, f) \in [0, 1]$, the optimal complex stem estimate $\hat{S}_k(t, f)$ for stem $k$ is calculated by normalizing against the total squared energy density:

$$\hat{S}_k(t, f) = \left( \frac{|M_k(t, f)|^2}{\sum_{j=0}^{3} |M_j(t, f)|^2 + \epsilon} \right) \cdot X_{\text{mix}}(t, f)$$

#### 2. Energy Conservation Guarantee
By construction of the Wiener normalization denominator, the sum of the four isolated complex stems identically reconstructs the input mixture:

$$\sum_{k=0}^{3} \hat{S}_k(t, f) = X_{\text{mix}}(t, f) \cdot \frac{\sum_{k=0}^{3} |M_k(t, f)|^2}{\sum_{j=0}^{3} |M_j(t, f)|^2 + \epsilon} \approx X_{\text{mix}}(t, f)$$

### 🚀 Usage Example

```cpp
import std;
import crescendo.tensor.core;
import crescendo.models.stem_unet;
import crescendo.dsp.wiener_filter;

using namespace crescendo::tensor;
using namespace crescendo::models;
using namespace crescendo::dsp;

// 1. Instantiate the 4-Stem Separation U-Net and Wiener Filter
UNet4StemSeparation<float> sep_unet;
WienerFilter4Stem<float> wiener_filter(1e-8f);

// 2. Ingest a complex STFT audio mixture [Batch=1, Channels=1, FreqBins=1025, Frames=512]
Tensor<std::complex<float>> complex_mix = /* ... from Phase 2 STFT ... */;
Tensor<float> mixture_magnitude         = /* ... |complex_mix| ... */;

// 3. Predict 4 independent soft-masks [Vocals, Drums, Bass, Other]
std::array<Tensor<float>, 4> stem_masks = sep_unet.forward_masks(mixture_magnitude);

// 4. Apply SIMD Multi-Channel Wiener Filtering to isolate complex stems
std::array<Tensor<std::complex<float>>, 4> isolated_stems = wiener_filter.separate(complex_mix, stem_masks);

// 5. Access individual isolated stems ready for Inverse STFT and WAV export
Tensor<std::complex<float>>& vocals_stft = isolated_stems[0];
Tensor<std::complex<float>>& drums_stft  = isolated_stems[1];
Tensor<std::complex<float>>& bass_stft   = isolated_stems[2];
Tensor<std::complex<float>>& other_stft  = isolated_stems[3];
```