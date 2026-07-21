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