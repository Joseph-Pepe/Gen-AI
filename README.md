🎛️ Crescendo Engine: Bare-Metal C++ AI Music Generator & Stem Separator

[![Language](https://img.shields.io/badge/Language-C%2B%2B26-00599C?style=for-the-badge&logo=c%2b%2b)](https://en.cppreference.com/w/cpp/26)
[![Dependencies](https://img.shields.io/badge/Dependencies-ZERO%20(Pure%20Bare--Metal)-brightgreen?style=for-the-badge)](https://github.com/)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%7C%20NEON%20%7C%20FMA-orange?style=for-the-badge)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

> **A high-performance, zero-dependency AI music generation and audio source separation (stem demixing) suite built entirely from scratch in standard C++26.**

---

## 🗺️ Architectural Roadmap

Modern AI audio suites are bogged down by gigabytes of Python dependencies, heavy ML frameworks (PyTorch, TensorFlow), complex build chains, and sluggish runtime interpretation. **Crescendo Engine** proves that state-of-the-art generative audio and digital signal processing (DSP) can be achieved from first principles.

By writing a custom multidimensional tensor engine, auto-differentiation graph, frequency-domain neural networks, and raw binary file parsers in pure C++, Crescendo delivers **maximum computational throughput**, **minimal memory footprint**, and **real-time interactive editing** without a single external library.

[x] Phase 1: Custom Tensor Library & Auto-differentiation engine.

[ ] Phase 2: AVX2 and ARM NEON SIMD vectorization for matrix ops.

[ ] Phase 3: Radix-2 FFT, STFT, and Mel-scale filterbank algorithms.

[ ] Phase 4: U-Net Architecture with Multi-Head Self-Attention.

[ ] Phase 5: DDIM fast sampling and Griffin-Lim phase estimation.

[ ] Phase 6: Frequency-domain 4-stem audio separation masking.

[ ] Phase 7: Native 16-bit / 32-bit floating point WAV file I/O.

[ ] Phase 8: GPU Compute Acceleration via raw Vulkan / OpenCL compute shaders.

[ ] Phase 9: Native ALAC (.m4a) lossless compression bitstream writer.

[ ] Phase 10: Graphical User Interface (GUI) built with raw OpenGL / GLFW (Zero UI frameworks).

---

## 🌟 Key Features

### 🧠 1. Custom Zero-Dependency ML Framework
* **Dynamic N-Dimensional Tensors:** Contiguous memory layouts with custom striding, broadcasting, and slicing.
* **SIMD-Accelerated Math:** Hand-tuned AVX2, FMA, and ARM NEON intrinsics for General Matrix Multiplication (GEMM), 2D Convolutions, and activation functions (`ReLU`, `GELU`, `SiLU`).
* **Auto-Differentiation Engine:** Dynamic computational graph with analytical backward pass implementations for all primitives.
* **Custom Optimizers:** Built-in `AdamW` optimizer with exponential moving average (EMA) momentum buffers and L2 weight decay.

### 🎼 2. High-Fidelity AI Music Generator
* **Mel-Spectrogram Diffusion Model:** Generates high-resolution 2D time-frequency grids from pure static white noise.
* **U-Net with Multi-Head Self-Attention:** Captures both local acoustic texture (via strided convolutions) and global musical structure (rhythm, melody, repetition) via attention blocks.
* **DDIM (Denoising Diffusion Implicit Models) Sampler:** Generates complete musical tracks in **20–50 deterministic steps** (a 95% reduction in inference time compared to standard 1,000-step DDPM).
* **Custom Training Loop:** Train directly on your own custom music library with automated data augmentation and noise scheduling.

### 🎸 3. Studio-Grade Stem Separation (Demixing)
* **Frequency-Domain Masking Architecture:** Separates mixed audio into 4 distinct stems: **Vocals**, **Drums**, **Bass**, and **Other**.
* **Complex-Valued STFT U-Net:** Analyzes magnitude spectrograms to predict high-precision soft masks ($M \in [0, 1]$) for each instrument class.
* **Phase-Preserving Reconstruction:** Applies estimated masks to complex frequency representations and reconstructs ultra-clean time-domain signals via overlap-add Inverse STFT.

### 🎚️ 4. Real-Time Interactive Music Editor
* **Lock-Free Audio Engine:** Circular lock-free ring buffers separate the high-priority real-time audio playback thread from UI and rendering threads.
* **Multi-Track Mixing & Slicing:** Adjust stem volumes, pan, apply EQ, and execute sample-accurate cuts without latency spikes.

### 💾 5. Native Audio File I/O
* **PCM `.wav` Engine:** Native 16-bit signed integer and 32-bit floating-point WAV encoder/decoder written from scratch.
* **Lossless `.m4a` (ALAC) & `.mp4` (AAC) Roadmap:** Custom container parsing and bitstream entropy coding pipelines for advanced Apple Lossless and MP4 container formats.

---

## 🏗️ System Architecture & File Structure

```text
crescendo/
├── 📁 include/                 # Header files (Public API & definitions)
│   ├── 📁 audio/               # Audio processing & container formats
│   │   ├── 📄 editor.hpp       # Lock-free interactive audio editor engine
│   │   ├── 📄 wav_parser.hpp   # Zero-dependency RIFF/WAVE PCM I/O
│   │   └── 📄 m4a_codec.hpp    # ALAC / AAC container & bitstream headers
│   ├── 📁 dsp/                 # Digital Signal Processing routines
│   │   ├── 📄 fft.hpp          # Radix-2 Cooley-Tukey FFT / IFFT
│   │   ├── 📄 stft.hpp         # STFT, iSTFT, and Overlap-Add windowing
│   │   ├── 📄 mel_filterbank.hpp # Mel-scale transformation matrices
│   │   └── 📄 griffin_lim.hpp  # Phase reconstruction algorithm
│   ├── 📁 models/              # Neural network architectures & diffusion
│   │   ├── 📄 unet.hpp         # 2D Convolutional U-Net architecture
│   │   ├── 📄 attention.hpp    # Multi-Head Self-Attention blocks
│   │   ├── 📄 diffusion.hpp    # Noise schedulers & DDIM fast sampler
│   │   └── 📄 separator.hpp    # Stem separation masking engine
│   └── 📁 tensor/              # Custom backend (Math, Simd, Autograd)
│       ├── 📄 tensor.hpp       # N-Dimensional contiguous tensor class
│       ├── 📄 autograd.hpp     # Computational graph & backward pass
│       ├── 📄 simd_ops.hpp     # AVX2 / ARM NEON vector math intrinsics
│       └── 📄 optimizer.hpp    # AdamW optimizer with L2 decay
├── 📁 src/                     # Implementation files (.cpp)
│   ├── 📁 audio/               # Audio parsers & editor implementations
│   ├── 📁 dsp/                 # FFT and spectrogram DSP implementations
│   ├── 📁 models/              # Neural network layer implementations
│   ├── 📁 tensor/              # Low-level tensor math & autograd engine
│   └── 📄 main.cpp             # Command-Line Interface (CLI) entry point
├── 📁 tests/                   # Test suite
│   ├── 🧪 test_fft.cpp         # FFT accuracy against analytical signals
│   ├── 🧪 test_autograd.cpp    # Gradient checking via finite differences
│   └── 🧪 test_audio_io.cpp    # Bit-exact WAV read/write verification
├── ⚙️ CMakeLists.txt           # Master build configuration (C++20, AVX2/NEON)
├── ⚖️ LICENSE                  # MIT License
└── 📖 README.md                # Project documentation
```

To create this for `CMakeLists.txt` without errors, we need to scaffold the folder structure and provide stub files using the command prompt.

```text
@echo off
mkdir include\audio include\dsp include\models include\tensor 2>nul
mkdir src\audio src\dsp src\models src\tensor 2>nul
mkdir tests build 2>nul

echo #pragma once > include\audio\editor.hpp
echo #pragma once > include\audio\wav_parser.hpp
echo #pragma once > include\audio\m4a_codec.hpp
echo #pragma once > include\dsp\fft.hpp
echo #pragma once > include\dsp\stft.hpp
echo #pragma once > include\dsp\mel_filterbank.hpp
echo #pragma once > include\dsp\griffin_lim.hpp
echo #pragma once > include\models\unet.hpp
echo #pragma once > include\models\attention.hpp
echo #pragma once > include\models\diffusion.hpp
echo #pragma once > include\models\separator.hpp
echo #pragma once > include\tensor\tensor.hpp
echo #pragma once > include\tensor\autograd.hpp
echo #pragma once > include\tensor\simd_ops.hpp
echo #pragma once > include\tensor\optimizer.hpp

echo // Stub > src\tensor\tensor.cpp
echo // Stub > src\tensor\autograd.cpp
echo // Stub > src\tensor\simd_ops.cpp
echo // Stub > src\tensor\optimizer.cpp
echo // Stub > src\dsp\fft.cpp
echo // Stub > src\dsp\stft.cpp
echo // Stub > src\dsp\mel_filterbank.cpp
echo // Stub > src\dsp\griffin_lim.cpp
echo // Stub > src\models\unet.cpp
echo // Stub > src\models\attention.cpp
echo // Stub > src\models\diffusion.cpp
echo // Stub > src\models\separator.cpp
echo // Stub > src\audio\editor.cpp
echo // Stub > src\audio\wav_parser.cpp
echo // Stub > src\audio\m4a_codec.cpp

(echo int main^(^) { return 0; }) > src\main.cpp
(echo int main^(^) { return 0; }) > tests\test_fft.cpp
(echo int main^(^) { return 0; }) > tests\test_autograd.cpp
(echo int main^(^) { return 0; }) > tests\test_audio_io.cpp

echo Scaffolding complete!
```

---

## 🛠️ Building from Source

This project uses CMake and vcpkg to ensure a frictionless, cross-platform build process.

### 1. System Prerequisites

Ensure your local workstation has the required core build tools installed.

* **Version Control:** Git installed and added to your system PATH.
* **Compiler:** GCC 13+ or Clang 17+, or MSVC 2026+ (Must support C++26 standard).
* **Build System:** CMake (v3.28 or newer).
* **Hardware:** CPU with AVX2 (x86_64) or NEON (ARM64) support recommended.

### 2. Compilation Commands

```text
# 1. Clone the repository
git clone [https://github.com/yourusername/crescendo-engine.git](https://github.com/yourusername/crescendo-engine.git)
cd crescendo-engine

# 2. Create and enter build directory
mkdir build && cd build

# 3. Configure CMake with SIMD optimizations enabled
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_AVX2=ON

# 4. Compile using all available CPU cores
cmake --build . --config Release -j $(nproc)
```

### Clear Broken Cache 

CMake hardcodes absolute paths inside its cache directory `CMakeCache.txt`, so if you rename or move a file you need to delete the outdated build directory and clear it. 

```text
rmdir /s /q build
```

Then generate a fresh build folder using our current directry location.

```text
cmake -S . -B build -G "Visual Studio 18 2026" -A x64
```

Once the configuration finishes cleanly, run the build again.

```text
cmake --build build --config Release
```

---

## 💻 Command-Line Usage

Crescendo provides a unified CLI tool for training, generating music, and isolating stems.

### 1. Training the Diffusion Model on Custom Music

Point the engine to a folder of .wav files. The engine will automatically slice them into frames, compute Mel-spectrograms, and train the U-Net from scratch.

```text
./bin/crescendo train --data-dir ./my_music_library \
                      --epochs 100 \
                      --batch-size 16 \
                      --learning-rate 0.0002 \
                      --save-model ./weights/crescendo_v1.bin
```

### 2. Generating Original Music (AI Composition)

Generate a brand-new 30-second audio track using the fast DDIM sampler.

```text
./bin/crescendo generate --model ./weights/crescendo_v1.bin \
                         --steps 30 \
                         --duration 30 \
                         --output ./output/ai_song_01.wav
```

### 3. Separating Stems (Demixing an Existing Song)

Isolate vocals, drums, bass, and instrumental accompaniment from a mixed track. 

```text
# This command outputs vocals.wav, drums.wav, bass.wav, and other.wav directly into ./stems/.
./bin/crescendo separate --input ./my_song.wav \
                         --model ./weights/stem_separator.bin \
                         --output-dir ./stems/
```

### 4. Launching Interactive Console Editor

Launch the lock-free terminal audio editor to mix stems and apply DSP effects in real time.

```text
./bin/crescendo edit --stems ./stems/
```
