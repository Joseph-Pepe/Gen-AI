---

## 🏗️ Phase 4: Generative Neural Network Backbone

The `crescendo::models` library provides the generative architecture responsible for learning musical structure, rhythm, melody, and instrument separation. 

Built from scratch in ISO C++26 without external machine learning dependencies, this phase implements a **2D Convolutional U-Net with Multi-Head Self-Attention**. To achieve hardware-accelerated processing speeds, spatial convolutions bypass slow scalar loops by utilizing an **`im2col` (Image-to-Column) transformation** coupled with Phase 3 SIMD vector kernels.

### ✨ Key Features

* **SIMD Convolutional Engine (`convolution2d.cppm`):** Extracts sliding 2D spatial patches into contiguous column matrices via `im2col`, executing convolutions and transposed convolutions as single AVX2/NEON Fused Multiply-Add matrix multiplications.
* **Spatial Group Normalization (`groupnorm.cppm`):** Divides channel feature maps into discrete groups to compute mean and variance independent of batch size, stabilizing small-batch audio training.
* **Multi-Head Self-Attention (`attention.cppm`):** Flattens 2D time-frequency grids into token sequences to evaluate global acoustic dependencies, allowing the model to maintain rhythmic coherence and repeating chorus structures across extended audio durations.
* **Skip-Connected U-Net (`unet.cppm`):** Combines strided downsampling encoders with nearest-neighbor bilinear upsampling decoders, concatenating intermediate feature maps across skip connections to preserve fine-grained high-frequency timbre.

### 🧮 Mathematical Foundation

#### 1. Scaled Dot-Product Attention
Given a tokenized spectrogram sequence $X \in \mathbb{R}^{L \times C}$ (where $L = H \times W$), the engine computes Query, Key, and Value projection matrices to evaluate attention across $h$ independent heads:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

#### 2. Spatial Group Normalization
For a channel group $G$ containing $C_G$ feature maps across spatial dimensions $H \times W$, activations are normalized and scaled via affine parameters $\gamma$ and $\beta$:

$$\hat{x}_i = \frac{x_i - \mu_G}{\sqrt{\sigma_G^2 + \epsilon}} \cdot \gamma + \beta$$

### 🚀 Usage Example

```cpp
import std;
import crescendo.tensor.core;
import crescendo.models.unet;

using namespace crescendo::tensor;
using namespace crescendo::models;

// 1. Instantiate the Generative U-Net Backbone
UNet2D<float> unet;

// 2. Ingest a 2D Log-Mel Spectrogram from Phase 2 [Batch, Channels, Mel-Bins, Time-Frames]
auto mel_spectrogram = Tensor<float>::random_normal({1, 1, 80, 80}, 0.0f, 0.5f);

// 3. Execute Forward Pass through Encoder, Attention Bottleneck, and Decoder
auto generated_spectrogram = unet.forward(mel_spectrogram);

// Verify output dimensions match input target
const auto& shape = generated_spectrogram.shape();
std::println("Generated Grid Dimensions: [{}, {}, {}, {}]", shape[0], shape[1], shape[2], shape[3]);
```