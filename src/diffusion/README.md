---

## 🌪️ Phase 5: Diffusion Noise Scheduler & DDIM Fast Sampler

The `crescendo::diffusion` library implements the generative scheduling engine responsible for synthesis loops. Rather than relying on traditional 1,000-step Markovian sampling paths which slow down CPU execution, this engine uses a deterministic **Denoising Diffusion Implicit Models (DDIM)** framework to perform high-fidelity audio synthesis in a fraction of the time.

### ✨ Key Features

* **Custom Vectorized Box-Muller Engine (`noise.cppm`):** Generates true standard normal Gaussian distributions $\mathcal{N}(0, I)$ directly in contiguous layout, avoiding the stack allocation overhead of runtime standard library wrappers.
* **Non-Linear Cosine Scheduler (`scheduler.cppm`):** Implements a continuous variance schedule to control noise injection, maintaining low-frequency acoustic properties during early degradation phases.
* **SIMD Blending Intrinsic Kernels:** Uses AVX2 `_mm256_fmadd_ps` and ARM NEON `vmlaq_f32` vectors to handle full forward tensor degradation loops in microsecond intervals.
* **Deterministic DDIM Fast Sampler (`ddim.cppm`):** Reformulates reverse trajectory math into an explicit ordinary differential equation, compressing total generation timelines from 1,000 down to **20 to 50 inference steps**.

### 🧮 Mathematical Foundation

#### 1. Forward Trajectory Degradation
Clean structural spectrogram tensors ($x_0$) are modified into partial noise shapes ($x_t$) at arbitrary training timesteps ($t$) using precalculated cumulative variance boundaries:

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

#### 2. Deterministic Reverse Update Loop
At each fast sampling checkpoint ($t$), the engine extracts the current predicted noise vector ($\epsilon_\theta$) from the U-Net backbone to compute the next historical state vector ($x_{t-\Delta t}$) without stochastic variance:

$$x_{t-\Delta t} = \sqrt{\bar{\alpha}_{t-\Delta t}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-\Delta t}} \cdot \epsilon_\theta(x_t, t)$$

### 🚀 Usage Example

```cpp
import std;
import crescendo.tensor.core;
import crescendo.models.unet;
import crescendo.diffusion.noise;
import crescendo.diffusion.scheduler;
import crescendo.diffusion.ddim;

using namespace crescendo::tensor;
using namespace crescendo::models;
using namespace crescendo::diffusion;

// 1. Initialize core diffusion modules
NormalGenerator<float> noise_engine;
CosineScheduler<float> variance_scheduler(1000);
DDIMSampler<float> ddim_sampler(variance_scheduler);
UNet2D<float> trained_unet;

// 2. Sample random Gaussian white noise seed tensor
auto latent_noise = noise_engine.generate({1, 1, 80, 80});

// 3. Run the deterministic fast sampler for 30 steps to construct a new spectrogram
auto generated_spectrogram = ddim_sampler.sample(trained_unet, latent_noise, 30);
```