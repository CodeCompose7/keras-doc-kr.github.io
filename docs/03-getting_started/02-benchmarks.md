---
layout: default
title: Keras 3 벤치마크
nav_order: 2
permalink: /getting_started/benchmarks/
parent: 시작하기
---

* 원본 링크 : [https://keras.io/getting_started/benchmarks/](https://keras.io/getting_started/benchmarks/){:target="_blank"}
* 최종 수정일 : 2024-03-29

# Keras 3 벤치마크
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Keras 3 벤치마크

We benchmark the three backends of Keras 3 ([TensorFlow](https://tensorflow.org/), [JAX](https://jax.readthedocs.io/en/latest/), [PyTorch](https://pytorch.org/)) alongside native PyTorch implementations ([HuggingFace](https://huggingface.co/) and [Meta Research](https://github.com/facebookresearch/)) and alongside Keras 2 with TensorFlow. Find code and setup details for reproducing our results [here](https://github.com/haifeng-jin/keras-benchmarks/tree/v0.0.3).

모델
------

We chose a set of popular computer vision and natural language processing models for both generative and non-generative AI tasks. See the table below for our selections.

**Table 1**: Models used in benchmarking.

|     | Non-Generative      | Generative             |
|-----|---------------------|------------------------|
| CV  | SegmentAnything[^1] | StableDiffusion[^2]    |
| NLP | BERT[^3]            | Gemma[^4], Mistral[^5] |

We leveraged pre-existing implementations from KerasCV and KerasNLP for the Keras versions of the models. For native PyTorch, we opted for the most popular options online:

*   BERT, Gemma, Mistral from HuggingFace Transformers
*   StableDiffusion from HuggingFace Diffusers
*   SegmentAnything from the original PyTorch implementation by Meta Research

We'll refer to this group as "Native PyTorch" in contrast to Keras 3 with PyTorch backend.

We employed synthetic data for all benchmarks. We used `bfloat16` precision for all LLM training and inferencing, and LoRA[^6] for all LLM training (fine-tuning). Based on the recommendations of the PyTorch team, we used `torch.compile(model, mode="reduce-overhead")` with native PyTorch implementations (with the exception of Gemma training and Mistral training due to incompatibility).

To measure out-of-the-box performance, we use high-level APIs (e.g. `Trainer()` from HuggingFace, plain PyTorch training loops and Keras `model.fit()`) with as little configuration as possible. Note that this is quite different from measuring an optimized implementation for a particular hardware/framework/model combination.

하드웨어
--------

All benchmarks are done with a single NVIDIA A100 GPU with 40GB of GPU memory on a Google Cloud Compute Engine of machine type `a2-highgpu-1g` with 12 vCPUs and 85GB host memory.

결과
-------

Table 2 displays benchmarking results in milliseconds per step. Each step involves training or predicting on a single data batch. Results are averaged over 100 steps, excluding the first, which includes model creation and compilation overhead.

For fair comparison, we use the same batch size across frameworks if it is the same model and task (fit or predict). However, for different models and tasks, due to their different sizes and architectures, we use different batch sizes to avoid either running out of memory (too large) or under GPU utilization (too small). A too small batch size would also unfairly make PyTorch look slow due to its Python overhead.

For large language models (Gemma and Mistral), we also used the same batch size since they are the same model type with similar number of parameters (7B). We also benchmarked text generation with batch size equal to 1 since it is widely requested by the users.

**Table 2**: Benchmarking results. The speed is measured in ms/step. Lower is better.


|  | Batch size | Native PyTorch | Keras 2 (TensorFlow) | Keras 3 (TensorFlow) | Keras 3  (JAX) | Keras 3 (PyTorch) | Keras 3 (best) |
|--|--|--|--|--|--|--|--|
| **SegmentAnything (fit)** | 1 | 1,233.25 | 386.93 | **355.25** | 361.69 | 1,388.87 | **355.25** |
| **SegmentAnything (predict)** | 4 | 1,476.87 | 1,859.27 | 438.50 | **376.34** | 1,720.96 | **376.34** |
| **Stable Diffusion (fit)** | 8 | 396.64 | 1,023.21 | 392.24 | **391.21** | 823.44 | **391.21** |
| **Stable Diffusion (predict)** | 13 | 759.05 | 649.71 | **616.04** | 627.27 | 1,337.17 | **616.04** |
| **BERT (fit)** | 32 | 214.73 | 486.00 | **214.49** | 222.37 | 808.68 | **214.49** |
| **BERT (predict)** | 256 | 739.46 | 470.12 | 466.01 | **418.72** | 1,865.98 | **865.29** |
| **Gemma (fit)** | 8 | 253.95 | NA | **232.52** | 273.67 | 525.15 | **232.52** |
| **Gemma (generate)** | 32 | 2,735.18 | NA | 1,134.91 | **1,128.21** | 7,952.67\* | **1,128.21** |
| **Gemma (generate)** | 1 | 1,618.85 | NA | 758.57 | **703.46** | 7,649.40\* | **703.46** |
| **Mistral (fit)** | 8 | 217.56 | NA | **185.92** | 213.22 | 452.12 | **185.92** |
| **Mistral (generate)** | 32 | 1,633.50 | NA | 966.06 | **957.25** | 10,932.59\* | **957.25** |
| **Mistral (generate)** | 1 | 1,554.79 | NA | 743.28 | **679.30** | 11,054.67\* | **679.30** |

\* _LLM inference with the PyTorch backend is abnormally slow at this time because KerasNLP uses static sequence padding, unlike HuggingFace. This will be addressed soon._

토론
----------

### 주요 발견 1: "최고의" 백엔드는 없습니다.

Each of the three backends of Keras offers unique strengths. Crucially, from a performance standpoint, there's no single backend that consistently outpaces the others. The fastest backend often depends on your specific model architecture.

This underscores the value of framework optionality when chasing optimal performance. Keras 3 empowers you to seamlessly switch backends, ensuring you find the ideal match for your model.

### 주요 발견 2: Keras 3은 참조 PyTorch 구현보다 지속적으로 빠릅니다.

The following figure compares the best-performing Keras 3 backend for each model with the corresponding reference native PyTorch implementation. We calculated the throughput (steps/ms) increase of Keras 3 over native PyTorch from Table 2. A 100% increase indicates Keras 3 is twice as fast, while 0% means both frameworks perform equally.

![Figure 1](https://i.imgur.com/03owEcn.png)

**Figure 1**: Keras 3 speedup over PyTorch measured in throughput (steps/ms)

Keras 3 with the best-performing backend outperformed the reference native PyTorch implementations for all the models. Notably, 5 out of 10 tasks demonstrated speedups exceeding 50%, with a maximum speedup of 290%.

### 주요 발견 3: Keras 3는 동급 최고의 "기본(out-of-the-box)" 성능을 제공합니다.

All Keras model implementations benchmarked here are plain implementations without any custom performance optimizations: they represent "out-of-the-box performance", the kind of performance that any Keras user should expect for their own models. While, for native PyTorch implementations, more performance optimizations are expected on the user's side.

Besides the numbers we shared above, we observed a significant performance boost (over 100%) in HuggingFace Diffusers' StableDiffusion inferencing between versions `0.3.0` and `0.25.0` (the version used in our benchmarks). Gemma in HuggingFace Transformers also has a significant improvement from `4.38.1` to `4.38.2` (the version used in our benchmarks). These performance improvements underscore HuggingFace's dedicated engineering efforts toward performance optimization.

Conversely, consider a less manually-optimized model like SegmentAnything, where we used the implementation provided by the research authors. Here, the performance gap compared to Keras is wider than most other models.

The takeaway here is that Keras offers exceptional out-of-the-box performance. You don't have to know all the tricks to make your model run faster.

### 주요 발견 4: Keras 3은 Keras 2보다 빠릅니다.

We also calculated the throughput (steps/ms) increase of Keras 3 (using its best-performing backend) over Keras 2 with TensorFlow from Table 1. Results are shown in the following figure.

![Figrue 2](https://i.imgur.com/jPncf0F.png)

**Figure 2**: Keras 3 speedup over Keras 2 measured in throughput (steps/ms)

Keras 3 consistently outperformed Keras 2 across all benchmarked models, with substantial speed increases in many cases. SegmentAnything inference saw a remarkable 380% boost, StableDiffusion training throughput increased by over 150%, and BERT training throughput rose by over 100%.

Importantly, you would still see a performance boost even if you simply upgrade to Keras 3 and continue using the TensorFlow backend. This is mainly because Keras 2 uses more TensorFlow fused ops directly, which may be sub-optimal for XLA compilation in certain use cases.

결론
-----------

Framework performance depends heavily on the specific model. Keras 3 empowers you to select the fastest framework for your task – an option almost always to outperform both Keras 2 and reference PyTorch implementations. Importantly, Keras 3 models deliver excellent out-of-the-box performance without requiring complex, low-level optimizations.

참조
----------

[^1]: Kirillov, Alexander, et al. "Segment anything." ICCV (2023).

[^2]: Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." CVPR (2022).

[^3]: Kenton, Jacob, et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL (2019).

[^4]: Banks, Jeanine, et al. "Gemma: Introducing new state-of-the-art open models." The Keyword, Google (2024).

[^5]: Jiang, Albert Q., et al. "Mistral 7B." arXiv preprint arXiv:2310.06825 (2023).

[^6]: Hu, Edward J., et al. "Lora: Low-rank adaptation of large language models." ICLR (2022).