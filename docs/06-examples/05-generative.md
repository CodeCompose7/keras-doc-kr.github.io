---
layout: default
title: 생성형 딥러닝
nav_order: 5
permalink: /examples/generative/
parent: 코드 예제
has_children: true
---

* 원본 링크 : [https://keras.io/examples/generative/](https://keras.io/examples/generative/){:target="_blank"}
* 최종 수정일 : 2024-04-02

# 생성형 딥러닝 (Generative Deep Learning)
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

### 이미지 생성
{: #image-generation}
<!-- ### Image generation -->

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[노이즈 제거 확산 암시적 모델 (Denoising Diffusion Implicit Models)]({% link docs/06-examples/05-generative/01-ddim.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[안정적인 Diffusion으로 잠재된 공간 걷기 (A walk through latent space with Stable Diffusion)]({% link docs/06-examples/05-generative/02-random_walks_with_stable_diffusion.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[DreamBooth]({% link docs/06-examples/05-generative/03-dreambooth.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[노이즈 제거 확산 확률론적 모델 (Denoising Diffusion Probabilistic Models)]({% link docs/06-examples/05-generative/04-ddpm.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[텍스트 반전을 통해 StableDiffusion의 새로운 개념 가르치기 (Teach StableDiffusion new concepts via Textual Inversion)]({% link docs/06-examples/05-generative/05-fine_tune_via_textual_inversion.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Stable Diffusion 미세 조정]({% link docs/06-examples/05-generative/06-finetune_stable_diffusion.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[변형 자동인코더 (Variational AutoEncoder)]({% link docs/06-examples/05-generative/07-vae.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Model.train\_step을 오버라이딩 하는 GAN (GAN overriding Model.train\_step)]({% link docs/06-examples/05-generative/08-dcgan_overriding_train_step.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[`Model.train_step`을 오버라이딩 하는 WGAN-GP (WGAN-GP overriding Model.train\_step)]({% link docs/06-examples/05-generative/09-wgan_gp.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[조건부 GAN (Conditional GAN)]({% link docs/06-examples/05-generative/10-conditional_gan.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[CycleGAN]({% link docs/06-examples/05-generative/11-cyclegan.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[적응형 판별자 보강을 통한 데이터 효율적 GAN (Data-efficient GANs with Adaptive Discriminator Augmentation)]({% link docs/06-examples/05-generative/12-gan_ada.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Deep Dream]({% link docs/06-examples/05-generative/13-deep_dream.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[조건부 이미지 생성을 위한 GauGAN (GauGAN for conditional image generation)]({% link docs/06-examples/05-generative/14-gaugan.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[PixelCNN]({% link docs/06-examples/05-generative/15-pixelcnn.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[StyleGAN으로 얼굴 이미지 생성 (Face image generation with StyleGAN)]({% link docs/06-examples/05-generative/16-stylegan.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[벡터화된 변형 자동 인코더 (Vector-Quantized Variational Autoencoders)]({% link docs/06-examples/05-generative/17-vq_vae.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

### 스타일 전이
{: #style-transfer}
<!-- ### Style transfer -->

V3
{: .label .label-green .mx-1}
[신경 스타일 전송 (Neural style transfer)]({% link docs/06-examples/05-generative/18-neural_style_transfer.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[AdaIN을 사용한 신경 스타일 전송 (Neural Style Transfer with AdaIN)]({% link docs/06-examples/05-generative/19-adain.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

### 텍스트 생성
{: #text-generation}
<!-- ### Text generation -->

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[KerasNLP를 사용한 GPT2 텍스트 생성 (GPT2 Text Generation with KerasNLP)]({% link docs/06-examples/05-generative/20-gpt2_text_generation_with_kerasnlp.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[KerasNLP로 처음부터 GPT 텍스트 생성하기 (GPT text generation from scratch with KerasNLP)]({% link docs/06-examples/05-generative/21-text_generation_gpt.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[미니어처 GPT로 텍스트 생성 (Text generation with a miniature GPT)]({% link docs/06-examples/05-generative/22-text_generation_with_miniature_gpt.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[LSTM을 사용한 문자 레벨 텍스트 생성 (Character-level text generation with LSTM)]({% link docs/06-examples/05-generative/23-lstm_character_level_text_generation.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[FNet을 사용한 텍스트 생성 (Text Generation using FNet)]({% link docs/06-examples/05-generative/24-text_generation_fnet.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

### 그래프 생성
{: #graph-generation}
<!-- ### Graph generation -->

V2
{: .label .label-yellow .mx-1}
[VAE를 사용한 약물 분자 생성 (Drug Molecule Generation with VAE)]({% link docs/06-examples/05-generative/25-molecule_generation.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[작은 분자 그래프 생성을 위한 R-GCN이 포함된 WGAN-GP (WGAN-GP with R-GCN for the generation of small molecular graphs)]({% link docs/06-examples/05-generative/26-wgan-graphs.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

### 기타
{: #other}
<!-- ### Other -->

V2
{: .label .label-yellow .mx-1}
[Real NVP를 사용한 밀도 추정 (Density estimation using Real NVP)]({% link docs/06-examples/05-generative/27-real_nvp.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}
