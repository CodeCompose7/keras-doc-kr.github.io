---
layout: default
title: 코드 예제
nav_order: 6
permalink: /examples/
has_children: true
---

* 원본 링크 : [https://keras.io/examples/](https://keras.io/examples/){:target="_blank"}
* 최종 수정일 : 2024-04-22

# 코드 예제
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

코드 예제 (Code examples)
=============

우리의 코드 예제는 300줄 미만의 짧은 코드이며, 수직적 딥러닝 워크플로우에 대한 집중적인 데모입니다.

모든 예제는 Jupyter 노트북으로 작성되었으며, 별도의 설정이 필요 없고 클라우드에서 실행되는 호스팅 노트북 환경인 [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb)에서 클릭 한 번으로 실행할 수 있습니다. Google Colab에는 GPU 및 TPU 런타임이 포함되어 있습니다.

★
{: .label .label-purple .mx-1}
시작하기 좋은 예제

V3
{: .label .label-green .mx-1}
Keras 3 예제

[컴퓨터 비전]({% link docs/06-examples/01-vision.md %})
------------------------------------

### 이미지 분류

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[처음부터 이미지 분류 (Image classification from scratch)]({% link docs/06-examples/01-vision/01-image_classification_from_scratch.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[간단한 MNIST convnet (Simple MNIST convnet)]({% link docs/06-examples/01-vision/02-mnist_convnet.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[EfficientNet으로 하는 미세 조정을 통한 이미지 분류 (Image classification via fine-tuning with EfficientNet)]({% link docs/06-examples/01-vision/03-image_classification_efficientnet_fine_tuning.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[비전 트랜스포머로 이미지 분류 (Image classification with Vision Transformer)]({% link docs/06-examples/01-vision/04-image_classification_with_vision_transformer.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[어텐션 기반 심층 다중 인스턴스 학습(MIL)을 사용한 분류 (Classification using Attention-based Deep Multiple Instance Learning)]({% link docs/06-examples/01-vision/05-attention_mil_classification.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[최신 MLP 모델을 사용한 이미지 분류 (Image classification with modern MLP models)]({% link docs/06-examples/01-vision/06-mlp_image_classification.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[이미지 분류를 위한 모바일 친화적인 트랜스포머 기반 모델 (A mobile-friendly Transformer-based model for image classification)]({% link docs/06-examples/01-vision/07-mobilevit.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[TPU에서 폐렴 분류 (Pneumonia Classification on TPU)]({% link docs/06-examples/01-vision/08-xray_classification_with_tpus.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[컴팩트 컨볼루션 트랜스포머 (Compact Convolutional Transformers)]({% link docs/06-examples/01-vision/09-cct.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[ConvMixer로 이미지 분류 (Image classification with ConvMixer)]({% link docs/06-examples/01-vision/10-convmixer.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[EANet(External Attention Transformer)을 사용한 이미지 분류 (Image classification with EANet (External Attention Transformer))]({% link docs/06-examples/01-vision/11-eanet.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[인볼루션 신경망 (Involutional neural networks)]({% link docs/06-examples/01-vision/12-involution.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Perceiver로 이미지 분류 (Image classification with Perceiver)]({% link docs/06-examples/01-vision/13-perceiver_image_classification.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Reptile을 사용한 퓨샷 학습 (Few-Shot learning with Reptile)]({% link docs/06-examples/01-vision/14-reptile.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[SimCLR을 사용한 대조 사전 트레이닝을 사용한 반지도 이미지 분류 (Semi-supervised image classification using contrastive pretraining with SimCLR)]({% link docs/06-examples/01-vision/15-semisupervised_simclr.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Swin 트랜스포머를 사용한 이미지 분류 (Image classification with Swin Transformers)]({% link docs/06-examples/01-vision/16-swin_transformers.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[소규모 데이터 세트에 대해 비전 트랜스포머 트레이닝 (Train a Vision Transformer on small datasets)]({% link docs/06-examples/01-vision/17-vit_small_ds.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[어텐션이 없는 비전 트랜스포머 (A Vision Transformer without Attention)]({% link docs/06-examples/01-vision/18-shiftvit.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[글로벌 컨텍스트 비전 트랜스포머를 이용한 이미지 분류 (Image Classification using Global Context Vision Transformer)]({% link docs/06-examples/01-vision/19-image_classification_using_global_context_vision_transformer.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[BigTransfer(BiT)를 사용한 이미지 분류 (Image Classification using BigTransfer (BiT))]({% link docs/06-examples/01-vision/20-bit.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 이미지 세그멘테이션

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[U-Net과 유사한 아키텍처를 사용한 이미지 세그멘테이션 (Image segmentation with a U-Net-like architecture)]({% link docs/06-examples/01-vision/21-oxford_pets_image_segmentation.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[DeepLabV3+를 사용한 다중 클래스 시맨틱 세그멘테이션 (Multiclass semantic segmentation using DeepLabV3+)]({% link docs/06-examples/01-vision/22-deeplabv3_plus.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[BASNet을 사용한 매우 정확한 경계 세그멘테이션 (Highly accurate boundaries segmentation using BASNet)]({% link docs/06-examples/01-vision/23-basnet_segmentation.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Composable 완전 컨볼루션 네트워크를 사용한 이미지 세그멘테이션 (Image Segmentation using Composable Fully-Convolutional Networks)]({% link docs/06-examples/01-vision/24-fully_convolutional_network.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 객체 감지

V2
{: .label .label-yellow .mx-1}
[RetinaNet을 이용한 객체 감지 (Object Detection with RetinaNet)]({% link docs/06-examples/01-vision/25-retinanet.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[전이 학습을 통한 키포인트 감지 (Keypoint Detection with Transfer Learning)]({% link docs/06-examples/01-vision/26-keypoint_detection.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[비전 트랜스포머를 사용한 객체 감지 (Object detection with Vision Transformers)]({% link docs/06-examples/01-vision/27-object_detection_using_vision_transformer.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 3D

V3
{: .label .label-green .mx-1}
[CT 스캔의 3D 이미지 분류 (3D image classification from CT scans)]({% link docs/06-examples/01-vision/28-3D_image_classification.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[단안 깊이 추정 (Monocular depth estimation)]({% link docs/06-examples/01-vision/29-depth_estimation.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[NeRF를 사용한 3D 체적 렌더링 (3D volumetric rendering with NeRF)]({% link docs/06-examples/01-vision/30-nerf.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[PointNet을 사용한 포인트 클라우드 세그멘테이션 (Point cloud segmentation with PointNet)]({% link docs/06-examples/01-vision/31-pointnet_segmentation.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[PointNet을 사용한 포인트 클라우드 분류 (Point cloud classification)]({% link docs/06-examples/01-vision/32-pointnet.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### OCR

V3
{: .label .label-green .mx-1}
[캡챠 읽기를 위한 OCR 모델 (OCR model for reading Captchas)]({% link docs/06-examples/01-vision/33-captcha_ocr.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[손글씨 인식 (Handwriting recognition)]({% link docs/06-examples/01-vision/34-handwriting_recognition.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 이미지 강화

V3
{: .label .label-green .mx-1}
[이미지 노이즈 제거를 위한 컨볼루셔널 오토인코더 (Convolutional autoencoder for image denoising)]({% link docs/06-examples/01-vision/35-autoencoder.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[MIRNet을 사용한 저조도 이미지 향상 (Low-light image enhancement using MIRNet)]({% link docs/06-examples/01-vision/36-mirnet.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Efficient Sub-Pixel CNN을 사용한 이미지 초해상도 (Image Super-Resolution using an Efficient Sub-Pixel CNN)]({% link docs/06-examples/01-vision/37-super_resolution_sub_pixel.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[단일 이미지 초해상도를 위한 향상된 깊은 Residual 네트워크 (Enhanced Deep Residual Networks for single-image super-resolution)]({% link docs/06-examples/01-vision/38-edsr.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[저조도 이미지 향상을 위한 Zero-DCE (Zero-DCE for low-light image enhancement)]({% link docs/06-examples/01-vision/39-zero_dce.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 데이터 보강

V3
{: .label .label-green .mx-1}
[이미지 분류를 위한 CutMix 데이터 보강 (CutMix data augmentation for image classification)]({% link docs/06-examples/01-vision/40-cutmix.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[이미지 분류를 위한 MixUp 보강 (MixUp augmentation for image classification)]({% link docs/06-examples/01-vision/41-mixup.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[견고성 향상을 위한 이미지 분류를 위한 RandAugment (RandAugment for Image Classification for Improved Robustness)]({% link docs/06-examples/01-vision/42-randaugment.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 이미지 & 텍스트

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[이미지 캡션 (Image captioning)]({% link docs/06-examples/01-vision/43-image_captioning.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[듀얼 인코더를 이용한 자연어 이미지 검색 (Natural language image search with a Dual Encoder)]({% link docs/06-examples/01-vision/44-nl_image_search.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 비전 모델 해석 가능성(interpretability)

V3
{: .label .label-green .mx-1}
[Convnets이 학습한 내용 시각화 (Visualizing what convnets learn)]({% link docs/06-examples/01-vision/45-visualizing_what_convnets_learn.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[통합 그래디언트를 통한 모델 해석 가능성 (Model interpretability with Integrated Gradients)]({% link docs/06-examples/01-vision/46-integrated_gradients.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[비전 트랜스포머 표현 조사 (Investigating Vision Transformer representations)]({% link docs/06-examples/01-vision/47-probing_vits.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Grad-CAM 클래스 활성화 시각화 (Grad-CAM class activation visualization)]({% link docs/06-examples/01-vision/48-grad_cam.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 이미지 유사도 검색

V2
{: .label .label-yellow .mx-1}
[중복에 가까운 이미지 검색 (Near-duplicate image search)]({% link docs/06-examples/01-vision/49-near_dup_search.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[시맨틱 이미지 클러스터링 (Semantic Image Clustering)]({% link docs/06-examples/01-vision/50-semantic_image_clustering.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[대비 손실이 있는 Siamese 네트워크를 사용한 이미지 유사도 추정 (Image similarity estimation using a Siamese Network with a contrastive loss)]({% link docs/06-examples/01-vision/51-siamese_contrastive.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[삼중(triplet) 손실이 있는 Siamese 네트워크를 사용한 이미지 유사도 추정 (Image similarity estimation using a Siamese Network with a triplet loss)]({% link docs/06-examples/01-vision/52-siamese_network.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[이미지 유사도 검색을 위한 메트릭 학습 (Metric learning for image similarity search)]({% link docs/06-examples/01-vision/53-metric_learning.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[TensorFlow Similarity를 사용한 이미지 유사도 검색을 위한 메트릭 학습 (Metric learning for image similarity search using TensorFlow Similarity)]({% link docs/06-examples/01-vision/54-metric_learning_tf_similarity.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[NNCLR을 사용한 자기 지도 대조 학습 (Self-supervised contrastive learning with NNCLR)]({% link docs/06-examples/01-vision/55-nnclr.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 비디오

V3
{: .label .label-green .mx-1}
[CNN-RNN 아키텍처를 사용한 비디오 분류 (Video Classification with a CNN-RNN Architecture)]({% link docs/06-examples/01-vision/56-video_classification.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[컨볼루션 LSTM을 사용한 다음 프레임 비디오 예측 (Next-Frame Video Prediction with Convolutional LSTMs)]({% link docs/06-examples/01-vision/57-conv_lstm.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[트랜스포머를 사용한 비디오 분류 (Video Classification with Transformers)]({% link docs/06-examples/01-vision/58-video_transformers.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[비디오 비전 트랜스포머 (Video Vision Transformer)]({% link docs/06-examples/01-vision/59-vivit.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 성능 레시피

V3
{: .label .label-green .mx-1}
[트레이닝 성능 향상을 위한 그래디언트 중앙화 (Gradient Centralization for Better Training Performance)]({% link docs/06-examples/01-vision/60-gradient_centralization.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[비전 트랜스포머에서 토큰화 학습하기 (Learning to tokenize in Vision Transformers)]({% link docs/06-examples/01-vision/61-token_learner.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[지식 증류 (Knowledge Distillation)]({% link docs/06-examples/01-vision/62-knowledge_distillation.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[FixRes: 트레이닝-테스트 해상도 불일치 수정 (FixRes: Fixing train-test resolution discrepancy)]({% link docs/06-examples/01-vision/63-fixres.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[LayerScale을 사용한 클래스 어텐션 이미지 트랜스포머 (Class Attention Image Transformers with LayerScale)]({% link docs/06-examples/01-vision/64-cait.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[통합 어텐션으로 컨볼루션 네트워크 강화 (Augmenting convnets with aggregated attention)]({% link docs/06-examples/01-vision/65-patch_convnet.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[컴퓨터 비전에서 리사이즈 학습 (Learning to Resize)]({% link docs/06-examples/01-vision/66-learnable_resizer.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 기타

V2
{: .label .label-yellow .mx-1}
[AdaMatch를 통한 반지도 및 도메인 적응 (Semi-supervision and domain adaptation with AdaMatch)]({% link docs/06-examples/01-vision/67-adamatch.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Contrastive SSL을 위한 Barlow Twins (Barlow Twins for Contrastive SSL)]({% link docs/06-examples/01-vision/68-barlow_twins.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[지도를 통한 일관성 트레이닝 (Consistency training with supervision)]({% link docs/06-examples/01-vision/69-consistency_training.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[증류식 비전 트랜스포머 (Distilling Vision Transformers)]({% link docs/06-examples/01-vision/70-deit.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[초점 변조(Focal Modulation): 셀프 어텐션을 대체하는 (Focal Modulation: A replacement for Self-Attention)]({% link docs/06-examples/01-vision/71-focal_modulation_network.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[이미지 분류를 위한 Forward-Forward 알고리즘 사용 (Using the Forward-Forward Algorithm for Image Classification)]({% link docs/06-examples/01-vision/72-forwardforward.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[자동 인코더를 사용한 마스크 이미지 모델링 (Masked image modeling with Autoencoders)]({% link docs/06-examples/01-vision/73-masked_image_modeling.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[🤗 트랜스포머로 무엇이든 모델 세그먼트 (Segment Anything Model with 🤗Transformers)]({% link docs/06-examples/01-vision/74-sam.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[SegFormer와 Hugging Face 트랜스포머를 사용한 시맨틱 세그멘테이션 (Semantic segmentation with SegFormer and Hugging Face Transformers)]({% link docs/06-examples/01-vision/75-segformer.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[SimSiam을 이용한 자기 지도 대조 학습 (Self-supervised contrastive learning with SimSiam)]({% link docs/06-examples/01-vision/76-simsiam.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[지도 대조 학습 (Supervised Contrastive Learning)]({% link docs/06-examples/01-vision/77-supervised-contrastive-learning.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Recurrence와 트랜스포머의 만남 (When Recurrence meets Transformers)]({% link docs/06-examples/01-vision/78-temporal_latent_bottleneck.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[YOLOV8 및 KerasCV를 통한 효율적인 객체 감지 (Efficient Object Detection with YOLOV8 and KerasCV)]({% link docs/06-examples/01-vision/79-yolov8.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

* * *

[자연어 처리 (Natural Language Processing)]({% link docs/06-examples/02-nlp.md %})
---------------------------------------------

### 텍스트 분류

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[처음부터 텍스트 분류 (Text classification from scratch)]({% link docs/06-examples/02-nlp/01-text_classification_from_scratch.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Active 학습을 사용한 분류 리뷰 (Review Classification using Active Learning)]({% link docs/06-examples/02-nlp/02-active_learning_review_classification.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[FNet을 사용한 텍스트 분류 (Text Classification using FNet)]({% link docs/06-examples/02-nlp/03-fnet_classification_with_keras_nlp.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[대규모 다중 레이블 텍스트 분류 (Large-scale multi-label text classification)]({% link docs/06-examples/02-nlp/04-multi_label_classification.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[트랜스포머로 텍스트 분류 (Text classification with Transformer)]({% link docs/06-examples/02-nlp/05-text_classification_with_transformer.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[스위치 트랜스포머로 텍스트 분류 (Text classification with Switch Transformer)]({% link docs/06-examples/02-nlp/06-text_classification_with_switch_transformer.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[의사 결정 포레스트와 사전 트레이닝된 임베딩을 사용한 텍스트 분류 (Text classification using Decision Forests and pretrained embeddings)]({% link docs/06-examples/02-nlp/07-tweet-classification-using-tfdf.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[사전 트레이닝된 단어 임베딩 사용 (Using pre-trained word embeddings)]({% link docs/06-examples/02-nlp/08-pretrained_word_embeddings.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[IMDB에 대한 양방향 LSTM (Bidirectional LSTM on IMDB)]({% link docs/06-examples/02-nlp/09-bidirectional_lstm_imdb.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[KerasNLP 및 tf.distribute를 사용한 데이터 병렬 트레이닝 (Data Parallel Training with KerasNLP and tf.distribute)]({% link docs/06-examples/02-nlp/10-data_parallel_training_with_keras_nlp.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 기계 번역

V3
{: .label .label-green .mx-1}
[KerasNLP를 사용한 영어-스페인어 번역 (English-to-Spanish translation with KerasNLP)]({% link docs/06-examples/02-nlp/11-neural_machine_translation_with_keras_nlp.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[시퀀스-to-시퀀스 트랜스포머를 사용한 영어-스페인어 번역 (English-to-Spanish translation with a sequence-to-sequence Transformer)]({% link docs/06-examples/02-nlp/12-neural_machine_translation_with_transformer.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[문자 레벨 recurrent 시퀀스-to-시퀀스 모델 (Character-level recurrent sequence-to-sequence model)]({% link docs/06-examples/02-nlp/13-lstm_seq2seq.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 연관성 예측 (Entailment prediction)

V2
{: .label .label-yellow .mx-1}
[멀티모달 수반 (Multimodal entailment)]({% link docs/06-examples/02-nlp/14-multimodal_entailment.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 명명된 엔티티 인식

V3
{: .label .label-green .mx-1}
[트랜스포머를 사용한 명명된 엔티티 인식 (Named Entity Recognition using Transformers)]({% link docs/06-examples/02-nlp/15-ner_transformers.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### Sequence-to-sequence

V2
{: .label .label-yellow .mx-1}
[BERT를 사용한 텍스트 추출 (Text Extraction with BERT)]({% link docs/06-examples/02-nlp/16-text_extraction_with_bert.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[숫자 덧셈을 수행하기 위한 시퀀스-to-시퀀스 학습 (Sequence to sequence learning for performing number addition)]({% link docs/06-examples/02-nlp/17-addition_rnn.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 텍스트 유사도 검색

V3
{: .label .label-green .mx-1}
[KerasNLP를 사용한 시맨틱 유사성 (Semantic Similarity with KerasNLP)]({% link docs/06-examples/02-nlp/18-semantic_similarity_with_keras_nlp.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[BERT를 사용한 시맨틱 유사성 (Semantic Similarity with BERT)]({% link docs/06-examples/02-nlp/19-semantic_similarity_with_bert.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Siamese RoBERTa 네트워크를 사용한 문장 임베딩 (Sentence embeddings using Siamese RoBERTa-networks)]({% link docs/06-examples/02-nlp/20-sentence_embeddings_with_sbert.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 언어 모델링

V2
{: .label .label-yellow .mx-1}
[BERT를 사용한 엔드투엔드 마스크 언어 모델링 (End-to-end Masked Language Modeling with BERT)]({% link docs/06-examples/02-nlp/21-masked_language_modeling.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Hugging Face 트랜스포머로 BERT 사전 트레이닝하기 (Pretraining BERT with Hugging Face Transformers)]({% link docs/06-examples/02-nlp/22-pretraining_BERT.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 효율적인 매개변수 미세 조정

V3
{: .label .label-green .mx-1}
[LoRA가 있는 GPT-2의 효율적인 파라미터 미세 조정 (Parameter-efficient fine-tuning of GPT-2 with LoRA)]({% link docs/06-examples/02-nlp/23-parameter_efficient_finetuning_of_gpt2_with_lora.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 기타

V2
{: .label .label-yellow .mx-1}
[BART를 사용한 추상적 텍스트 요약 (Abstractive Text Summarization with BART)]({% link docs/06-examples/02-nlp/24-abstractive_summarization_with_bart.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[🤗 트랜스포머 및 TPU를 사용하여 처음부터 언어 모델 트레이닝하기 (Training a language model from scratch with 🤗 Transformers and TPUs)]({% link docs/06-examples/02-nlp/25-mlm_training_tpus.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[전이 학습으로 객관식 과제 (MultipleChoice Task with Transfer Learning)]({% link docs/06-examples/02-nlp/26-multiple_choice_task_with_transfer_learning.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Hugging Face 트랜스포머로 질문 답변하기 (Question Answering with Hugging Face Transformers)]({% link docs/06-examples/02-nlp/27-question_answering.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Hugging Face 트랜스포머를 사용한 추상적 요약 (Abstractive Summarization with Hugging Face Transformers)]({% link docs/06-examples/02-nlp/28-t5_hf_summarization.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

* * *

[구조화된 데이터]({% link docs/06-examples/03-structured_data.md %})
---------------------------------------------

### 구조화된 데이터 분류

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[FeatureSpace를 사용한 구조화된 데이터 분류 (Structured data classification with FeatureSpace)]({% link docs/06-examples/03-structured_data/01-structured_data_classification_with_feature_space.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[불균형 분류: 신용 카드 사기 탐지 (Imbalanced classification: credit card fraud detection)]({% link docs/06-examples/03-structured_data/02-imbalanced_classification.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[처음부터 구조화된 데이터 분류 (Structured data classification from scratch)]({% link docs/06-examples/03-structured_data/03-structured_data_classification_from_scratch.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[와이드, 딥, 크로스 네트워크를 통한 구조화된 데이터 학습 (Structured data learning with Wide, Deep, and Cross networks)]({% link docs/06-examples/03-structured_data/04-wide_deep_cross_networks.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Gated Residual 및 변수 선택 네트워크를 사용한 분류 (Classification with Gated Residual and Variable Selection Networks)]({% link docs/06-examples/03-structured_data/05-classification_with_grn_and_vsn.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[TensorFlow 의사 결정 포리스트를 사용한 분류 (Classification with TensorFlow Decision Forests)]({% link docs/06-examples/03-structured_data/06-classification_with_tfdf.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[신경 의사 결정 포레스트를 사용한 분류 (Classification with Neural Decision Forests)]({% link docs/06-examples/03-structured_data/07-deep_neural_decision_forests.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[TabTransformer를 사용한 구조화된 데이터 학습 (Structured data learning with TabTransformer)]({% link docs/06-examples/03-structured_data/08-tabtransformer.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 추천

V3
{: .label .label-green .mx-1}
[영화 추천을 위한 Collaborative 필터링 (Collaborative Filtering for Movie Recommendations)]({% link docs/06-examples/03-structured_data/09-collaborative_filtering_movielens.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[트랜스포머 기반 추천 시스템 (A Transformer-based recommendation system)]({% link docs/06-examples/03-structured_data/10-movielens_recommendations_transformers.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 기타

V2
{: .label .label-yellow .mx-1}
[FeatureSpace 고급 사용 사례 (FeatureSpace advanced use cases)]({% link docs/06-examples/03-structured_data/11-feature_space_advanced.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

* * *

[타임시리즈]({% link docs/06-examples/04-timeseries.md %})
-----------------------------------

### 타임시리즈 분류

★
{: .label .label-purple .mx-1}
V3
{: .label .label-green .mx-1}
[처음부터 시계열 분류 (Timeseries classification from scratch)]({% link docs/06-examples/04-timeseries/01-timeseries_classification_from_scratch.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[트랜스포머 모델을 사용한 시계열 분류 (Timeseries classification with a Transformer model)]({% link docs/06-examples/04-timeseries/02-timeseries_classification_transformer.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[행동 식별을 위한 뇌파 신호 분류 (Electroencephalogram Signal Classification for action identification)]({% link docs/06-examples/04-timeseries/03-eeg_signal_classification.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[결제 카드 사기 탐지를 위한 이벤트 분류 (Event classification for payment card fraud detection)]({% link docs/06-examples/04-timeseries/04-event_classification_for_payment_card_fraud_detection.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 이상 징후 탐지

V3
{: .label .label-green .mx-1}
[자동 인코더를 사용한 시계열 이상 탐지 (Timeseries anomaly detection using an Autoencoder)]({% link docs/06-examples/04-timeseries/05-timeseries_anomaly_detection.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

### 타임시리즈 예측

V3
{: .label .label-green .mx-1}
[그래프 신경망과 LSTM을 사용한 트래픽 예측 (Traffic forecasting using graph neural networks and LSTM)]({% link docs/06-examples/04-timeseries/06-timeseries_traffic_forecasting.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[날씨 예측을 위한 시계열 예측 (Timeseries forecasting for weather prediction)]({% link docs/06-examples/04-timeseries/07-timeseries_weather_forecasting.md %})
{: .d-inline .v-align-middle}

.
{: .lh-0 .my-0 .opacity-0}

* * *

[생성형 딥러닝]({% link docs/06-examples/05-generative.md %})
-------------------------------------------------

### 이미지 생성

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
[안정적인 확산 미세 조정 (Fine-tuning Stable Diffusion)]({% link docs/06-examples/05-generative/06-finetune_stable_diffusion.md %})
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
[적응형 판별자 증강을 통한 데이터 효율적 GAN (Data-efficient GANs with Adaptive Discriminator Augmentation)]({% link docs/06-examples/05-generative/12-gan_ada.md %})
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

V2
{: .label .label-yellow .mx-1}
[Real NVP를 사용한 밀도 추정 (Density estimation using Real NVP)]({% link docs/06-examples/05-generative/27-real_nvp.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

* * *

[오디오 데이터]({% link docs/06-examples/06-audio.md %})
------------------------------

### 음성 인식

V3
{: .label .label-green .mx-1}
[트랜스포머를 통한 자동 음성 인식 (Automatic Speech Recognition with Transformer)]({% link docs/06-examples/06-audio/01-transformer_asr.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

### 기타

V2
{: .label .label-yellow .mx-1}
[CTC를 사용한 자동 음성 인식 (Automatic Speech Recognition using CTC)]({% link docs/06-examples/06-audio/02-ctc_asr.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[특징 매칭을 사용한 MelGAN 기반 스펙트로그램 반전 (MelGAN-based spectrogram inversion using feature matching)]({% link docs/06-examples/06-audio/03-melgan_spectrogram_inversion.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[화자 인식 (Speaker Recognition)]({% link docs/06-examples/06-audio/04-speaker_recognition_using_cnn.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[전이 학습을 사용한 영어 화자 억양 인식 (English speaker accent recognition using Transfer Learning)]({% link docs/06-examples/06-audio/05-uk_ireland_accent_recognition.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Hugging Face 트랜스포머를 사용한 오디오 분류 (Audio Classification with Hugging Face Transformers)]({% link docs/06-examples/06-audio/06-wav2vec2_audiocls.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

* * *

[Reinforcement Learning]({% link docs/06-examples/07-rl.md %})
---------------------------------------

### RL 알고리즘

V3
{: .label .label-green .mx-1}
[Actor Critic Method]({% link docs/06-examples/07-rl/01-actor_critic_cartpole.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Proximal Policy Optimization]({% link docs/06-examples/07-rl/02-ppo_cartpole.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Deep Q-Learning for Atari Breakout]({% link docs/06-examples/07-rl/03-deep_q_network_breakout.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

### 기타

V2
{: .label .label-yellow .mx-1}
[Deep Deterministic Policy Gradient (DDPG)]({% link docs/06-examples/07-rl/04-ddpg_pendulum.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

* * *

[Graph Data]({% link docs/06-examples/08-graph.md %})
------------------------------

### 그래프 데이터 (Graph Data)

V2
{: .label .label-yellow .mx-1}
[Graph attention network (GAT) for node classification]({% link docs/06-examples/08-graph/01-gat_node_classification.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Node Classification with Graph Neural Networks]({% link docs/06-examples/08-graph/02-gnn_citations.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Message-passing neural network (MPNN) for molecular property prediction]({% link docs/06-examples/08-graph/03-mpnn-molecular-graphs.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Graph representation learning with node2vec]({% link docs/06-examples/08-graph/04-node2vec_movielens.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

* * *

[Quick Keras Recipes]({% link docs/06-examples/09-keras_recipes.md %})
-----------------------------------------------

### 서비스

V3
{: .label .label-green .mx-1}
[Serving TensorFlow models with TFServing]({% link docs/06-examples/09-keras_recipes/01-tf_serving.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

### Keras 사용 팁

V3
{: .label .label-green .mx-1}
[Keras debugging tips]({% link docs/06-examples/09-keras_recipes/02-debugging_tips.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Customizing the convolution operation of a Conv2D layer]({% link docs/06-examples/09-keras_recipes/03-subclassing_conv_layers.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Trainer pattern]({% link docs/06-examples/09-keras_recipes/04-trainer_pattern.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Endpoint layer pattern]({% link docs/06-examples/09-keras_recipes/05-endpoint_layer_pattern.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Reproducibility in Keras Models]({% link docs/06-examples/09-keras_recipes/06-reproducibility_recipes.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Writing Keras Models With TensorFlow NumPy]({% link docs/06-examples/09-keras_recipes/07-tensorflow_numpy_models.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Simple custom layer example: Antirectifier]({% link docs/06-examples/09-keras_recipes/08-antirectifier.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Packaging Keras models for wide distribution using Functional Subclassing]({% link docs/06-examples/09-keras_recipes/09-packaging_keras_models_for_wide_distribution.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

### ML 모범 사례

V3
{: .label .label-green .mx-1}
[Estimating required sample size for model training]({% link docs/06-examples/09-keras_recipes/10-sample_size_estimate.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Memory-efficient embeddings for recommendation systems]({% link docs/06-examples/09-keras_recipes/11-memory_efficient_embeddings.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V3
{: .label .label-green .mx-1}
[Creating TFRecords]({% link docs/06-examples/09-keras_recipes/12-creating_tfrecords.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

### 기타

V2
{: .label .label-yellow .mx-1}
[Approximating non-Function Mappings with Mixture Density Networks]({% link docs/06-examples/09-keras_recipes/13-approximating_non_function_mappings.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Probabilistic Bayesian Neural Networks]({% link docs/06-examples/09-keras_recipes/14-bayesian_neural_networks.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Knowledge distillation recipes]({% link docs/06-examples/09-keras_recipes/15-better_knowledge_distillation.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[Evaluating and exporting scikit-learn metrics in a Keras callback]({% link docs/06-examples/09-keras_recipes/16-sklearn_metric_callbacks.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

V2
{: .label .label-yellow .mx-1}
[How to train a Keras model on TFRecord files]({% link docs/06-examples/09-keras_recipes/17-tfrecord.md %})
{: .d-inline .v-align-middle}
.
{: .lh-0 .my-0 .opacity-0}

* * *

Adding a new code example
-------------------------

We welcome new code examples! Here are our rules:

*   They should be shorter than 300 lines of code (comments may be as long as you want).
*   They should demonstrate modern Keras best practices.
*   They should be substantially different in topic from all examples listed above.
*   They should be extensively documented & commented.

New examples are added via Pull Requests to the [keras.io repository](https://github.com/keras-team/keras-io). They must be submitted as a `.py` file that follows a specific format. They are usually generated from Jupyter notebooks. See the [`tutobooks` documentation](https://github.com/keras-team/keras-io/blob/master/README.md) for more details.

If you would like to convert a Keras 2 example to Keras 3, please open a Pull Request to the [keras.io repository](https://github.com/keras-team/keras-io).
