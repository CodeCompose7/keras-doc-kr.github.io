---
layout: default
title: Multi-GPU distributed training with JAX
nav_order: 15
permalink: /guides/distributed_training_with_jax/
parent: 개발자 가이드
---

* 원본 링크 : [https://keras.io/guides/distributed_training_with_jax/](https://keras.io/guides/distributed_training_with_jax/){:target="_blank"}
* 최종 수정일 : 2024-03-29

# Multi-GPU distributed training with JAX
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

**저자:** [fchollet](https://twitter.com/fchollet)  
**생성일:** 2023/07/11  
**최종편집일:** 2023/07/11  
**설명:** Guide to multi-GPU/TPU training for Keras models with JAX.

[Colab에서 보기](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/distributed_training_with_jax.ipynb){: .btn .btn-blue }
[GitHub 소스](https://github.com/keras-team/keras-io/blob/master/guides/distributed_training_with_jax.py){: .btn .btn-blue }

----