---
layout: default
title: Writing a training loop from scratch in JAX
nav_order: 8
permalink: /guides/writing_a_custom_training_loop_in_jax/
parent: 개발자 가이드
---

* 원본 링크 : [https://keras.io/guides/writing_a_custom_training_loop_in_jax/](https://keras.io/guides/writing_a_custom_training_loop_in_jax/){:target="_blank"}
* 최종 수정일 : 2024-03-29

# Writing a training loop from scratch in JAX
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

**저자:** [fchollet](https://twitter.com/fchollet)  
**생성일:** 2023/06/25  
**최종편집일:** 2023/06/25  
**설명:** Writing low-level training & evaluation loops in JAX.

[Colab에서 보기](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/writing_a_custom_training_loop_in_jax.ipynb){: .btn .btn-blue }
[GitHub 소스](https://github.com/keras-team/keras-io/blob/master/guides/writing_a_custom_training_loop_in_jax.py){: .btn .btn-blue }

----