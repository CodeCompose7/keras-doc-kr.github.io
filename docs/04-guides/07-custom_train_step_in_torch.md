---
layout: default
title: Customizing what happens in `fit()` with PyTorch
nav_order: 7
permalink: /guides/custom_train_step_in_torch/
parent: 개발자 가이드
---

* 원본 링크 : [https://keras.io/guides/custom_train_step_in_torch/](https://keras.io/guides/custom_train_step_in_torch/){:target="_blank"}
* 최종 수정일 : 2024-03-29

# Customizing what happens in `fit()` with PyTorch
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

**저자:** [fchollet](https://twitter.com/fchollet)  
**생성일:** 2023/06/27  
**최종편집일:** 2023/06/27  
**설명:** Overriding the training step of the Model class with PyTorch.

[Colab에서 보기](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/custom_train_step_in_torch.ipynb){: .btn .btn-blue }
[GitHub 소스](https://github.com/keras-team/keras-io/blob/master/guides/custom_train_step_in_torch.py){: .btn .btn-blue }

----