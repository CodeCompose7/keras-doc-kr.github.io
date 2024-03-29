---
layout: default
title: Distributed hyperparameter tuning
nav_order: 2
permalink: /guides/keras_tuner/distributed_tuning/
parent: 하이퍼파라미터 튜닝
grand_parent: 개발자 가이드
---

* 원본 링크 : [https://keras.io/guides/keras_tuner/distributed_tuning/](https://keras.io/guides/keras_tuner/distributed_tuning/){:target="_blank"}
* 최종 수정일 : 2024-03-29

# Distributed hyperparameter tuning
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

**저자:** Tom O'Malley, Haifeng Jin  
**생성일:** 2019/10/24  
**최종편집일:** 2021/06/02  
**설명:** Tuning the hyperparameters of the models with multiple GPUs and multiple machines.

[Colab에서 보기](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_tuner/distributed_tuning.ipynb){: .btn .btn-blue }
[GitHub 소스](https://github.com/keras-team/keras-io/blob/master/guides/keras_tuner/distributed_tuning.py){: .btn .btn-blue }

```shell
!pip install keras-tuner -q
```

----