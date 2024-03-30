---
layout: default
title: Tune hyperparameters in your custom training loop
nav_order: 3
permalink: /guides/keras_tuner/custom_tuner/
parent: KerasTuner
grand_parent: 개발자 가이드
---

* 원본 링크 : [https://keras.io/guides/keras_tuner/custom_tuner/](https://keras.io/guides/keras_tuner/custom_tuner/){:target="_blank"}
* 최종 수정일 : 2024-03-29

# Tune hyperparameters in your custom training loop
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

**저자:** Tom O'Malley, Haifeng Jin  
**생성일:** 2019/10/28  
**최종편집일:** 2022/01/12  
**설명:** Use `HyperModel.fit()` to tune training hyperparameters (such as batch size).

[Colab에서 보기](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_tuner/custom_tuner.ipynb){: .btn .btn-blue }
[GitHub 소스](https://github.com/keras-team/keras-io/blob/master/guides/keras_tuner/custom_tuner.py){: .btn .btn-blue }

```shell
!pip install keras-tuner -q
```

----