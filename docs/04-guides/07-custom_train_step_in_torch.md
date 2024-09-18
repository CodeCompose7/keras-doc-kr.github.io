---
layout: default
title: PyTorch에서 `fit()`의 동작을 커스터마이즈하기
nav_order: 7
permalink: /guides/custom_train_step_in_torch/
parent: 개발자 가이드
---

* 원본 링크 : [https://keras.io/guides/custom_train_step_in_torch/](https://keras.io/guides/custom_train_step_in_torch/){:target="_blank"}
* 최종 수정일 : 2024-09-18

# PyTorch에서 `fit()`의 동작을 커스터마이즈하기 (Customizing what happens in `fit()` with PyTorch)
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

**저자:** [fchollet](https://twitter.com/fchollet)  
**생성일:** 2023/06/27  
**최종편집일:** 2024/08/01  
**설명:** PyTorch의 Model 클래스에서 트레이닝 스텝 오버라이딩하기

[Colab에서 보기](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/custom_train_step_in_torch.ipynb){: .btn .btn-blue }
[GitHub 소스](https://github.com/keras-team/keras-io/blob/master/guides/custom_train_step_in_torch.py){: .btn .btn-blue }

----

## 소개
{: #introduction}
<!-- ## Introduction -->

지도 학습을 할 때, `fit()`을 사용하면 모든 것이 원활하게 작동합니다.

모든 작은 부분까지 제어해야 할 때는, 
처음부터 완전히 직접 트레이닝 루프를 작성할 수 있습니다.

하지만 커스텀 트레이닝 알고리즘이 필요하지만, 
콜백, 빌트인 분산 지원, 또는 스텝 퓨징(step fusing)과 같은, 
`fit()`의 편리한 기능을 여전히 활용하고 싶다면 어떻게 해야 할까요?

Keras의 핵심 원칙 중 하나는 **복잡성의 점진적 공개**입니다. 
낮은 레벨의 워크플로우로 점진적으로 진입할 수 있어야 합니다. 
높은 레벨의 기능이 정확히 당신의 사용 사례와 맞지 않더라도, 갑자기 막히지 않아야 합니다. 
높은 레벨의 편의성을 유지하면서도 작은 세부사항에 대한 더 많은 제어를 얻을 수 있어야 합니다.

`fit()`이 수행하는 작업을 커스터마이즈해야 할 때는, 
**`Model` 클래스의 트레이닝 스텝 함수를 오버라이드**해야 합니다. 
이 함수는 `fit()`이 각 데이터 배치마다 호출하는 함수입니다. 
그러면 평소처럼 `fit()`을 호출할 수 있고, 여러분의 학습 알고리즘이 실행될 것입니다.

이 패턴은 Functional API로 모델을 구축하는 것을 방해하지 않는다는 점에 유의하세요. 
`Sequential` 모델이든, Functional API 모델이든, 서브클래싱된 모델이든, 
이 작업을 수행할 수 있습니다.

이것이 어떻게 작동하는지 살펴보겠습니다.

------------------------------------------------------------------------

## 셋업
{: #setup}
<!-- ## Setup -->

```python
import os

# 이 가이드는 torch 백엔드에서만 실행할 수 있습니다.
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
from keras import layers
import numpy as np
```

------------------------------------------------------------------------

## 첫 번째 간단한 예제
{: #a-first-simple-example}
<!-- ## A first simple example -->

간단한 예제부터 시작해봅시다:

- [`keras.Model`](/api/models/model#model-class)를 서브클래싱하는 새로운 클래스를 생성합니다.
- 메서드 `train_step(self, data)`만 오버라이드합니다.
- 메트릭 이름(손실을 포함하여)을 현재 값에 매핑하는 딕셔너리를 반환합니다.

입력 인자 `data`는 트레이닝 데이터로서 `fit`에 전달되는 것입니다:

- `fit(x, y, ...)`를 호출하여 NumPy 배열을 전달하면, 
  `data`는 튜플 `(x, y)`가 됩니다.
- `fit(dataset, ...)`를 호출하여 `torch.utils.data.DataLoader` 또는 [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)를 전달하면, 
  `data`는 각 배치마다 `dataset`에 의해 생성(yielded)되는 것입니다.

`train_step()` 메서드의 본문에서, 여러분이 이미 익숙한 일반적인 트레이닝 업데이트를 구현합니다. 
중요한 점은, **`self.compute_loss()`를 통해 손실을 계산한다는 것**인데, 
이는 `compile()`에 전달된 손실 함수들을 감싸고 있습니다.

마찬가지로, `self.metrics`로부터의 메트릭에 대해 `metric.update_state(y, y_pred)`를 호출하여, 
`compile()`에 전달된 메트릭의 상태를 업데이트하고, 
마지막에 `self.metrics`에서 결과를 조회하여 현재 값을 가져옵니다.

```python
class CustomModel(keras.Model):
    def train_step(self, data):
        # 데이터를 언패킹합니다. 
        # 그 구조는 모델과 `fit()`에 전달한 것에 따라 달라집니다.
        x, y = data

        # 이전 트레이닝 스텝에서 남은 가중치의 그래디언트를 지우기 위해
        # torch.nn.Module.zero_grad()를 호출합니다.
        self.zero_grad()

        # 손실 계산
        y_pred = self(x, training=True)  # Forward pass
        loss = self.compute_loss(y=y, y_pred=y_pred)

        # 손실에 대해 torch.Tensor.backward()를 호출하여
        # 가중치의 그래디언트를 계산합니다.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # 가중치 업데이트
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # 메트릭 업데이트 (손실을 추적하는 메트릭 포함)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # 메트릭 이름을 현재 값에 매핑하는 딕셔너리를 반환합니다.
        # 이는 손실을 포함한다는 점에 유의하세요. (self.metrics에서 추적됨)
        return {m.name: m.result() for m in self.metrics}
```

이것을 시도해봅시다:

```python
# CustomModel의 인스턴스를 생성하고 컴파일합니다.
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 평소처럼 `fit`을 사용합니다.
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=3)
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Epoch 1/3    
    32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - mae: 0.3410 - loss: 0.1772
Epoch 2/3
    32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - mae: 0.3336 - loss: 0.1695
Epoch 3/3
    32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - mae: 0.3170 - loss: 0.1511

<keras.src.callbacks.history.History at 0x7f48a3255710>
```

</details>

------------------------------------------------------------------------

## 더 낮은 레벨로 내려가기
{: #going-lower-level}
<!-- ## Going lower-level -->

당연히, `compile()`에서 손실 함수를 전달하지 않고, 
대신 `train_step`에서 모든 작업을 *수동으로* 처리할 수 있습니다. 
메트릭도 마찬가지입니다.

다음은 옵티마이저 설정만을 위해 `compile()`을 사용하는, 더 낮은 레벨의 예제입니다:

- `__init__()`에서, 손실과 MAE 점수를 추적하기 위한 `Metric` 인스턴스를 만듭니다.
- 이 메트릭들의 상태를 (메트릭에 대해 `update_state()` 호출하여) 업데이트하는,
  커스텀 `train_step()`을 구현하고, 
  그런 다음 진행률 표시줄에 표시하거나 콜백으로 전달하기 위해, 
  현재 평균 값을 반환하도록 `result()`를 통해 조회합니다.
- 각 에포크 사이에 메트릭에 대해 `reset_states()`를 호출해야 한다는 점을 유의하세요! 
  그렇지 않으면, `result()`를 호출하면 트레이닝 시작 이후의 평균이 반환되는데, 
  우리는 일반적으로 에포크별 평균을 사용합니다. 
  다행히도 프레임워크는 이를 자동으로 처리해줍니다: 
  모델의 `metrics` 속성에 초기화하려는 메트릭을 나열하기만 하면 됩니다. 
  모델은 각 `fit()` 에포크의 시작 시 또는 `evaluate()` 호출의 시작 시에, 
  여기에 나열된 모든 객체에 대해 `reset_states()`를 호출합니다.

```python
class CustomModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.loss_fn = keras.losses.MeanSquaredError()

    def train_step(self, data):
        x, y = data

        # 이전 트레이닝 스텝에서 남은 가중치의 그래디언트를 지우기 위해
        # torch.nn.Module.zero_grad()를 호출합니다.
        self.zero_grad()

        # 손실 계산
        y_pred = self(x, training=True)  # 순전파
        loss = self.loss_fn(y, y_pred)

        # 손실에 대해 torch.Tensor.backward()를 호출하여 가중치의
        # 그래디언트를 계산합니다.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # 가중치 업데이트
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # 자체 메트릭 계산
        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        return {
            "loss": self.loss_tracker.result(),
            "mae": self.mae_metric.result(),
        }

    @property
    def metrics(self):
        # 여기에 `Metric` 객체를 나열하여, 
        # 각 에포크의 시작 시 또는 `evaluate()`의 시작 시,
        # 자동으로 `reset_states()`가 호출되도록 합니다.
        return [self.loss_tracker, self.mae_metric]


# CustomModel의 인스턴스 생성
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)

# 여기서 손실이나 메트릭을 전달하지 않습니다.
model.compile(optimizer="adam")

# 평소처럼 `fit`을 사용합니다 -- 콜백 등을 사용할 수 있습니다.
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.fit(x, y, epochs=5)
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Epoch 1/5
    32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.6173 - mae: 0.6607
Epoch 2/5
    32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.2340 - mae: 0.3883
Epoch 3/5
    32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1922 - mae: 0.3517
Epoch 4/5
    32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - loss: 0.1802 - mae: 0.3411
Epoch 5/5
    32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 3ms/step - loss: 0.1862 - mae: 0.3505

<keras.src.callbacks.history.History at 0x7f48975ccbd0>
```

</details>

------------------------------------------------------------------------

## `sample_weight` & `class_weight` 지원
{: #supporting-sample_weight-class_weight-}
<!-- ## Supporting `sample_weight` & `class_weight` -->

첫 번째 기본 예제에서 샘플 가중치에 대해 언급하지 않은 것을 눈치채셨을 겁니다. 
`sample_weight`와 `class_weight`를 `fit()` 인자로 지원하려면, 
간단히 다음과 같이 하면 됩니다:

- `data` 인자에서 `sample_weight`를 언팩합니다.
- `compute_loss`와 `update_state`에 이를 전달합니다. 
  (물론, 손실 및 메트릭에 대해 `compile()`을 사용하지 않는다면, 수동으로 적용할 수도 있습니다.)
- 끝입니다.

```python
class CustomModel(keras.Model):
    def train_step(self, data):
        # 데이터를 언팩합니다. 그 구조는 모델과
        # `fit()`에 전달한 것에 따라 달라집니다.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # 이전 트레이닝 스텝에서 남은 가중치의 그래디언트를 지우기 위해
        # torch.nn.Module.zero_grad()를 호출합니다.
        self.zero_grad()


        # 손실 계산
        y_pred = self(x, training=True)  # 순전파
        loss = self.compute_loss(
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )

        # 손실에 대해 torch.Tensor.backward()를 호출하여,
        # 가중치의 그래디언트를 계산합니다.
        loss.backward()

        trainable_weights = [v for v in self.trainable_weights]
        gradients = [v.value.grad for v in trainable_weights]

        # 가중치 업데이트
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        # 메트릭 업데이트 (손실을 추적하는 메트릭 포함)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)

        # 메트릭 이름을 현재 값에 매핑하는 딕셔너리를 반환합니다.
        # 이는 손실을 포함한다는 점에 유의하세요. (self.metrics에서 추적됨)
        return {m.name: m.result() for m in self.metrics}


# CustomModel의 인스턴스를 생성하고 컴파일합니다
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# 이제 sample_weight 인자를 사용할 수 있습니다.
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
sw = np.random.random((1000, 1))
model.fit(x, y, sample_weight=sw, epochs=3)
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Epoch 1/3
    32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - mae: 0.3216 - loss: 0.0827
Epoch 2/3
    32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - mae: 0.3156 - loss: 0.0803
Epoch 3/3
    32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - mae: 0.3085 - loss: 0.0760

<keras.src.callbacks.history.History at 0x7f48975d7bd0>
```

</details>

------------------------------------------------------------------------

## 사용자 정의 평가 단계 제공하기
{: #providing-your-own-evaluation-step}
<!-- ## Providing your own evaluation step -->

`model.evaluate()` 호출에 대해서도 동일한 작업을 하고 싶다면 어떻게 해야 할까요? 
그런 경우, `test_step`을 정확히 동일한 방식으로 오버라이드하면 됩니다. 
다음은 그 예시입니다:

```python
class CustomModel(keras.Model):
    def test_step(self, data):
        # 데이터를 언팩합니다
        x, y = data
        # 예측값 계산
        y_pred = self(x, training=False)
        # 손실을 추적하는 메트릭을 업데이트합니다.
        loss = self.compute_loss(y=y, y_pred=y_pred)
        # 메트릭을 업데이트합니다.
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # 메트릭 이름을 현재 값에 매핑하는 딕셔너리를 반환합니다.
        # 이는 손실을 포함한다는 점에 유의하세요. (self.metrics에서 추적됨)
        return {m.name: m.result() for m in self.metrics}


# CustomModel의 인스턴스를 생성합니다.
inputs = keras.Input(shape=(32,))
outputs = keras.layers.Dense(1)(inputs)
model = CustomModel(inputs, outputs)
model.compile(loss="mse", metrics=["mae"])

# 우리의 커스텀 test_step으로 평가합니다.
x = np.random.random((1000, 32))
y = np.random.random((1000, 1))
model.evaluate(x, y)
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
1/32 \[37m━━━━━━━━━━━━━━━━━━━━ 0s 2ms/step - mae: 0.8706 - loss: 0.9344

32/32 ━━━━━━━━━━━━━━━━━━━━ 0s 1ms/step - mae: 0.8959 - loss: 0.9952

[1.0077838897705078, 0.8984771370887756]
```

</details>

------------------------------------------------------------------------

## 마무리: 엔드 투 엔드 GAN 예제
{: #wrapping-up-an-end-to-end-gan-example}
<!-- ## Wrapping up: an end-to-end GAN example -->

엔드 투 엔드 예제를 통해 지금까지 배운 모든 것을 활용해 봅시다.

다음과 같은 구성 요소를 고려하겠습니다:

- 28x28x1 이미지를 생성하는 생성자 네트워크
- 28x28x1 이미지를 두 개의 클래스("가짜"와 "진짜")로 분류하는 판별자 네트워크
- 각 네트워크에 대한 옵티마이저 하나
- 판별자를 트레이닝하기 위한 손실 함수

```python
# 판별자 생성
discriminator = keras.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.GlobalMaxPooling2D(),
        layers.Dense(1),
    ],
    name="discriminator",
)

# 생성자 생성
latent_dim = 128
generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        # 7x7x128 맵으로 reshape 하기 위해 128개의 계수를 생성합니다.
        layers.Dense(7 * 7 * 128),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
    ],
    name="generator",
)
```

여기에는 `compile()`을 자신의 시그니처로 사용하고, 
`train_step`에서 전체 GAN 알고리즘을 17줄로 구현한, 
기능이 완전한(feature-complete) GAN 클래스가 있습니다:

```python
class GAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_tracker = keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = keras.metrics.Mean(name="g_loss")
        self.seed_generator = keras.random.SeedGenerator(1337)
        self.built = True

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(real_images, tuple) or isinstance(real_images, list):
            real_images = real_images[0]
        # 잠재 공간에서 랜덤 포인트 샘플링
        batch_size = real_images.shape[0]
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # 가짜 이미지를 생성합니다.
        generated_images = self.generator(random_latent_vectors)

        # 이를 진짜 이미지와 결합합니다.
        real_images = torch.tensor(real_images, device=device)
        combined_images = torch.concat([generated_images, real_images], axis=0)

        # 진짜와 가짜 이미지를 구분하는 레이블을 조합합니다.
        labels = torch.concat(
            [
                torch.ones((batch_size, 1), device=device),
                torch.zeros((batch_size, 1), device=device),
            ],
            axis=0,
        )
        # 레이블에 랜덤 노이즈를 추가합니다. - 중요한 트릭입니다!
        labels += 0.05 * keras.random.uniform(labels.shape, seed=self.seed_generator)

        # 판별자를 트레이닝합니다.
        self.zero_grad()
        predictions = self.discriminator(combined_images)
        d_loss = self.loss_fn(labels, predictions)
        d_loss.backward()
        grads = [v.value.grad for v in self.discriminator.trainable_weights]
        with torch.no_grad():
            self.d_optimizer.apply(grads, self.discriminator.trainable_weights)

        # 잠재 공간에서 랜덤 포인트 샘플링
        random_latent_vectors = keras.random.normal(
            shape=(batch_size, self.latent_dim), seed=self.seed_generator
        )

        # "모든 진짜 이미지 (all real images)"라고 말하는 레이블을 조합합니다.
        misleading_labels = torch.zeros((batch_size, 1), device=device)

        # 생성자를 트레이닝합니다. (판별자의 가중치를 업데이트하면 안됩니다)
        self.zero_grad()
        predictions = self.discriminator(self.generator(random_latent_vectors))
        g_loss = self.loss_fn(misleading_labels, predictions)
        grads = g_loss.backward()
        grads = [v.value.grad for v in self.generator.trainable_weights]
        with torch.no_grad():
            self.g_optimizer.apply(grads, self.generator.trainable_weights)

        # 메트릭을 업데이트하고 그 값을 반환합니다.
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {
            "d_loss": self.d_loss_tracker.result(),
            "g_loss": self.g_loss_tracker.result(),
        }
```

이를 테스트해봅시다:

```python
# 데이터셋을 준비합니다. 우리는 MNIST 숫자의 트레이닝과 테스트 데이터를 모두 사용합니다.
batch_size = 64
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype("float32") / 255.0
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))

# TensorDataset 생성
dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(all_digits), torch.from_numpy(all_digits)
)

# DataLoader 생성
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)

gan.fit(dataloader, epochs=1)
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
    1094/1094 ━━━━━━━━━━━━━━━━━━━━ 394s 360ms/step - d_loss: 0.2436 - g_loss: 4.7259
<keras.src.callbacks.history.History at 0x7f489760a490>
```

</details>

딥러닝의 기본 개념은 간단한데, 왜 그 구현은 고통스러워야 할까요?
