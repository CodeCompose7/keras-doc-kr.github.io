---
layout: default
title: KerasTuner 시작하기
nav_order: 1
permalink: /guides/keras_tuner/getting_started/
parent: KerasTuner
grand_parent: 개발자 가이드
---

* 원본 링크 : [https://keras.io/guides/keras_tuner/getting_started/](https://keras.io/guides/keras_tuner/getting_started/){:target="_blank"}
* 최종 수정일 : 2024-09-19

# KerasTuner 시작하기 (Getting started with KerasTuner)
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

**저자:** Luca Invernizzi, James Long, Francois Chollet, Tom O'Malley, Haifeng Jin  
**생성일:** 2019/05/31  
**최종편집일:** 2021/10/27  
**설명:** 모델 하이퍼파라미터 튜닝을 위한 KerasTuner 사용 기본 사항.

[Colab에서 보기](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_tuner/getting_started.ipynb){: .btn .btn-blue }
[GitHub 소스](https://github.com/keras-team/keras-io/blob/master/guides/keras_tuner/getting_started.py){: .btn .btn-blue }

```shell
!pip install keras-tuner -q
```

----

## 소개
{: #introduction}
<!-- ## Introduction -->

KerasTuner는 범용 하이퍼파라미터 튜닝 라이브러리입니다. 
Keras 워크플로우와의 강력한 통합을 제공하지만, 이에 국한되지 않습니다. 
scikit-learn 모델을 튜닝하거나 다른 작업에도 사용할 수 있습니다. 
이 튜토리얼에서는 KerasTuner를 사용하여 모델 아키텍처, 트레이닝 과정 및 데이터 전처리 단계를 튜닝하는 방법을 배울 것입니다. 
간단한 예제부터 시작해봅시다.

------------------------------------------------------------------------

## 모델 아키텍처 튜닝
{: #tune-the-model-architecture}
<!-- ## Tune the model architecture -->

먼저, 컴파일된 Keras 모델을 반환하는 함수를 작성해야 합니다. 
이 함수는 모델을 빌드할 때 하이퍼파라미터를 정의하기 위한 `hp` 인자를 받습니다.

### 검색 공간 정의
{: #define-the-search-space}
<!-- ### Define the search space -->

다음 코드 예제에서는, 두 개의 `Dense` 레이어로 구성된 Keras 모델을 정의합니다. 
첫 번째 `Dense` 레이어의 유닛 수를 튜닝하려고 합니다. 
이를 위해 `hp.Int('units', min_value=32, max_value=512, step=32)`로 정수형 하이퍼파라미터를 정의합니다. 
이 하이퍼파라미터는 32에서 512까지의 범위를 가지며, 샘플링할 때 최소 단위는 32입니다.

```python
import keras
from keras import layers


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            # 하이퍼파라미터 정의
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
```

모델이 성공적으로 빌드되는지 빠르게 테스트할 수 있습니다.

```python
import keras_tuner

build_model(keras_tuner.HyperParameters())
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
<Sequential name=sequential, built=False>
```

</details>

여러 가지 하이퍼파라미터를 정의할 수도 있습니다. 
다음 코드에서는 `Dropout` 레이어를 사용할지 여부를 `hp.Boolean()`으로 튜닝하고, 
활성화 함수는 `hp.Choice()`로 선택하며, 
옵티마이저의 학습률은 `hp.Float()`로 튜닝합니다.

```python
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(
        layers.Dense(
            # 유닛 수 튜닝.
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            # 사용할 활성화 함수 튜닝.
            activation=hp.Choice("activation", ["relu", "tanh"]),
        )
    )
    # 드롭아웃 사용 여부 튜닝.
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    # 옵티마이저 학습률을 하이퍼파라미터로 정의.
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


build_model(keras_tuner.HyperParameters())
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
<Sequential name=sequential_1, built=False>
```

</details>

아래에 보이는 것처럼, 하이퍼파라미터는 실제 값입니다. 
사실, 이는 실제 값을 반환하는 함수일 뿐입니다. 
예를 들어, `hp.Int()`는 `int` 값을 반환합니다. 
따라서, 이를 변수, for 루프, 또는 if 조건문에 넣을 수 있습니다.

```python
hp = keras_tuner.HyperParameters()
print(hp.Int("units", min_value=32, max_value=512, step=32))
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
32
```

</details>

하이퍼파라미터를 미리 정의하고 Keras 코드를 별도의 함수에 둘 수도 있습니다.

```python
def call_existing_code(units, activation, dropout, lr):
    model = keras.Sequential()
    model.add(layers.Flatten())
    model.add(layers.Dense(units=units, activation=activation))
    if dropout:
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(hp):
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    # 하이퍼파라미터 값을 사용하여 기존 모델 빌드 코드를 호출합니다.
    model = call_existing_code(
        units=units, activation=activation, dropout=dropout, lr=lr
    )
    return model


build_model(keras_tuner.HyperParameters())
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
<Sequential name=sequential_2, built=False>
```

</details>

각 하이퍼파라미터는 이름(첫 번째 인자)으로 고유하게 식별됩니다. 
서로 다른 `Dense` 레이어에서 유닛 수를 별도의 하이퍼파라미터로 조정하려면, 
그들에게 `f"units_{i}"`와 같이 다른 이름을 부여하면 됩니다.

또한, 이는 조건부 하이퍼파라미터를 생성하는 예시이기도 합니다. 
`Dense` 레이어에서 유닛 수를 지정하는 많은 하이퍼파라미터가 존재하며, 
이러한 하이퍼파라미터의 수는 레이어 수에 따라 달라집니다. 
레이어 수 자체도 하나의 하이퍼파라미터이기 때문에, 
전체 하이퍼파라미터 수는 시도할 때마다 다를 수 있습니다. 
어떤 하이퍼파라미터는 특정 조건이 충족될 때만 사용됩니다. 
예를 들어, `units_3`는 `num_layers`가 3보다 클 때만 사용됩니다. 
KerasTuner를 사용하면, 모델을 생성하는 동안 이러한 하이퍼파라미터를 동적으로 쉽게 정의할 수 있습니다.

```python
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    # 레이어 수를 튜닝합니다.
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                # 각 레이어의 유닛 수를 개별적으로 튜닝합니다.
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


build_model(keras_tuner.HyperParameters())
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
<Sequential name=sequential_3, built=False>
```

</details>

### 검색 시작하기
{: #start-the-search}
<!-- ### Start the search -->

탐색 공간을 정의한 후, 탐색을 실행할 튜너 클래스를 선택해야 합니다. 
`RandomSearch`, `BayesianOptimization`, `Hyperband` 중 하나를 선택할 수 있으며, 
이는 각각 다른 튜닝 알고리즘에 해당합니다. 
여기서는 `RandomSearch`를 예로 사용합니다.

튜너를 초기화하려면 여러 인자를 지정해야 합니다.

-   `hypermodel`. 모델을 빌드하는 함수로, 여기서는 `build_model`이 해당됩니다.
-   `objective`. 최적화할 목표의 이름입니다. (빌트인 지표의 경우, 최소화 또는 최대화 여부는 자동으로 추론됩니다)
    이 튜토리얼 후반부에서 커스텀 지표를 사용하는 방법을 소개할 예정입니다.
-   `max_trials`. 탐색 중 실행할 총 실험 횟수입니다.
-   `executions_per_trial`. 각 실험에서 구축되고 fit 되어야 하는 모델의 수입니다. 
    서로 다른 실험은 서로 다른 하이퍼파라미터 값을 가집니다. 
    동일한 실험 내에서는 동일한 하이퍼파라미터 값을 가집니다. 
    각 실험에서 여러 번 실행하는 목적은 결과의 분산을 줄여 모델의 성능을 더 정확하게 평가할 수 있도록 하기 위함입니다. 
    더 빠르게 결과를 얻고 싶다면, `executions_per_trial=1`로 설정할 수 있습니다. 
    (각 모델 구성에 대해 한 번의 트레이닝 라운드만 실행)
-   `overwrite`. 동일한 디렉토리에서 이전 결과를 덮어쓸지 아니면 이전 탐색을 다시 시작할지를 제어합니다. 
    여기서는 `overwrite=True`로 설정하여 새 탐색을 시작하고 이전 결과를 무시합니다.
-   `directory`. 탐색 결과를 저장할 디렉토리 경로입니다.
-   `project_name`. `directory` 내에 저장할 하위 디렉토리의 이름입니다.

```python
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="my_dir",
    project_name="helloworld",
)
```

탐색 공간 요약을 출력할 수 있습니다:

```python
tuner.search_space_summary()
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Search space summary
Default search space size: 5
num_layers (Int)
{'default': None, 'conditions': [], 'min_value': 1, 'max_value': 3, 'step': 1, 'sampling': 'linear'}
units_0 (Int)
{'default': None, 'conditions': [], 'min_value': 32, 'max_value': 512, 'step': 32, 'sampling': 'linear'}
activation (Choice)
{'default': 'relu', 'conditions': [], 'values': ['relu', 'tanh'], 'ordered': False}
dropout (Boolean)
{'default': False, 'conditions': []}
lr (Float)
{'default': 0.0001, 'conditions': [], 'min_value': 0.0001, 'max_value': 0.01, 'step': None, 'sampling': 'log'}
```

</details>

탐색을 시작하기 전에, MNIST 데이터셋을 준비합니다.

```python
import keras
import numpy as np

(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x[:-10000]
x_val = x[-10000:]
y_train = y[:-10000]
y_val = y[-10000:]

x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

그런 다음, 최적의 하이퍼파라미터 구성을 탐색합니다. 
`search`에 전달된 모든 인자는 각 실행에서 `model.fit()`에 전달됩니다. 
모델을 평가하기 위해 `validation_data`를 반드시 전달해야 합니다.

```python
tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Trial 3 Complete [00h 00m 19s]
val_accuracy: 0.9665500223636627

Best val_accuracy So Far: 0.9665500223636627
Total elapsed time: 00h 00m 40s
```

</details>

탐색(`search`)이 진행되는 동안, 
모델 빌드 함수는 서로 다른 실험에서 서로 다른 하이퍼파라미터 값으로 호출됩니다. 
각 실험에서 튜너는 새로운 하이퍼파라미터 값을 생성하여 모델을 빌드합니다. 
그런 다음 모델을 fit하고 평가합니다. 이때 메트릭이 기록됩니다. 
튜너는 탐색 공간을 점진적으로 탐색하며, 결국 좋은 하이퍼파라미터 값을 찾습니다.

### 결과 조회
{: #query-the-results}
<!-- ### Query the results -->

탐색이 완료되면, 최적의 모델을 조회할 수 있습니다. 
모델은 `validation_data`에 대해 평가한 결과 가장 성능이 좋은 에포크에서 저장됩니다.

```python
# 상위 2개의 모델을 가져옵니다.
models = tuner.get_best_models(num_models=2)
best_model = models[0]
best_model.summary()
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
/usr/local/python/3.10.13/lib/python3.10/site-packages/keras/src/saving/saving_lib.py:388: UserWarning: Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 18 variables. 
    trackable.load_own_variables(weights_store.get(inner_path))
/usr/local/python/3.10.13/lib/python3.10/site-packages/keras/src/saving/saving_lib.py:388: UserWarning: Skipping variable loading for optimizer 'adam', because it has 2 variables whereas the saved optimizer has 10 variables. 
    trackable.load_own_variables(weights_store.get(inner_path))

Model: "sequential"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ flatten (Flatten)               │ (32, 784)                 │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (Dense)                   │ (32, 416)                 │    326,560 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_1 (Dense)                 │ (32, 512)                 │    213,504 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_2 (Dense)                 │ (32, 32)                  │     16,416 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (Dropout)               │ (32, 32)                  │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense_3 (Dense)                 │ (32, 10)                  │        330 │
└─────────────────────────────────┴───────────────────────────┴────────────┘

    Total params: 556,810 (2.12 MB)

    Trainable params: 556,810 (2.12 MB)

    Non-trainable params: 0 (0.00 B)
```

</details>

탐색 결과 요약을 출력할 수도 있습니다.

```python
tuner.results_summary()
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Results summary
Results in my_dir/helloworld
Showing 10 best trials
Objective(name="val_accuracy", direction="max")

Trial 2 summary
Hyperparameters:
num_layers: 3
units_0: 416
activation: relu
dropout: True
lr: 0.0001324166048504802
units_1: 512
units_2: 32
Score: 0.9665500223636627

Trial 0 summary
Hyperparameters:
num_layers: 1
units_0: 128
activation: tanh
dropout: False
lr: 0.001425162921397599
Score: 0.9623999893665314

Trial 1 summary
Hyperparameters:
num_layers: 2
units_0: 512
activation: tanh
dropout: True
lr: 0.0010584293918512798
units_1: 32
Score: 0.9606499969959259
```

</details>

`my_dir/helloworld` 폴더, 즉 `directory/project_name`에 자세한 로그, 체크포인트 등을 찾을 수 있습니다.

또한, TensorBoard와 HParams 플러그인을 사용하여 튜닝 결과를 시각화할 수 있습니다. 
자세한 내용은 [이 링크]({{ site.baseurl }}/guides/keras_tuner/visualize_tuning/)를 참고하세요.

### 모델 재트레이닝
{: #retrain-the-model}
<!-- ### Retrain the model -->

전체 데이터셋을 사용하여 모델을 다시 트레이닝하고 싶다면, 
최적의 하이퍼파라미터를 가져와 직접 모델을 재트레이닝할 수 있습니다.

```python
# 최적의 2개의 하이퍼파라미터를 가져옵니다.
best_hps = tuner.get_best_hyperparameters(5)
# 최적의 하이퍼파라미터로 모델을 빌드합니다.
model = build_model(best_hps[0])
# 전체 데이터셋으로 트레이닝합니다.
x_all = np.concatenate((x_train, x_val))
y_all = np.concatenate((y_train, y_val))
model.fit(x=x_all, y=y_all, epochs=1)
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
1/1875 [37m━━━━━━━━━━━━━━━━━━━━  17:57 575ms/step - accuracy: 0.1250 - loss: 2.3113

29/1875 \[37m━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.1753 - loss: 2.2296

63/1875 \[37m━━━━━━━━━━━━━━━━━━━━ 3s 2ms/step - accuracy: 0.2626 - loss: 2.1206

96/1875 ━\[37m━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.3252 - loss: 2.0103

130/1875 ━\[37m━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.3745 - loss: 1.9041

164/1875 ━\[37m━━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.4139 - loss: 1.8094

199/1875 ━━\[37m━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.4470 - loss: 1.7246

235/1875 ━━\[37m━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.4752 - loss: 1.6493

270/1875 ━━\[37m━━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.4982 - loss: 1.5857

305/1875 ━━━\[37m━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.5182 - loss: 1.5293

339/1875 ━━━\[37m━━━━━━━━━━━━━━━━━ 2s 2ms/step - accuracy: 0.5354 - loss: 1.4800

374/1875 ━━━\[37m━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.5513 - loss: 1.4340

409/1875 ━━━━\[37m━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.5656 - loss: 1.3924

444/1875 ━━━━\[37m━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.5785 - loss: 1.3545

478/1875 ━━━━━\[37m━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.5899 - loss: 1.3208

513/1875 ━━━━━\[37m━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.6006 - loss: 1.2887

548/1875 ━━━━━\[37m━━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6104 - loss: 1.2592

583/1875 ━━━━━━\[37m━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6195 - loss: 1.2318

618/1875 ━━━━━━\[37m━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6279 - loss: 1.2063

653/1875 ━━━━━━\[37m━━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6358 - loss: 1.1823

688/1875 ━━━━━━━\[37m━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6431 - loss: 1.1598

723/1875 ━━━━━━━\[37m━━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6500 - loss: 1.1387

758/1875 ━━━━━━━━\[37m━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6564 - loss: 1.1189

793/1875 ━━━━━━━━\[37m━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6625 - loss: 1.1002

828/1875 ━━━━━━━━\[37m━━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6682 - loss: 1.0826

863/1875 ━━━━━━━━━\[37m━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6736 - loss: 1.0658

899/1875 ━━━━━━━━━\[37m━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6788 - loss: 1.0495

935/1875 ━━━━━━━━━\[37m━━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6838 - loss: 1.0339

970/1875 ━━━━━━━━━━\[37m━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6885 - loss: 1.0195

1005/1875 ━━━━━━━━━━\[37m━━━━━━━━━━ 1s 1ms/step - accuracy: 0.6929 - loss: 1.0058

1041/1875 ━━━━━━━━━━━\[37m━━━━━━━━━ 1s 1ms/step - accuracy: 0.6972 - loss: 0.9923

1076/1875 ━━━━━━━━━━━\[37m━━━━━━━━━ 1s 1ms/step - accuracy: 0.7012 - loss: 0.9798

1111/1875 ━━━━━━━━━━━\[37m━━━━━━━━━ 1s 1ms/step - accuracy: 0.7051 - loss: 0.9677

1146/1875 ━━━━━━━━━━━━\[37m━━━━━━━━ 1s 1ms/step - accuracy: 0.7088 - loss: 0.9561

1182/1875 ━━━━━━━━━━━━\[37m━━━━━━━━ 1s 1ms/step - accuracy: 0.7124 - loss: 0.9446

1218/1875 ━━━━━━━━━━━━\[37m━━━━━━━━ 0s 1ms/step - accuracy: 0.7159 - loss: 0.9336

1254/1875 ━━━━━━━━━━━━━\[37m━━━━━━━ 0s 1ms/step - accuracy: 0.7193 - loss: 0.9230

1289/1875 ━━━━━━━━━━━━━\[37m━━━━━━━ 0s 1ms/step - accuracy: 0.7225 - loss: 0.9131

1324/1875 ━━━━━━━━━━━━━━\[37m━━━━━━ 0s 1ms/step - accuracy: 0.7255 - loss: 0.9035

1359/1875 ━━━━━━━━━━━━━━\[37m━━━━━━ 0s 1ms/step - accuracy: 0.7284 - loss: 0.8943

1394/1875 ━━━━━━━━━━━━━━\[37m━━━━━━ 0s 1ms/step - accuracy: 0.7313 - loss: 0.8853

1429/1875 ━━━━━━━━━━━━━━━\[37m━━━━━ 0s 1ms/step - accuracy: 0.7341 - loss: 0.8767

1465/1875 ━━━━━━━━━━━━━━━\[37m━━━━━ 0s 1ms/step - accuracy: 0.7368 - loss: 0.8680

1500/1875 ━━━━━━━━━━━━━━━━\[37m━━━━ 0s 1ms/step - accuracy: 0.7394 - loss: 0.8599

1535/1875 ━━━━━━━━━━━━━━━━\[37m━━━━ 0s 1ms/step - accuracy: 0.7419 - loss: 0.8520

1570/1875 ━━━━━━━━━━━━━━━━\[37m━━━━ 0s 1ms/step - accuracy: 0.7443 - loss: 0.8444

1605/1875 ━━━━━━━━━━━━━━━━━\[37m━━━ 0s 1ms/step - accuracy: 0.7467 - loss: 0.8370

1639/1875 ━━━━━━━━━━━━━━━━━\[37m━━━ 0s 1ms/step - accuracy: 0.7489 - loss: 0.8299

1674/1875 ━━━━━━━━━━━━━━━━━\[37m━━━ 0s 1ms/step - accuracy: 0.7511 - loss: 0.8229

1707/1875 ━━━━━━━━━━━━━━━━━━\[37m━━ 0s 1ms/step - accuracy: 0.7532 - loss: 0.8164

1741/1875 ━━━━━━━━━━━━━━━━━━\[37m━━ 0s 1ms/step - accuracy: 0.7552 - loss: 0.8099

1774/1875 ━━━━━━━━━━━━━━━━━━\[37m━━ 0s 1ms/step - accuracy: 0.7572 - loss: 0.8038

1809/1875 ━━━━━━━━━━━━━━━━━━━\[37m━ 0s 1ms/step - accuracy: 0.7592 - loss: 0.7975

1843/1875 ━━━━━━━━━━━━━━━━━━━\[37m━ 0s 1ms/step - accuracy: 0.7611 - loss: 0.7915

1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - accuracy: 0.7629 - loss: 0.7858

<keras.src.callbacks.history.History at 0x7f31883d9e10>
```

</details>

------------------------------------------------------------------------

## 모델 트레이닝 튜닝하기
{: #tune-model-training}
<!-- ## Tune model training -->

모델 빌딩 프로세스를 튜닝하려면, `HyperModel` 클래스를 서브클래싱해야 합니다. 
이를 통해 하이퍼모델을 쉽게 공유하고 재사용할 수 있습니다.

모델 빌딩과 트레이닝 프로세스를 각각 튜닝하려면, 
`HyperModel.build()`와 `HyperModel.fit()`을 오버라이드해야 합니다. 
`HyperModel.build()` 메서드는 하이퍼파라미터를 사용하여, 
Keras 모델을 생성하고 반환하는 모델 빌딩 함수와 동일합니다.

`HyperModel.fit()`에서는, `HyperModel.build()`에서 반환된 모델, `hp`, 
그리고 `search()`에 전달된 모든 인자에 접근할 수 있습니다. 
모델을 트레이닝한 후 트레이닝 기록을 반환해야 합니다.

다음 코드에서는, `model.fit()`에서 `shuffle` 인자를 튜닝합니다.

일반적으로 에포크 수를 튜닝할 필요는 없습니다. 
왜냐하면 `model.fit()`에 내장된 콜백이 전달되어, 
`validation_data`로 평가된 가장 좋은 에포크에서 모델을 저장하기 때문입니다.

> **참고**: `**kwargs`는 항상 `model.fit()`에 전달되어야 합니다. 
> 여기에는 모델 저장 및 TensorBoard 플러그인을 위한 콜백이 포함되어 있기 때문입니다.

```python
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                units=hp.Int("units", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
        model.add(layers.Dense(10, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            # 각 에포크에서 데이터를 셔플할지 여부를 튜닝.
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )
```

다시 한 번, 코드를 빠르게 확인하여 제대로 작동하는지 확인할 수 있습니다.

```python
hp = keras_tuner.HyperParameters()
hypermodel = MyHyperModel()
model = hypermodel.build(hp)
hypermodel.fit(hp, model, np.random.rand(100, 28, 28), np.random.rand(100, 10))
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
1/4 ━━━━━\[37m━━━━━━━━━━━━━━━ 0s 279ms/step - accuracy: 0.0000e+00 - loss: 12.2230

4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 108ms/step - accuracy: 0.0679 - loss: 11.9568

4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 109ms/step - accuracy: 0.0763 - loss: 11.8941

<keras.src.callbacks.history.History at 0x7f318865c100>
```

</details>

------------------------------------------------------------------------

## 데이터 전처리 튜닝하기
{: #tune-data-preprocessing}
<!-- ## Tune data preprocessing -->

데이터 전처리를 튜닝하려면, `HyperModel.fit()`에서, 
인수로부터 데이터셋에 접근할 수 있도록, 추가적인 단계를 추가하면 됩니다. 
다음 코드에서는, 트레이닝 전에 데이터를 정규화할지 여부를 튜닝합니다. 
이번에는 `x`와 `y`를 함수 시그니처에 명시적으로 넣었는데, 
이는 우리가 이 값들을 사용해야 하기 때문입니다.

```python
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential()
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                units=hp.Int("units", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
        model.add(layers.Dense(10, activation="softmax"))
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, x, y, **kwargs):
        if hp.Boolean("normalize"):
            x = layers.Normalization()(x)
        return model.fit(
            x,
            y,
            # 각 에포크에서 데이터를 셔플할지 여부를 튜닝.
            shuffle=hp.Boolean("shuffle"),
            **kwargs,
        )


hp = keras_tuner.HyperParameters()
hypermodel = MyHyperModel()
model = hypermodel.build(hp)
hypermodel.fit(hp, model, np.random.rand(100, 28, 28), np.random.rand(100, 10))
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
1/4 ━━━━━\[37m━━━━━━━━━━━━━━━ 0s 276ms/step - accuracy: 0.1250 - loss: 12.0090

4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 94ms/step - accuracy: 0.0994 - loss: 12.1242

4/4 ━━━━━━━━━━━━━━━━━━━━ 1s 95ms/step - accuracy: 0.0955 - loss: 12.1594

<keras.src.callbacks.history.History at 0x7f31ba836200>
```

</details>

하이퍼파라미터가 `build()`와 `fit()` 모두에서 사용되는 경우, 
`build()`에서 정의하고 `hp.get(hp_name)`을 사용하여, 
`fit()`에서 해당 값을 가져올 수 있습니다. 
이미지 크기를 예로 들어보겠습니다. 
이 값은 `build()`에서 입력 크기로 사용되며, 
`fit()`의 데이터 전처리 단계에서 이미지를 자를 때 사용됩니다.

```python
class MyHyperModel(keras_tuner.HyperModel):
    def build(self, hp):
        image_size = hp.Int("image_size", 10, 28)
        inputs = keras.Input(shape=(image_size, image_size))
        outputs = layers.Flatten()(inputs)
        outputs = layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )(outputs)
        outputs = layers.Dense(10, activation="softmax")(outputs)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, hp, model, x, y, validation_data=None, **kwargs):
        if hp.Boolean("normalize"):
            x = layers.Normalization()(x)
        image_size = hp.get("image_size")
        cropped_x = x[:, :image_size, :image_size, :]
        if validation_data:
            x_val, y_val = validation_data
            cropped_x_val = x_val[:, :image_size, :image_size, :]
            validation_data = (cropped_x_val, y_val)
        return model.fit(
            cropped_x,
            y,
            # 각 에포크에서 데이터를 셔플할지 여부를 튜닝.
            shuffle=hp.Boolean("shuffle"),
            validation_data=validation_data,
            **kwargs,
        )


tuner = keras_tuner.RandomSearch(
    MyHyperModel(),
    objective="val_accuracy",
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="tune_hypermodel",
)

tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Trial 3 Complete [00h 00m 04s]
val_accuracy: 0.9567000269889832

Best val_accuracy So Far: 0.9685999751091003
Total elapsed time: 00h 00m 13s
```

</details>

### 모델 재트레이닝하기
{: #retrain-the-model}
<!-- ### Retrain the model -->

`HyperModel`을 사용하면 최적의 모델을 직접 재트레이닝할 수도 있습니다.

```python
hypermodel = MyHyperModel()
best_hp = tuner.get_best_hyperparameters()[0]
model = hypermodel.build(best_hp)
hypermodel.fit(best_hp, model, x_all, y_all, epochs=1)
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
1/1875 [37m━━━━━━━━━━━━━━━━━━━━  9:00 289ms/step - accuracy: 0.0000e+00 - loss: 2.4352

52/1875 \[37m━━━━━━━━━━━━━━━━━━━━ 1s 996us/step - accuracy: 0.6035 - loss: 1.3521

110/1875 ━\[37m━━━━━━━━━━━━━━━━━━━ 1s 925us/step - accuracy: 0.7037 - loss: 1.0231

171/1875 ━\[37m━━━━━━━━━━━━━━━━━━━ 1s 890us/step - accuracy: 0.7522 - loss: 0.8572

231/1875 ━━\[37m━━━━━━━━━━━━━━━━━━ 1s 877us/step - accuracy: 0.7804 - loss: 0.7590

291/1875 ━━━\[37m━━━━━━━━━━━━━━━━━ 1s 870us/step - accuracy: 0.7993 - loss: 0.6932

350/1875 ━━━\[37m━━━━━━━━━━━━━━━━━ 1s 867us/step - accuracy: 0.8127 - loss: 0.6467

413/1875 ━━━━\[37m━━━━━━━━━━━━━━━━ 1s 856us/step - accuracy: 0.8238 - loss: 0.6079

476/1875 ━━━━━\[37m━━━━━━━━━━━━━━━ 1s 848us/step - accuracy: 0.8326 - loss: 0.5774

535/1875 ━━━━━\[37m━━━━━━━━━━━━━━━ 1s 849us/step - accuracy: 0.8394 - loss: 0.5536

600/1875 ━━━━━━\[37m━━━━━━━━━━━━━━ 1s 841us/step - accuracy: 0.8458 - loss: 0.5309

661/1875 ━━━━━━━\[37m━━━━━━━━━━━━━ 1s 840us/step - accuracy: 0.8511 - loss: 0.5123

723/1875 ━━━━━━━\[37m━━━━━━━━━━━━━ 0s 837us/step - accuracy: 0.8559 - loss: 0.4955

783/1875 ━━━━━━━━\[37m━━━━━━━━━━━━ 0s 838us/step - accuracy: 0.8600 - loss: 0.4811

847/1875 ━━━━━━━━━\[37m━━━━━━━━━━━ 0s 834us/step - accuracy: 0.8640 - loss: 0.4671

912/1875 ━━━━━━━━━\[37m━━━━━━━━━━━ 0s 830us/step - accuracy: 0.8677 - loss: 0.4544

976/1875 ━━━━━━━━━━\[37m━━━━━━━━━━ 0s 827us/step - accuracy: 0.8709 - loss: 0.4429

1040/1875 ━━━━━━━━━━━\[37m━━━━━━━━━ 0s 825us/step - accuracy: 0.8738 - loss: 0.4325

1104/1875 ━━━━━━━━━━━\[37m━━━━━━━━━ 0s 822us/step - accuracy: 0.8766 - loss: 0.4229

1168/1875 ━━━━━━━━━━━━\[37m━━━━━━━━ 0s 821us/step - accuracy: 0.8791 - loss: 0.4140

1233/1875 ━━━━━━━━━━━━━\[37m━━━━━━━ 0s 818us/step - accuracy: 0.8815 - loss: 0.4056

1296/1875 ━━━━━━━━━━━━━\[37m━━━━━━━ 0s 817us/step - accuracy: 0.8837 - loss: 0.3980

1361/1875 ━━━━━━━━━━━━━━\[37m━━━━━━ 0s 815us/step - accuracy: 0.8858 - loss: 0.3907

1424/1875 ━━━━━━━━━━━━━━━\[37m━━━━━ 0s 814us/step - accuracy: 0.8877 - loss: 0.3840

1488/1875 ━━━━━━━━━━━━━━━\[37m━━━━━ 0s 813us/step - accuracy: 0.8895 - loss: 0.3776

1550/1875 ━━━━━━━━━━━━━━━━\[37m━━━━ 0s 813us/step - accuracy: 0.8912 - loss: 0.3718

1613/1875 ━━━━━━━━━━━━━━━━━\[37m━━━ 0s 813us/step - accuracy: 0.8928 - loss: 0.3662

1678/1875 ━━━━━━━━━━━━━━━━━\[37m━━━ 0s 811us/step - accuracy: 0.8944 - loss: 0.3607

1744/1875 ━━━━━━━━━━━━━━━━━━\[37m━━ 0s 809us/step - accuracy: 0.8959 - loss: 0.3555

1810/1875 ━━━━━━━━━━━━━━━━━━━\[37m━ 0s 808us/step - accuracy: 0.8973 - loss: 0.3504

1874/1875 ━━━━━━━━━━━━━━━━━━━\[37m━ 0s 807us/step - accuracy: 0.8987 - loss: 0.3457

1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 808us/step - accuracy: 0.8987 - loss: 0.3456

<keras.src.callbacks.history.History at 0x7f31884b3070>
```

</details>

------------------------------------------------------------------------

## 튜닝 목표 설정하기
{: #specify-the-tuning-objective}
<!-- ## Specify the tuning objective -->

지금까지의 모든 예제에서는 검증 정확도(`"val_accuracy"`)를 사용하여 최적의 모델을 선택했습니다. 
사실, 튜닝 목표로 사용할 수 있는 메트릭은 무엇이든 가능합니다. 
가장 일반적으로 사용되는 메트릭은 검증 손실인 `"val_loss"`입니다.

### 빌트인 메트릭을 목표로 사용하기
{: #built-in-metric-as-the-objective}
<!-- ### Built-in metric as the objective -->

Keras에는 목표로 사용할 수 있는 빌트인 메트릭이 많이 있습니다. 
[빌트인 메트릭 리스트]({{ site.baseurl }}/api/metrics/)를 참조하세요.

빌트인 메트릭을 목표로 사용하려면, 다음 단계를 따르세요:

-   모델을 내장 메트릭으로 컴파일하세요. 
    예를 들어, `MeanAbsoluteError()`를 사용하려면, 
    `metrics=[MeanAbsoluteError()]`로 모델을 컴파일해야 합니다. 
    또한 해당 메트릭의 이름 문자열을 사용할 수도 있습니다: `metrics=["mean_absolute_error"]`. 
    메트릭의 이름 문자열은 항상 클래스 이름을 스네이크 케이스로 변환한 형식입니다.

<!-- -->

-   목표 이름 문자열을 식별하세요. 
    목표 이름 문자열은 항상 `f"val_{metric_name_string}"` 형식입니다. 
    예를 들어, 검증 데이터에 대해 평가한 평균 절대 오차의 목표 이름 문자열은 `"val_mean_absolute_error"`가 됩니다.

<!-- -->

-   이를 [`keras_tuner.Objective`]({{ site.baseurl }}/api/keras_tuner/tuners/objective#objective-class)로 감싸세요. 
    일반적으로 목표를 [`keras_tuner.Objective`]({{ site.baseurl }}/api/keras_tuner/tuners/objective#objective-class) 객체로 감싸서 목표를 최적화할 방향을 지정해야 합니다. 
    예를 들어, 평균 절대 오차를 최소화하려면, 
    `keras_tuner.Objective("val_mean_absolute_error", "min")`을 사용할 수 있습니다. 
    방향은 `"min"` 또는 `"max"` 중 하나여야 합니다.

<!-- -->

-   튜너에 래핑된 목표를 전달하세요.

다음은 최소한의 코드 예시입니다.

```python
def build_regressor(hp):
    model = keras.Sequential(
        [
            layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
            layers.Dense(units=1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        # 목표는 메트릭 중 하나입니다.
        metrics=[keras.metrics.MeanAbsoluteError()],
    )
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_regressor,
    # 목표 이름과 방향.
    # 이름은 f"val_{snake_case_metric_class_name}" 형식입니다.
    objective=keras_tuner.Objective("val_mean_absolute_error", direction="min"),
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="built_in_metrics",
)

tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1)),
)

tuner.results_summary()
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Trial 3 Complete [00h 00m 01s]
val_mean_absolute_error: 0.39589792490005493

Best val_mean_absolute_error So Far: 0.34321871399879456
Total elapsed time: 00h 00m 03s
Results summary
Results in my_dir/built_in_metrics
Showing 10 best trials
Objective(name="val_mean_absolute_error", direction="min")

Trial 1 summary
Hyperparameters:
units: 32
Score: 0.34321871399879456

Trial 2 summary
Hyperparameters:
units: 128
Score: 0.39589792490005493

Trial 0 summary
Hyperparameters:
units: 96
Score: 0.5005304217338562
```

</details>

### 커스텀 메트릭을 튜닝 목표로 사용하기
{: #custom-metric-as-the-objective}
<!-- ### Custom metric as the objective -->

커스텀 메트릭을 구현하여 하이퍼파라미터 탐색의 목표로 사용할 수 있습니다. 
여기서는 예시로 평균 제곱 오차(MSE)를 사용하겠습니다. 
먼저, [`keras.metrics.Metric`]({{ site.baseurl }}/api/metrics/base_metric#metric-class)를 서브클래싱하여 MSE 메트릭을 구현합니다. 
`super().__init__()`의 `name` 인수를 사용해 메트릭의 이름을 지정하는 것을 잊지 마세요. 
이 이름은 나중에 사용됩니다. 
참고로, MSE는 사실 빌트인 메트릭이며, 
[`keras.metrics.MeanSquaredError`]({{ site.baseurl }}/api/metrics/regression_metrics#meansquarederror-class)를 통해 불러올 수 있습니다. 
이 예시는 커스텀 메트릭을 하이퍼파라미터 탐색 목표로 사용하는 방법을 보여줍니다.

커스텀 메트릭을 구현하는 방법에 대한 자세한 내용은 [이 튜토리얼]({{ site.baseurl }}/api/metrics/#creating-custom-metrics)을 참조하세요. 
만약 `update_state(y_true, y_pred, sample_weight)`와는 다른 함수 시그니처를 사용하는 메트릭을 만들고자 한다면, 
[이 튜토리얼]({{ site.baseurl }}/guides/customizing_what_happens_in_fit/#going-lowerlevel)를 따라, 
`train_step()` 메서드를 재정의할 수 있습니다.

```python
from keras import ops


class CustomMetric(keras.metrics.Metric):
    def __init__(self, **kwargs):
        # 메트릭 이름을 "custom_metric"으로 지정합니다.
        super().__init__(name="custom_metric", **kwargs)
        self.sum = self.add_weight(name="sum", initializer="zeros")
        self.count = self.add_weight(name="count", dtype="int32", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = ops.square(y_true - y_pred)
        count = ops.shape(y_true)[0]
        if sample_weight is not None:
            sample_weight = ops.cast(sample_weight, self.dtype)
            values *= sample_weight
            count *= sample_weight
        self.sum.assign_add(ops.sum(values))
        self.count.assign_add(count)

    def result(self):
        return self.sum / ops.cast(self.count, "float32")

    def reset_state(self):
        self.sum.assign(0)
        self.count.assign(0)
```

커스텀 목표로 검색을 실행합니다.

```python
def build_regressor(hp):
    model = keras.Sequential(
        [
            layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
            layers.Dense(units=1),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="mean_squared_error",
        # 커스텀 메트릭을 metrics에 추가합니다.
        metrics=[CustomMetric()],
    )
    return model


tuner = keras_tuner.RandomSearch(
    hypermodel=build_regressor,
    # 목표의 이름과 방향을 지정합니다.
    objective=keras_tuner.Objective("val_custom_metric", direction="min"),
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="custom_metrics",
)

tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1)),
)

tuner.results_summary()
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Trial 3 Complete [00h 00m 01s]
val_custom_metric: 0.2830956280231476

Best val_custom_metric So Far: 0.2529197633266449
Total elapsed time: 00h 00m 02s
Results summary
Results in my_dir/custom_metrics
Showing 10 best trials
Objective(name="val_custom_metric", direction="min")

Trial 0 summary
Hyperparameters:
units: 32
Score: 0.2529197633266449

Trial 2 summary
Hyperparameters:
units: 128
Score: 0.2830956280231476

Trial 1 summary
Hyperparameters:
units: 96
Score: 0.4656866192817688
```

</details>

커스텀 목표가 커스텀 메트릭으로 표현하기 어려운 경우, 
`HyperModel.fit()`에서 직접 모델을 평가하고 목표 값을 반환할 수도 있습니다. 
이 경우 목표 값은 기본적으로 최소화됩니다. 
이러한 경우 튜너를 초기화할 때 `objective`를 지정할 필요가 없습니다. 
그러나 이 경우 메트릭 값은 Keras 로그에서 추적되지 않으며, KerasTuner 로그에만 기록됩니다. 
따라서 이 값들은 Keras 메트릭을 사용하는 TensorBoard 뷰에서 표시되지 않습니다.

```python
class HyperRegressor(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential(
            [
                layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
                layers.Dense(units=1),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
        )
        return model

    def fit(self, hp, model, x, y, validation_data, **kwargs):
        model.fit(x, y, **kwargs)
        x_val, y_val = validation_data
        y_pred = model.predict(x_val)
        # 최소화할 단일 float 값을 반환합니다.
        return np.mean(np.abs(y_pred - y_val))


tuner = keras_tuner.RandomSearch(
    hypermodel=HyperRegressor(),
    # 목표를 지정할 필요가 없습니다.
    # 목표는 `HyperModel.fit()`의 반환 값입니다.
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="custom_eval",
)
tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1)),
)

tuner.results_summary()
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Trial 3 Complete [00h 00m 01s]
default_objective: 0.6571611521766413

Best default_objective So Far: 0.40719249752993525
Total elapsed time: 00h 00m 02s
Results summary
Results in my_dir/custom_eval
Showing 10 best trials
Objective(name="default_objective", direction="min")

Trial 1 summary
Hyperparameters:
units: 128
Score: 0.40719249752993525

Trial 0 summary
Hyperparameters:
units: 96
Score: 0.4992297225533352

Trial 2 summary
Hyperparameters:
units: 32
Score: 0.6571611521766413
```

</details>

KerasTuner에서 여러 메트릭을 추적하면서 그중 하나만 목표로 사용할 경우, 
메트릭 이름을 키로 하고 메트릭 값을 값으로 하는 딕셔너리를 반환할 수 있습니다. 
예를 들어, `{"metric_a": 1.0, "metric_b": 2.0}`을 반환하고, 
키 중 하나를 목표 이름으로 사용할 수 있습니다. 
예를 들어, `keras_tuner.Objective("metric_a", "min")`와 같이 설정합니다.

```python
class HyperRegressor(keras_tuner.HyperModel):
    def build(self, hp):
        model = keras.Sequential(
            [
                layers.Dense(units=hp.Int("units", 32, 128, 32), activation="relu"),
                layers.Dense(units=1),
            ]
        )
        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
        )
        return model

    def fit(self, hp, model, x, y, validation_data, **kwargs):
        model.fit(x, y, **kwargs)
        x_val, y_val = validation_data
        y_pred = model.predict(x_val)
        # KerasTuner가 추적할 메트릭 딕셔너리를 반환합니다.
        return {
            "metric_a": -np.mean(np.abs(y_pred - y_val)),
            "metric_b": np.mean(np.square(y_pred - y_val)),
        }


tuner = keras_tuner.RandomSearch(
    hypermodel=HyperRegressor(),
    # 목표는 딕셔너리의 키 중 하나입니다.
    # 음의 MAE를 최대화, 즉 MAE를 최소화합니다.
    objective=keras_tuner.Objective("metric_a", "max"),
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="custom_eval_dict",
)
tuner.search(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    validation_data=(np.random.rand(20, 10), np.random.rand(20, 1)),
)

tuner.results_summary()
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Trial 3 Complete [00h 00m 01s]
metric_a: -0.39470441501524833

Best metric_a So Far: -0.3836997988261662
Total elapsed time: 00h 00m 02s
Results summary
Results in my_dir/custom_eval_dict
Showing 10 best trials
Objective(name="metric_a", direction="max")

Trial 1 summary
Hyperparameters:
units: 64
Score: -0.3836997988261662

Trial 2 summary
Hyperparameters:
units: 32
Score: -0.39470441501524833

Trial 0 summary
Hyperparameters:
units: 96
Score: -0.46081380465766364
```

</details>

------------------------------------------------------------------------

## 엔드 투 엔드 워크플로우 튜닝
{: #tune-end-to-end-workflows}
<!-- ## Tune end-to-end workflows -->

일부 경우에는, 코드를 빌드 및 fit 함수로 맞추는 것이 어려울 수 있습니다. 
이 경우 `Tuner.run_trial()`을 재정의하여 엔드 투 엔드 워크플로우를 한곳에 유지할 수 있으며, 
이를 통해 트라이얼을 완전히 제어할 수 있습니다. 
이를 일종의 블랙박스 옵티마이저로 간주할 수 있습니다.

### 어떤 함수이든 튜닝
{: #tune-any-function}
<!-- ### Tune any function -->

예를 들어, `f(x)=x*x+1`을 최소화하는 `x` 값을 찾을 수 있습니다. 
아래 코드에서는 `x`를 하이퍼파라미터로 정의하고, `f(x)`를 목표 값으로 반환합니다. 
튜너를 초기화할 때, `hypermodel`과 `objective` 인수는 생략할 수 있습니다.

```python
class MyTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        # 트라이얼에서 hp 가져오기
        hp = trial.hyperparameters
        # "x"를 하이퍼파라미터로 정의
        x = hp.Float("x", min_value=-1.0, max_value=1.0)
        # 최소화할 목표 값 반환
        return x * x + 1


tuner = MyTuner(
    # hypermodel이나 objective를 지정하지 않음
    max_trials=20,
    overwrite=True,
    directory="my_dir",
    project_name="tune_anything",
)

# run_trial()에서 사용하지 않으면, search()에 아무것도 전달할 필요 없음
tuner.search()
print(tuner.get_best_hyperparameters()[0].get("x"))
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Trial 20 Complete [00h 00m 00s]
default_objective: 1.6547719581194267

Best default_objective So Far: 1.0013236767905302
Total elapsed time: 00h 00m 00s
0.03638236922645777
```

</details>

### Keras 코드 분리 유지
{: #keep-keras-code-separate}
<!-- ### Keep Keras code separate -->

Keras 코드를 변경하지 않고 그대로 유지하면서 KerasTuner를 사용하여 튜닝할 수 있습니다. 
Keras 코드를 수정할 수 없는 경우에 유용합니다.

이 방식은 더 많은 유연성을 제공합니다. 
모델 빌드 및 트레이닝 코드를 따로 분리할 필요가 없습니다. 
그러나, 이 워크플로우는 모델 저장이나 TensorBoard 플러그인과의 연결을 제공하지는 않습니다.

모델을 저장하려면, 각 트라이얼을 고유하게 식별하는 문자열인 `trial.trial_id`를 사용하여, 
서로 다른 경로를 구성해 각 트라이얼에서 생성된 모델을 저장할 수 있습니다.

```python
import os


def keras_code(units, optimizer, saving_path):
    # 모델 빌드
    model = keras.Sequential(
        [
            layers.Dense(units=units, activation="relu"),
            layers.Dense(units=1),
        ]
    )
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
    )

    # 데이터 준비
    x_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)
    x_val = np.random.rand(20, 10)
    y_val = np.random.rand(20, 1)

    # 모델 트레이닝 및 평가
    model.fit(x_train, y_train)

    # 모델 저장
    model.save(saving_path)

    # 목표 값으로 단일 float를 반환.
    # {metric_name: metric_value} 형식의 딕셔너리를 반환할 수도 있습니다.
    y_pred = model.predict(x_val)
    return np.mean(np.abs(y_pred - y_val))


class MyTuner(keras_tuner.RandomSearch):
    def run_trial(self, trial, **kwargs):
        hp = trial.hyperparameters
        return keras_code(
            units=hp.Int("units", 32, 128, 32),
            optimizer=hp.Choice("optimizer", ["adam", "adadelta"]),
            saving_path=os.path.join("/tmp", f"{trial.trial_id}.keras"),
        )


tuner = MyTuner(
    max_trials=3,
    overwrite=True,
    directory="my_dir",
    project_name="keep_code_separate",
)
tuner.search()
# 모델 재트레이닝
best_hp = tuner.get_best_hyperparameters()[0]
keras_code(**best_hp.values, saving_path="/tmp/best_model.keras")
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Trial 3 Complete [00h 00m 00s]
default_objective: 0.18014027375230962

Best default_objective So Far: 0.18014027375230962
Total elapsed time: 00h 00m 03s

1/4 ━━━━━\[37m━━━━━━━━━━━━━━━ 0s 172ms/step - loss: 0.5030

4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 60ms/step - loss: 0.5288

4/4 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step - loss: 0.5367

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 27ms/step

1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 28ms/step

0.5918120126201316
```

</details>

------------------------------------------------------------------------

## KerasTuner에는 사전 제작된 튜닝 가능한 애플리케이션 HyperResNet 및 HyperXception이 포함되어 있습니다.
{: #kerastuner-includes-pre-made-tunable-applications-hyperresnet-and-hyperxception}
<!-- ## KerasTuner includes pre-made tunable applications: HyperResNet and HyperXception -->

이들은 컴퓨터 비전을 위한 즉시 사용 가능한 하이퍼모델입니다.

이 모델들은 `loss="categorical_crossentropy"`와 `metrics=["accuracy"]`로 사전 컴파일되어 있습니다.

```python
from keras_tuner.applications import HyperResNet

hypermodel = HyperResNet(input_shape=(28, 28, 1), classes=10)

tuner = keras_tuner.RandomSearch(
    hypermodel,
    objective="val_accuracy",
    max_trials=2,
    overwrite=True,
    directory="my_dir",
    project_name="built_in_hypermodel",
)
```
