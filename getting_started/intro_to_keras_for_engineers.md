---
layout: default
title: 엔지니어를 위한 Keras 소개
nav_order: 1
permalink: /getting_started/intro_to_keras_for_engineers
parent: 시작하기
---

* 원본 링크 : [https://keras.io/getting_started/intro_to_keras_for_engineers/](https://keras.io/getting_started/intro_to_keras_for_engineers/)
* 최종 수정일 : 2024-03-28

# 엔지니어를 위한 Keras 소개
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

**저자:** [fchollet](https://twitter.com/fchollet)  
**생성일:** 2023/07/10  
**최종편집일:** 2023/07/10  
**설명:** Keras 3와의 첫 접촉

[Colab에서 보기](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/intro_to_keras_for_engineers.ipynb){: .btn .btn-blue }
[GitHub 소스](https://github.com/keras-team/keras-io/blob/master/guides/intro_to_keras_for_engineers.py){: .btn .btn-blue }

----

## 소개

Keras 3는 TensorFlow, JAX 및 PyTorch와 상호 호환되는 딥러닝 프레임워크입니다. 이 노트북에서는 주요 Keras 3 워크플로우를 안내합니다.

----

## 셋업

여기서는 JAX 백엔드를 사용하지만, 아래 문자열을 ``"tensorflow"`` 또는 ``"torch"``로 수정하고, "Restart runtime"을 누르면, 노트북 전체가 똑같이 실행됩니다! 이 전체 가이드는 백엔드에 구애받지 않습니다.

```python
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

# Keras는 백엔드가 구성된 후에 import 되어야 한다는 점에 유의하세요. 
# 패키지를 import 한 후에는, 백엔드를 변경할 수 없습니다.

import keras
```

----

## 첫 번째 예시: MNIST 컨브넷

MNIST 숫자를 분류하기 위해 convnet을 트레이닝하는 ML의 Hello World부터 시작하겠습니다.

다음은 데이터입니다:

```python
# 데이터를 로드하고 트레이닝 세트와 테스트 세트로 분할하기
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# 이미지를 [0, 1] 범위로 조정하기
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# 이미지가 (28, 28, 1) 모양인지 확인하기
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")
```

```
x_train shape: (60000, 28, 28, 1)
y_train shape: (60000,)
60000 train samples
10000 test samples
```

이것이 우리의 모델입니다.

Keras에서 제공하는 다양한 모델 빌드 옵션은 다음과 같습니다:

*   [Sequential API](https://keras.io/guides/sequential_model/) (아래에 우리가 사용한 것)
*   [Functional API](https://keras.io/guides/functional_api/) (가장 일반적임)
*   [서브클래싱을 통해 당신만의 모델 작성하기](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) (for advanced use cases)

```python
# 모델 파라미터
num_classes = 10
input_shape = (28, 28, 1)

model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)
```

이것이 우리 모델의 Summary 입니다.:

```python
model.summary()
```

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape              ┃    Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 26, 26, 64)        │        640 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_1 (Conv2D)               │ (None, 24, 24, 64)        │     36,928 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 12, 12, 64)        │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_2 (Conv2D)               │ (None, 10, 10, 128)       │     73,856 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ conv2d_3 (Conv2D)               │ (None, 8, 8, 128)         │    147,584 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ global_average_pooling2d        │ (None, 128)               │          0 │
│ (GlobalAveragePooling2D)        │                           │            │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dropout (Dropout)               │ (None, 128)               │          0 │
├─────────────────────────────────┼───────────────────────────┼────────────┤
│ dense (Dense)                   │ (None, 10)                │      1,290 │
└─────────────────────────────────┴───────────────────────────┴────────────┘
 Total params: 260,298 (1016.79 KB)
 Trainable params: 260,298 (1016.79 KB)
 Non-trainable params: 0 (0.00 B)
```

`compile()` 메서드를 사용하여 옵티마이저, 손실 함수, 모니터링할 메트릭을 지정합니다. JAX 및 TensorFlow 백엔드에서는 기본적으로 XLA 컴파일이 켜져 있다는 점에 유의하세요.

```python
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)
```

모델을 트레이닝하고 평가해 보겠습니다. 보지 않은 데이터에 대한 일반화를 모니터링하기 위해 트레이닝 중에 데이터의 15%에 해당하는 검증 분할을 따로 설정하겠습니다.

```python
batch_size = 128
epochs = 20

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="model_at_epoch_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)
score = model.evaluate(x_test, y_test, verbose=0)
```

```
Epoch 1/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 74s 184ms/step - acc: 0.4980 - loss: 1.3832 - val_acc: 0.9609 - val_loss: 0.1513
Epoch 2/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 74s 186ms/step - acc: 0.9245 - loss: 0.2487 - val_acc: 0.9702 - val_loss: 0.0999
Epoch 3/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 70s 175ms/step - acc: 0.9515 - loss: 0.1647 - val_acc: 0.9816 - val_loss: 0.0608
Epoch 4/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 69s 174ms/step - acc: 0.9622 - loss: 0.1247 - val_acc: 0.9833 - val_loss: 0.0541
Epoch 5/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 68s 171ms/step - acc: 0.9685 - loss: 0.1083 - val_acc: 0.9860 - val_loss: 0.0468
Epoch 6/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 70s 176ms/step - acc: 0.9710 - loss: 0.0955 - val_acc: 0.9897 - val_loss: 0.0400
Epoch 7/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 69s 172ms/step - acc: 0.9742 - loss: 0.0853 - val_acc: 0.9888 - val_loss: 0.0388
Epoch 8/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 68s 169ms/step - acc: 0.9789 - loss: 0.0738 - val_acc: 0.9902 - val_loss: 0.0387
Epoch 9/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 75s 187ms/step - acc: 0.9789 - loss: 0.0691 - val_acc: 0.9907 - val_loss: 0.0341
Epoch 10/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 77s 194ms/step - acc: 0.9806 - loss: 0.0636 - val_acc: 0.9907 - val_loss: 0.0348
Epoch 11/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 74s 186ms/step - acc: 0.9812 - loss: 0.0610 - val_acc: 0.9926 - val_loss: 0.0271
Epoch 12/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 219s 550ms/step - acc: 0.9820 - loss: 0.0590 - val_acc: 0.9912 - val_loss: 0.0294
Epoch 13/20
 399/399 ━━━━━━━━━━━━━━━━━━━━ 70s 176ms/step - acc: 0.9843 - loss: 0.0504 - val_acc: 0.9918 - val_loss: 0.0316
```

트레이닝 중에, 각 에포크가 끝날 때마다 모델을 저장했습니다. 아래처럼 모델을 최신 상태로 저장할 수도 있습니다:

```python
model.save("final_model.keras")
```

그리고 이렇게 다시 로드합니다:

```python
model = keras.saving.load_model("final_model.keras")
```

다음으로, `predict()`를 사용하여 클래스 확률 예측을 쿼리할 수 있습니다:

```python
predictions = model.predict(x_test)
```

```
 313/313 ━━━━━━━━━━━━━━━━━━━━ 3s 9ms/step
```

기본 사항은 여기까지입니다!

----

## 크로스 프레임워크 사용자 정의 컴포넌트 작성

Keras enables you to write custom Layers, Models, Metrics, Losses, and Optimizers that work across TensorFlow, JAX, and PyTorch with the same codebase. Let's take a look at custom layers first.

The `keras.ops` namespace contains:

*   An implementation of the NumPy API, e.g. [`keras.ops.stack`](/api/ops/numpy#stack-function) or [`keras.ops.matmul`](/api/ops/numpy#matmul-function).
*   A set of neural network specific ops that are absent from NumPy, such as [`keras.ops.conv`](/api/ops/nn#conv-function) or [`keras.ops.binary_crossentropy`](/api/ops/nn#binarycrossentropy-function).

Let's make a custom `Dense` layer that works with all backends:

```python
class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, name=None):
        super().__init__(name=name)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer=keras.initializers.GlorotNormal(),
            name="kernel",
            trainable=True,
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer=keras.initializers.Zeros(),
            name="bias",
            trainable=True,
        )

    def call(self, inputs):
        # Use Keras ops to create backend-agnostic layers/metrics/etc.
        x = keras.ops.matmul(inputs, self.w) + self.b
        return self.activation(x)
```

Next, let's make a custom `Dropout` layer that relies on the `keras.random` namespace:

```python
class MyDropout(keras.layers.Layer):
    def __init__(self, rate, name=None):
        super().__init__(name=name)
        self.rate = rate
        # Use seed_generator for managing RNG state.
        # It is a state element and its seed variable is
        # tracked as part of `layer.variables`.
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        # Use `keras.random` for random ops.
        return keras.random.dropout(inputs, self.rate, seed=self.seed_generator)
```

Next, let's write a custom subclassed model that uses our two custom layers:

```python
class MyModel(keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv_base = keras.Sequential(
            [
                keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                keras.layers.MaxPooling2D(pool_size=(2, 2)),
                keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
                keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
                keras.layers.GlobalAveragePooling2D(),
            ]
        )
        self.dp = MyDropout(0.5)
        self.dense = MyDense(num_classes, activation="softmax")

    def call(self, x):
        x = self.conv_base(x)
        x = self.dp(x)
        return self.dense(x)
```

Let's compile it and fit it:

```python
model = MyModel(num_classes=10)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=1,  # For speed
    validation_split=0.15,
)
```

```
 399/399 ━━━━━━━━━━━━━━━━━━━━ 70s 174ms/step - acc: 0.5104 - loss: 1.3473 - val_acc: 0.9256 - val_loss: 0.2484

<keras.src.callbacks.history.History at 0x105608670>
```

----

## Training models on arbitrary data sources

All Keras models can be trained and evaluated on a wide variety of data sources, independently of the backend you're using. This includes:

*   NumPy arrays
*   Pandas dataframes
*   TensorFlow [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) objects
*   PyTorch `DataLoader` objects
*   Keras `PyDataset` objects

They all work whether you're using TensorFlow, JAX, or PyTorch as your Keras backend.

Let's try it out with PyTorch `DataLoaders`:

```python
import torch

# Create a TensorDataset
train_torch_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_train), torch.from_numpy(y_train)
)
val_torch_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_test), torch.from_numpy(y_test)
)

# Create a DataLoader
train_dataloader = torch.utils.data.DataLoader(
    train_torch_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_torch_dataset, batch_size=batch_size, shuffle=False
)

model = MyModel(num_classes=10)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)
model.fit(train_dataloader, epochs=1, validation_data=val_dataloader)
```

```
 469/469 ━━━━━━━━━━━━━━━━━━━━ 81s 172ms/step - acc: 0.5502 - loss: 1.2550 - val_acc: 0.9419 - val_loss: 0.1972

<keras.src.callbacks.history.History at 0x2b3385480>
```

Now let's try this out with [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data):

```python
import tensorflow as tf

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)
test_dataset = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

model = MyModel(num_classes=10)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)
model.fit(train_dataset, epochs=1, validation_data=test_dataset)
```

```
 469/469 ━━━━━━━━━━━━━━━━━━━━ 81s 172ms/step - acc: 0.5771 - loss: 1.1948 - val_acc: 0.9229 - val_loss: 0.2502

<keras.src.callbacks.history.History at 0x2b33e7df0>
```

----

## Further reading

This concludes our short overview of the new multi-backend capabilities of Keras 3. Next, you can learn about:

### How to customize what happens in `fit()`

Want to implement a non-standard training algorithm yourself but still want to benefit from the power and usability of `fit()`? It's easy to customize `fit()` to support arbitrary use cases:

*   [Customizing what happens in `fit()` with TensorFlow](http://keras.io/guides/custom_train_step_in_tensorflow/)
*   [Customizing what happens in `fit()` with JAX](http://keras.io/guides/custom_train_step_in_jax/)
*   [Customizing what happens in `fit()` with PyTorch](http://keras.io/guides/custom_train_step_in_torch/)

* * *

How to write custom training loops
----------------------------------

*   [Writing a training loop from scratch in TensorFlow](http://keras.io/guides/writing_a_custom_training_loop_in_tensorflow/)
*   [Writing a training loop from scratch in JAX](http://keras.io/guides/writing_a_custom_training_loop_in_jax/)
*   [Writing a training loop from scratch in PyTorch](http://keras.io/guides/writing_a_custom_training_loop_in_torch/)

* * *

How to distribute training
--------------------------

*   [Guide to distributed training with TensorFlow](http://keras.io/guides/distributed_training_with_tensorflow/)
*   [JAX distributed training example](https://github.com/keras-team/keras/blob/master/examples/demo_jax_distributed.py)
*   [PyTorch distributed training example](https://github.com/keras-team/keras/blob/master/examples/demo_torch_multi_gpu.py)

Enjoy the library! 🚀