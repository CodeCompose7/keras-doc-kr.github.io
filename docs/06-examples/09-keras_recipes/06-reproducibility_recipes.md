---
layout: default
title: Keras 모델의 재현성
nav_order: 06+00
permalink: /examples/keras_recipes/reproducibility_recipes/
parent: 빠른 Keras 레시피
grand_parent: 코드 예제
---

* 원본 링크 : [https://keras.io/examples/keras_recipes/reproducibility_recipes/](https://keras.io/examples/keras_recipes/reproducibility_recipes/){:target="_blank"}
* 최종 수정일 : 2024-04-22

# Keras 모델의 재현성 (Reproducibility in Keras Models)
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

**저자:** [Frightera](https://github.com/Frightera)  
**생성일:** 2023/05/05  
**최종편집일:** 2023/05/05  
**설명:** Demonstration of random weight initialization and reproducibility in Keras models.

[Colab에서 보기](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/keras_recipes/ipynb/reproducibility_recipes.ipynb){: .btn .btn-blue }
[GitHub 소스](https://github.com/keras-team/keras-io/blob/master/examples/keras_recipes/reproducibility_recipes.py){: .btn .btn-blue }

ⓘ 이 예제는 Keras 3을 사용합니다.
{: .label .label-green .px-10}

----

## Introduction

This example demonstrates how to control randomness in Keras models. Sometimes you may want to reproduce the exact same results across runs, for experimentation purposes or to debug a problem.

* * *

Setup
-----

```python
import json
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras import initializers

# Set the seed using keras.utils.set_random_seed. This will set:
# 1) `numpy` seed
# 2) backend random seed
# 3) `python` random seed
keras.utils.set_random_seed(812)

# If using TensorFlow, this will make GPU ops as deterministic as possible,
# but it will affect the overall performance, so be mindful of that.
tf.config.experimental.enable_op_determinism()
```

* * *

Weight initialization in Keras
------------------------------

Most of the layers in Keras have `kernel_initializer` and `bias_initializer` parameters. These parameters allow you to specify the strategy used for initializing the weights of layer variables. The following built-in initializers are available as part of `keras.initializers`:

```python
initializers_list = [
    initializers.RandomNormal,
    initializers.RandomUniform,
    initializers.TruncatedNormal,
    initializers.VarianceScaling,
    initializers.GlorotNormal,
    initializers.GlorotUniform,
    initializers.HeNormal,
    initializers.HeUniform,
    initializers.LecunNormal,
    initializers.LecunUniform,
    initializers.Orthogonal,
]
```

In a reproducible model, the weights of the model should be initialized with same values in subsequent runs. First, we'll check how initializers behave when they are called multiple times with same `seed` value.

```python
for initializer in initializers_list:
    print(f"Running {initializer}")

    for iteration in range(2):
        # In order to get same results across multiple runs from an initializer,
        # you can specify a seed value.
        result = float(initializer(seed=42)(shape=(1, 1)))
        print(f"\tIteration --> {iteration} // Result --> {result}")
    print("\n")
```

```
Running <class 'keras.src.initializers.random_initializers.RandomNormal'>
    Iteration --> 0 // Result --> 0.05609520897269249
    Iteration --> 1 // Result --> 0.05609520897269249

Running <class 'keras.src.initializers.random_initializers.RandomUniform'>
    Iteration --> 0 // Result --> 0.03690483793616295
    Iteration --> 1 // Result --> 0.03690483793616295

Running <class 'keras.src.initializers.random_initializers.TruncatedNormal'>
    Iteration --> 0 // Result --> 0.05230803042650223
    Iteration --> 1 // Result --> 0.05230803042650223

Running <class 'keras.src.initializers.random_initializers.VarianceScaling'>
    Iteration --> 0 // Result --> 1.1893247365951538
    Iteration --> 1 // Result --> 1.1893247365951538

Running <class 'keras.src.initializers.random_initializers.GlorotNormal'>
    Iteration --> 0 // Result --> 1.1893247365951538
    Iteration --> 1 // Result --> 1.1893247365951538

Running <class 'keras.src.initializers.random_initializers.GlorotUniform'>
    Iteration --> 0 // Result --> 1.2784210443496704
    Iteration --> 1 // Result --> 1.2784210443496704

Running <class 'keras.src.initializers.random_initializers.HeNormal'>
    Iteration --> 0 // Result --> 1.6819592714309692
    Iteration --> 1 // Result --> 1.6819592714309692

Running <class 'keras.src.initializers.random_initializers.HeUniform'>
    Iteration --> 0 // Result --> 1.8079603910446167
    Iteration --> 1 // Result --> 1.8079603910446167

Running <class 'keras.src.initializers.random_initializers.LecunNormal'>
    Iteration --> 0 // Result --> 1.1893247365951538
    Iteration --> 1 // Result --> 1.1893247365951538

Running <class 'keras.src.initializers.random_initializers.LecunUniform'>
    Iteration --> 0 // Result --> 1.2784210443496704
    Iteration --> 1 // Result --> 1.2784210443496704

Running <class 'keras.src.initializers.random_initializers.OrthogonalInitializer'>
    Iteration --> 0 // Result --> 1.0
    Iteration --> 1 // Result --> 1.0
```

Now, let's inspect how two different initializer objects behave when they are have the same seed value.

```python
# Setting the seed value for an initializer will cause two different objects
# to produce same results.
glorot_normal_1 = keras.initializers.GlorotNormal(seed=42)
glorot_normal_2 = keras.initializers.GlorotNormal(seed=42)

input_dim, neurons = 3, 5

# Call two different objects with same shape
result_1 = glorot_normal_1(shape=(input_dim, neurons))
result_2 = glorot_normal_2(shape=(input_dim, neurons))

# Check if the results are equal.
equal = np.allclose(result_1, result_2)
print(f"Are the results equal? {equal}")
```

```
Are the results equal? True
```

If the seed value is not set (or different seed values are used), two different objects will produce different results. Since the random seed is set at the beginning of the notebook, the results will be same in the sequential runs. This is related to the [`keras.utils.set_random_seed`](/api/utils/python_utils#setrandomseed-function).

```python
glorot_normal_3 = keras.initializers.GlorotNormal()
glorot_normal_4 = keras.initializers.GlorotNormal()

# Let's call the initializer.
result_3 = glorot_normal_3(shape=(input_dim, neurons))

# Call the second initializer.
result_4 = glorot_normal_4(shape=(input_dim, neurons))

equal = np.allclose(result_3, result_4)
print(f"Are the results equal? {equal}")
```

```
Are the results equal? False
```

`result_3` and `result_4` will be different, but when you run the notebook again, `result_3` will have identical values to the ones in the previous run. Same goes for `result_4`.

* * *

Reproducibility in model training process
-----------------------------------------

If you want to reproduce the results of a model training process, you need to control the randomness sources during the training process. In order to show a realistic example, this section utilizes [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) using parallel map and shuffle operations.

In order to start, let's create a simple function which returns the history object of the Keras model.

```python
def train_model(train_data: tf.data.Dataset, test_data: tf.data.Dataset) -> dict:
    model = keras.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    # model.fit has a `shuffle` parameter which has a default value of `True`.
    # If you are using array-like objects, this will shuffle the data before
    # training. This argument is ignored when `x` is a generator or
    # [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
    history = model.fit(train_data, epochs=2, validation_data=test_data)

    print(f"Model accuracy on test data: {model.evaluate(test_data)[1] * 100:.2f}%")

    return history.history


# Load the MNIST dataset
(train_images, train_labels), (
    test_images,
    test_labels,
) = keras.datasets.mnist.load_data()

# Construct tf.data.Dataset objects
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
```

Remember we called `tf.config.experimental.enable_op_determinism()` at the beginning of the function. This makes the [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) operations deterministic. However, making [`tf.data`](https://www.tensorflow.org/api_docs/python/tf/data) operations deterministic comes with a performance cost. If you want to learn more about it, please check this [official guide](https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism#determinism_and_tfdata).

Small summary what's going on here. Models have `kernel_initializer` and `bias_initializer` parameters. Since we set random seeds using [`keras.utils.set_random_seed`](/api/utils/python_utils#setrandomseed-function) in the beginning of the notebook, the initializers will produce same results in the sequential runs. Additionally, TensorFlow operations have now become deterministic. Frequently, you will be utilizing GPUs that have thousands of hardware threads which causes non-deterministic behavior to occur.

```python
def prepare_dataset(image, label):
    # Cast and normalize the image
    image = tf.cast(image, tf.float32) / 255.0

    # Expand the channel dimension
    image = tf.expand_dims(image, axis=-1)

    # Resize the image
    image = tf.image.resize(image, (32, 32))

    return image, label
```

[`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) objects have a `shuffle` method which shuffles the data. This method has a `buffer_size` parameter which controls the size of the buffer. If you set this value to `len(train_images)`, the whole dataset will be shuffled. If the buffer size is equal to the length of the dataset, then the elements will be shuffled in a completely random order.

Main drawback of setting the buffer size to the length of the dataset is that filling the buffer can take a while depending on the size of the dataset.

Here is a small summary of what's going on here: 1) The `shuffle()` method creates a buffer of the specified size. 2) The elements of the dataset are randomly shuffled and placed into the buffer. 3) The elements of the buffer are then returned in a random order.

Since `tf.config.experimental.enable_op_determinism()` is enabled and we set random seeds using [`keras.utils.set_random_seed`](/api/utils/python_utils#setrandomseed-function) in the beginning of the notebook, the `shuffle()` method will produce same results in the sequential runs.

```python
# Prepare the datasets, batch-map --> vectorized operations
train_data = (
    train_ds.shuffle(buffer_size=len(train_images))
    .batch(batch_size=64)
    .map(prepare_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_data = (
    test_ds.batch(batch_size=64)
    .map(prepare_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)
```

Train the model for the first time.

```python
history = train_model(train_data, test_data)
```

```
Epoch 1/2
 938/938 ━━━━━━━━━━━━━━━━━━━━ 26s 27ms/step - accuracy: 0.5418 - loss: 1.2867 - val_accuracy: 0.9291 - val_loss: 0.2303
Epoch 2/2
 938/938 ━━━━━━━━━━━━━━━━━━━━ 25s 26ms/step - accuracy: 0.9075 - loss: 0.2983 - val_accuracy: 0.9583 - val_loss: 0.1343
 157/157 ━━━━━━━━━━━━━━━━━━━━ 1s 4ms/step - accuracy: 0.9512 - loss: 0.1559
Model accuracy on test data: 95.83%
```

Let's save our results into a JSON file, and restart the kernel. After restarting the kernel, we should see the same results as the previous run, this includes metrics and loss values both on the training and test data.

```python
# Save the history object into a json file
with open("history.json", "w") as fp:
    json.dump(history, fp)
```

Do not run the cell above in order not to overwrite the results. Execute the model training cell again and compare the results.

```python
with open("history.json", "r") as fp:
    history_loaded = json.load(fp)
```

Compare the results one by one. You will see that they are equal.

```python
for key in history.keys():
    for i in range(len(history[key])):
        if not np.allclose(history[key][i], history_loaded[key][i]):
            print(f"{key} not equal")
```

* * *

Conclusion
----------

In this tutorial, you learned how to control the randomness sources in Keras and TensorFlow. You also learned how to reproduce the results of a model training process.

If you want to initialize the model with the same weights everytime, you need to set `kernel_initializer` and `bias_initializer` parameters of the layers and provide a `seed` value to the initializer.

There still may be some inconsistencies due to numerical error accumulation such as using `recurrent_dropout` in RNN layers.

Reproducibility is subject to the environment. You'll get the same results if you run the notebook or the code on the same machine with the same environment.