---
layout: default
title: KerasNLP 및 tf.distribute를 사용한 데이터 병렬 트레이닝
nav_order: 10+00
permalink: /examples/nlp/data_parallel_training_with_keras_nlp/
parent: 자연어 처리
grand_parent: 코드 예제
---

* 원본 링크 : [https://keras.io/examples/nlp/data_parallel_training_with_keras_nlp/](https://keras.io/examples/nlp/data_parallel_training_with_keras_nlp/){:target="_blank"}
* 최종 수정일 : 2024-04-12

# KerasNLP 및 tf.distribute를 사용한 데이터 병렬 트레이닝 (Data Parallel Training with KerasNLP and tf.distribute)
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

**저자:** Anshuman Mishra  
**생성일:** 2023/07/07  
**최종편집일:** 2023/07/07  
**설명:** Data Parallel training with KerasNLP and tf.distribute.

[Colab에서 보기](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/data_parallel_training_with_keras_nlp.ipynb){: .btn .btn-blue }
[GitHub 소스](https://github.com/keras-team/keras-io/blob/master/examples/nlp/data_parallel_training_with_keras_nlp.py){: .btn .btn-blue }

ⓘ 이 예제는 Keras 3을 사용합니다.
{: .label .label-green .px-10}

----

## Introduction

Distributed training is a technique used to train deep learning models on multiple devices or machines simultaneously. It helps to reduce training time and allows for training larger models with more data. KerasNLP is a library that provides tools and utilities for natural language processing tasks, including distributed training.

In this tutorial, we will use KerasNLP to train a BERT-based masked language model (MLM) on the wikitext-2 dataset (a 2 million word dataset of wikipedia articles). The MLM task involves predicting the masked words in a sentence, which helps the model learn contextual representations of words.

This guide focuses on data parallelism, in particular synchronous data parallelism, where each accelerator (a GPU or TPU) holds a complete replica of the model, and sees a different partial batch of the input data. Partial gradients are computed on each device, aggregated, and used to compute a global gradient update.

Specifically, this guide teaches you how to use the [`tf.distribute`](https://www.tensorflow.org/api_docs/python/tf/distribute) API to train Keras models on multiple GPUs, with minimal changes to your code, in the following two setups:

*   On multiple GPUs (typically 2 to 8) installed on a single machine (single host, multi-device training). This is the most common setup for researchers and small-scale industry workflows.
*   On a cluster of many machines, each hosting one or multiple GPUs (multi-worker distributed training). This is a good setup for large-scale industry workflows, e.g. training high-resolution text summarization models on billion word datasets on 20-100 GPUs.

```python
!pip install -q --upgrade keras-nlp
!pip install -q --upgrade keras  # Upgrade to Keras 3.
```

* * *

Imports
-------

```python
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
import keras_nlp
```

Before we start any training, let's configure our single GPU to show up as two logical devices.

When you are training with two or more physical GPUs, this is totally uncessary. This is just a trick to show real distributed training on the default colab GPU runtime, which has only one GPU available.

```python
!nvidia-smi --query-gpu=memory.total --format=csv,noheader
```

```python
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.set_logical_device_configuration(
    physical_devices[0],
    [
        tf.config.LogicalDeviceConfiguration(memory_limit=15360 // 2),
        tf.config.LogicalDeviceConfiguration(memory_limit=15360 // 2),
    ],
)

logical_devices = tf.config.list_logical_devices("GPU")
logical_devices

EPOCHS = 3
```

```
24576 MiB
```

To do single-host, multi-device synchronous training with a Keras model, you would use the [`tf.distribute.MirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy) API. Here's how it works:

*   Instantiate a `MirroredStrategy`, optionally configuring which specific devices you want to use (by default the strategy will use all GPUs available).
*   Use the strategy object to open a scope, and within this scope, create all the Keras objects you need that contain variables. Typically, that means **creating & compiling the model** inside the distribution scope.
*   Train the model via `fit()` as usual.

```python
strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")
```

```
INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1')
Number of devices: 2
```

Base batch size and learning rate

```python
base_batch_size = 32
base_learning_rate = 1e-4
```

Calculate scaled batch size and learning rate

```python
scaled_batch_size = base_batch_size * strategy.num_replicas_in_sync
scaled_learning_rate = base_learning_rate * strategy.num_replicas_in_sync
```

Now, we need to download and preprocess the wikitext-2 dataset. This dataset will be used for pretraining the BERT model. We will filter out short lines to ensure that the data has enough context for training.

```python
keras.utils.get_file(
    origin="https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip",
    extract=True,
)
wiki_dir = os.path.expanduser("~/.keras/datasets/wikitext-2/")

# Load wikitext-103 and filter out short lines.
wiki_train_ds = (
    tf.data.TextLineDataset(
        wiki_dir + "wiki.train.tokens",
    )
    .filter(lambda x: tf.strings.length(x) > 100)
    .shuffle(buffer_size=500)
    .batch(scaled_batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
wiki_val_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.valid.tokens")
    .filter(lambda x: tf.strings.length(x) > 100)
    .shuffle(buffer_size=500)
    .batch(scaled_batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
wiki_test_ds = (
    tf.data.TextLineDataset(wiki_dir + "wiki.test.tokens")
    .filter(lambda x: tf.strings.length(x) > 100)
    .shuffle(buffer_size=500)
    .batch(scaled_batch_size)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
```

In the above code, we download the wikitext-2 dataset and extract it. Then, we define three datasets: wiki\_train\_ds, wiki\_val\_ds, and wiki\_test\_ds. These datasets are filtered to remove short lines and are batched for efficient training.

It's a common practice to use a decayed learning rate in NLP training/tuning. We'll use `PolynomialDecay` schedule here.

```python
total_training_steps = sum(1 for _ in wiki_train_ds.as_numpy_iterator()) * EPOCHS
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=scaled_learning_rate,
    decay_steps=total_training_steps,
    end_learning_rate=0.0,
)


class PrintLR(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(
            f"\nLearning rate for epoch {epoch + 1} is {model_dist.optimizer.learning_rate.numpy()}"
        )
```

Let's also make a callback to TensorBoard, this will enable visualization of different metrics while we train the model in later part of this tutorial. We put all the callbacks together as follows:

```python
callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir="./logs"),
    PrintLR(),
]


print(tf.config.list_physical_devices("GPU"))
```

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

With the datasets prepared, we now initialize and compile our model and optimizer within the `strategy.scope()`:

```python
with strategy.scope():
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    model_dist = keras_nlp.models.BertMaskedLM.from_preset("bert_tiny_en_uncased")

    # This line just sets pooled_dense layer as non-trainiable, we do this to avoid
    # warnings of this layer being unused
    model_dist.get_layer("bert_backbone").get_layer("pooled_dense").trainable = False

    model_dist.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.AdamW(learning_rate=scaled_learning_rate),
        weighted_metrics=[keras.metrics.SparseCategoricalAccuracy()],
        jit_compile=False,
    )

    model_dist.fit(
        wiki_train_ds, validation_data=wiki_val_ds, epochs=EPOCHS, callbacks=callbacks
    )
```

```
Epoch 1/3
Learning rate for epoch 1 is 0.00019999999494757503
 239/239 ━━━━━━━━━━━━━━━━━━━━ 43s 136ms/step - loss: 3.7009 - sparse_categorical_accuracy: 0.1499 - val_loss: 1.1509 - val_sparse_categorical_accuracy: 0.3485
Epoch 2/3
 239/239 ━━━━━━━━━━━━━━━━━━━━ 0s 122ms/step - loss: 2.6094 - sparse_categorical_accuracy: 0.5284
Learning rate for epoch 2 is 0.00019999999494757503
 239/239 ━━━━━━━━━━━━━━━━━━━━ 32s 133ms/step - loss: 2.6038 - sparse_categorical_accuracy: 0.5274 - val_loss: 0.9812 - val_sparse_categorical_accuracy: 0.4006
Epoch 3/3
 239/239 ━━━━━━━━━━━━━━━━━━━━ 0s 123ms/step - loss: 2.3564 - sparse_categorical_accuracy: 0.6053
Learning rate for epoch 3 is 0.00019999999494757503
 239/239 ━━━━━━━━━━━━━━━━━━━━ 32s 134ms/step - loss: 2.3514 - sparse_categorical_accuracy: 0.6040 - val_loss: 0.9213 - val_sparse_categorical_accuracy: 0.4230
```

After fitting our model under the scope, we evaluate it normally!

```python
model_dist.evaluate(wiki_test_ds)
```

```
 29/29 ━━━━━━━━━━━━━━━━━━━━ 3s 60ms/step - loss: 1.9197 - sparse_categorical_accuracy: 0.8527

[0.9470901489257812, 0.4373602867126465]
```

For distributed training across multiple machines (as opposed to training that only leverages multiple devices on a single machine), there are two distribution strategies you could use: `MultiWorkerMirroredStrategy` and `ParameterServerStrategy`:

*   [`tf.distribute.MultiWorkerMirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy) implements a synchronous CPU/GPU multi-worker solution to work with Keras-style model building and training loop, using synchronous reduction of gradients across the replicas.
*   [`tf.distribute.experimental.ParameterServerStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/ParameterServerStrategy) implements an asynchronous CPU/GPU multi-worker solution, where the parameters are stored on parameter servers, and workers update the gradients to parameter servers asynchronously.

### Further reading

1.  [TensorFlow distributed training guide](https://www.tensorflow.org/guide/distributed_training)
2.  [Tutorial on multi-worker training with Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)
3.  [MirroredStrategy docs](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)
4.  [MultiWorkerMirroredStrategy docs](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy)
5.  [Distributed training in tf.keras with Weights & Biases](https://towardsdatascience.com/distributed-training-in-tf-keras-with-w-b-ccf021f9322e)
