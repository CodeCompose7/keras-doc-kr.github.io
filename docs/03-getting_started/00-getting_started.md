---
layout: default
title: 시작하기
nav_order: 3
permalink: /getting_started
has_children: true
---

* 원본 링크 : [https://keras.io/getting_started/](https://keras.io/getting_started/){:target="_blank"}
* 최종 수정일 : 2024-03-28

# Keras로 시작하기
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## 학습 리소스
------------------

Keras 소개 한 페이지짜리 가이드를 찾고 계신 머신러닝 엔지니어이신가요? [엔지니어를 위한 Keras 소개](getting_started/intro_to_keras_for_engineers/) 가이드를 읽어보세요.

Keras 3와 그 기능에 대해 더 자세히 알고 싶으신가요? [Keras 3 출시 발표](/)를 참조하세요.

Keras API의 다양한 부분에 대한 심층적인 사용법을 다루는 자세한 가이드를 찾고 계신가요? [Keras 개발자 가이드](guides/)를 읽어보세요.

다양한 사용 사례에서 Keras가 실제로 작동하는 모습을 보여주는 튜토리얼을 찾고 계신가요? 컴퓨터 비전, 자연어 처리, 생성 AI 분야에서 Keras 모범 사례를 보여주는 150개 이상의 잘 설명된 노트북인 [Keras 코드 예제](examples/)를 참조하세요.

----

## 케라스 3 설치하기

PyPI에서 다음을 통해 Keras를 설치할 수 있습니다:

```shell
pip install --upgrade keras
```

다음을 통해 로컬 Keras 버전 번호를 확인할 수 있습니다:

```python
import keras
print(keras.__version__)
```

Keras 3를 사용하려면 백엔드 프레임워크(JAX, TensorFlow 또는 PyTorch)도 설치해야 합니다:

* [JAX 설치하기](https://jax.readthedocs.io/en/latest/installation.html)
* [TensorFlow 설치](https://www.tensorflow.org/install)
* [PyTorch 설치](https://pytorch.org/get-started/locally/)

텐서플로우 2.15를 설치한 경우, 이후 케라스 3를 재설치해야 합니다. 그 이유는 `tensorflow==2.15`가 Keras 설치를 `keras==2.15`로 덮어쓰기 때문입니다. 텐서플로우 2.16 버전부터는 기본적으로 케라스 3이 설치되므로 이 단계는 필요하지 않습니다.

### Installing KerasCV and KerasNLP

KerasCV and KerasNLP can be installed via pip:

```shell
pip install --upgrade keras-cv
pip install --upgrade keras-nlp
pip install --upgrade keras
```

----

## Configuring your backend

You can export the environment variable `KERAS_BACKEND` or you can edit your local config file at `~/.keras/keras.json` to configure your backend. Available backend options are: `"jax"`, `"tensorflow"`, `"torch"`. Example:

```
export KERAS_BACKEND="jax"
```

In Colab, you can do:

```python
import os
os.environ["KERAS_BACKEND"] = "jax"
import keras
```

**Note:** The backend must be configured before importing Keras, and the backend cannot be changed after the package has been imported.

### GPU dependencies

#### Colab or Kaggle

If you are running on Colab or Kaggle, the GPU should already be configured, with the correct CUDA version. Installing a newer version of CUDA on Colab or Kaggle is typically not possible. Even though pip installers exist, they rely on a pre-installed NVIDIA driver and there is no way to update the driver on Colab or Kaggle.

#### Universal GPU environment

If you want to attempt to create a "universal environment" where any backend can use the GPU, we recommend following [the dependency versions used by Colab](https://colab.sandbox.google.com/drive/13cpd3wCwEHpsmypY9o6XB6rXgBm5oSxu) (which seeks to solve this exact problem). You can install the CUDA driver [from here](https://developer.nvidia.com/cuda-downloads), then pip install backends by following their respective CUDA installation instructions: [Installing JAX](https://jax.readthedocs.io/en/latest/installation.html), [Installing TensorFlow](https://www.tensorflow.org/install), [Installing PyTorch](https://pytorch.org/get-started/locally/)

#### Most stable GPU environment

This setup is recommended if you are a Keras contributor and are running Keras tests. It installs all backends but only gives GPU access to one backend at a time, avoiding potentially conflicting dependency requirements between backends. You can use the following backend-specific requirements files:

*   [requirements-jax-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-jax-cuda.txt)
*   [requirements-tensorflow-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-tensorflow-cuda.txt)
*   [requirements-torch-cuda.txt](https://github.com/keras-team/keras/blob/master/requirements-torch-cuda.txt)

These install all CUDA-enabled dependencies via pip. They expect a NVIDIA driver to be preinstalled. We recommend a clean python environment for each backend to avoid CUDA version mismatches. As an example, here is how to create a JAX GPU environment with [Conda](https://docs.conda.io/en/latest/):

```shell
conda create -y -n keras-jax python=3.10
conda activate keras-jax
pip install -r requirements-jax-cuda.txt
pip install --upgrade keras
```

----

## TensorFlow + Keras 2 backwards compatibility

From TensorFlow 2.0 to TensorFlow 2.15 (included), doing `pip install tensorflow` will also install the corresponding version of Keras 2 – for instance, `pip install tensorflow==2.14.0` will install `keras==2.14.0`. That version of Keras is then available via both `import keras` and `from tensorflow import keras` (the [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras) namespace).

Starting with TensorFlow 2.16, doing `pip install tensorflow` will install Keras 3. When you have TensorFlow >= 2.16 and Keras 3, then by default `from tensorflow import keras` ([`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras)) will be Keras 3.

Meanwhile, the legacy Keras 2 package is still being released regularly and is available on PyPI as `tf_keras` (or equivalently `tf-keras` – note that `-` and `_` are equivalent in PyPI package names). To use it, you can install it via `pip install tf_keras` then import it via `import tf_keras as keras`.

Should you want [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras) to stay on Keras 2 after upgrading to TensorFlow 2.16+, you can configure your TensorFlow installation so that [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras) points to `tf_keras`. To achieve this:

1.  Make sure to install `tf_keras`. Note that TensorFlow does not install it by default.
2.  Export the environment variable `TF_USE_LEGACY_KERAS=1`.

There are several ways to export the environment variable:

1.  You can simply run the shell command `export TF_USE_LEGACY_KERAS=1` before launching the Python interpreter.
2.  You can add `export TF_USE_LEGACY_KERAS=1` to your `.bashrc` file. That way the variable will still be exported when you restart your shell.
3.  You can start your Python script with:

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

These lines would need to be before any `import tensorflow` statement.

----

## Compatibility matrix

### JAX compatibility

The following Keras + JAX versions are compatible with each other:

*   `jax==0.4.20` & `keras~=3.0`

### TensorFlow compatibility

The following Keras + TensorFlow versions are compatible with each other:

To use Keras 2:

*   `tensorflow~=2.13.0` & `keras~=2.13.0`
*   `tensorflow~=2.14.0` & `keras~=2.14.0`
*   `tensorflow~=2.15.0` & `keras~=2.15.0`

To use Keras 3:

*   `tensorflow~=2.16.1` & `keras~=3.0`

### PyTorch compatibility

The following Keras + PyTorch versions are compatible with each other:

*   `torch~=2.1.0` & `keras~=3.0`