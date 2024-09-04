---
layout: default
title: 회귀 손실
nav_order: 02+00
permalink: /api/losses/regression_losses/
parent: Losses
grand_parent: Keras 3 API 문서
---

* 원본 링크 : [https://keras.io/api/losses/regression_losses/](https://keras.io/api/losses/regression_losses/){:target="_blank"}
* 최종 수정일 : 2024-09-05

# 회귀 손실 (Regression losses)
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

### `MeanSquaredError` class
<!-- ### `MeanSquaredError` class -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L42){: .btn .btn-outline }

```python
keras.losses.MeanSquaredError(
    reduction="sum_over_batch_size", name="mean_squared_error", dtype=None
)
```

Computes the mean of squares of errors between labels and predictions.

Formula:

```python
loss = mean(square(y_true - y_pred))
```

**Arguments**

*   **reduction**: Type of reduction to apply to the loss. In almost all cases this should be `"sum_over_batch_size"`. Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
*   **name**: Optional name for the loss instance.
*   **dtype**: The dtype of the loss's computations. Defaults to `None`, which means using `keras.backend.floatx()`. `keras.backend.floatx()` is a `"float32"` unless set to different value (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is provided, then the `compute_dtype` will be utilized.

* * *

### `MeanAbsoluteError` class
<!-- ### `MeanAbsoluteError` class -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L78){: .btn .btn-outline }

```python
keras.losses.MeanAbsoluteError(
    reduction="sum_over_batch_size", name="mean_absolute_error", dtype=None
)
```

Computes the mean of absolute difference between labels and predictions.

Formula:

```python
loss = mean(abs(y_true - y_pred))
```

**Arguments**

*   **reduction**: Type of reduction to apply to the loss. In almost all cases this should be `"sum_over_batch_size"`. Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
*   **name**: Optional name for the loss instance.
*   **dtype**: The dtype of the loss's computations. Defaults to `None`, which means using `keras.backend.floatx()`. `keras.backend.floatx()` is a `"float32"` unless set to different value (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is provided, then the `compute_dtype` will be utilized.

* * *

### `MeanAbsolutePercentageError` class
<!-- ### `MeanAbsolutePercentageError` class -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L114){: .btn .btn-outline }

```python
keras.losses.MeanAbsolutePercentageError(
    reduction="sum_over_batch_size", name="mean_absolute_percentage_error", dtype=None
)
```

Computes the mean absolute percentage error between `y_true` & `y_pred`.

Formula:

```python
loss = 100 * mean(abs((y_true - y_pred) / y_true))
```

**Arguments**

*   **reduction**: Type of reduction to apply to the loss. In almost all cases this should be `"sum_over_batch_size"`. Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
*   **name**: Optional name for the loss instance.
*   **dtype**: The dtype of the loss's computations. Defaults to `None`, which means using `keras.backend.floatx()`. `keras.backend.floatx()` is a `"float32"` unless set to different value (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is provided, then the `compute_dtype` will be utilized.

* * *

### `MeanSquaredLogarithmicError` class
<!-- ### `MeanSquaredLogarithmicError` class -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L153){: .btn .btn-outline }

```python
keras.losses.MeanSquaredLogarithmicError(
    reduction="sum_over_batch_size", name="mean_squared_logarithmic_error", dtype=None
)
```

Computes the mean squared logarithmic error between `y_true` & `y_pred`.

Formula:

```python
loss = mean(square(log(y_true + 1) - log(y_pred + 1)))
```

**Arguments**

*   **reduction**: Type of reduction to apply to the loss. In almost all cases this should be `"sum_over_batch_size"`. Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
*   **name**: Optional name for the loss instance.
*   **dtype**: The dtype of the loss's computations. Defaults to `None`, which means using `keras.backend.floatx()`. `keras.backend.floatx()` is a `"float32"` unless set to different value (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is provided, then the `compute_dtype` will be utilized.

* * *

### `CosineSimilarity` class
<!-- ### `CosineSimilarity` class -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L192){: .btn .btn-outline }

```python
keras.losses.CosineSimilarity(
    axis=-1, reduction="sum_over_batch_size", name="cosine_similarity", dtype=None
)
```

Computes the cosine similarity between `y_true` & `y_pred`.

Note that it is a number between -1 and 1. When it is a negative number between -1 and 0, 0 indicates orthogonality and values closer to -1 indicate greater similarity. This makes it usable as a loss function in a setting where you try to maximize the proximity between predictions and targets. If either `y_true` or `y_pred` is a zero vector, cosine similarity will be 0 regardless of the proximity between predictions and targets.

Formula:

```python
loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
```

**Arguments**

*   **axis**: The axis along which the cosine similarity is computed (the features axis). Defaults to `-1`.
*   **reduction**: Type of reduction to apply to the loss. In almost all cases this should be `"sum_over_batch_size"`. Supported options are `"sum"`, `"sum_over_batch_size"` or `None`.
*   **name**: Optional name for the loss instance.
*   **dtype**: The dtype of the loss's computations. Defaults to `None`, which means using `keras.backend.floatx()`. `keras.backend.floatx()` is a `"float32"` unless set to different value (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is provided, then the `compute_dtype` will be utilized.

* * *

### `mean_squared_error` function
<!-- ### `mean_squared_error` function -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L1267){: .btn .btn-outline }

```python
keras.losses.mean_squared_error(y_true, y_pred)
```

Computes the mean squared error between labels and predictions.

Formula:

```python
loss = mean(square(y_true - y_pred), axis=-1)
```

**Example**

```python
>>> y_true = np.random.randint(0, 2, size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.mean_squared_error(y_true, y_pred)
```

**Arguments**

*   **y\_true**: Ground truth values with shape = `[batch_size, d0, .. dN]`.
*   **y\_pred**: The predicted values with shape = `[batch_size, d0, .. dN]`.

**Returns**

Mean squared error values with shape = `[batch_size, d0, .. dN-1]`.

* * *

### `mean_absolute_error` function
<!-- ### `mean_absolute_error` function -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L1306){: .btn .btn-outline }

```python
keras.losses.mean_absolute_error(y_true, y_pred)
```

Computes the mean absolute error between labels and predictions.

```python
loss = mean(abs(y_true - y_pred), axis=-1)
```

**Arguments**

*   **y\_true**: Ground truth values with shape = `[batch_size, d0, .. dN]`.
*   **y\_pred**: The predicted values with shape = `[batch_size, d0, .. dN]`.

**Returns**

Mean absolute error values with shape = `[batch_size, d0, .. dN-1]`.

**Example**

```python
>>> y_true = np.random.randint(0, 2, size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.mean_absolute_error(y_true, y_pred)
```

* * *

### `mean_absolute_percentage_error` function
<!-- ### `mean_absolute_percentage_error` function -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L1343){: .btn .btn-outline }

```python
keras.losses.mean_absolute_percentage_error(y_true, y_pred)
```

Computes the mean absolute percentage error between `y_true` & `y_pred`.

Formula:

```python
loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)
```

Division by zero is prevented by dividing by `maximum(y_true, epsilon)` where `epsilon = keras.backend.epsilon()` (default to `1e-7`).

**Arguments**

*   **y\_true**: Ground truth values with shape = `[batch_size, d0, .. dN]`.
*   **y\_pred**: The predicted values with shape = `[batch_size, d0, .. dN]`.

**Returns**

Mean absolute percentage error values with shape = `[batch_size, d0, .. dN-1]`.

**Example**

```python
>>> y_true = np.random.random(size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.mean_absolute_percentage_error(y_true, y_pred)
```

* * *

### `mean_squared_logarithmic_error` function
<!-- ### `mean_squared_logarithmic_error` function -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L1389){: .btn .btn-outline }

```python
keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
```

Computes the mean squared logarithmic error between `y_true` & `y_pred`.

Formula:

```python
loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)
```

Note that `y_pred` and `y_true` cannot be less or equal to 0. Negative values and 0 values will be replaced with `keras.backend.epsilon()` (default to `1e-7`).

**Arguments**

*   **y\_true**: Ground truth values with shape = `[batch_size, d0, .. dN]`.
*   **y\_pred**: The predicted values with shape = `[batch_size, d0, .. dN]`.

**Returns**

Mean squared logarithmic error values with shape = `[batch_size, d0, .. dN-1]`.

**Example**

```python
>>> y_true = np.random.randint(0, 2, size=(2, 3))
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
```

* * *

### `cosine_similarity` function
<!-- ### `cosine_similarity` function -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L1436){: .btn .btn-outline }

```python
keras.losses.cosine_similarity(y_true, y_pred, axis=-1)
```

Computes the cosine similarity between labels and predictions.

Formula:

```python
loss = -sum(l2_norm(y_true) * l2_norm(y_pred))
```

Note that it is a number between -1 and 1. When it is a negative number between -1 and 0, 0 indicates orthogonality and values closer to -1 indicate greater similarity. This makes it usable as a loss function in a setting where you try to maximize the proximity between predictions and targets. If either `y_true` or `y_pred` is a zero vector, cosine similarity will be 0 regardless of the proximity between predictions and targets.

**Arguments**

*   **y\_true**: Tensor of true targets.
*   **y\_pred**: Tensor of predicted targets.
*   **axis**: Axis along which to determine similarity. Defaults to `-1`.

**Returns**

Cosine similarity tensor.

**Example**

```python
>>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
>>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
>>> loss = keras.losses.cosine_similarity(y_true, y_pred, axis=-1)
[-0., -0.99999994, 0.99999994]
```

* * *

### `Huber` class
<!-- ### `Huber` class -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L242){: .btn .btn-outline }

```python
keras.losses.Huber(
    delta=1.0, reduction="sum_over_batch_size", name="huber_loss", dtype=None
)
```

Computes the Huber loss between `y_true` & `y_pred`.

Formula:

```python
for x in error:
    if abs(x) <= delta:
        loss.append(0.5 * x^2)
    elif abs(x) > delta:
        loss.append(delta * abs(x) - 0.5 * delta^2)

loss = mean(loss, axis=-1)
```

See: [Huber loss](https://en.wikipedia.org/wiki/Huber_loss).

**Arguments**

*   **delta**: A float, the point where the Huber loss function changes from a quadratic to linear.
*   **reduction**: Type of reduction to apply to loss. Options are `"sum"`, `"sum_over_batch_size"` or `None`. Defaults to `"sum_over_batch_size"`.
*   **name**: Optional name for the instance.
*   **dtype**: The dtype of the loss's computations. Defaults to `None`, which means using `keras.backend.floatx()`. `keras.backend.floatx()` is a `"float32"` unless set to different value (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is provided, then the `compute_dtype` will be utilized.

* * *

### `huber` function
<!-- ### `huber` function -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L1476){: .btn .btn-outline }

```python
keras.losses.huber(y_true, y_pred, delta=1.0)
```

Computes Huber loss value.

Formula:

```python
for x in error:
    if abs(x) <= delta:
        loss.append(0.5 * x^2)
    elif abs(x) > delta:
        loss.append(delta * abs(x) - 0.5 * delta^2)

loss = mean(loss, axis=-1)
```

See: [Huber loss](https://en.wikipedia.org/wiki/Huber_loss).

**Example**

```python
>>> y_true = [[0, 1], [0, 0]]
>>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
>>> loss = keras.losses.huber(y_true, y_pred)
0.155
```

**Arguments**

*   **y\_true**: tensor of true targets.
*   **y\_pred**: tensor of predicted targets.
*   **delta**: A float, the point where the Huber loss function changes from a quadratic to linear. Defaults to `1.0`.

**Returns**

Tensor with one scalar loss entry per sample.

* * *

### `LogCosh` class
<!-- ### `LogCosh` class -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L288){: .btn .btn-outline }

```python
keras.losses.LogCosh(reduction="sum_over_batch_size", name="log_cosh", dtype=None)
```

Computes the logarithm of the hyperbolic cosine of the prediction error.

Formula:

```python
error = y_pred - y_true
logcosh = mean(log((exp(error) + exp(-error))/2), axis=-1)
```

where x is the error `y_pred - y_true`.

**Arguments**

*   **reduction**: Type of reduction to apply to loss. Options are `"sum"`, `"sum_over_batch_size"` or `None`. Defaults to `"sum_over_batch_size"`.
*   **name**: Optional name for the instance.
*   **dtype**: The dtype of the loss's computations. Defaults to `None`, which means using `keras.backend.floatx()`. `keras.backend.floatx()` is a `"float32"` unless set to different value (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is provided, then the `compute_dtype` will be utilized.

* * *

### `log_cosh` function
<!-- ### `log_cosh` function -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/losses/losses.py#L1526){: .btn .btn-outline }

```python
keras.losses.log_cosh(y_true, y_pred)
```

Logarithm of the hyperbolic cosine of the prediction error.

Formula:

```python
loss = mean(log(cosh(y_pred - y_true)), axis=-1)
```

Note that `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly like the mean squared error, but will not be so strongly affected by the occasional wildly incorrect prediction.

**Example**

```python
>>> y_true = [[0., 1.], [0., 0.]]
>>> y_pred = [[1., 1.], [0., 0.]]
>>> loss = keras.losses.log_cosh(y_true, y_pred)
0.108
```

**Arguments**

*   **y\_true**: Ground truth values with shape = `[batch_size, d0, .. dN]`.
*   **y\_pred**: The predicted values with shape = `[batch_size, d0, .. dN]`.

**Returns**

Logcosh error values with shape = `[batch_size, d0, .. dN-1]`.

* * *