---
layout: default
title: └ 9) UpSampling2D 레이어
nav_order: 14+09
permalink: /api/layers/reshaping_layers/up_sampling2d/
parent: Keras 레이어 API
grand_parent: Keras 3 API 문서
---

* 원본 링크 : [https://keras.io/api/layers/reshaping_layers/up_sampling2d/](https://keras.io/api/layers/reshaping_layers/up_sampling2d/){:target="_blank"}
* 최종 수정일 : 2024-09-05

# UpSampling2D 레이어 (UpSampling2D layer)
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

### `UpSampling2D` class
<!-- ### `UpSampling2D` class -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/layers/reshaping/up_sampling2d.py#L9){: .btn .btn-outline }

```python
keras.layers.UpSampling2D(
    size=(2, 2), data_format=None, interpolation="nearest", **kwargs
)
```

Upsampling layer for 2D inputs.

The implementation uses interpolative resizing, given the resize method (specified by the `interpolation` argument). Use `interpolation=nearest` to repeat the rows and columns of the data.

**Example**

```python
>>> input_shape = (2, 2, 1, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> print(x)
[[[[ 0  1  2]]
  [[ 3  4  5]]]
 [[[ 6  7  8]]
  [[ 9 10 11]]]]
>>> y = keras.layers.UpSampling2D(size=(1, 2))(x)
>>> print(y)
[[[[ 0  1  2]
   [ 0  1  2]]
  [[ 3  4  5]
   [ 3  4  5]]]
 [[[ 6  7  8]
   [ 6  7  8]]
  [[ 9 10 11]
   [ 9 10 11]]]]
```

**Arguments**

*   **size**: Int, or tuple of 2 integers. The upsampling factors for rows and columns.
*   **data\_format**: A string, one of `"channels_last"` (default) or `"channels_first"`. The ordering of the dimensions in the inputs. `"channels_last"` corresponds to inputs with shape `(batch_size, height, width, channels)` while `"channels_first"` corresponds to inputs with shape `(batch_size, channels, height, width)`. When unspecified, uses `image_data_format` value found in your Keras config file at `~/.keras/keras.json` (if exists) else `"channels_last"`. Defaults to `"channels_last"`.
*   **interpolation**: A string, one of `"bicubic"`, `"bilinear"`, `"lanczos3"`, `"lanczos5"`, `"nearest"`.

**Input shape**

4D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, rows, cols, channels)` - If `data_format` is `"channels_first"`: `(batch_size, channels, rows, cols)`

**Output shape**

4D tensor with shape: - If `data_format` is `"channels_last"`: `(batch_size, upsampled_rows, upsampled_cols, channels)` - If `data_format` is `"channels_first"`: `(batch_size, channels, upsampled_rows, upsampled_cols)`

* * *