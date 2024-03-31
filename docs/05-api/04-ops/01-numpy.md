---
layout: default
title: NumPy ops
nav_order: 01+00
permalink: /api/ops/numpy/
parent: Ops API
grand_parent: Keras 3 API 문서
---

* 원본 링크 : [https://keras.io/api/ops/numpy/](https://keras.io/api/ops/numpy/){:target="_blank"}
* 최종 수정일 : 2024-03-29

# NumPy ops
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---


### `absolute` function

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L229)

`keras.ops.absolute(x)`

Compute the absolute value element-wise.

`keras.ops.abs` is a shorthand for this function.

**Arguments**

*   **x**: Input tensor.

**Returns**

An array containing the absolute value of each element in `x`.

**Example**

`>>> x = keras.ops.convert_to_tensor([-1.2, 1.2]) >>> keras.ops.absolute(x) array([1.2, 1.2], dtype=float32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L282)

### `add` function

`keras.ops.add(x1, x2)`

Add arguments element-wise.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

The tensor containing the element-wise sum of `x1` and `x2`.

**Examples**

`>>> x1 = keras.ops.convert_to_tensor([1, 4]) >>> x2 = keras.ops.convert_to_tensor([5, 6]) >>> keras.ops.add(x1, x2) array([6, 10], dtype=int32)`

[`keras.ops.add`](/api/ops/numpy#add-function) also broadcasts shapes:

`>>> x1 = keras.ops.convert_to_tensor( ...     [[5, 4], ...      [5, 6]] ... ) >>> x2 = keras.ops.convert_to_tensor([5, 6]) >>> keras.ops.add(x1, x2) array([[10 10]        [10 12]], shape=(2, 2), dtype=int32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L341)

### `all` function

`keras.ops.all(x, axis=None, keepdims=False)`

Test whether all array elements along a given axis evaluate to `True`.

**Arguments**

*   **x**: Input tensor.
*   **axis**: An integer or tuple of integers that represent the axis along which a logical AND reduction is performed. The default (`axis=None`) is to perform a logical AND over all the dimensions of the input array. `axis` may be negative, in which case it counts for the last to the first axis.
*   **keepdims**: If `True`, axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. Defaults to`False`.

**Returns**

The tensor containing the logical AND reduction over the `axis`.

**Examples**

`>>> x = keras.ops.convert_to_tensor([True, False]) >>> keras.ops.all(x) array(False, shape=(), dtype=bool)`

`>>> x = keras.ops.convert_to_tensor([[True, False], [True, True]]) >>> keras.ops.all(x, axis=0) array([ True False], shape=(2,), dtype=bool)`

`keepdims=True` outputs a tensor with dimensions reduced to one.

`>>> x = keras.ops.convert_to_tensor([[True, False], [True, True]]) >>> keras.ops.all(x, keepdims=True) array([[False]], shape=(1, 1), dtype=bool)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L464)

### `amax` function

`keras.ops.amax(x, axis=None, keepdims=False)`

Returns the maximum of an array or maximum value along an axis.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis along which to compute the maximum. By default (`axis=None`), find the maximum value in all the dimensions of the input array.
*   **keepdims**: If `True`, axes which are reduced are left in the result as dimensions that are broadcast to the size of the original input tensor. Defaults to `False`.

**Returns**

An array with the maximum value. If `axis=None`, the result is a scalar value representing the maximum element in the entire array. If `axis` is given, the result is an array with the maximum values along the specified axis.

**Examples**

`>>> x = keras.ops.convert_to_tensor([[1, 3, 5], [2, 3, 6]]) >>> keras.ops.amax(x) array(6, dtype=int32)`

`>>> x = keras.ops.convert_to_tensor([[1, 6, 8], [1, 5, 2]]) >>> keras.ops.amax(x, axis=0) array([1, 6, 8], dtype=int32)`

`>>> x = keras.ops.convert_to_tensor([[1, 6, 8], [1, 5, 2]]) >>> keras.ops.amax(x, axis=1, keepdims=True) array([[8], [5]], dtype=int32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L519)

### `amin` function

`keras.ops.amin(x, axis=None, keepdims=False)`

Returns the minimum of an array or minimum value along an axis.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis along which to compute the minimum. By default (`axis=None`), find the minimum value in all the dimensions of the input array.
*   **keepdims**: If `True`, axes which are reduced are left in the result as dimensions that are broadcast to the size of the original input tensor. Defaults to `False`.

**Returns**

An array with the minimum value. If `axis=None`, the result is a scalar value representing the minimum element in the entire array. If `axis` is given, the result is an array with the minimum values along the specified axis.

**Examples**

`>>> x = keras.ops.convert_to_tensor([1, 3, 5, 2, 3, 6]) >>> keras.ops.amin(x) array(1, dtype=int32)`

`>>> x = keras.ops.convert_to_tensor([[1, 6, 8], [7, 5, 3]]) >>> keras.ops.amin(x, axis=0) array([1,5,3], dtype=int32)`

`>>> x = keras.ops.convert_to_tensor([[1, 6, 8], [7, 5, 3]]) >>> keras.ops.amin(x, axis=1, keepdims=True) array([[1],[3]], dtype=int32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L405)

### `any` function

`keras.ops.any(x, axis=None, keepdims=False)`

Test whether any array element along a given axis evaluates to `True`.

**Arguments**

*   **x**: Input tensor.
*   **axis**: An integer or tuple of integers that represent the axis along which a logical OR reduction is performed. The default (`axis=None`) is to perform a logical OR over all the dimensions of the input array. `axis` may be negative, in which case it counts for the last to the first axis.
*   **keepdims**: If `True`, axes which are reduced are left in the result as dimensions with size one. With this option, the result will broadcast correctly against the input array. Defaults to`False`.

**Returns**

The tensor containing the logical OR reduction over the `axis`.

**Examples**

`>>> x = keras.ops.convert_to_tensor([True, False]) >>> keras.ops.any(x) array(True, shape=(), dtype=bool)`

`>>> x = keras.ops.convert_to_tensor([[True, False], [True, True]]) >>> keras.ops.any(x, axis=0) array([ True  True], shape=(2,), dtype=bool)`

`keepdims=True` outputs a tensor with dimensions reduced to one.

`>>> x = keras.ops.convert_to_tensor([[True, False], [True, True]]) >>> keras.ops.all(x, keepdims=True) array([[False]], shape=(1, 1), dtype=bool)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L590)

### `append` function

`keras.ops.append(x1, x2, axis=None)`

Append tensor `x2` to the end of tensor `x1`.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.
*   **axis**: Axis along which tensor `x2` is appended to tensor `x1`. If `None`, both tensors are flattened before use.

**Returns**

A tensor with the values of `x2` appended to `x1`.

**Examples**

`>>> x1 = keras.ops.convert_to_tensor([1, 2, 3]) >>> x2 = keras.ops.convert_to_tensor([[4, 5, 6], [7, 8, 9]]) >>> keras.ops.append(x1, x2) array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int32)`

When `axis` is specified, `x1` and `x2` must have compatible shapes.

`>>> x1 = keras.ops.convert_to_tensor([[1, 2, 3], [4, 5, 6]]) >>> x2 = keras.ops.convert_to_tensor([[7, 8, 9]]) >>> keras.ops.append(x1, x2, axis=0) array([[1, 2, 3],         [4, 5, 6],         [7, 8, 9]], dtype=int32) >>> x3 = keras.ops.convert_to_tensor([7, 8, 9]) >>> keras.ops.append(x1, x3, axis=0) Traceback (most recent call last):     ... TypeError: Cannot concatenate arrays with different numbers of dimensions: got (2, 3), (3,).`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L651)

### `arange` function

`keras.ops.arange(start, stop=None, step=1, dtype=None)`

Return evenly spaced values within a given interval.

`arange` can be called with a varying number of positional arguments: \* `arange(stop)`: Values are generated within the half-open interval `[0, stop)` (in other words, the interval including start but excluding stop). \* `arange(start, stop)`: Values are generated within the half-open interval `[start, stop)`. \* `arange(start, stop, step)`: Values are generated within the half-open interval `[start, stop)`, with spacing between values given by step.

**Arguments**

*   **start**: Integer or real, representing the start of the interval. The interval includes this value.
*   **stop**: Integer or real, representing the end of the interval. The interval does not include this value, except in some cases where `step` is not an integer and floating point round-off affects the length of `out`. Defaults to `None`.
*   **step**: Integer or real, represent the spacing between values. For any output `out`, this is the distance between two adjacent values, `out[i+1] - out[i]`. The default step size is 1. If `step` is specified as a position argument, `start` must also be given.
*   **dtype**: The type of the output array. If `dtype` is not given, infer the data type from the other input arguments.

**Returns**

Tensor of evenly spaced values. For floating point arguments, the length of the result is `ceil((stop - start)/step)`. Because of floating point overflow, this rule may result in the last element of out being greater than stop.

**Examples**

`>>> keras.ops.arange(3) array([0, 1, 2], dtype=int32)`

`>>> keras.ops.arange(3.0) array([0., 1., 2.], dtype=float32)`

`>>> keras.ops.arange(3, 7) array([3, 4, 5, 6], dtype=int32)`

`>>> keras.ops.arange(3, 7, 2) array([3, 5], dtype=int32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L713)

### `arccos` function

`keras.ops.arccos(x)`

Trigonometric inverse cosine, element-wise.

The inverse of `cos` so that, if `y = cos(x)`, then `x = arccos(y)`.

**Arguments**

*   **x**: Input tensor.

**Returns**

Tensor of the angle of the ray intersecting the unit circle at the given x-coordinate in radians `[0, pi]`.

**Example**

`>>> x = keras.ops.convert_to_tensor([1, -1]) >>> keras.ops.arccos(x) array([0.0, 3.1415927], dtype=float32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L749)

### `arccosh` function

`keras.ops.arccosh(x)`

Inverse hyperbolic cosine, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor of same shape as x.

**Example**

`>>> x = keras.ops.convert_to_tensor([10, 100]) >>> keras.ops.arccosh(x) array([2.993223, 5.298292], dtype=float32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L783)

### `arcsin` function

`keras.ops.arcsin(x)`

Inverse sine, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Tensor of the inverse sine of each element in `x`, in radians and in the closed interval `[-pi/2, pi/2]`.

**Example**

`>>> x = keras.ops.convert_to_tensor([1, -1, 0]) >>> keras.ops.arcsin(x) array([ 1.5707964, -1.5707964,  0.], dtype=float32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L818)

### `arcsinh` function

`keras.ops.arcsinh(x)`

Inverse hyperbolic sine, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor of same shape as `x`.

**Example**

`>>> x = keras.ops.convert_to_tensor([1, -1, 0]) >>> keras.ops.arcsinh(x) array([0.88137364, -0.88137364, 0.0], dtype=float32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L852)

### `arctan` function

`keras.ops.arctan(x)`

Trigonometric inverse tangent, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Tensor of the inverse tangent of each element in `x`, in the interval `[-pi/2, pi/2]`.

**Example**

`>>> x = keras.ops.convert_to_tensor([0, 1]) >>> keras.ops.arctan(x) array([0., 0.7853982], dtype=float32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L891)

### `arctan2` function

`keras.ops.arctan2(x1, x2)`

Element-wise arc tangent of `x1/x2` choosing the quadrant correctly.

The quadrant (i.e., branch) is chosen so that `arctan2(x1, x2)` is the signed angle in radians between the ray ending at the origin and passing through the point `(1, 0)`, and the ray ending at the origin and passing through the point `(x2, x1)`. (Note the role reversal: the "y-coordinate" is the first function parameter, the "x-coordinate" is the second.) By IEEE convention, this function is defined for `x2 = +/-0` and for either or both of `x1` and `x2` `= +/-inf`.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

Tensor of angles in radians, in the range `[-pi, pi]`.

**Examples**

Consider four points in different quadrants:

`>>> x = keras.ops.convert_to_tensor([-1, +1, +1, -1]) >>> y = keras.ops.convert_to_tensor([-1, -1, +1, +1]) >>> keras.ops.arctan2(y, x) * 180 / numpy.pi array([-135., -45., 45., 135.], dtype=float32)`

Note the order of the parameters. `arctan2` is defined also when x2=0 and at several other points, obtaining values in the range `[-pi, pi]`:

`>>> keras.ops.arctan2( ...     keras.ops.array([1., -1.]), ...     keras.ops.array([0., 0.]), ... ) array([ 1.5707964, -1.5707964], dtype=float32) >>> keras.ops.arctan2( ...     keras.ops.array([0., 0., numpy.inf]), ...     keras.ops.array([+0., -0., numpy.inf]), ... ) array([0., 3.1415925, 0.7853982], dtype=float32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L949)

### `arctanh` function

`keras.ops.arctanh(x)`

Inverse hyperbolic tangent, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor of same shape as `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L980)

### `argmax` function

`keras.ops.argmax(x, axis=None)`

Returns the indices of the maximum values along an axis.

**Arguments**

*   **x**: Input tensor.
*   **axis**: By default, the index is into the flattened tensor, otherwise along the specified axis.

**Returns**

Tensor of indices. It has the same shape as `x`, with the dimension along `axis` removed.

**Example**

`>>> x = keras.ops.arange(6).reshape(2, 3) + 10 >>> x array([[10, 11, 12],        [13, 14, 15]], dtype=int32) >>> keras.ops.argmax(x) array(5, dtype=int32) >>> keras.ops.argmax(x, axis=0) array([1, 1, 1], dtype=int32) >>> keras.ops.argmax(x, axis=1) array([2, 2], dtype=int32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1026)

### `argmin` function

`keras.ops.argmin(x, axis=None)`

Returns the indices of the minium values along an axis.

**Arguments**

*   **x**: Input tensor.
*   **axis**: By default, the index is into the flattened tensor, otherwise along the specified axis.

**Returns**

Tensor of indices. It has the same shape as `x`, with the dimension along `axis` removed.

**Example**

`>>> x = keras.ops.arange(6).reshape(2, 3) + 10 >>> x array([[10, 11, 12],        [13, 14, 15]], dtype=int32) >>> keras.ops.argmin(x) array(0, dtype=int32) >>> keras.ops.argmin(x, axis=0) array([0, 0, 0], dtype=int32) >>> keras.ops.argmin(x, axis=1) array([0, 0], dtype=int32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1070)

### `argsort` function

`keras.ops.argsort(x, axis=-1)`

Returns the indices that would sort a tensor.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis along which to sort. Defaults to`-1` (the last axis). If `None`, the flattened tensor is used.

**Returns**

Tensor of indices that sort `x` along the specified `axis`.

**Examples**

One dimensional array:

`>>> x = keras.ops.array([3, 1, 2]) >>> keras.ops.argsort(x) array([1, 2, 0], dtype=int32)`

Two-dimensional array:

`>>> x = keras.ops.array([[0, 3], [3, 2], [4, 5]]) >>> x array([[0, 3],        [3, 2],        [4, 5]], dtype=int32) >>> keras.ops.argsort(x, axis=0) array([[0, 1],        [1, 0],        [2, 2]], dtype=int32) >>> keras.ops.argsort(x, axis=1) array([[0, 1],        [1, 0],        [0, 1]], dtype=int32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1116)

### `array` function

`keras.ops.array(x, dtype=None)`

Create a tensor.

**Arguments**

*   **x**: Input tensor.
*   **dtype**: The desired data-type for the tensor.

**Returns**

A tensor.

**Examples**

`>>> keras.ops.array([1, 2, 3]) array([1, 2, 3], dtype=int32)`

`>>> keras.ops.array([1, 2, 3], dtype="float32") array([1., 2., 3.], dtype=float32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1184)

### `average` function

`keras.ops.average(x, axis=None, weights=None)`

Compute the weighted average along the specified axis.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Integer along which to average `x`. The default, `axis=None`, will average over all of the elements of the input tensor. If axis is negative it counts from the last to the first axis.
*   **weights**: Tensor of wieghts associated with the values in `x`. Each value in `x` contributes to the average according to its associated weight. The weights array can either be 1-D (in which case its length must be the size of a along the given axis) or of the same shape as `x`. If `weights=None` (default), then all data in `x` are assumed to have a weight equal to one.
*   \_\_ The 1-D calculation is\_\_: `avg = sum(a * weights) / sum(weights)`. The only constraint on weights is that `sum(weights)` must not be 0.

**Returns**

Return the average along the specified axis.

**Examples**

`>>> data = keras.ops.arange(1, 5) >>> data array([1, 2, 3, 4], dtype=int32) >>> keras.ops.average(data) array(2.5, dtype=float32) >>> keras.ops.average( ...     keras.ops.arange(1, 11), ...     weights=keras.ops.arange(10, 0, -1) ... ) array(4., dtype=float32)`

`>>> data = keras.ops.arange(6).reshape((3, 2)) >>> data array([[0, 1],        [2, 3],        [4, 5]], dtype=int32) >>> keras.ops.average( ...     data, ...     axis=1, ...     weights=keras.ops.array([1./4, 3./4]) ... ) array([0.75, 2.75, 4.75], dtype=float32) >>> keras.ops.average( ...     data, ...     weights=keras.ops.array([1./4, 3./4]) ... ) Traceback (most recent call last):     ... ValueError: Axis must be specified when shapes of a and weights differ.`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1264)

### `bincount` function

`keras.ops.bincount(x, weights=None, minlength=0)`

Count the number of occurrences of each value in a tensor of integers.

**Arguments**

*   **x**: Input tensor. It must be of dimension 1, and it must only contain non-negative integer(s).
*   **weights**: Weight tensor. It must have the same length as `x`. The default value is `None`. If specified, `x` is weighted by it, i.e. if `n = x[i]`, `out[n] += weight[i]` instead of the default behavior `out[n] += 1`.
*   **minlength**: An integer. The default value is 0. If specified, there will be at least this number of bins in the output tensor. If greater than `max(x) + 1`, each value of the output at an index higher than `max(x)` is set to 0.

**Returns**

1D tensor where each element gives the number of occurrence(s) of its index value in x. Its length is the maximum between `max(x) + 1` and minlength.

**Examples**

`>>> x = keras.ops.array([1, 2, 2, 3], dtype="uint8") >>> keras.ops.bincount(x) array([0, 1, 2, 1], dtype=int32) >>> weights = x / 2 >>> weights array([0.5, 1., 1., 1.5], dtype=float64) >>> keras.ops.bincount(x, weights=weights) array([0., 0.5, 2., 1.5], dtype=float64) >>> minlength = (keras.ops.max(x).numpy() + 1) + 2 # 6 >>> keras.ops.bincount(x, minlength=minlength) array([0, 1, 2, 1, 0, 0], dtype=int32)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1319)

### `broadcast_to` function

`keras.ops.broadcast_to(x, shape)`

Broadcast a tensor to a new shape.

**Arguments**

*   **x**: The tensor to broadcast.
*   **shape**: The shape of the desired tensor. A single integer `i` is interpreted as `(i,)`.

**Returns**

A tensor with the desired shape.

**Examples**

`>>> x = keras.ops.array([1, 2, 3]) >>> keras.ops.broadcast_to(x, (3, 3)) array([[1, 2, 3],        [1, 2, 3],        [1, 2, 3]])`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1361)

### `ceil` function

`keras.ops.ceil(x)`

Return the ceiling of the input, element-wise.

The ceil of the scalar `x` is the smallest integer `i`, such that `i >= x`.

**Arguments**

*   **x**: Input tensor.

**Returns**

The ceiling of each element in `x`, with float dtype.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1395)

### `clip` function

`keras.ops.clip(x, x_min, x_max)`

Clip (limit) the values in a tensor.

Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of `[0, 1]` is specified, values smaller than 0 become 0, and values larger than 1 become 1.

**Arguments**

*   **x**: Input tensor.
*   **x\_min**: Minimum value.
*   **x\_max**: Maximum value.

**Returns**

The clipped tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1452)

### `concatenate` function

`keras.ops.concatenate(xs, axis=0)`

Join a sequence of tensors along an existing axis.

**Arguments**

*   **xs**: The sequence of tensors to concatenate.
*   **axis**: The axis along which the tensors will be joined. Defaults to `0`.

**Returns**

The concatenated tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1506)

### `conj` function

`keras.ops.conj(x)`

Shorthand for [`keras.ops.conjugate`](/api/ops/numpy#conjugate-function).

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1482)

### `conjugate` function

`keras.ops.conjugate(x)`

Returns the complex conjugate, element-wise.

The complex conjugate of a complex number is obtained by changing the sign of its imaginary part.

[`keras.ops.conj`](/api/ops/numpy#conj-function) is a shorthand for this function.

**Arguments**

*   **x**: Input tensor.

**Returns**

The complex conjugate of each element in `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1521)

### `copy` function

`keras.ops.copy(x)`

Returns a copy of `x`.

**Arguments**

*   **x**: Input tensor.

**Returns**

A copy of `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1549)

### `cos` function

`keras.ops.cos(x)`

Cosine, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

The corresponding cosine values.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1577)

### `cosh` function

`keras.ops.cosh(x)`

Hyperbolic cosine, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor of same shape as `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1610)

### `count_nonzero` function

`keras.ops.count_nonzero(x, axis=None)`

Counts the number of non-zero values in `x` along the given `axis`.

If no axis is specified then all non-zeros in the tensor are counted.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis or tuple of axes along which to count the number of non-zeros. Defaults to `None`.

**Returns**

int or tensor of ints.

**Examples**

`>>> x = keras.ops.array([[0, 1, 7, 0], [3, 0, 2, 19]]) >>> keras.ops.count_nonzero(x) 5 >>> keras.ops.count_nonzero(x, axis=0) array([1, 1, 2, 1], dtype=int64) >>> keras.ops.count_nonzero(x, axis=1) array([2, 3], dtype=int64)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1692)

### `cross` function

`keras.ops.cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None)`

Returns the cross product of two (arrays of) vectors.

The cross product of `x1` and `x2` in R^3 is a vector perpendicular to both `x1` and `x2`. If `x1` and `x2` are arrays of vectors, the vectors are defined by the last axis of `x1` and `x2` by default, and these axes can have dimensions 2 or 3.

Where the dimension of either `x1` or `x2` is 2, the third component of the input vector is assumed to be zero and the cross product calculated accordingly.

In cases where both input vectors have dimension 2, the z-component of the cross product is returned.

**Arguments**

*   **x1**: Components of the first vector(s).
*   **x2**: Components of the second vector(s).
*   **axisa**: Axis of `x1` that defines the vector(s). Defaults to `-1`.
*   **axisb**: Axis of `x2` that defines the vector(s). Defaults to `-1`.
*   **axisc**: Axis of the result containing the cross product vector(s). Ignored if both input vectors have dimension 2, as the return is scalar. By default, the last axis.
*   **axis**: If defined, the axis of `x1`, `x2` and the result that defines the vector(s) and cross product(s). Overrides `axisa`, `axisb` and `axisc`.

Note: Torch backend does not support two dimensional vectors, or the arguments `axisa`, `axisb` and `axisc`. Use `axis` instead.

**Returns**

Vector cross product(s).

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1764)

### `cumprod` function

`keras.ops.cumprod(x, axis=None, dtype=None)`

Return the cumulative product of elements along a given axis.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis along which the cumulative product is computed. By default the input is flattened.
*   **dtype**: dtype of returned tensor. Defaults to x.dtype.

**Returns**

Output tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1803)

### `cumsum` function

`keras.ops.cumsum(x, axis=None, dtype=None)`

Returns the cumulative sum of elements along a given axis.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis along which the cumulative sum is computed. By default the input is flattened.
*   **dtype**: dtype of returned tensor. Defaults to x.dtype.

**Returns**

Output tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1856)

### `diag` function

`keras.ops.diag(x, k=0)`

Extract a diagonal or construct a diagonal array.

**Arguments**

*   **x**: Input tensor. If `x` is 2-D, returns the k-th diagonal of `x`. If `x` is 1-D, return a 2-D tensor with `x` on the k-th diagonal.
*   **k**: The diagonal to consider. Defaults to `0`. Use `k > 0` for diagonals above the main diagonal, and `k < 0` for diagonals below the main diagonal.

**Returns**

The extracted diagonal or constructed diagonal tensor.

**Examples**

`>>> from keras import ops >>> x = ops.arange(9).reshape((3, 3)) >>> x array([[0, 1, 2],        [3, 4, 5],        [6, 7, 8]])`

`>>> ops.diag(x) array([0, 4, 8]) >>> ops.diag(x, k=1) array([1, 5]) >>> ops.diag(x, k=-1) array([3, 7])`

`>>> ops.diag(ops.diag(x))) array([[0, 0, 0],        [0, 4, 0],        [0, 0, 8]])`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L1937)

### `diagonal` function

`keras.ops.diagonal(x, offset=0, axis1=0, axis2=1)`

Return specified diagonals.

If `x` is 2-D, returns the diagonal of `x` with the given offset, i.e., the collection of elements of the form `x[i, i+offset]`.

If `x` has more than two dimensions, the axes specified by `axis1` and `axis2` are used to determine the 2-D sub-array whose diagonal is returned.

The shape of the resulting array can be determined by removing `axis1` and `axis2` and appending an index to the right equal to the size of the resulting diagonals.

**Arguments**

*   **x**: Input tensor.
*   **offset**: Offset of the diagonal from the main diagonal. Can be positive or negative. Defaults to `0`.(main diagonal).
*   **axis1**: Axis to be used as the first axis of the 2-D sub-arrays. Defaults to `0`.(first axis).
*   **axis2**: Axis to be used as the second axis of the 2-D sub-arrays. Defaults to `1` (second axis).

**Returns**

Tensor of diagonals.

**Examples**

`>>> from keras import ops >>> x = ops.arange(4).reshape((2, 2)) >>> x array([[0, 1],        [2, 3]]) >>> x.diagonal() array([0, 3]) >>> x.diagonal(1) array([1])`

`>>> x = ops.arange(8).reshape((2, 2, 2)) >>> x array([[[0, 1],         [2, 3]],        [[4, 5],         [6, 7]]]) >>> x.diagonal(0, 0, 1) array([[0, 6],        [1, 7]])`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2016)

### `diff` function

`keras.ops.diff(a, n=1, axis=-1)`

Calculate the n-th discrete difference along the given axis.

The first difference is given by `out[i] = a[i+1] - a[i]` along the given axis, higher differences are calculated by using `diff` recursively.

**Arguments**

*   **a**: Input tensor.
*   **n**: The number of times values are differenced. Defaults to `1`.
*   **axis**: Axis to compute discrete difference(s) along. Defaults to `-1`.(last axis).

**Returns**

Tensor of diagonals.

**Examples**

`>>> from keras import ops >>> x = ops.convert_to_tensor([1, 2, 4, 7, 0]) >>> ops.diff(x) array([ 1,  2,  3, -7]) >>> ops.diff(x, n=2) array([  1,   1, -10])`

`>>> x = ops.convert_to_tensor([[1, 3, 6, 10], [0, 5, 6, 8]]) >>> ops.diff(x) array([[2, 3, 4],        [5, 1, 2]]) >>> ops.diff(x, axis=0) array([[-1,  2,  0, -2]])`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2065)

### `digitize` function

`keras.ops.digitize(x, bins)`

Returns the indices of the bins to which each value in `x` belongs.

**Arguments**

*   **x**: Input array to be binned.
*   **bins**: Array of bins. It has to be one-dimensional and monotonically increasing.

**Returns**

Output array of indices, of same shape as `x`.

**Example**

`>>> x = np.array([0.0, 1.0, 3.0, 1.6]) >>> bins = np.array([0.0, 3.0, 4.5, 7.0]) >>> keras.ops.digitize(x, bins) array([1, 1, 2, 1])`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5533)

### `divide` function

`keras.ops.divide(x1, x2)`

Divide arguments element-wise.

[`keras.ops.true_divide`](/api/ops/numpy#truedivide-function) is an alias for this function.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

Output tensor, the quotient `x1/x2`, element-wise.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5572)

### `divide_no_nan` function

`keras.ops.divide_no_nan(x1, x2)`

Safe element-wise division which returns 0 where the denominator is 0.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

The quotient `x1/x2`, element-wise, with zero where x2 is zero.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2128)

### `dot` function

`keras.ops.dot(x1, x2)`

Dot product of two tensors.

*   If both `x1` and `x2` are 1-D tensors, it is inner product of vectors (without complex conjugation).
*   If both `x1` and `x2` are 2-D tensors, it is matrix multiplication.
*   If either `x1` or `x2` is 0-D (scalar), it is equivalent to `x1 * x2`.
*   If `x1` is an N-D tensor and `x2` is a 1-D tensor, it is a sum product over the last axis of `x1` and `x2`.
*   If `x1` is an N-D tensor and `x2` is an M-D tensor (where `M>=2`), it is a sum product over the last axis of `x1` and the second-to-last axis of `x2`: `dot(x1, x2)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])`.

**Arguments**

*   **x1**: First argument.
*   **x2**: Second argument.

Note: Torch backend does not accept 0-D tensors as arguments.

**Returns**

Dot product of `x1` and `x2`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2336)

### `einsum` function

`keras.ops.einsum(subscripts, *operands)`

Evaluates the Einstein summation convention on the operands.

**Arguments**

*   **subscripts**: Specifies the subscripts for summation as comma separated list of subscript labels. An implicit (classical Einstein summation) calculation is performed unless the explicit indicator `->` is included as well as subscript labels of the precise output form.
*   **operands**: The operands to compute the Einstein sum of.

**Returns**

The calculation based on the Einstein summation convention.

**Example**

`>>> from keras import ops >>> a = ops.arange(25).reshape(5, 5) >>> b = ops.arange(5) >>> c = ops.arange(6).reshape(2, 3)`

Trace of a matrix:

`>>> ops.einsum("ii", a) 60 >>> ops.einsum(a, [0, 0]) 60 >>> ops.trace(a) 60`

Extract the diagonal:

`>>> ops.einsum("ii -> i", a) array([ 0,  6, 12, 18, 24]) >>> ops.einsum(a, [0, 0], [0]) array([ 0,  6, 12, 18, 24]) >>> ops.diag(a) array([ 0,  6, 12, 18, 24])`

Sum over an axis:

`>>> ops.einsum("ij -> i", a) array([ 10,  35,  60,  85, 110]) >>> ops.einsum(a, [0, 1], [0]) array([ 10,  35,  60,  85, 110]) >>> ops.sum(a, axis=1) array([ 10,  35,  60,  85, 110])`

For higher dimensional tensors summing a single axis can be done with ellipsis:

`>>> ops.einsum("...j -> ...", a) array([ 10,  35,  60,  85, 110]) >>> np.einsum(a, [..., 1], [...]) array([ 10,  35,  60,  85, 110])`

Compute a matrix transpose or reorder any number of axes:

`>>> ops.einsum("ji", c) array([[0, 3],        [1, 4],        [2, 5]]) >>> ops.einsum("ij -> ji", c) array([[0, 3],        [1, 4],        [2, 5]]) >>> ops.einsum(c, [1, 0]) array([[0, 3],        [1, 4],        [2, 5]]) >>> ops.transpose(c) array([[0, 3],        [1, 4],        [2, 5]])`

Matrix vector multiplication:

`>>> ops.einsum("ij, j", a, b) array([ 30,  80, 130, 180, 230]) >>> ops.einsum(a, [0, 1], b, [1]) array([ 30,  80, 130, 180, 230]) >>> ops.einsum("...j, j", a, b) array([ 30,  80, 130, 180, 230])`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2434)

### `empty` function

`keras.ops.empty(shape, dtype=None)`

Return a tensor of given shape and type filled with uninitialized data.

**Arguments**

*   **shape**: Shape of the empty tensor.
*   **dtype**: Desired data type of the empty tensor.

**Returns**

The empty tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2459)

### `equal` function

`keras.ops.equal(x1, x2)`

Returns `(x1 == x2)` element-wise.

**Arguments**

*   **x1**: Tensor to compare.
*   **x2**: Tensor to compare.

**Returns**

Output tensor, element-wise comparison of `x1` and `x2`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2486)

### `exp` function

`keras.ops.exp(x)`

Calculate the exponential of all elements in the input tensor.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor, element-wise exponential of `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2522)

### `expand_dims` function

`keras.ops.expand_dims(x, axis)`

Expand the shape of a tensor.

Insert a new axis at the `axis` position in the expanded tensor shape.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Position in the expanded axes where the new axis (or axes) is placed.

**Returns**

Output tensor with the number of dimensions increased.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2558)

### `expm1` function

`keras.ops.expm1(x)`

Calculate `exp(x) - 1` for all elements in the tensor.

**Arguments**

*   **x**: Input values.

**Returns**

Output tensor, element-wise exponential minus one.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5998)

### `eye` function

`keras.ops.eye(N, M=None, k=0, dtype=None)`

Return a 2-D tensor with ones on the diagonal and zeros elsewhere.

**Arguments**

*   **N**: Number of rows in the output.
*   **M**: Number of columns in the output. If `None`, defaults to `N`.
*   **k**: Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal.
*   **dtype**: Data type of the returned tensor.

**Returns**

Tensor with ones on the k-th diagonal and zeros elsewhere.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2585)

### `flip` function

`keras.ops.flip(x, axis=None)`

Reverse the order of elements in the tensor along the given axis.

The shape of the tensor is preserved, but the elements are reordered.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis or axes along which to flip the tensor. The default, `axis=None`, will flip over all of the axes of the input tensor.

**Returns**

Output tensor with entries of `axis` reversed.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2618)

### `floor` function

`keras.ops.floor(x)`

Return the floor of the input, element-wise.

The floor of the scalar `x` is the largest integer `i`, such that `i <= x`.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor, element-wise floor of `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L6031)

### `floor_divide` function

`keras.ops.floor_divide(x1, x2)`

Returns the largest integer smaller or equal to the division of inputs.

**Arguments**

*   **x1**: Numerator.
*   **x2**: Denominator.

**Returns**

Output tensor, `y = floor(x1/x2)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2644)

### `full` function

`keras.ops.full(shape, fill_value, dtype=None)`

Return a new tensor of given shape and type, filled with `fill_value`.

**Arguments**

*   **shape**: Shape of the new tensor.
*   **fill\_value**: Fill value.
*   **dtype**: Desired data type of the tensor.

**Returns**

Output tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2668)

### `full_like` function

`keras.ops.full_like(x, fill_value, dtype=None)`

Return a full tensor with the same shape and type as the given tensor.

**Arguments**

*   **x**: Input tensor.
*   **fill\_value**: Fill value.
*   **dtype**: Overrides data type of the result.

**Returns**

Tensor of `fill_value` with the same shape and type as `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2756)

### `get_item` function

`keras.ops.get_item(x, key)`

Return `x[key]`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2775)

### `greater` function

`keras.ops.greater(x1, x2)`

Return the truth value of `x1 > x2` element-wise.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

Output tensor, element-wise comparison of `x1` and `x2`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2802)

### `greater_equal` function

`keras.ops.greater_equal(x1, x2)`

Return the truth value of `x1 >= x2` element-wise.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

Output tensor, element-wise comparison of `x1` and `x2`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2850)

### `hstack` function

`keras.ops.hstack(xs)`

Stack tensors in sequence horizontally (column wise).

This is equivalent to concatenation along the first axis for 1-D tensors, and along the second axis for all other tensors.

**Arguments**

*   **xs**: Sequence of tensors.

**Returns**

The tensor formed by stacking the given tensors.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2877)

### `identity` function

`keras.ops.identity(n, dtype=None)`

Return the identity tensor.

The identity tensor is a square tensor with ones on the main diagonal and zeros elsewhere.

**Arguments**

*   **n**: Number of rows (and columns) in the `n x n` output tensor.
*   **dtype**: Data type of the output tensor.

**Returns**

The identity tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2903)

### `imag` function

`keras.ops.imag(x)`

Return the imaginary part of the complex argument.

**Arguments**

*   **x**: Input tensor.

**Returns**

The imaginary component of the complex argument.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2929)

### `isclose` function

`keras.ops.isclose(x1, x2)`

Return whether two tensors are element-wise almost equal.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

Output boolean tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2953)

### `isfinite` function

`keras.ops.isfinite(x)`

Return whether a tensor is finite, element-wise.

Real values are finite when they are not NaN, not positive infinity, and not negative infinity. Complex values are finite when both their real and imaginary parts are finite.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output boolean tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L2980)

### `isinf` function

`keras.ops.isinf(x)`

Test element-wise for positive or negative infinity.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output boolean tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3003)

### `isnan` function

`keras.ops.isnan(x)`

Test element-wise for NaN and return result as a boolean tensor.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output boolean tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3029)

### `less` function

`keras.ops.less(x1, x2)`

Return the truth value of `x1 < x2` element-wise.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

Output tensor, element-wise comparison of `x1` and `x2`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3056)

### `less_equal` function

`keras.ops.less_equal(x1, x2)`

Return the truth value of `x1 <= x2` element-wise.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

Output tensor, element-wise comparison of `x1` and `x2`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3129)

### `linspace` function

`keras.ops.linspace(     start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0 )`

Return evenly spaced numbers over a specified interval.

Returns `num` evenly spaced samples, calculated over the interval `[start, stop]`.

The endpoint of the interval can optionally be excluded.

**Arguments**

*   **start**: The starting value of the sequence.
*   **stop**: The end value of the sequence, unless `endpoint` is set to `False`. In that case, the sequence consists of all but the last of `num + 1` evenly spaced samples, so that `stop` is excluded. Note that the step size changes when `endpoint` is `False`.
*   **num**: Number of samples to generate. Defaults to `50`. Must be non-negative.
*   **endpoint**: If `True`, `stop` is the last sample. Otherwise, it is not included. Defaults to`True`.
*   **retstep**: If `True`, return `(samples, step)`, where `step` is the spacing between samples.
*   **dtype**: The type of the output tensor.
*   **axis**: The axis in the result to store the samples. Relevant only if start or stop are array-like. Defaults to `0`.

Note: Torch backend does not support `axis` argument.

**Returns**

A tensor of evenly spaced numbers. If `retstep` is `True`, returns `(samples, step)`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3189)

### `log` function

`keras.ops.log(x)`

Natural logarithm, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor, element-wise natural logarithm of `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3217)

### `log10` function

`keras.ops.log10(x)`

Return the base 10 logarithm of the input tensor, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor, element-wise base 10 logarithm of `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3246)

### `log1p` function

`keras.ops.log1p(x)`

Returns the natural logarithm of one plus the `x`, element-wise.

Calculates `log(1 + x)`.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor, element-wise natural logarithm of `1 + x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3276)

### `log2` function

`keras.ops.log2(x)`

Base-2 logarithm of `x`, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor, element-wise base-2 logarithm of `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3307)

### `logaddexp` function

`keras.ops.logaddexp(x1, x2)`

Logarithm of the sum of exponentiations of the inputs.

Calculates `log(exp(x1) + exp(x2))`.

**Arguments**

*   **x1**: Input tensor.
*   **x2**: Input tensor.

**Returns**

Output tensor, element-wise logarithm of the sum of exponentiations of the inputs.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3337)

### `logical_and` function

`keras.ops.logical_and(x1, x2)`

Computes the element-wise logical AND of the given input tensors.

Zeros are treated as `False` and non-zeros are treated as `True`.

**Arguments**

*   **x1**: Input tensor.
*   **x2**: Input tensor.

**Returns**

Output tensor, element-wise logical AND of the inputs.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3368)

### `logical_not` function

`keras.ops.logical_not(x)`

Computes the element-wise NOT of the given input tensor.

Zeros are treated as `False` and non-zeros are treated as `True`.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor, element-wise logical NOT of the input.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3401)

### `logical_or` function

`keras.ops.logical_or(x1, x2)`

Computes the element-wise logical OR of the given input tensors.

Zeros are treated as `False` and non-zeros are treated as `True`.

**Arguments**

*   **x1**: Input tensor.
*   **x2**: Input tensor.

**Returns**

Output tensor, element-wise logical OR of the inputs.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L6058)

### `logical_xor` function

`keras.ops.logical_xor(x1, x2)`

Compute the truth value of `x1 XOR x2`, element-wise.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

Output boolean tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3471)

### `logspace` function

`keras.ops.logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0)`

Returns numbers spaced evenly on a log scale.

In linear space, the sequence starts at `base ** start` and ends with `base ** stop` (see `endpoint` below).

**Arguments**

*   **start**: The starting value of the sequence.
*   **stop**: The final value of the sequence, unless `endpoint` is `False`. In that case, `num + 1` values are spaced over the interval in log-space, of which all but the last (a sequence of length `num`) are returned.
*   **num**: Number of samples to generate. Defaults to `50`.
*   **endpoint**: If `True`, `stop` is the last sample. Otherwise, it is not included. Defaults to`True`.
*   **base**: The base of the log space. Defaults to `10`.
*   **dtype**: The type of the output tensor.
*   **axis**: The axis in the result to store the samples. Relevant only if start or stop are array-like.

Note: Torch backend does not support `axis` argument.

**Returns**

A tensor of evenly spaced samples on a log scale.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3533)

### `matmul` function

`keras.ops.matmul(x1, x2)`

Matrix product of two tensors.

*   If both tensors are 1-dimensional, the dot product (scalar) is returned.
*   If either tensor is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
*   If the first tensor is 1-D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
*   If the second tensor is 1-D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.

**Arguments**

*   **x1**: First tensor.
*   **x2**: Second tensor.

**Returns**

Output tensor, matrix product of the inputs.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3580)

### `max` function

`keras.ops.max(x, axis=None, keepdims=False, initial=None)`

Return the maximum of a tensor or maximum along an axis.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis or axes along which to operate. By default, flattened input is used.
*   **keepdims**: If this is set to `True`, the axes which are reduced are left in the result as dimensions with size one. Defaults to`False`.
*   **initial**: The minimum value of an output element. Defaults to`None`.

**Returns**

Maximum of `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3622)

### `maximum` function

`keras.ops.maximum(x1, x2)`

Element-wise maximum of `x1` and `x2`.

**Arguments**

*   **x1**: First tensor.
*   **x2**: Second tensor.

**Returns**

Output tensor, element-wise maximum of `x1` and `x2`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5838)

### `mean` function

`keras.ops.mean(x, axis=None, keepdims=False)`

Compute the arithmetic mean along the specified axes.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis or axes along which the means are computed. The default is to compute the mean of the flattened tensor.
*   **keepdims**: If this is set to `True`, the axes which are reduced are left in the result as dimensions with size one.

**Returns**

Output tensor containing the mean values.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3660)

### `median` function

`keras.ops.median(x, axis=None, keepdims=False)`

Compute the median along the specified axis.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis or axes along which the medians are computed. Defaults to `axis=None` which is to compute the median(s) along a flattened version of the array.
*   **keepdims**: If this is set to `True`, the axes which are reduce are left in the result as dimensions with size one.

**Returns**

The output tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3714)

### `meshgrid` function

`keras.ops.meshgrid(*x, indexing="xy")`

Creates grids of coordinates from coordinate vectors.

Given `N` 1-D tensors `T0, T1, ..., TN-1` as inputs with corresponding lengths `S0, S1, ..., SN-1`, this creates an `N` N-dimensional tensors `G0, G1, ..., GN-1` each with shape `(S0, ..., SN-1)` where the output `Gi` is constructed by expanding `Ti` to the result shape.

**Arguments**

*   **x**: 1-D tensors representing the coordinates of a grid.
*   **indexing**: `"xy"` or `"ij"`. "xy" is cartesian; `"ij"` is matrix indexing of output. Defaults to `"xy"`.

**Returns**

Sequence of N tensors.

**Example**

`>>> from keras import ops >>> x = ops.array([1, 2, 3]) >>> y = ops.array([4, 5, 6])`

`>>> grid_x, grid_y = ops.meshgrid(x, y, indexing="ij") >>> grid_x array([[1, 1, 1],        [2, 2, 2],        [3, 3, 3]]) >>> grid_y array([[4, 5, 6],        [4, 5, 6],        [4, 5, 6]])`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3773)

### `min` function

`keras.ops.min(x, axis=None, keepdims=False, initial=None)`

Return the minimum of a tensor or minimum along an axis.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis or axes along which to operate. By default, flattened input is used.
*   **keepdims**: If this is set to `True`, the axes which are reduced are left in the result as dimensions with size one. Defaults to`False`.
*   **initial**: The maximum value of an output element. Defaults to`None`.

**Returns**

Minimum of `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3815)

### `minimum` function

`keras.ops.minimum(x1, x2)`

Element-wise minimum of `x1` and `x2`.

**Arguments**

*   **x1**: First tensor.
*   **x2**: Second tensor.

**Returns**

Output tensor, element-wise minimum of `x1` and `x2`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3848)

### `mod` function

`keras.ops.mod(x1, x2)`

Returns the element-wise remainder of division.

**Arguments**

*   **x1**: First tensor.
*   **x2**: Second tensor.

**Returns**

Output tensor, element-wise remainder of division.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3908)

### `moveaxis` function

`keras.ops.moveaxis(x, source, destination)`

Move axes of a tensor to new positions.

Other axes remain in their original order.

**Arguments**

*   **x**: Tensor whose axes should be reordered.
*   **source**: Original positions of the axes to move. These must be unique.
*   **destination**: Destinations positions for each of the original axes. These must also be unique.

**Returns**

Tensor with moved axes.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5496)

### `multiply` function

`keras.ops.multiply(x1, x2)`

Multiply arguments element-wise.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

Output tensor, element-wise product of `x1` and `x2`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3933)

### `nan_to_num` function

`keras.ops.nan_to_num(x)`

Replace NaN with zero and infinity with large finite numbers.

**Arguments**

*   **x**: Input data.

**Returns**

`x`, with non-finite values replaced.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3961)

### `ndim` function

`keras.ops.ndim(x)`

Return the number of dimensions of a tensor.

**Arguments**

*   **x**: Input tensor.

**Returns**

The number of dimensions in `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5661)

### `negative` function

`keras.ops.negative(x)`

Numerical negative, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor, `y = -x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L3981)

### `nonzero` function

`keras.ops.nonzero(x)`

Return the indices of the elements that are non-zero.

**Arguments**

*   **x**: Input tensor.

**Returns**

Indices of elements that are non-zero.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/linalg.py#L272)

### `norm` function

`keras.ops.norm(x, ord=None, axis=None, keepdims=False)`

Matrix or vector norm.

This function is able to return one of eight different matrix norms, or one of an infinite number of vector norms (described below), depending on the value of the `ord` parameter.

**Arguments**

*   **x**: Input tensor.
*   **ord**: Order of the norm (see table under Notes). The default is `None`.
*   **axis**: If `axis` is an integer, it specifies the axis of `x` along which to compute the vector norms. If `axis` is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of these matrices are computed.
*   **keepdims**: If this is set to `True`, the axes which are reduced are left in the result as dimensions with size one.

Note: For values of `ord < 1`, the result is, strictly speaking, not a mathematical 'norm', but it may still be useful for various numerical purposes. The following norms can be calculated: - For matrices: - `ord=None`: Frobenius norm - `ord="fro"`: Frobenius norm - `ord="nuc"`: nuclear norm - `ord=np.inf`: `max(sum(abs(x), axis=1))` - `ord=-np.inf`: `min(sum(abs(x), axis=1))` - `ord=0`: not supported - `ord=1`: `max(sum(abs(x), axis=0))` - `ord=-1`: `min(sum(abs(x), axis=0))` - `ord=2`: 2-norm (largest sing. value) - `ord=-2`: smallest singular value - other: not supported - For vectors: - `ord=None`: 2-norm - `ord="fro"`: not supported - `ord="nuc"`: not supported - `ord=np.inf`: `max(abs(x))` - `ord=-np.inf`: `min(abs(x))` - `ord=0`: `sum(x != 0)` - `ord=1`: as below - `ord=-1`: as below - `ord=2`: as below - `ord=-2`: as below - other: `sum(abs(x)**ord)**(1./ord)`

**Returns**

Norm of the matrix or vector(s).

**Example**

`>>> x = keras.ops.reshape(keras.ops.arange(9, dtype="float32") - 4, (3, 3)) >>> keras.ops.linalg.norm(x) 7.7459664`

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4005)

### `not_equal` function

`keras.ops.not_equal(x1, x2)`

Return `(x1 != x2)` element-wise.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

Output tensor, element-wise comparsion of `x1` and `x2`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5973)

### `ones` function

`keras.ops.ones(shape, dtype=None)`

Return a new tensor of given shape and type, filled with ones.

**Arguments**

*   **shape**: Shape of the new tensor.
*   **dtype**: Desired data type of the tensor.

**Returns**

Tensor of ones with the given shape and dtype.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4031)

### `ones_like` function

`keras.ops.ones_like(x, dtype=None)`

Return a tensor of ones with the same shape and type of `x`.

**Arguments**

*   **x**: Input tensor.
*   **dtype**: Overrides the data type of the result.

**Returns**

A tensor of ones with the same shape and type as `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4101)

### `outer` function

`keras.ops.outer(x1, x2)`

Compute the outer product of two vectors.

Given two vectors `x1` and `x2`, the outer product is:

`out[i, j] = x1[i] * x2[j]`

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

Outer product of `x1` and `x2`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4177)

### `pad` function

`keras.ops.pad(x, pad_width, mode="constant", constant_values=None)`

Pad a tensor.

**Arguments**

*   **x**: Tensor to pad.
*   **pad\_width**: Number of values padded to the edges of each axis. `((before_1, after_1), ...(before_N, after_N))` unique pad widths for each axis. `((before, after),)` yields same before and after pad for each axis. `(pad,)` or `int` is a shortcut for `before = after = pad` width for all axes.
*   **mode**: One of `"constant"`, `"edge"`, `"linear_ramp"`, `"maximum"`, `"mean"`, `"median"`, `"minimum"`, `"reflect"`, `"symmetric"`, `"wrap"`, `"empty"`, `"circular"`. Defaults to`"constant"`.
*   **constant\_values**: value to pad with if `mode == "constant"`. Defaults to `0`. A `ValueError` is raised if not None and `mode != "constant"`.

Note: Torch backend only supports modes `"constant"`, `"reflect"`, `"symmetric"` and `"circular"`. Only Torch backend supports `"circular"` mode.

Note: Tensorflow backend only supports modes `"constant"`, `"reflect"` and `"symmetric"`.

**Returns**

Padded tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5636)

### `power` function

`keras.ops.power(x1, x2)`

First tensor elements raised to powers from second tensor, element-wise.

**Arguments**

*   **x1**: The bases.
*   **x2**: The exponents.

**Returns**

Output tensor, the bases in `x1` raised to the exponents in `x2`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4251)

### `prod` function

`keras.ops.prod(x, axis=None, keepdims=False, dtype=None)`

Return the product of tensor elements over a given axis.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis or axes along which a product is performed. The default, `axis=None`, will compute the product of all elements in the input tensor.
*   **keepdims**: If this is set to `True`, the axes which are reduce are left in the result as dimensions with size one.
*   **dtype**: Data type of the returned tensor.

**Returns**

Product of elements of `x` over the given axis or axes.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4300)

### `quantile` function

`keras.ops.quantile(x, q, axis=None, method="linear", keepdims=False)`

Compute the q-th quantile(s) of the data along the specified axis.

**Arguments**

*   **x**: Input tensor.
*   **q**: Probability or sequence of probabilities for the quantiles to compute. Values must be between 0 and 1 inclusive.
*   **axis**: Axis or axes along which the quantiles are computed. Defaults to `axis=None` which is to compute the quantile(s) along a flattened version of the array.
*   **method**: A string specifies the method to use for estimating the quantile. Available methods are `"linear"`, `"lower"`, `"higher"`, `"midpoint"`, and `"nearest"`. Defaults to `"linear"`. If the desired quantile lies between two data points `i < j`:
    *   `"linear"`: `i + (j - i) * fraction`, where fraction is the fractional part of the index surrounded by `i` and `j`.
    *   `"lower"`: `i`.
    *   `"higher"`: `j`.
    *   `"midpoint"`: `(i + j) / 2`
    *   `"nearest"`: `i` or `j`, whichever is nearest.
*   **keepdims**: If this is set to `True`, the axes which are reduce are left in the result as dimensions with size one.

**Returns**

The quantile(s). If `q` is a single probability and `axis=None`, then the result is a scalar. If multiple probabilies levels are given, first axis of the result corresponds to the quantiles. The other axes are the axes that remain after the reduction of `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4353)

### `ravel` function

`keras.ops.ravel(x)`

Return a contiguous flattened tensor.

A 1-D tensor, containing the elements of the input, is returned.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4379)

### `real` function

`keras.ops.real(x)`

Return the real part of the complex argument.

**Arguments**

*   **x**: Input tensor.

**Returns**

The real component of the complex argument.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4402)

### `reciprocal` function

`keras.ops.reciprocal(x)`

Return the reciprocal of the argument, element-wise.

Calculates `1/x`.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor, element-wise reciprocal of `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4458)

### `repeat` function

`keras.ops.repeat(x, repeats, axis=None)`

Repeat each element of a tensor after themselves.

**Arguments**

*   **x**: Input tensor.
*   **repeats**: The number of repetitions for each element.
*   **axis**: The axis along which to repeat values. By default, use the flattened input array, and return a flat output array.

**Returns**

Output tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4492)

### `reshape` function

`keras.ops.reshape(x, newshape)`

Gives a new shape to a tensor without changing its data.

**Arguments**

*   **x**: Input tensor.
*   **newshape**: The new shape should be compatible with the original shape. One shape dimension can be -1 in which case the value is inferred from the length of the array and remaining dimensions.

**Returns**

The reshaped tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4523)

### `roll` function

`keras.ops.roll(x, shift, axis=None)`

Roll tensor elements along a given axis.

Elements that roll beyond the last position are re-introduced at the first.

**Arguments**

*   **x**: Input tensor.
*   **shift**: The number of places by which elements are shifted.
*   **axis**: The axis along which elements are shifted. By default, the array is flattened before shifting, after which the original shape is restored.

**Returns**

Output tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4557)

### `round` function

`keras.ops.round(x, decimals=0)`

Evenly round to the given number of decimals.

**Arguments**

*   **x**: Input tensor.
*   **decimals**: Number of decimal places to round to. Defaults to `0`.

**Returns**

Output tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4582)

### `sign` function

`keras.ops.sign(x)`

Returns a tensor with the signs of the elements of `x`.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor of same shape as `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4611)

### `sin` function

`keras.ops.sin(x)`

Trigonometric sine, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor of same shape as `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4640)

### `sinh` function

`keras.ops.sinh(x)`

Hyperbolic sine, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor of same shape as `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4663)

### `size` function

`keras.ops.size(x)`

Return the number of elements in a tensor.

**Arguments**

*   **x**: Input tensor.

**Returns**

Number of elements in `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4690)

### `sort` function

`keras.ops.sort(x, axis=-1)`

Sorts the elements of `x` along a given axis in ascending order.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis along which to sort. If `None`, the tensor is flattened before sorting. Defaults to `-1`; the last axis.

**Returns**

Sorted tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4752)

### `split` function

`keras.ops.split(x, indices_or_sections, axis=0)`

Split a tensor into chunks.

**Arguments**

*   **x**: Input tensor.
*   **indices\_or\_sections**: If an integer, N, the tensor will be split into N equal sections along `axis`. If a 1-D array of sorted integers, the entries indicate indices at which the tensor will be split along `axis`.
*   **axis**: Axis along which to split. Defaults to `0`.

Note: A split does not have to result in equal division when using Torch backend.

**Returns**

A list of tensors.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5718)

### `sqrt` function

`keras.ops.sqrt(x)`

Return the non-negative square root of a tensor, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor, the non-negative square root of `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5688)

### `square` function

`keras.ops.square(x)`

Return the element-wise square of the input.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor, the square of `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5762)

### `squeeze` function

`keras.ops.squeeze(x, axis=None)`

Remove axes of length one from `x`.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Select a subset of the entries of length one in the shape.

**Returns**

The input tensor with all or a subset of the dimensions of length 1 removed.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4808)

### `stack` function

`keras.ops.stack(x, axis=0)`

Join a sequence of tensors along a new axis.

The `axis` parameter specifies the index of the new axis in the dimensions of the result.

**Arguments**

*   **x**: A sequence of tensors.
*   **axis**: Axis along which to stack. Defaults to `0`.

**Returns**

The stacked tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4849)

### `std` function

`keras.ops.std(x, axis=None, keepdims=False)`

Compute the standard deviation along the specified axis.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis along which to compute standard deviation. Default is to compute the standard deviation of the flattened tensor.
*   **keepdims**: If this is set to `True`, the axes which are reduced are left in the result as dimensions with size one.

**Returns**

Output tensor containing the standard deviation values.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5462)

### `subtract` function

`keras.ops.subtract(x1, x2)`

Subtract arguments element-wise.

**Arguments**

*   **x1**: First input tensor.
*   **x2**: Second input tensor.

**Returns**

Output tensor, element-wise difference of `x1` and `x2`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5922)

### `sum` function

`keras.ops.sum(x, axis=None, keepdims=False)`

Sum of a tensor over the given axes.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis or axes along which the sum is computed. The default is to compute the sum of the flattened tensor.
*   **keepdims**: If this is set to `True`, the axes which are reduced are left in the result as dimensions with size one.

**Returns**

Output tensor containing the sum.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4887)

### `swapaxes` function

`keras.ops.swapaxes(x, axis1, axis2)`

Interchange two axes of a tensor.

**Arguments**

*   **x**: Input tensor.
*   **axis1**: First axis.
*   **axis2**: Second axis.

**Returns**

A tensor with the axes swapped.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4927)

### `take` function

`keras.ops.take(x, indices, axis=None)`

Take elements from a tensor along an axis.

**Arguments**

*   **x**: Source tensor.
*   **indices**: The indices of the values to extract.
*   **axis**: The axis over which to select values. By default, the flattened input tensor is used.

**Returns**

The corresponding tensor of values.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4979)

### `take_along_axis` function

`keras.ops.take_along_axis(x, indices, axis=None)`

Select values from `x` at the 1-D `indices` along the given axis.

**Arguments**

*   **x**: Source tensor.
*   **indices**: The indices of the values to extract.
*   **axis**: The axis over which to select values. By default, the flattened input tensor is used.

**Returns**

The corresponding tensor of values.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5016)

### `tan` function

`keras.ops.tan(x)`

Compute tangent, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor of same shape as `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5045)

### `tanh` function

`keras.ops.tanh(x)`

Hyperbolic tangent, element-wise.

**Arguments**

*   **x**: Input tensor.

**Returns**

Output tensor of same shape as `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5105)

### `tensordot` function

`keras.ops.tensordot(x1, x2, axes=2)`

Compute the tensor dot product along specified axes.

**Arguments**

*   **x1**: First tensor.
*   **x2**: Second tensor.
*   **axes**: - If an integer, N, sum over the last N axes of `x1` and the first N axes of `x2` in order. The sizes of the corresponding axes must match. - Or, a list of axes to be summed over, first sequence applying to `x1`, second to `x2`. Both sequences must be of the same length.

**Returns**

The tensor dot product of the inputs.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5152)

### `tile` function

`keras.ops.tile(x, repeats)`

Repeat `x` the number of times given by `repeats`.

If `repeats` has length `d`, the result will have dimension of `max(d, x.ndim)`.

If `x.ndim < d`, `x` is promoted to be d-dimensional by prepending new axes.

If `x.ndim > d`, `repeats` is promoted to `x.ndim` by prepending 1's to it.

**Arguments**

*   **x**: Input tensor.
*   **repeats**: The number of repetitions of `x` along each axis.

**Returns**

The tiled output tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5201)

### `trace` function

`keras.ops.trace(x, offset=0, axis1=0, axis2=1)`

Return the sum along diagonals of the tensor.

If `x` is 2-D, the sum along its diagonal with the given offset is returned, i.e., the sum of elements `x[i, i+offset]` for all `i`.

If a has more than two dimensions, then the axes specified by `axis1` and `axis2` are used to determine the 2-D sub-arrays whose traces are returned.

The shape of the resulting tensor is the same as that of `x` with `axis1` and `axis2` removed.

**Arguments**

*   **x**: Input tensor.
*   **offset**: Offset of the diagonal from the main diagonal. Can be both positive and negative. Defaults to `0`.
*   **axis1**: Axis to be used as the first axis of the 2-D sub-arrays. Defaults to `0`.(first axis).
*   **axis2**: Axis to be used as the second axis of the 2-D sub-arrays. Defaults to `1` (second axis).

**Returns**

If `x` is 2-D, the sum of the diagonal is returned. If `x` has larger dimensions, then a tensor of sums along diagonals is returned.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5795)

### `transpose` function

`keras.ops.transpose(x, axes=None)`

Returns a tensor with `axes` transposed.

**Arguments**

*   **x**: Input tensor.
*   **axes**: Sequence of integers. Permutation of the dimensions of `x`. By default, the order of the axes are reversed.

**Returns**

`x` with its axes permuted.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5245)

### `tri` function

`keras.ops.tri(N, M=None, k=0, dtype=None)`

Return a tensor with ones at and below a diagonal and zeros elsewhere.

**Arguments**

*   **N**: Number of rows in the tensor.
*   **M**: Number of columns in the tensor.
*   **k**: The sub-diagonal at and below which the array is filled. `k = 0` is the main diagonal, while `k < 0` is below it, and `k > 0` is above. The default is 0.
*   **dtype**: Data type of the returned tensor. The default is "float32".

**Returns**

Tensor with its lower triangle filled with ones and zeros elsewhere. `T[i, j] == 1` for `j <= i + k`, 0 otherwise.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5276)

### `tril` function

`keras.ops.tril(x, k=0)`

Return lower triangle of a tensor.

For tensors with `ndim` exceeding 2, `tril` will apply to the final two axes.

**Arguments**

*   **x**: Input tensor.
*   **k**: Diagonal above which to zero elements. Defaults to `0`. the main diagonal. `k < 0` is below it, and `k > 0` is above it.

**Returns**

Lower triangle of `x`, of same shape and data type as `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5308)

### `triu` function

`keras.ops.triu(x, k=0)`

Return upper triangle of a tensor.

For tensors with `ndim` exceeding 2, `triu` will apply to the final two axes.

**Arguments**

*   **x**: Input tensor.
*   **k**: Diagonal below which to zero elements. Defaults to `0`. the main diagonal. `k < 0` is below it, and `k > 0` is above it.

**Returns**

Upper triangle of `x`, of same shape and data type as `x`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5609)

### `true_divide` function

`keras.ops.true_divide(x1, x2)`

Alias for [`keras.ops.divide`](/api/ops/numpy#divide-function).

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5876)

### `var` function

`keras.ops.var(x, axis=None, keepdims=False)`

Compute the variance along the specified axes.

**Arguments**

*   **x**: Input tensor.
*   **axis**: Axis or axes along which the variance is computed. The default is to compute the variance of the flattened tensor.
*   **keepdims**: If this is set to `True`, the axes which are reduced are left in the result as dimensions with size one.

**Returns**

Output tensor containing the variance.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5340)

### `vdot` function

`keras.ops.vdot(x1, x2)`

Return the dot product of two vectors.

If the first argument is complex, the complex conjugate of the first argument is used for the calculation of the dot product.

Multidimensional tensors are flattened before the dot product is taken.

**Arguments**

*   **x1**: First input tensor. If complex, its complex conjugate is taken before calculation of the dot product.
*   **x2**: Second input tensor.

**Returns**

Output tensor.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5389)

### `vstack` function

`keras.ops.vstack(xs)`

Stack tensors in sequence vertically (row wise).

**Arguments**

*   **xs**: Sequence of tensors.

**Returns**

Tensor formed by stacking the given tensors.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5421)

### `where` function

`keras.ops.where(condition, x1=None, x2=None)`

Return elements chosen from `x1` or `x2` depending on `condition`.

**Arguments**

*   **condition**: Where `True`, yield `x1`, otherwise yield `x2`.
*   **x1**: Values from which to choose when `condition` is `True`.
*   **x2**: Values from which to choose when `condition` is `False`.

**Returns**

A tensor with elements from `x1` where `condition` is `True`, and elements from `x2` where `condition` is `False`.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L5950)

### `zeros` function

`keras.ops.zeros(shape, dtype=None)`

Return a new tensor of given shape and type, filled with zeros.

**Arguments**

*   **shape**: Shape of the new tensor.
*   **dtype**: Desired data type of the tensor.

**Returns**

Tensor of zeros with the given shape and dtype.

* * *

[\[source\]](https://github.com/keras-team/keras/tree/v3.1.1//keras/ops/numpy.py#L4057)

### `zeros_like` function

`keras.ops.zeros_like(x, dtype=None)`

Return a tensor of zeros with the same shape and type as `x`.

**Arguments**

*   **x**: Input tensor.
*   **dtype**: Overrides the data type of the result.

**Returns**

A tensor of zeros with the same shape and type as `x`.

* * *