---
layout: default
title: MultiHeadAttention 레이어
nav_order: 13+02
permalink: /api/layers/attention_layers/multi_head_attention/
parent: Attention 레이어
grand_parent: Keras 레이어 API
---

* 원본 링크 : [https://keras.io/api/layers/attention_layers/multi_head_attention/](https://keras.io/api/layers/attention_layers/multi_head_attention/){:target="_blank"}
* 최종 수정일 : 2024-09-05

# MultiHeadAttention 레이어 (MultiHeadAttention layer)
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

### `MultiHeadAttention` 클래스
<!-- ### `MultiHeadAttention` class -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/layers/attention/multi_head_attention.py#L19){: .btn .btn-outline }

```python
keras.layers.MultiHeadAttention(
    num_heads,
    key_dim,
    value_dim=None,
    dropout=0.0,
    use_bias=True,
    output_shape=None,
    attention_axes=None,
    kernel_initializer="glorot_uniform",
    bias_initializer="zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    seed=None,
    **kwargs
)
```

MultiHeadAttention layer.

This is an implementation of multi-headed attention as described in the paper "Attention is all you Need" [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762). If `query`, `key,` `value` are the same, then this is self-attention. Each timestep in `query` attends to the corresponding sequence in `key`, and returns a fixed-width vector.

This layer first projects `query`, `key` and `value`. These are (effectively) a list of tensors of length `num_attention_heads`, where the corresponding shapes are `(batch_size, <query dimensions>, key_dim)`, `(batch_size, <key/value dimensions>, key_dim)`, `(batch_size, <key/value dimensions>, value_dim)`.

Then, the query and key tensors are dot-producted and scaled. These are softmaxed to obtain attention probabilities. The value tensors are then interpolated by these probabilities, then concatenated back to a single tensor.

Finally, the result tensor with the last dimension as `value_dim` can take a linear projection and return.

**인수**

*   **num\_heads**: Number of attention heads.
*   **key\_dim**: Size of each attention head for query and key.
*   **value\_dim**: Size of each attention head for value.
*   **dropout**: Dropout probability.
*   **use\_bias**: Boolean, whether the dense layers use bias vectors/matrices.
*   **output\_shape**: The expected shape of an output tensor, besides the batch and sequence dims. If not specified, projects back to the query feature dim (the query input's last dimension).
*   **attention\_axes**: axes over which the attention is applied. `None` means attention over all axes, but batch, heads, and features.
*   **kernel\_initializer**: Initializer for dense layer kernels.
*   **bias\_initializer**: Initializer for dense layer biases.
*   **kernel\_regularizer**: Regularizer for dense layer kernels.
*   **bias\_regularizer**: Regularizer for dense layer biases.
*   **activity\_regularizer**: Regularizer for dense layer activity.
*   **kernel\_constraint**: Constraint for dense layer kernels.
*   **bias\_constraint**: Constraint for dense layer kernels.
*   **seed**: Optional integer to seed the dropout layer.

**호출 인수**

*   **query**: Query tensor of shape `(B, T, dim)`, where `B` is the batch size, `T` is the target sequence length, and dim is the feature dimension.
*   **value**: Value tensor of shape `(B, S, dim)`, where `B` is the batch size, `S` is the source sequence length, and dim is the feature dimension.
*   **key**: Optional key tensor of shape `(B, S, dim)`. If not given, will use `value` for both `key` and `value`, which is the most common case.
*   **attention\_mask**: a boolean mask of shape `(B, T, S)`, that prevents attention to certain positions. The boolean mask specifies which query elements can attend to which key elements, 1 indicates attention and 0 indicates no attention. Broadcasting can happen for the missing batch dimensions and the head dimension.
*   **return\_attention\_scores**: A boolean to indicate whether the output should be `(attention_output, attention_scores)` if `True`, or `attention_output` if `False`. Defaults to `False`.
*   **training**: Python boolean indicating whether the layer should behave in training mode (adding dropout) or in inference mode (no dropout). Will go with either using the training mode of the parent layer/model, or `False` (inference) if there is no parent layer.
*   **use\_causal\_mask**: A boolean to indicate whether to apply a causal mask to prevent tokens from attending to future tokens (e.g., used in a decoder Transformer).

**반환**

*   **attention\_output**: The result of the computation, of shape `(B, T, E)`, where `T` is for target sequence shapes and `E` is the query input last dimension if `output_shape` is `None`. Otherwise, the multi-head outputs are projected to the shape specified by `output_shape`.
*   **attention\_scores**: (Optional) multi-head attention coefficients over attention axes.

* * *