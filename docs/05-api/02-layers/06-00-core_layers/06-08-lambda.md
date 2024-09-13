---
layout: default
title: Lambda 레이어
nav_order: 06+08
permalink: /api/layers/core_layers/lambda/
parent: 코어 레이어
grand_parent: Keras 레이어 API
---

* 원본 링크 : [https://keras.io/api/layers/core_layers/lambda/](https://keras.io/api/layers/core_layers/lambda/){:target="_blank"}
* 최종 수정일 : 2024-09-05

# Lambda 레이어 (Lambda layer)
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

### `Lambda` 클래스
{: #lambda-class}
<!-- ### `Lambda` class -->

[소스](https://github.com/keras-team/keras/tree/v3.5.0/keras/src/layers/core/lambda_layer.py#L12){: .btn .btn-outline }


```python
keras.layers.Lambda(function, output_shape=None, mask=None, arguments=None, **kwargs)
```

임의의 표현식을 `Layer` 객체로 래핑합니다.

`Lambda` 레이어는 Sequential 및 Functional API 모델을 구성할 때, 
임의의 표현식을 `Layer`로 사용할 수 있도록 존재합니다. 
`Lambda` 레이어는 간단한 작업이나 빠른 실험에 가장 적합합니다. 
고급 사용 사례의 경우, `Layer`의 새 하위 클래스를 작성하는 것이 좋습니다.

경고: `Lambda` 레이어에는 (역)직렬화 제한이 있습니다!

`Lambda` 레이어를 사용하는 대신, 
`Layer`를 하위 클래스화하는 주된 이유는 모델을 저장하고 검사하기 위해서입니다. 
`Lambda` 레이어는 Python 바이트코드를 직렬화하여 저장되는데, 
이는 근본적으로 이식성이 없고 잠재적으로 안전하지 않습니다. 
저장된 동일한 환경에서만 로드해야 합니다. 
하위 클래스화된 레이어는 `get_config()` 메서드를 재정의하여, 
보다 이식성 있는 방식으로 저장할 수 있습니다. 
하위 클래스화된 레이어에 의존하는 모델은 시각화하고 추론하기가 더 쉽습니다.

**예제**

```python
# add a x -> x^2 layer
model.add(Lambda(lambda x: x ** 2))
```

**인수**

*   **function**: 평가할 함수. 
    *   첫 번째 인수로 입력 텐서를 취합니다.
*   **output\_shape**: 함수에서 예상되는 출력 모양. 
    *   이 인수는 명시적으로 제공되지 않으면, 일반적으로 추론할 수 있습니다. 
    *   튜플 또는 함수가 될 수 있습니다. 
        *   튜플인 경우, 첫 번째 차원만 지정합니다. 샘플 차원은 입력과 동일하다고 가정합니다. `output_shape = (input_shape[0], ) + output_shape` 또는 입력이 `None`이고 샘플 차원도 `None`입니다. `output_shape = (None, ) + output_shape`. 
        *   함수인 경우, 전체 모양을 입력 모양의 함수로 지정합니다. `output_shape = f(input_shape)`.
*   **mask**: None (마스킹 없음을 나타냄) 또는 `compute_mask` 레이어 메서드와 동일한 시그니처를 가진 호출 가능 객체, 또는 입력이 무엇이든 출력 마스크로 반환되는 텐서입니다.
*   **arguments**: 함수에 전달할 키워드 인수의 Optional 딕셔너리.

* * *