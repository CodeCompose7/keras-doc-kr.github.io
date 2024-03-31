---
layout: default
title: Keras 3
nav_order: 1
permalink: /
---

* 원본 링크 : [https://keras.io/keras_3/](https://keras.io/keras_3/){:target="_blank"}
* 최종 수정일 : 2024-03-30

# Keras 3.0 소개

[시작하기]({{ site.baseurl }}/getting_started){: .btn .btn-blue }
[API 문서]({{ site.baseurl }}/api){: .btn .btn-blue }
[가이드]({{ site.baseurl }}/guides){: .btn .btn-blue }
[GitHub](https://github.com/keras-team/keras/){: .btn .btn-blue }

5개월간의 광범위한 공개 베타 테스트 끝에, Keras 3.0의 공식 출시를 발표하게 되어 기쁘게 생각합니다. Keras 3는 완전히 새롭게 재작성된 Keras로, JAX, TensorFlow 또는 PyTorch 위에서 Keras 워크플로를 실행할 수 있으며, 완전히 새로운 대규모 모델 트레이닝 및 배포 기능을 제공합니다. 현재 목표에 따라 가장 적합한 프레임워크를 선택하고, 다른 프레임워크로 전환할 수 있습니다. 또한 Keras를 로우레벨 크로스 프레임워크 언어로 사용하여, 하나의 코드베이스로 JAX, TensorFlow 또는 PyTorch의 기본 워크플로에서 사용할 수 있는 레이어, 모델 또는 메트릭과 같은 사용자 지정 구성 요소를 개발할 수도 있습니다.

----

## 멀티 프레임워크 머신 러닝에 오신 것을 환영합니다.

뛰어난 UX, API 디자인, 디버깅 기능에 집중하여 빠른 속도로 개발할 수 있는 Keras의 장점은 이미 잘 알고 계실 것입니다. 또한 250만 명 이상의 개발자가 선택했으며, Waymo 자율 주행 차량 및 YouTube 추천 엔진과 같은 세계에서 가장 정교하고 대규모의 ML 시스템을 구동하는 등 수많은 테스트를 거친 프레임워크입니다. 그렇다면 새로운 멀티 백엔드 Keras 3를 사용하면 어떤 추가적인 이점이 있을까요?

*  **당신의 모델에 항상 최상의 성능 제공** 벤치마크 결과, JAX는 일반적으로 GPU, TPU, CPU에서 최고의 트레이닝 및 추론 성능을 제공하지만, 비-XLA TensorFlow가 GPU에서 더 빠른 경우도 있기 때문에 결과는 모델마다 다릅니다. _당신의 코드를 변경하지 않고도_ 당신의 모델에 가장 적합한 성능을 제공하는 백엔드를 동적으로 선택할 수 있으므로, 달성 가능한 최고의 효율로 트레이닝하고 서비스를 제공할 수 있습니다.
*  **모델에 대한 에코시스템 옵션 잠금 해제** 모든 Keras 3 모델은 PyTorch `Module`로 인스턴스화할 수 있으며, TensorFlow `SavedModel`로 내보낼 수 있고, 상태없는 JAX 함수로 인스턴스화할 수 있습니다. 즉, PyTorch 에코시스템 패키지, 모든 범위의 TensorFlow 배포 및 프로덕션 도구(예: TF-Serving, TF.js 및 TFLite), JAX 대규모 TPU 트레이닝 인프라와 함께 Keras 3 모델을 사용할 수 있습니다. Keras 3 API를 사용해 하나의 `model.py`를 작성하고, ML 세계의 모든 것을 이용할 수 있습니다.
*  **JAX로 대규모 모델 병렬 처리 및 데이터 병렬 처리 활용** Keras 3에는 현재 JAX 백엔드용으로 구현된 새로운 배포 API인 `keras.distribution` 네임스페이스가 포함되어 있습니다. (곧 TensorFlow 및 PyTorch 백엔드에 제공될 예정) 이를 통해 임의의 모델 규모와 클러스터 규모에서, 모델 병렬 처리, 데이터 병렬 처리, 그리고 이 두 가지의 조합을 쉽게 수행할 수 있습니다. 모델 정의, 학습 로직, 샤딩 구성이 모두 서로 분리되어 있기 때문에, 배포 워크플로우를 개발하기 쉽고 유지 관리가 쉽습니다. [스타터 가이드]({{ site.baseurl }}/guides/distribution/)를 참조하세요.
*  **오픈소스 모델 릴리스의 도달 범위를 극대화하세요.** 사전 트레이닝된 모델을 릴리스하고 싶으신가요? 최대한 많은 사람이 사용할 수 있기를 원하시나요? 순수 TensorFlow 또는 PyTorch로 구현하면 커뮤니티의 약 절반이 사용할 수 있습니다. Keras 3로 구현하면 선택한 프레임워크에 관계없이 누구나 즉시 사용할 수 있습니다.(Keras 사용자가 아니더라도) 추가 개발 비용 없이 두 배의 효과를 얻을 수 있습니다.
*  **모든 소스의 데이터 파이프라인을 사용하세요.** Keras 3 `fit()`/`evaluate()`/`predict()` 루틴은 사용 중인 백엔드에 관계없이 `tf.data.Dataset` 객체, PyTorch `DataLoader` 객체, NumPy 배열, Pandas 데이터 프레임과 호환이 가능합니다. PyTorch `DataLoader`에서 Keras 3 + TensorFlow 모델을 트레이닝하거나, `tf.data.Dataset`에서 Keras 3 + PyTorch 모델을 트레이닝할 수 있습니다.

---

## JAX, TensorFlow, PyTorch에서 사용할 수 있는 전체 Keras API.

Keras 3는 전체 Keras API를 구현하여 100개 이상의 레이어, 수십 개의 메트릭, 손실 함수, 최적화 도구 및 콜백, Keras 트레이닝 및 평가 루프, Keras 저장 및 직렬화 인프라 등 전체 Keras API를 TensorFlow, JAX 및 PyTorch에서 사용할 수 있도록 합니다. 여러분이 잘 알고 사랑하는 모든 API가 여기에 있습니다.

빌트인 레이어만 사용하는 모든 Keras 모델은 지원되는 모든 백엔드에서 즉시 작동합니다. 실제로, 빌트인 레이어만 사용하는 기존 `tf.keras` 모델은 JAX와 PyTorch에서 _바로_ 실행을 시작할 수 있습니다! 코드베이스에 완전히 새로운 기능이 추가된 것입니다.

![](https://s3.amazonaws.com/keras.io/img/keras_3/cross_framework_keras_3.jpg)

---

## 멀티 프레임워크 레이어, 모델, 메트릭 작성...

Keras 3를 사용하면 모든 프레임워크에서 동일하게 작동하는 구성 요소(임의의 사용자 정의 레이어 또는 사전 트레이닝된 모델 등)를 만들 수 있습니다. 특히, Keras 3에서는 모든 백엔드에서 작동하는 `keras.ops` 네임스페이스에 액세스할 수 있습니다. 여기에는 다음이 포함됩니다:

* **NumPy API의  완전한 구현** "NumPy와 유사한" 것이 아니라, 말 그대로 동일한 함수와 동일한 인수를 사용하는 NumPy API의  완전한 구현입니다. `ops.matmul`, `ops.sum`, `ops.stack`, `ops.einsum` 등을 사용할 수 있습니다.
* **NumPy에는 없는  신경망 전용 함수 집합** NumPy에는 없는  신경망 전용 함수 집합으로, `ops.softmax`, `ops.binary_crossentropy`, `ops.conv` 등이 있습니다.
 
사용자 정의 레이어, 사용자 정의 손실, 사용자 정의 메트릭, 사용자 정의 옵티마이저는 `keras.ops`의 ops만 사용하는 한, **동일한 코드를 사용해 JAX, PyTorch, TensorFlow에서 작동합니다.** 즉, 하나의 컴포넌트 구현(예: 단일 체크포인트 파일과 함께 단일 `model.py`)만 유지하면, 모든 프레임워크에서 정확히 동일한 수치로 사용할 수 있습니다.

![](https://s3.amazonaws.com/keras.io/img/keras_3/custom_component_authoring_keras_3.jpg)

---

## ...모든 JAX, TensorFlow, PyTorch 워크플로우에서 원활하게 작동합니다.

Keras 3는 Keras 모델, Keras 옵티마이저, Keras 손실 및 메트릭을 정의하고, `fit()`, `evaluate()` 및 `predict()`를 호출하는 Keras 중심 워크플로우만을 위한 것이 아닙니다. 또한 낮은 레벨의 백엔드 네이티브 워크플로우와도 원활하게 작동하도록 설계되었습니다. 즉, Keras 모델(또는 손실이나 메트릭과 같은 다른 구성 요소)을 가져와 JAX 트레이닝 루프, TensorFlow 트레이닝 루프 또는 PyTorch 트레이닝 루프에서 사용하거나, JAX 또는 PyTorch 모델의 일부로서 충돌없이 시작할 수 있습니다. Keras 3는 이전에 TensorFlow에서 `tf.keras`가 제공했던 것과 동일한 수준의 저레벨 구현 유연성을 JAX 및 PyTorch에서 제공합니다.

다음을 수행할 수 있습니다:

* `optax` 옵티마이저, `jax.grad`, `jax.jit`, `jax.pmap`을 사용해 저레벨 JAX 트레이닝 루프를 작성하여, Keras 모델을 트레이닝할 수 있습니다.
* `tf.GradientTape` 및 `tf.distribute`를 사용하여, Keras 모델을 트레이닝하는 저레벨 TensorFlow 훈련 루프를 작성합니다.
* 낮은 레벨의 PyTorch 트레이닝 루프를 작성하여, `torch.optim` 옵티마이저, `torch` 손실 함수 및 `torch.nn.parallel.DistributedDataParallel` 래퍼를 사용하여 Keras 모델을 트레이닝합니다.
* PyTorch `Module`에서 Keras 레이어를 사용하세요. (왜냐하면 그들은 `Module` 인스턴스이기도 하므로!)
* Keras 모델에서 PyTorch `Module`을 Keras 레이어인 것처럼 사용하세요.
* 등

![](https://s3.amazonaws.com/keras.io/img/keras-core/custom_training_loops.jpg)

---

## 대규모 데이터 병렬처리와 모델 병렬처리를 위한 새로운 배포 API.

우리가 작업해 온 모델의 규모가 점점 더 커지고 있기 때문에, 우리는 다중 장치 모델 샤딩 문제에 대한 Kerasic 솔루션을 제공하고자 했습니다. 우리가 설계한 API는 모델 정의, 트레이닝 로직, 샤딩 구성을 서로 완전히 분리하여, 단일 기기에서 실행되는 것처럼 모델을 작성할 수 있습니다. 그런 다음 모델을 트레이닝할 때가 되면 임의의 샤딩 구성을 임의의 모델에 추가할 수 있습니다.

데이터 병렬 처리(작은 모델을 여러 디바이스에서 동일하게 복제)는 단 두 줄로 처리할 수 있습니다:

![](https://s3.amazonaws.com/keras.io/img/keras_3/keras_3_data_parallel.jpg)

모델 병렬화를 사용하면 모델 변수와 중간 출력 텐서에 대한 샤딩 레이아웃을 여러 개의 명명된 차원을 따라 지정할 수 있습니다. 일반적인 경우, 사용 가능한 장치를 2D 그리드(*장치 메시(device mesh)*라고 함)로 구성하여, 첫 번째 차원은 데이터 병렬 처리에 사용하고, 두 번째 차원은 모델 병렬 처리에 사용합니다. 그런 다음 모델 차원을 따라 샤딩되고, 데이터 차원을 따라 복제되도록 모델을 구성할 수 있습니다.

API를 사용하면 정규 표현식을 통해 모든 변수와 모든 출력 텐서의 레이아웃을 구성할 수 있습니다. 이를 통해 전체 변수 범주에 대해 동일한 레이아웃을 빠르게 지정할 수 있습니다.

![](https://s3.amazonaws.com/keras.io/img/keras_3/keras_3_model_parallel.jpg)

새로운 배포 API는 다중 백엔드를 위한 것이지만, 당분간은 JAX 백엔드에서만 사용할 수 있습니다. 텐서플로우와 파이토치 지원은 곧 제공될 예정입니다. [이 가이드]({{ site.baseurl }}/guides/distribution/)로 시작하세요!

---

## 사전 트레이닝된 모델.

Keras 3에서 지금 바로 사용할 수 있는 다양한 사전 트레이닝된 모델이 있습니다.

모든 백엔드에서 40개의 Keras 애플리케이션 모델(`keras.applications` 네임스페이스)을 사용할 수 있습니다. [KerasCV]({{ site.baseurl }}/api/keras_cv/) 및 [KerasNLP]({{ site.baseurl }}/api/keras_nlp/)의 방대한 사전 트레이닝된 모델도 모든 백엔드에서 작동합니다. 여기에는 다음이 포함됩니다:

* BERT
* OPT
* Whisper
* T5
* StableDiffusion
* YOLOv8
* SegmentAnything
* 등

--- 

## 모든 백엔드에서 크로스 프레임워크 데이터 파이프라인을 지원합니다.

멀티 프레임워크 ML은 또한 멀티 프레임워크 데이터 로딩 및 전처리를 의미합니다. Keras 3 모델은 JAX, PyTorch, TensorFlow 백엔드 중 어떤 것을 사용하든 상관없이, 다양한 데이터 파이프라인을 사용하여 트레이닝할 수 있습니다. 그냥 작동합니다.

* `tf.data.Dataset` 파이프라인: 확장 가능한 프로덕션 ML을 위한 레퍼런스.
* `torch.utils.data.DataLoader` 객체.
* NumPy 배열 및 Pandas 데이터 프레임.
* Keras 자체 `keras.utils.PyDataset` 객체.

---

## 복잡성의 점진적 공개.

*복잡성의 점진적 공개*는 Keras API의 핵심 설계 원칙입니다. Keras는 하나의 "진정한" 모델 구축 및 트레이닝 방식을 따르도록 강요하지 않습니다. 대신, 매우 높은 레벨부터 매우 낮은 레벨까지 다양한 사용자 프로필에 따라 다양한 워크플로우를 지원합니다.

즉, `Sequential` 및 `Functional` 모델을 사용하고 `fit()`으로 트레이닝하는 등 간단한 워크플로로 시작할 수 있으며, 유연성이 더 필요한 경우, 이전 코드 대부분을 재사용하면서 다양한 구성 요소를 쉽게 사용자 지정할 수 있습니다. 요구 사항이 더 구체화되어도 갑자기 복잡성의 절벽으로 떨어지지 않고, 다른 도구 세트로 전환할 필요가 없습니다.

이러한 원칙을 모든 백엔드에 적용했습니다. 예를 들어, 트레이닝 루프를 처음부터 새로 작성할 필요없이, `train_step` 메서드를 재정의하는 것만으로, 트레이닝 루프에서 일어나는 일을 사용자 정의하면서, `fit()`의 강력한 기능을 활용할 수 있습니다.

PyTorch 및 TensorFlow에서 작동하는 방식은 다음과 같습니다:

![](https://s3.amazonaws.com/keras.io/img/keras-core/customizing_fit.jpg)

그리고 [여기 링크]({{ site.baseurl }}/guides/custom_train_step_in_jax/)는 JAX 버전에 대한 링크입니다.

---

## 레이어, 모델, 메트릭, 옵티마이저를 위한 새로운 상태없는 API.

[함수형 프로그래밍](https://en.wikipedia.org/wiki/Functional_programming)을 좋아하시나요? 정말 마음에 드실 겁니다.

이제 Keras의 모든 상태 있는 객체(즉, 트레이닝 또는 평가 중에 업데이트되는 숫자 변수를 소유하는 객체)에 상태 없는 API가 제공되어, (완전히 상태없음을 요구하는) JAX 함수에서 사용할 수 있게 되었습니다:

* 모든 레이어와 모델에는 `__call__()`을 미러링하는 `stateless_call()` 메서드가 있습니다.
* 모든 옵티마이저에는 `apply()`를 미러링하는 `stateless_apply()` 메서드가 있습니다.
* 모든 메트릭에는 `update_state()`를 미러링하는 `stateless_update_state()` 메서드와 `result()`를 미러링하는 `stateless_result()` 메서드가 있습니다.

이러한 메서드에는 부수효과(side-effects)가 전혀 없습니다: 대상 객체의 상태 변수의 현재 값을 입력으로 받고, 업데이트 값을 출력의 일부로 반환합니다. 예를 들어, :

```python
outputs, updated_non_trainable_variables = layer.stateless_call(
    trainable_variables,
    non_trainable_variables,
    inputs,
)
```

이러한 메서드는 직접 구현할 필요가 없습니다. 상태 있는 버전(예: `call()` 또는 `update_state()`)을 구현하기만 하면 자동으로 사용할 수 있습니다.

* * *

Keras 2에서 Keras 3으로 이동하기
------------------------------
Keras 3는 Keras 2와의 역호환성이 뛰어나며, [여기](https://github.com/keras-team/keras/issues/18467)에 나열된 몇 가지 예외를 제외하고, Keras 2의 전체 공개 API 표면을 구현합니다. 대부분의 사용자는 Keras 3에서 Keras 스크립트를 실행하기 위해 코드를 변경할 필요가 없습니다.

대규모 코드베이스는 위에 나열된 예외 중 하나가 발생할 가능성이 높고, private API 또는 deprecated API(`tf.compat.v1.keras` 네임스페이스, `experimental` 네임스페이스, `keras.src` private 네임스페이스)를 사용했을 가능성이 높으므로 일부 코드 변경이 필요할 수 있습니다. Keras 3로 마이그레이션하는 데 도움을 드리기 위해, 발생할 수 있는 모든 문제에 대한 빠른 수정 사항이 포함된 전체 [마이그레이션 가이드]({{ site.baseurl }}/guides/migrating_to_keras_3/)를 공개하고 있습니다.

또한 Keras 3의 변경 사항을 무시하고, TensorFlow와 Keras 2를 계속 사용할 수 있는 옵션도 있습니다. 이는 활발하게 개발되지는 않았지만, 업데이트된 종속성으로 계속 실행해야 하는 프로젝트에 좋은 옵션이 될 수 있습니다. 두 가지 가능성이 있습니다:

1.  독립형 패키지로 `keras`에 액세스하고 있었다면, `pip install tf_keras`를 통해 설치할 수 있는 Python 패키지 `tf_keras`를 사용하도록 전환하세요. 코드와 API는 완전히 변경되지 않았으며, 패키지 이름만 다른 Keras 2.15입니다. 저희는 `tf_keras`의 버그를 계속 수정하고 정기적으로 새 버전을 출시할 예정입니다. 그러나 패키지가 유지 관리 모드에 있으므로, 새로운 기능이나 성능 개선은 추가되지 않습니다.

2.  `tf.keras`를 통해 `keras`에 액세스하는 경우, TensorFlow 2.16까지는 즉각적인 변경 사항이 없습니다. TensorFlow 2.16 이상에서는 기본적으로 Keras 3을 사용합니다. TensorFlow 2.16+에서 Keras 2를 계속 사용하려면, 먼저 `tf_keras`를 설치한 다음, 환경 변수 `TF_USE_LEGACY_KERAS=1`을 export 할 수 있습니다. 이렇게 하면 TensorFlow 2.16+가 로컬에 설치된 `tf_keras` 패키지로 tf.keras를 확인하도록 지시합니다. 그러나 이것은 자신의 코드에만 영향을 미치는 것이 아니라, Python 프로세스에서 `tf.keras`를 임포트하는 모든 패키지에 영향을 미칠 수 있다는 점에 유의하세요. 변경 사항이 자신의 코드에만 영향을 미치도록 하려면, `tf_keras` 패키지를 사용해야 합니다.

* * *

라이브러리를 즐겨보세요!
------------------

새로운 Keras를 사용해 보시고 멀티 프레임워크 ML을 활용하여 워크플로를 개선해 보시기를 기대합니다. 이슈, 충돌 지점, 기능 요청, 성공 사례 등 여러분의 의견을 듣고 싶으니 언제든 알려주세요!

* * *

FAQ
---

#### Q: Keras 3는 기존 Keras 2와 호환되나요?

예. `tf.keras`로 개발된 코드는 일반적으로 (TensorFlow 백엔드로) Keras 3에서 그대로 실행할 수 있습니다. 주의해야 할 몇 가지 비호환성이 있으며, 모두 이 [마이그레이션 가이드]({{ site.baseurl }}/guides/migrating_to_keras_3/)에서 다루고 있습니다.

`tf.keras`와 Keras 3의 API를 나란히 사용하는 것은, 완전히 다른 엔진에서 실행되는 서로 다른 패키지이기 때문에 **가능하지** 않습니다.

### Q: 기존 Keras 2에서 개발된 사전 트레이닝된 모델이 Keras 3에서도 작동하나요?

일반적으로 그렇습니다. 모든 `tf.keras` 모델은 TensorFlow 백엔드로 Keras 3에서 바로 작동합니다. (`.keras` v3 형식으로 저장해야 함) 또한, 모델에 빌트인 Keras 레이어만 사용하는 경우, JAX 및 PyTorch 백엔드가 있는 Keras 3에서도 바로 작동합니다.

모델에 TensorFlow API를 사용하여 작성된 사용자 정의 레이어가 포함된 경우, 일반적으로 코드를 백엔드에 구애받지 않도록 쉽게 변환할 수 있습니다. 예를 들어, Keras 애플리케이션의 40개 레거시 `tf.keras` 모델을 모두 백엔드에 구애받지 않도록 변환하는 데 몇 시간밖에 걸리지 않았습니다.

### Q: Keras 3 모델을 한 백엔드에 저장하고, 다른 백엔드에서 다시 로드할 수 있나요?

예, 가능합니다. 저장된 `.keras` 파일에는 백엔드 특화 기능이 전혀 없습니다. 저장된 Keras 모델은 프레임워크에 구애받지 않으며, 어떤 백엔드에서든 다시 로드할 수 있습니다.

그러나, 사용자 정의 구성 요소가 포함된 모델을 다른 백엔드로 다시 로드하려면 백엔드에 구애받지 않는 API(예: `keras.ops`)를 사용하여 사용자 정의 구성 요소를 구현해야 한다는 점에 유의하세요.

### Q: `tf.data` 파이프라인 내에서 Keras 3 구성 요소를 사용할 수 있나요?

예. TensorFlow 백엔드를 사용하면 Keras 3는 `tf.data`와 완벽하게 호환됩니다. (예: `Sequential` 모델을 `tf.data` 파이프라인에 `.map()`할 수 있음)

다른 백엔드를 사용하는 Keras 3는 `tf.data`에 대한 지원이 제한적입니다. 임의의 레이어나 모델을 `tf.data` 파이프라인으로 `.map()` 할 수 없습니다. 그러나, `tf.data`와 함께 특정 Keras 3 전처리 레이어(예: `IntegerLookup` 또는 `CategoryEncoding`)는 사용할 수 있습니다.

`tf.data` 파이프라인(Keras를 사용하지 않는)을 사용하여 `.fit()`, `.evaluate()` 또는 `.predict()`에 대한 호출을 제공하는 경우, 모든 백엔드에서 즉시 작동합니다.

### Q: Keras 3 모델은 다른 백엔드에서 실행해도 동일하게 작동하나요?

예, 백엔드 간에 수치는 동일합니다. 하지만 다음 주의 사항에 유의하세요:

* RNG 동작은 백엔드마다 다릅니다. (seeding 후에도 결과는 각 백엔드에서 결정론적(deterministic)이지만, 백엔드마다 다를 수 있습니다) 따라서, 무작위 가중치 초기화 값과 드롭아웃 값은 백엔드마다 다를 수 있습니다.
* 부동소수점 구현의 특성상, 결과는 함수 실행당 float32의 `1e-7` 정밀도까지만 동일합니다. 따라서, 오랜 시간 모델을 트레이닝할 경우, 작은 수치 차이가 누적되어 결국 눈에 띄는 수치 차이가 발생할 수 있습니다.
* PyTorch에서는 비대칭 패딩(asymmetric padding)을 사용한 평균 풀링을 지원하지 않기 때문에, `padding="same"`로 평균 풀링 레이어를 사용하면, 테두리 행/열의 숫자가 달라질 수 있습니다. 실제로는 자주 발생하지 않으며, 40개의 Keras 애플리케이션 비전 모델 중 단 하나만 영향을 받았습니다.

### Q: Keras 3는 분산 트레이닝을 지원하나요?

예. 데이터 병렬 배포는 JAX, TensorFlow 및 PyTorch에서 기본적으로 지원됩니다. 모델 병렬 배포는 JAX의 경우 `keras.distribution` API를 통해 즉시 지원됩니다.

**TensorFlow 사용 시:**

Keras 3는 `tf.distribute`와 호환됩니다. Distribution Strategy scope를 열고, 그 안에서 모델을 생성/트레이닝하기만 하면 됩니다. [여기 예시가 있습니다]({{ site.baseurl }}/guides/distributed_training_with_tensorflow/).

**PyTorch 사용 시:**

Keras 3는 PyTorch의 `DistributedDataParallel` 유틸리티와 호환됩니다. [여기 예시가 있습니다]({{ site.baseurl }}/guides/distributed_training_with_torch/).

**JAX 사용 시:**

JAX에서는 `keras.distribution` API를 사용하여 데이터 병렬 배포와 모델 병렬 배포를 모두 수행할 수 있습니다. 예를 들어, 데이터 병렬 배포를 수행하려면 다음 코드 조각만 필요합니다:

```python
distribution = keras.distribution.DataParallel(devices=keras.distribution.list_devices())
keras.distribution.set_distribution(distribution)
```

모델 병렬 배포에 대해서는 [다음 가이드]({{ site.baseurl }}/guides/distribution/)를 참조하세요.

또한 `jax.sharding`과 같은 JAX API를 통해 직접 트레이닝을 배포할 수도 있습니다. [여기 예시가 있습니다]({{ site.baseurl }}/guides/distributed_training_with_jax/).

### Q: 사용자 정의 Keras 레이어를 기본 PyTorch `Modules` 또는 Flax `Modules`에서 사용할 수 있나요?

Keras API(예: `keras.ops` 네임스페이스)만 사용하여 작성된 경우, 예, Keras 레이어는 기본 PyTorch 및 JAX 코드에서 바로 작동합니다. PyTorch에서는 다른 PyTorch `Module`처럼 케라스 레이어를 사용하면 됩니다. JAX에서는 상태없는 레이어 API, 즉 `layer.stateless_call()`을 사용해야 합니다.

### Q: 앞으로 더 많은 백엔드를 추가할 예정인가요? 프레임워크 XYZ는 어떻게 되나요?

대상 프레임워크에 대규모 사용자 기반이 있거나, 기타 고유한 기술적 이점이 있는 경우, 새로운 백엔드를 추가할 수 있습니다. 그러나, 새로운 백엔드를 추가하고 유지 관리하는 것은 큰 부담이 되므로, 각각의 새로운 백엔드 후보를 사례별로 신중하게 검토할 것이며, 새로운 백엔드를 많이 추가하지는 않을 것입니다. 아직 제대로 확립되지 않은 새로운 프레임워크는 추가하지 않을 것입니다. 현재 [Mojo](https://www.modular.com/mojo)로 작성된 백엔드를 추가하는 방안을 검토 중입니다. 이 기능이 유용하다고 생각되면, Mojo 팀에 알려주시기 바랍니다.
