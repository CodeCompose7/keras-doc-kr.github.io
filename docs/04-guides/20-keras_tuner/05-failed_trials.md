---
layout: default
title: KerasTuner에서 실패한 시도 처리하기
nav_order: 5
permalink: /guides/keras_tuner/failed_trials/
parent: KerasTuner
grand_parent: 개발자 가이드
---

* 원본 링크 : [https://keras.io/guides/keras_tuner/failed_trials/](https://keras.io/guides/keras_tuner/failed_trials/){:target="_blank"}
* 최종 수정일 : 2024-09-20

# KerasTuner에서 실패한 시도 처리하기 (Handling failed trials in KerasTuner)
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

**저자:** Haifeng Jin  
**생성일:** 2023/02/28  
**최종편집일:** 2023/02/28  
**설명:** KerasTuner의 장애 허용(fault tolerance) 구성의 기본 사항.

[Colab에서 보기](https://colab.research.google.com/github/keras-team/keras-io/blob/master/guides/ipynb/keras_tuner/failed_trials.ipynb){: .btn .btn-blue }
[GitHub 소스](https://github.com/keras-team/keras-io/blob/master/guides/keras_tuner/failed_trials.py){: .btn .btn-blue }

----

## 소개
{: #introduction}
<!-- ## Introduction -->

KerasTuner 프로그램은 각 모델의 트레이닝 시간이 오래 걸릴 수 있기 때문에, 
실행 시간이 길어질 수 있습니다. 
일부 시도가 랜덤하게 실패한다고 해서, 프로그램이 중단되지 않도록 처리해야 합니다.

이 가이드에서는 KerasTuner에서 실패한 시도를 처리하는 방법을 설명합니다. 다음 내용을 포함합니다:

-   탐색 중 실패한 시도를 허용(tolerate)하는 방법
-   모델을 빌드하고 평가하는 동안, 시도를 실패로 표시하는 방법
-   `FatalError`를 발생시켜 탐색을 중단하는 방법

------------------------------------------------------------------------

## 셋업
{: #setup}
<!-- ## Setup -->

```python
!pip install keras-tuner -q

import keras
from keras import layers
import keras_tuner
import numpy as np
```

------------------------------------------------------------------------

## 실패한 시도 허용하기(Tolerate)
{: #tolerate-failed-trials}
<!-- ## Tolerate failed trials -->

우리는 튜너를 초기화할 때 `max_retries_per_trial` 및 `max_consecutive_failed_trials` 인수를 사용할 것입니다.

`max_retries_per_trial`은 시도가 계속 실패할 때, 
최대 몇 번 다시 시도할지를 제어합니다. 
예를 들어, 값이 3으로 설정되면, 
시도는 총 4번(1번 실패한 실행 + 3번의 재시도)이 실행된 후에, 
최종적으로 실패로 표시됩니다. 기본 값은 0입니다.

`max_consecutive_failed_trials`은 연속해서 실패한 시도가 몇 번 발생해야, 
탐색이 중단될지를 제어합니다. 
예를 들어, 값이 3으로 설정되어 있고, 
시도 2, 3, 4가 모두 실패하면 탐색이 중단됩니다. 
그러나 값이 3으로 설정되어 있고, 시도 2, 3, 5, 6이 실패하더라도, 
연속적으로 실패한 것이 아니기 때문에 탐색이 중단되지 않습니다. 
기본 값은 3입니다.

다음 코드는 이 두 가지 인수가 실제로 어떻게 작동하는지를 보여줍니다.

-   2개의 dense 레이어에서 유닛 수에 대한 2개의 하이퍼파라미터로 탐색 공간을 정의합니다.
-   두 레이어의 유닛 곱이 800을 초과하면, 모델이 너무 크다는 `ValueError`를 발생시킵니다.

```python
def build_model(hp):
    # Dense 레이어의 유닛에 대한 2개의 하이퍼파라미터 정의
    units_1 = hp.Int("units_1", 10, 40, step=10)
    units_2 = hp.Int("units_2", 10, 30, step=10)

    # 모델 정의
    model = keras.Sequential(
        [
            layers.Dense(units=units_1, input_shape=(20,)),
            layers.Dense(units=units_2),
            layers.Dense(units=1),
        ]
    )
    model.compile(loss="mse")

    # 모델이 너무 클 경우 오류 발생
    num_params = model.count_params()
    if num_params > 1200:
        raise ValueError(f"Model too large! It contains {num_params} params.")
    return model
```

튜너를 다음과 같이 설정합니다.

-   `max_retries_per_trial=3`으로 설정합니다.
-   `max_consecutive_failed_trials=8`로 설정합니다.
-   모든 하이퍼파라미터 값 조합을 열거하기 위해, `GridSearch`를 사용합니다.

```python
tuner = keras_tuner.GridSearch(
    hypermodel=build_model,
    objective="val_loss",
    overwrite=True,
    max_retries_per_trial=3,
    max_consecutive_failed_trials=8,
)

# 랜덤 데이터를 사용하여 모델을 트레이닝합니다.
tuner.search(
    x=np.random.rand(100, 20),
    y=np.random.rand(100, 1),
    validation_data=(
        np.random.rand(100, 20),
        np.random.rand(100, 1),
    ),
    epochs=10,
)

# 결과를 출력합니다.
tuner.results_summary()
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Trial 12 Complete [00h 00m 00s]

Best val_loss So Far: 0.12375041842460632
Total elapsed time: 00h 00m 08s
Results summary
Results in ./untitled_project
Showing 10 best trials
Objective(name="val_loss", direction="min")

Trial 0003 summary
Hyperparameters:
units_1: 20
units_2: 10
Score: 0.12375041842460632

Trial 0001 summary
Hyperparameters:
units_1: 10
units_2: 20
Score: 0.12741881608963013

Trial 0002 summary
Hyperparameters:
units_1: 10
units_2: 30
Score: 0.13982832431793213

Trial 0000 summary
Hyperparameters:
units_1: 10
units_2: 10
Score: 0.1433391124010086

Trial 0005 summary
Hyperparameters:
units_1: 20
units_2: 30
Score: 0.14747518301010132

Trial 0006 summary
Hyperparameters:
units_1: 30
units_2: 10
Score: 0.15092280507087708

Trial 0004 summary
Hyperparameters:
units_1: 20
units_2: 20
Score: 0.21962997317314148

Trial 0007 summary
Hyperparameters:
units_1: 30
units_2: 20
Traceback (most recent call last):
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/base_tuner.py", line 273, in _try_run_and_update_trial
    self._run_and_update_trial(trial, *fit_args, **fit_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/base_tuner.py", line 238, in _run_and_update_trial
    results = self.run_trial(trial, *fit_args, **fit_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 314, in run_trial
    obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 232, in _build_and_fit_model
    model = self._try_build(hp)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 164, in _try_build
    model = self._build_hypermodel(hp)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 155, in _build_hypermodel
    model = self.hypermodel.build(hp)
    File "/tmp/ipykernel_21713/966577796.py", line 19, in build_model
    raise ValueError(f"Model too large! It contains {num_params} params.")
ValueError: Model too large! It contains 1271 params.

Trial 0008 summary
Hyperparameters:
units_1: 30
units_2: 30
Traceback (most recent call last):
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/base_tuner.py", line 273, in _try_run_and_update_trial
    self._run_and_update_trial(trial, *fit_args, **fit_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/base_tuner.py", line 238, in _run_and_update_trial
    results = self.run_trial(trial, *fit_args, **fit_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 314, in run_trial
    obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 232, in _build_and_fit_model
    model = self._try_build(hp)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 164, in _try_build
    model = self._build_hypermodel(hp)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 155, in _build_hypermodel
    model = self.hypermodel.build(hp)
    File "/tmp/ipykernel_21713/966577796.py", line 19, in build_model
    raise ValueError(f"Model too large! It contains {num_params} params.")
ValueError: Model too large! It contains 1591 params.

Trial 0009 summary
Hyperparameters:
units_1: 40
units_2: 10
Traceback (most recent call last):
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/base_tuner.py", line 273, in _try_run_and_update_trial
    self._run_and_update_trial(trial, *fit_args, **fit_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/base_tuner.py", line 238, in _run_and_update_trial
    results = self.run_trial(trial, *fit_args, **fit_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 314, in run_trial
    obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 232, in _build_and_fit_model
    model = self._try_build(hp)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 164, in _try_build
    model = self._build_hypermodel(hp)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 155, in _build_hypermodel
    model = self.hypermodel.build(hp)
    File "/tmp/ipykernel_21713/966577796.py", line 19, in build_model
    raise ValueError(f"Model too large! It contains {num_params} params.")
ValueError: Model too large! It contains 1261 params.
```

</details>

------------------------------------------------------------------------

## 실패한 시도로 표시하기
{: #mark-a-trial-as-failed}
<!-- ## Mark a trial as failed -->

모델이 너무 클 때는 재시도할 필요가 없습니다. 
같은 하이퍼파라미터로 몇 번을 시도하더라도, 항상 너무 큰 모델이 될 것입니다.

이를 처리하기 위해 `max_retries_per_trial=0`으로 설정할 수 있지만, 
이 경우 어떤 오류가 발생하더라도 재시도를 하지 않습니다. 
우리는 여전히 예기치 않은 오류에 대해서는 재시도를 원할 수도 있습니다. 
이 상황을 더 잘 처리할 방법이 있을까요?

우리는 `FailedTrialError`를 발생시켜 재시도를 건너뛸 수 있습니다. 
이 오류가 발생하면 해당 시도는 재시도되지 않습니다. 
다른 오류가 발생할 경우에는 여전히 재시도가 실행됩니다. 다음은 그 예시입니다.

```python
def build_model(hp):
    # 두 개의 Dense 레이어에서 사용할 유닛 수에 대한 2개의 하이퍼파라미터 정의
    units_1 = hp.Int("units_1", 10, 40, step=10)
    units_2 = hp.Int("units_2", 10, 30, step=10)

    # 모델 정의
    model = keras.Sequential(
        [
            layers.Dense(units=units_1, input_shape=(20,)),
            layers.Dense(units=units_2),
            layers.Dense(units=1),
        ]
    )
    model.compile(loss="mse")

    # 모델이 너무 클 경우 오류 발생
    num_params = model.count_params()
    if num_params > 1200:
        # 이 오류가 발생하면 재시도를 건너뜁니다.
        raise keras_tuner.errors.FailedTrialError(
            f"모델이 너무 큽니다! {num_params}개의 파라미터를 포함합니다."
        )
    return model

# 튜너 설정
tuner = keras_tuner.GridSearch(
    hypermodel=build_model,
    objective="val_loss",
    overwrite=True,
    max_retries_per_trial=3,
    max_consecutive_failed_trials=8,
)

# 랜덤 데이터를 사용하여 모델을 트레이닝합니다.
tuner.search(
    x=np.random.rand(100, 20),
    y=np.random.rand(100, 1),
    validation_data=(
        np.random.rand(100, 20),
        np.random.rand(100, 1),
    ),
    epochs=10,
)

# 결과를 출력합니다.
tuner.results_summary()
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Trial 12 Complete [00h 00m 00s]

Best val_loss So Far: 0.08265472948551178
Total elapsed time: 00h 00m 05s
Results summary
Results in ./untitled_project
Showing 10 best trials
Objective(name="val_loss", direction="min")

Trial 0002 summary
Hyperparameters:
units_1: 10
units_2: 30
Score: 0.08265472948551178

Trial 0005 summary
Hyperparameters:
units_1: 20
units_2: 30
Score: 0.11731438338756561

Trial 0006 summary
Hyperparameters:
units_1: 30
units_2: 10
Score: 0.13600358366966248

Trial 0004 summary
Hyperparameters:
units_1: 20
units_2: 20
Score: 0.1465979516506195

Trial 0000 summary
Hyperparameters:
units_1: 10
units_2: 10
Score: 0.15967626869678497

Trial 0001 summary
Hyperparameters:
units_1: 10
units_2: 20
Score: 0.1646396517753601

Trial 0003 summary
Hyperparameters:
units_1: 20
units_2: 10
Score: 0.1696309596300125

Trial 0007 summary
Hyperparameters:
units_1: 30
units_2: 20
Traceback (most recent call last):
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/base_tuner.py", line 273, in _try_run_and_update_trial
    self._run_and_update_trial(trial, *fit_args, **fit_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/base_tuner.py", line 238, in _run_and_update_trial
    results = self.run_trial(trial, *fit_args, **fit_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 314, in run_trial
    obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 232, in _build_and_fit_model
    model = self._try_build(hp)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 164, in _try_build
    model = self._build_hypermodel(hp)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 155, in _build_hypermodel
    model = self.hypermodel.build(hp)
    File "/tmp/ipykernel_21713/2463037569.py", line 20, in build_model
    raise keras_tuner.errors.FailedTrialError(
keras_tuner.src.errors.FailedTrialError: Model too large! It contains 1271 params.

Trial 0008 summary
Hyperparameters:
units_1: 30
units_2: 30
Traceback (most recent call last):
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/base_tuner.py", line 273, in _try_run_and_update_trial
    self._run_and_update_trial(trial, *fit_args, **fit_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/base_tuner.py", line 238, in _run_and_update_trial
    results = self.run_trial(trial, *fit_args, **fit_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 314, in run_trial
    obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 232, in _build_and_fit_model
    model = self._try_build(hp)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 164, in _try_build
    model = self._build_hypermodel(hp)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 155, in _build_hypermodel
    model = self.hypermodel.build(hp)
    File "/tmp/ipykernel_21713/2463037569.py", line 20, in build_model
    raise keras_tuner.errors.FailedTrialError(
keras_tuner.src.errors.FailedTrialError: Model too large! It contains 1591 params.

Trial 0009 summary
Hyperparameters:
units_1: 40
units_2: 10
Traceback (most recent call last):
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/base_tuner.py", line 273, in _try_run_and_update_trial
    self._run_and_update_trial(trial, *fit_args, **fit_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/base_tuner.py", line 238, in _run_and_update_trial
    results = self.run_trial(trial, *fit_args, **fit_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 314, in run_trial
    obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 232, in _build_and_fit_model
    model = self._try_build(hp)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 164, in _try_build
    model = self._build_hypermodel(hp)
    File "/home/codespace/.local/lib/python3.10/site-packages/keras_tuner/src/engine/tuner.py", line 155, in _build_hypermodel
    model = self.hypermodel.build(hp)
    File "/tmp/ipykernel_21713/2463037569.py", line 20, in build_model
    raise keras_tuner.errors.FailedTrialError(
keras_tuner.src.errors.FailedTrialError: Model too large! It contains 1261 params.
```

</details>

------------------------------------------------------------------------

## 검색을 프로그래밍 방식으로 중단하기
{: #terminate-the-search-programmatically}
<!-- ## Terminate the search programmatically -->

코드에 버그가 있을 경우 즉시 검색을 중단하고 버그를 수정해야 합니다. 
정의된 조건이 충족되었을 때, 검색을 프로그래밍 방식으로 중단할 수 있습니다. 
`FatalError` (또는 그 하위 클래스인 `FatalValueError`, `FatalTypeError`, `FatalRuntimeError`)를 발생시키면, 
`max_consecutive_failed_trials` 인수와 상관없이 검색이 중단됩니다.

다음은 모델이 너무 클 때 검색을 중단하는 예시입니다.

```python
def build_model(hp):
    # 두 개의 Dense 레이어에서 사용할 유닛 수에 대한 2개의 하이퍼파라미터 정의
    units_1 = hp.Int("units_1", 10, 40, step=10)
    units_2 = hp.Int("units_2", 10, 30, step=10)

    # 모델 정의
    model = keras.Sequential(
        [
            layers.Dense(units=units_1, input_shape=(20,)),
            layers.Dense(units=units_2),
            layers.Dense(units=1),
        ]
    )
    model.compile(loss="mse")

    # 모델이 너무 클 경우 오류 발생
    num_params = model.count_params()
    if num_params > 1200:
        # 이 오류가 발생하면 검색이 중단됩니다.
        raise keras_tuner.errors.FatalError(
            f"모델이 너무 큽니다! {num_params}개의 파라미터를 포함합니다."
        )
    return model


tuner = keras_tuner.GridSearch(
    hypermodel=build_model,
    objective="val_loss",
    overwrite=True,
    max_retries_per_trial=3,
    max_consecutive_failed_trials=8,
)

try:
    # 랜덤 데이터를 사용하여 모델을 트레이닝합니다.
    tuner.search(
        x=np.random.rand(100, 20),
        y=np.random.rand(100, 1),
        validation_data=(
            np.random.rand(100, 20),
            np.random.rand(100, 1),
        ),
        epochs=10,
    )
except keras_tuner.errors.FatalError:
    print("검색이 중단되었습니다.")
```

<details markdown="block">
<summary>결과를 보려면 클릭하세요.</summary>

```
Trial 7 Complete [00h 00m 01s]
val_loss: 0.14219732582569122

Best val_loss So Far: 0.09755773097276688
Total elapsed time: 00h 00m 04s

Search: Running Trial #8

Value             |Best Value So Far |Hyperparameter
30                |10                |units_1
20                |20                |units_2

The search is terminated.
```

</details>

------------------------------------------------------------------------

## 주요 내용 정리
{: #takeaways}
<!-- ## Takeaways -->

이 가이드에서는, KerasTuner에서 실패한 실험을 처리하는 방법을 배웠습니다:

- `max_retries_per_trial`을 사용하여, 실패한 실험에 대한 재시도 횟수를 지정합니다.
- `max_consecutive_failed_trials`을 사용하여, 허용할 수 있는 최대 연속 실패 실험 횟수를 지정합니다.
- `FailedTrialError`를 발생시켜, 실험을 실패로 바로 표시하고, 재시도를 건너뜁니다.
- `FatalError`, `FatalValueError`, `FatalTypeError`, `FatalRuntimeError`를 발생시켜, 검색을 즉시 종료합니다.
