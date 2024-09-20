---
layout: default
title: KerasNLP - 자연어 워크플로우
nav_order: 9
permalink: /keras_nlp/
has_children: false
---

* 원본 링크 : [https://keras.io/keras_nlp/](https://keras.io/keras_nlp/){:target="_blank"}
* 최종 수정일 : 2024-09-21

# KerasNLP
{: .no_toc }

## 목차
{: .no_toc .text-delta }

1. TOC
{:toc}

---

# KerasNLP
{: #kerasnlp}
<!-- # KerasNLP -->

KerasNLP is a natural language processing library that works natively with TensorFlow, JAX, or PyTorch. Built on Keras 3, these models, layers, metrics, and tokenizers can be trained and serialized in any framework and re-used in another without costly migrations.

KerasNLP supports users through their entire development cycle. Our workflows are built from modular components that have state-of-the-art preset weights when used out-of-the-box and are easily customizable when more control is needed.

This library is an extension of the core Keras API; all high-level modules are [`Layers`]({{ site.baseurl }}/api/layers/) or [`Models`]({{ site.baseurl }}/api/models/) that receive that same level of polish as core Keras. If you are familiar with Keras, congratulations! You already understand most of KerasNLP.

See our [Getting Started guide]({{ site.baseurl }}/guides/keras_nlp/getting_started) to start learning our API. We welcome [contributions](https://github.com/keras-team/keras-nlp/blob/master/CONTRIBUTING.md).

------------------------------------------------------------------------

## Quick links
{: #quick-links}
<!-- ## Quick links -->

-   [KerasNLP API reference]({{ site.baseurl }}/api/keras_nlp/)
-   [KerasNLP on GitHub](https://github.com/keras-team/keras-nlp)
-   [List of available pre-trained models]({{ site.baseurl }}/api/keras_nlp/models/)

## Guides
{: #guides}
<!-- ## Guides -->

-   [Getting Started with KerasNLP]({{ site.baseurl }}/guides/keras_nlp/getting_started/)
-   [Uploading Models with KerasNLP]({{ site.baseurl }}/guides/keras_nlp/upload/)
-   [Pretraining a Transformer from scratch]({{ site.baseurl }}/guides/keras_nlp/transformer_pretraining/)

## Examples
{: #examples}
<!-- ## Examples -->

-   [GPT-2 text generation]({{ site.baseurl }}/examples/generative/gpt2_text_generation_with_kerasnlp/)
-   [Parameter-efficient fine-tuning of GPT-2 with LoRA]({{ site.baseurl }}/examples/nlp/parameter_efficient_finetuning_of_gpt2_with_lora/)
-   [Semantic Similarity]({{ site.baseurl }}/examples/nlp/semantic_similarity_with_keras_nlp/)
-   [Sentence embeddings using Siamese RoBERTa-networks]({{ site.baseurl }}/examples/nlp/sentence_embeddings_with_sbert/)
-   [Data Parallel Training with tf.distribute]({{ site.baseurl }}/examples/nlp/data_parallel_training_with_keras_nlp/)
-   [English-to-Spanish translation]({{ site.baseurl }}/examples/nlp/neural_machine_translation_with_keras_nlp/)
-   [GPT text generation from scratch]({{ site.baseurl }}/examples/generative/text_generation_gpt/)
-   [Text Classification using FNet]({{ site.baseurl }}/examples/nlp/fnet_classification_with_keras_nlp/)

------------------------------------------------------------------------

## Installation
{: #installation}
<!-- ## Installation -->

KerasNLP supports both Keras 2 and Keras 3. We recommend Keras 3 for all new users, as it enables using KerasNLP models and layers with JAX, TensorFlow and PyTorch.

### Keras 2 Installation
{: #keras-2-installation}
<!-- ### Keras 2 Installation -->

To install the latest KerasNLP release with Keras 2, simply run:

```shell
pip install --upgrade keras-nlp
```

### Keras 3 Installation
{: #keras-3-installation}
<!-- ### Keras 3 Installation -->

There are currently two ways to install Keras 3 with KerasNLP. To install the stable versions of KerasNLP and Keras 3, you should install Keras 3 **after** installing KerasNLP. This is a temporary step while TensorFlow is pinned to Keras 2, and will no longer be necessary after TensorFlow 2.16.

```shell
pip install --upgrade keras-nlp
pip install --upgrade keras
```

To install the latest nightly changes for both KerasNLP and Keras, you can use our nightly package.

```shell
pip install --upgrade keras-nlp-nightly
```

**Note:** Keras 3 will not function with TensorFlow 2.14 or earlier.

See [Getting started with Keras]({{ site.baseurl }}/getting_started/) for more information on installing Keras generally and compatibility with different frameworks.

------------------------------------------------------------------------

## Quickstart
{: #quickstart}
<!-- ## Quickstart -->

Fine-tune BERT on a small sentiment analysis task using the [`keras_nlp.models`]({{ site.baseurl }}/api/keras_nlp/models/) API:

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

import keras_nlp
import tensorflow_datasets as tfds

imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16,
)
# Load a BERT model.
classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_base_en_uncased", 
    num_classes=2,
)
# Fine-tune on IMDb movie reviews.
classifier.fit(imdb_train, validation_data=imdb_test)
# Predict two new examples.
classifier.predict(["What an amazing movie!", "A total waste of my time."])
```

------------------------------------------------------------------------

## Compatibility
{: #compatibility}
<!-- ## Compatibility -->

We follow [Semantic Versioning](https://semver.org/), and plan to provide backwards compatibility guarantees both for code and saved models built with our components. While we continue with pre-release `0.y.z` development, we may break compatibility at any time and APIs should not be consider stable.

## Disclaimer
{: #disclaimer}
<!-- ## Disclaimer -->

KerasNLP provides access to pre-trained models via the `keras_nlp.models` API. These pre-trained models are provided on an "as is" basis, without warranties or conditions of any kind. The following underlying models are provided by third parties, and subject to separate licenses: BART, DeBERTa, DistilBERT, GPT-2, OPT, RoBERTa, Whisper, and XLM-RoBERTa.

## Citing KerasNLP
{: #citing-kerasnlp}
<!-- ## Citing KerasNLP -->

If KerasNLP helps your research, we appreciate your citations. Here is the BibTeX entry:

```latex
@misc{kerasnlp2022,
    title={KerasNLP},
    author={Watson, Matthew, and Qian, Chen, and Bischof, Jonathan and Chollet, 
    Fran\c{c}ois and others},
    year={2022},
    howpublished={\url{https://github.com/keras-team/keras-nlp}},
}
```
