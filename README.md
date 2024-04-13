# Keras 문서 페이지

## Keras

Keras 한글 문서 페이지는 GitHub Page에서 볼 수 있습니다.

* [GitHub Page에서 보기](https://codecompose7.github.io/keras-doc-kr.github.io/)


## 주의사항

### 링크

링크는 다음과 같이 기록해야 합니다.

```markdown
Keras의 핵심 데이터 구조는 **레이어(layers)**와 **모델(models)**입니다. 가장 간단한 모델 유형은 선형 레이어 스택인 [`Sequential` 모델]({% link docs/04-guides/02-sequential_model.md %})입니다.
```

{% link docs/04-guides/02-sequential_model.md %} 파일 명을 사용하세요.

`permalink`를 사용할 수도 있습니다. 하지만 이 경우, `{{ site.baseurl }}/` 이 부분을 앞에 붙여주세요.

```markdown
가장 간단한 모델 유형은 선형 레이어 스택인 [`Sequential` 모델]({{ site.baseurl }}/guides/sequential_model/)입니다.
```

### permalink 링크

```markdown
---
layout: default
title: Keras 3 API 문서
nav_order: 5
permalink: /api/
has_children: true
---
```

이 부분의 `permalink:` 끝은 항상 `/`로 끝나야 합니다. 이에 대한 정규식 검색어는 다음과 같습니다.

```
^permalink: [^\s]+[^\/\s]$
```

### 이탤릭체

markdown에서 이탤릭체는 `_italic_` 대신 `*italic*`을 사용하세요.

### 코드 블록

markdown에서 코드 블록은 ` ``` ` 로 사용하세요. 다만, 사용 언어를 붙여주세요.

` ```python ` 이나 ` ```latex `, ` ```shell ` 등으로 사용합니다.

아무 것도 붙지 않은 ` ``` ` 은 결과 항목으로 사용됩니다.

### 참조 사이트

* [Keras에 대해](https://keras.io/about/)
* [DeepL](https://www.deepl.com/)
* [HTML to Markdown](https://codebeautify.org/html-to-markdown)
* [Online latex editor](https://latexeditor.lagrida.com/)
* [Bulk Image Downloader From Url List](https://chromewebstore.google.com/detail/kekkjfalendfifgchoceceoahjnejmgf?hl=en-GB)

## Jekyll 생성

### 로컬 미리보기

로컬 호스팅을 위한 jekyll 실행 방법입니다.

```shell
bundle exec jekyll serve --baseurl ''
```

### 페이지 참조

* [https://github.com/just-the-docs/just-the-docs](https://github.com/just-the-docs/just-the-docs)
* [https://just-the-docs.com/](https://just-the-docs.com/)
