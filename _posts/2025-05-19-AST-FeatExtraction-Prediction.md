---
title: AST를 사용한 오디오 피처 추출 및 오디오 라벨 예측
description: AST 논문 이해와 허깅페이스 라이브러리를 사용한 오디오 피처 추출과 라벨 예측하기
date: 2024-05-19 20:00:00 +0900
categories: [Capstone Research, Realted Work]
tags: [AST, Audio, Label Prediction]
pin: true
math: true
---

안녕하세요 :)

오늘은 제가 진행 중인 졸업 연구에서 참고한 논문인 **AST: Audio Spectrogram Transformer** (Gong et al., 2021)을 소개하고,  
허깅페이스에서 모델과 라이브러리를 가져와 직접 오디오 피처 추출과 오디오 라벨 예측을 하는 방법을 소개하려고 합니다.

현재 저는 비디오 기반 행동 인식 도메인에서 **오디오 modality가 누락**인 상황에 모델이 **성능을 robust하게 유지**할 수 있는 방법론을 연구 중입니다.  
특히 **오디오 modality의 결손을 대체**하거나 **복원**할 수 있는 **피처 생성 기반 접근**을 실험하고 있으며, 이를 위해 고품질의 오디오 표현 추출이 중요합니다.

연구를 진행하며 AST를 이해하고 사용한 내용을 공유 드리고자 합니다.

논문은 [**여기**](https://arxiv.org/pdf/2104.01778)를 눌러 이동해주세요.

---

# **AST: Audio Spectrogram Transformer**

**AST (Audio Spectrogram Transformer)**는 기존의 CNN 기반 오디오 분류 모델들과 달리,  
순수하게 **self-attention 기반**으로만 구성된 **최초의 오디오 분류 모델**입니다.  
이 모델은 **Vision Transformer (ViT)** 구조를 오디오 spectrogram에 적용한 것으로,  
convolution 없이도 **전역적인 문맥 정보**를 효과적으로 학습할 수 있다는 것이 주요 특징입니다.

기존의 **CNN**을 기반으로 하는 오디오 분류 모델은 **local feature를 추출**하는 데는 강하지만,  
**long-range context**는 잘 다루지 못합니다.  
이후 **CNN + Attention의 hybrid 구조**가 연구되었지만,  
여전히 CNN의 **inductive bias**에 의한 한계가 존재했습니다.

이에 따라 주어진 논문은 ViT처럼 **순수한 attention-only 모델이 오디오에서도 잘 작동하는가?**  
라는 질문에 답하기 위해 **AST**를 제안하였습니다.

---

## **Architecture**

AST는 오디오를 spectrogram으로 변환하고, 이를 이미지처럼 취급해 처리하는 모델입니다.  
핵심 구조는 다음과 같습니다.

![AST Architecture](https://www.researchgate.net/publication/355232609/figure/fig1/AS:1079151175962625@1634301098423/AST-architecture-overview-taken-from-10-The-input-spectrogram-is-rearranged-into-a.ppm)

1. **Log-Mel Spectrogram 생성**  
   - 입력 waveform을 128-dimensional log-Mel spectrogram으로 변환  
   - 오디오 길이에 따라 `[T, 128]` 형태의 2D 시계열 이미지가 생성됨

2. **Patch Embedding**  
   - Spectrogram을 **patch 단위 (e.g. 16×16)**로 나누어  
     각 패치를 **flatten + linear projection → 768차원 임베딩**  
   - 이 과정은 ViT에서의 image patch embedding과 동일

3. **[CLS] Token 삽입 + Positional Encoding 추가**  
   - 전체 시퀀스를 요약하기 위한 `[CLS]` token을 맨 앞에 추가  
   - 각 patch 위치를 인식할 수 있도록 **1D positional embedding** 덧붙임

4. **Transformer Encoder (Multi-head Self-Attention)**  
   - 전통적인 ViT와 동일한 구조의 Transformer encoder 사용  
   - CNN 없이 **전역적인 시퀀스 간 관계(context)**를 직접 학습

5. **[CLS] Token 추출 (또는 전체 시퀀스 활용)**  
   - 최종 출력 중 `[CLS]` 벡터를 **오디오 전체의 표현**으로 사용하거나  
     필요에 따라 patch-level 시퀀스 출력 전체를 사용할 수도 있음

---

## **Hugging Face의 AST 모델 사용하기**

**Hugging Face**에서는 **사전학습 된 AST 모델**을 제공합니다.  
관련 문서는 [**여기**](https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)를 눌러 이동해주세요.

모델을 불러오는 것부터 시작하겠습니다.

불러올 모델은 AudioSet 데이터셋으로 fine-tuned된 AST입니다.  
이 모델은 **모델 라벨 분류**를 위해 학습되었으나,  
Transformer의 아웃풋 **[CLS] 임베딩**을 **오디오 피처**로 활용 가능합니다.

```python
pretrained_model = "MIT/ast-finetuned-audioset-10-10-0.4593"
```

모델을 불러와 사용할 주요 라이브러리는 다음과 같습니다:

```python
from transformers import ASTFeatureExtractor, ASTModel
```

- **ASTFeatureExtractor**: raw waveform → log-mel spectrogram + normalization + padding  
  이 라이브러리는 오디오의 raw waveform을 log-mel spectrogram으로 변환하며,  
  정규화와 패딩 처리를 자동으로 수행합니다. 이 과정을 통해 `ASTModel`의 input을 준비하게 됩니다.

- **ASTModel**: `ASTFeatureExtractor`로 처리된 오디오 피처를 input으로 받아  
  Transformer를 통해 임베딩을 추출합니다.

---

## **오디오 피처 추출 및 라벨 예측 코드**

1. 먼저, **오디오 파일**(.wav)을 **mono 채널**로 불러와 **16kHz**로 **샘플링**합니다.

```python
import soundfile as sf
waveform, sr = sf.read(path, dtype='float32', always_2d=True)
waveform = torch.from_numpy(waveform[:, 0])  # mono 채널 사용
```

2. 다음으로 Hugging Face의 **ASTFeatureExtractor**를 사용해 오디오를  
   **log-Mel Spectrogram**으로 변환하고, **normalization**과 **padding**을 자동 처리합니다.

```python
from transformers import ASTFeatureExtractor

feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

inputs = feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")
```

3. 이렇게 처리한 log-Mel Spectrogram은 Transformer의 input으로 사용되고,  
   output으로는 `[CLS]` 임베딩이 추출됩니다.

```python
from transformers import ASTModel

model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model.eval()

with torch.no_grad():
    outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
```

4. Transformer의 output 피처의 차원을 축소해 label prediction에 사용 가능하도록 준비합니다.

```python
projector = torch.nn.Linear(768, 128)
feature = projector(cls_embedding).squeeze(0).cpu().numpy()
```

---

## **오디오 라벨 예측 (Classification)**

5. 준비된 피처를 input으로 `ASTForAudioClassification`을 사용하면  
   **오디오 라벨 예측**을 수행할 수 있습니다.  
   불러온 사전 학습 모델에 사용된 데이터셋은 **AudioSet**으로,  
   **527개의 클래스**가 존재하며 예측 결과는 **multi-label classification**입니다.

```python
from transformers import ASTForAudioClassification

clf_model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
clf_model.eval()

with torch.no_grad():
    outputs = clf_model(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits)

topk = torch.topk(probs, k=5)
print("Top 5 label indices:", topk.indices)
print("Top 5 probabilities:", topk.values)
```

이렇게 코드를 실습하면 **multi-label prediction 결과**를 확인할 수 있습니다.

이는 오디오 피처가 **어떤 의미적 정보를 담고 있는지 해석**하는 데 유용하며,  
현재 진행 중인 연구 맥락에서 이 예측 라벨은  
**오디오 생성**, **비디오-오디오 정합성 평가**,  
**의미 기반 분류기 학습** 등 다양한 downstream task에 활용될 수 있습니다.

---

감사합니다 :)