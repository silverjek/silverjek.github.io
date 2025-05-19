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

오늘은 제가 진행 중인 졸업 연구에서 참고한 논문인 **AST: Audio Spectrogram Transformer** (Gong et al., 2021)을 소개하고, 허깅페이스에서 모델과 라이브러리를 가져와 직접 오디오 피처 추출과 오디오 라벨 예측을 하는 방법을 소개하려고 합니다.

현재 저는 비디오 기반 행동 인식 도메인에서 **오디오 modality가 누락**인 상황에 모델이 **성능을 robust하게 유지**할 수 있는 방법론을 연구 중입니다.
특히 **오디오 modality의 결손을 대체**하거나 **복원**할 수 있는 **피처 생성 기반 접근**을 실험하고 있으며, 이를 위해 고품질의 오디오 표현 추출이 중요합니다.

연구를 진행하며 AST를 이해하고 사용한 내용을 공유 드리고자 합니다.

논문은 [**여기**](https://arxiv.org/pdf/2104.01778)를 눌러 이동해주세요.

# **AST: Audio Spectrogram Transformer**

**AST (Audio Spectrogram Transformer)**는 기존의 CNN 기반 오디오 분류 모델들과 달리, 순수하게 **self-attention 기반**으로만 구성된 **최초의 오디오 분류 모델**입니다.
이 모델은 **Vision Transformer (ViT)** 구조를 오디오 spectrogram에 적용한 것으로, convolution 없이도 **전역적인 문맥 정보**를 효과적으로 학습할 수 있다는 것이 주요 특징입니다.

기존의 **CNN**을 기반으로 하는 오디오 분류 모델은 **local feature를 추출**하는 데는 강하지만, **long-range context**는 잘 다루지 못합니다. 이후 **CNN + Attention의 hybrid 구조**가 연구되었지만, 여전히 CNN의 **inductive bias**에 의한 한계가 존재했습니다.

이에 따라 주어진 논문은 ViT처럼 **순수한 attention-only 모델**이 오디오에서도 잘 작동하는가? 라는 질문에 답하기 위헤 **AST**를 제안하였습니다.

## **Architecture**
AST는 오디오를 spectrogram으로 변환하고, 이를 이미지처럼 취급해 처리하는 모델입니다. 핵심 구조는 다음과 같습니다.

![AST](/_posts/20250519/AST.png){: width="972" height="589" .w-75 .normal}

1.	**Log-Mel Spectrogram** 생성
	•	입력 waveform을 128-dimensional log-Mel spectrogram으로 변환
	•	오디오 길이에 따라 [T, 128] 형태의 2D 시계열 이미지가 생성됨
2.	**Patch Embedding**
	•	Spectrogram을 **patch 단위 (e.g. 16×16)**로 나누어 **각 패치를 flatten + linear projection → 768차원 임베딩**
	•	이 과정은 ViT에서의 image patch embedding과 동일
3.	**[CLS] Token** 삽입 + **Positional Encoding** 추가
	•	전체 시퀀스를 요약하기 위한 [CLS] token을 맨 앞에 추가
	•	각 patch 위치를 인식할 수 있도록 1D positional embedding 덧붙임
4.	**Transformer Encoder** (Multi-head Self-Attention)
	•	전통적인 ViT와 동일한 구조의 Transformer encoder 사용
	•	CNN 없이 **전역적인 시퀀스 간 관계(context)**를 직접 학습
5.	**[CLS] Token** 추출 (또는 전체 시퀀스 활용)
	•	최종 출력 중 [CLS] 벡터를 오디오 전체의 표현으로 사용하거나 필요에 따라 patch-level 시퀀스 출력 전체를 사용할 수도 있음