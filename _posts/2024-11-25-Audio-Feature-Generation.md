---
title: Audio Feature Generation
description: 선행 논문을 기반으로 한 LSTM-based 오디오 피쳐 생성하기
date: 2024-11-22 09:00:00 +0900
categories: [Capstone Research, Pilot Study]
tags: [Machine Learning, LSTM, CNN]
pin: true
math: true
---

이 글은 이화여자대학교 졸업 프로젝트 연구 트랙 진행 상황 아카이빙을 위한 기록입니다.

현재 저희 팀은 multimodal 데이터를 활용한 action recognition을 도메인으로 연구를 진행 중 입니다. 특히 특정 모달리티가 누락되었을 경우 행동 인식의 성능이 저하 될 수 있음을 주요 문제 상황으로 정의하고 데이터를 생성, 모델의 성능을 개선하는 방법을 탐구 중에 있습니다.

현재 연구의 시작 단계로 기반이 되는 선행 논문, Audio Feature Generation for Missing Modality Problem in Video Action Recognition (Lee et al., 2019)[^footnote]의 실험을 재현 중에 있습니다. 재현을 통해 데이터 처리와 모델에 대해 이해하고 저희 연구의 novelty를 위한 기술을 적용하기 위한 기반을 마련했습니다.

선행 논문에서  Moments in Time(하이퍼링크) 데이터 셋을 사용했습니다. 해당 데이터 셋은 행동 인식을 위한 3초의 짧은 비디오를 action class로 라벨링 했으며 people/animal action, objects, natural phenomena를 다양하게 포함하고 있습니다.

본 포스팅에서는 OpenCV를 사용한 비디오 데이터 전처리, PyTorch를 사용한 특징 추출, 그리고 논문에서 제안한 LSTM-based sound feature generator의 first branch인 classification branch를 구현하는 방법을 다루겠습니다.

# Data Processing (데이터 전처리) & Feature Extraction (특징 추출)

먼저, raw data인 비디오를 처리하고 feature를 추출하는 플로우에 대한 이해가 필요합니다. 현재 재현 중인 논문의 2.1. LSTM-based sound feature generator 에서는 feature extractor를 다음과 같은 구조로 제안합니다.

$$ v_R = \text{cnn}_{\phi_r}(I_R), \quad v_F = \text{cnn}_{\phi_f}(I_F) $$

$I_R$: 비디오를 프레임 단위로 쪼갠 RGB images의 집합
$I_F$: RGB images 사이의 상대적 움직임을 포착한 이미지의 집합 
$v_R$: I_R을 CNN을 통해 추출한 특징 벡터의 집합
$v_F$: I_F를 CNN을 통해 추출한 특징 벡터의 집합

논문의 3.2 Implementation Details를 참고하면 pre-trained TSN을 사용했음을 알 수 있지만, 추후 연구 기술 적용을 위해 간단하게 모델을 재현 중이기 때문에 본 포스팅에서는 TSN의 기반인 CNN을 사용할 것 입니다.

## Data Processing

먼저 raw data인 비디오를 $I_R$과 $I_F$로 처리하기 위해서 OpenCV를 사용했습니다. 
<div align="center">
  <img src="https://learning.rc.virginia.edu/courses/opencv/opencv_logo.png" alt="OpenCV Logo" style="width:50%;"/>
</div>


```python
#RGB, optical flow 뽑는 코드 넣기
```

해당 코드는 OpenCV의 ~~~를 사용해 6fps(초당 6프레임)으로 비디오를 RGB images로 저장하고 그로부터 프레임 간 Optical Flow images를 저장하는 과정입니다. 또한 처리 속도 개선을 위해 multiprocessing을 사용했습니다.

## Feature Extraction

이제 각 영상마다 $I_R$과 $I_F$가 준비 되었으니 CNN을 사용해 피쳐 벡터로 변환해야 합니다. Backbone은 PyTorch의 ResNet50을 사용했으며, 해당 모델은 각 이미지에 대한 feature를 [1, 2048]로 반환합니다.


```python
#cnn 피쳐 뽑는 코드 넣기
```

해당 코드는 - f는 각 영상의 추출된 RGB images 개수라고 할 때에 - 모든 raw video에 대해 $I_R$ 은 [f, 2048] 형태의 $v_R$, $I_F$은 [f-1, 2048] 형태의 $v_R$로 반환합니다. 그리고 각 벡터를 .npy 파일로 저장했습니다.
또한 논문에서는 $v_R$과 $v_F$를 concat하고 있어 다음과 같은 코드로 spatial feature와 temporal feature를 결합한 각 영상에 대한 $v_C$를 .npy 파일로 저장해 LSTM 모델에의 피쳐 사용을 위한 준비를 마쳤습니다.

[^footnote]: Lee, H.-C., Lin, C.-Y., Hsu, P.-C., & Hsu, W. H. (2019). Audio feature generation for missing modality problem in video action recognition. ICASSP 2019 - IEEE International Conference on Acoustics, Speech and Signal Processing, 3956–3960. https://ieeexplore.ieee.org/document/8682513?utm_source=chatgpt.com