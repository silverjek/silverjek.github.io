---
title: Audio Feature Generation
description: LSTM-based sound feature generator - classification branch 구현하기
date: 2024-11-22 09:00:00 +0900
categories: [Capstone Research, Pilot Study]
tags: [Machine Learning, LSTM, CNN]
pin: true
math: true
---

이 글은 이화여자대학교 졸업 프로젝트 연구 트랙 진행 상황 아카이빙을 위한 기록입니다.

현재 저희 팀은 **multimodal 데이터를 활용한 action recognition**을 도메인으로 연구를 진행 중 입니다. 특히 **특정 모달리티가 누락**되었을 경우 **행동 인식의 성능이 저하** 될 수 있음을 주요 문제 상황으로 정의하고 **누락된 데이터를 생성**, **모델의 성능을 개선**하는 방법을 탐구 중에 있습니다.

현재 연구의 시작 단계로 기반이 되는 선행 논문, Audio Feature Generation for Missing Modality Problem in Video Action Recognition (Lee et al., 2019)[^footnote]의 실험을 재현 중에 있습니다. 재현을 통해 데이터 처리와 모델에 대해 이해하고 저희 연구의 novelty를 위한 기술을 적용하기 위한 기반을 마련했습니다.

선행 논문에서  [**Moments in Time**](http://moments.csail.mit.edu/) 데이터 셋을 사용했습니다. 해당 데이터 셋은 행동 인식을 위해 3초의 짧은 비디오를 304개의 action class로 라벨링 했으며 people&animal action, objects, 그리고 natural phenomena를 다양하게 포함하고 있습니다.
데이터 셋 용량 및 초기 실험의 편의/속도를 위해 10개의 클래스에서 각 1,000개의 영상을 랜덤으로 추출해 총 10,000개의 데이터로 실험을 진행했습니다.

본 포스팅에서는 OpenCV를 사용한 비디오 데이터 전처리, PyTorch를 사용한 특징 추출, 그리고 논문에서 제안한 LSTM-based sound feature generator의 first branch인 **classification branch**를 구현하는 방법을 다루겠습니다.

# Data Processing (데이터 전처리) <br> & Feature Extraction (특징 추출)

먼저, raw data인 비디오를 처리하고 feature를 추출하는 플로우에 대한 이해가 필요합니다. 현재 재현 중인 논문의 2.1. LSTM-based sound feature generator 에서는 **feature extractor**를 다음과 같은 구조로 제안합니다.

$$ v_R = \text{cnn}_{\phi_r}(I_R), \quad v_F = \text{cnn}_{\phi_f}(I_F) $$

$I_R$: 비디오를 프레임 단위로 쪼갠 RGB images의 집합

$I_F$: RGB images 사이의 상대적 움직임을 포착한 이미지의 집합 

$v_R$: I_R을 CNN을 통해 추출한 특징 벡터의 집합, **spatial feature**

$v_F$: I_F를 CNN을 통해 추출한 특징 벡터의 집합, **temporal feature**

논문의 3.2 Implementation Details를 참고하면 pre-trained TSN을 사용했음을 알 수 있지만, 추후 연구 기술 적용을 위해 간단하게 모델을 재현 중이기 때문에 본 포스팅에서는 TSN의 기반인 CNN을 사용할 것 입니다.

## Data Processing

먼저 raw data인 비디오를 $I_R$과 $I_F$로 처리하기 위해서 OpenCV를 사용했습니다. 또한 데이터 처리 과정에 NumPy를 사용했습니다.

```python
import os
import cv2
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor

def extract_frames(video_path, output_dir, target_fps=6):
    """Extract frames from a video and save as images at the specified FPS."""
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / target_fps)
    frame_count = 0
    saved_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames to {output_dir}")

def calculate_optical_flow(frames_dir, output_dir):
    """Calculate optical flow from frames and save as images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if len(frame_files) < 2:
        print(f"Not enough frames in {frames_dir} for optical flow calculation.")
        return

    prev_frame = cv2.imread(frame_files[0], cv2.IMREAD_GRAYSCALE)

    for i in range(1, len(frame_files)):
        curr_frame = cv2.imread(frame_files[i], cv2.IMREAD_GRAYSCALE)
        
        # Ensure frames are properly loaded
        if prev_frame is None or curr_frame is None:
            print(f"Error reading frames in {frames_dir}. Skipping frame {i}.")
            continue

        # Optical Flow calculation
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = np.uint8(angle * 180 / np.pi / 2)  # Hue: direction
        hsv[..., 1] = 255  # Saturation: constant
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value: magnitude

        # Convert HSV to BGR for saving
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        output_path = os.path.join(output_dir, f"flow_{i:04d}.jpg")
        cv2.imwrite(output_path, flow_bgr)

        prev_frame = curr_frame

    print(f"Optical flow saved in {output_dir}")

def process_video(video_path, rgb_dir, flow_dir, class_name, target_fps=6):
    """Process a single video for frame extraction and optical flow calculation."""
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    class_rgb_dir = os.path.join(rgb_dir, class_name)
    class_flow_dir = os.path.join(flow_dir, class_name)
    frames_dir = os.path.join(class_rgb_dir, video_id)
    optical_flow_dir = os.path.join(class_flow_dir, video_id)

    extract_frames(video_path, frames_dir, target_fps=target_fps)
    calculate_optical_flow(frames_dir, optical_flow_dir)

def process_class_videos(class_path, class_name, rgb_dir, flow_dir, target_fps=6, max_videos=1000):
    video_files = [f for f in os.listdir(class_path) if f.endswith((".mp4", ".avi", ".mkv"))]
    if len(video_files) > max_videos:
        video_files = random.sample(video_files, max_videos)  # Randomly select 1000 videos

    for video_file in video_files:
        video_path = os.path.join(class_path, video_file)
        try:
            process_video(video_path, rgb_dir, flow_dir, class_name, target_fps=target_fps)
        except Exception as e:
            print(f"Error processing video {video_file}: {e}")

if __name__ == "__main__":
    raw_data_dir = "Audio-Feature-Generation/data/raw/training"
    rgb_dir = "Audio-Feature-Generation/data/processed/rgb"
    flow_dir = "Audio-Feature-Generation/data/processed/flow"

    max_videos_per_class = 1000
    target_fps = 6

    with ThreadPoolExecutor() as executor:
        futures = []
        for class_dir in os.listdir(raw_data_dir):
            class_path = os.path.join(raw_data_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            futures.append(executor.submit(process_class_videos, class_path, class_dir, rgb_dir, flow_dir, target_fps, max_videos_per_class))

        for future in futures:
            future.result()

    print("All videos processed.")
```

해당 코드는 OpenCV의 ~~~를 사용해 6fps(초당 6프레임)으로 비디오를 RGB images로 저장하고 그로부터 프레임 간 Optical Flow images를 저장하는 과정입니다.
논문에 Optical Flow 처리 방식에 대해 명확히 언급되어 있지 않아 Farneback 방식으로 처리했습니다. 또한 처리 속도 개선을 위해 multiprocessing을 사용했습니다.

## Feature Extraction

이제 각 영상마다 $I_R$과 $I_F$가 준비 되었으니 CNN을 사용해 피쳐 벡터로 변환해야 합니다. Backbone은 PyTorch의 ResNet50을 사용했으며, 해당 모델은 각 이미지에 대한 feature를 [1, 2048]로 반환합니다.
즉, 각 영상의 프레임 이미지가 t개 존재한다면 해당 영상의 feature vector는 [t, 2048] 형태입니다.

```python
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import cv2
from preprocess import process_video

def load_images(image_dir):
    """Load images from a directory and preprocess them for CNN."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images = []
    filenames = sorted(os.listdir(image_dir))
    for filename in filenames:
        img_path = os.path.join(image_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = transform(img)
        images.append(img_tensor)

    return torch.stack(images)

def extract_features(model, images):
    """Extract feature vectors from images using a CNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    images = images.to(device)

    with torch.no_grad():
        features = model(images)

    return features.cpu().numpy()

def process_features(frames_dir, optical_flow_dir, cnn_model):
    """Extract and concatenate features for RGB and Optical Flow."""
    rgb_images = load_images(frames_dir)
    optical_flow_images = load_images(optical_flow_dir)

    rgb_features = extract_features(cnn_model, rgb_images)
    optical_flow_features = extract_features(cnn_model, optical_flow_images)

    min_length = min(rgb_features.shape[0], optical_flow_features.shape[0])
    combined_features = np.concatenate((rgb_features[:min_length], optical_flow_features[:min_length]), axis=-1)

    return combined_features

def save_combined_features(output_dir, video_id, combined_features):
    """Save combined features to a .npy file."""
    output_path = os.path.join(output_dir, f"{video_id}_combined.npy")
    np.save(output_path, combined_features)
    print(f"Combined features saved: {output_path}")

# Example usage in a main script
if __name__ == "__main__":
    raw_data_dir = "Audio-Feature-Generation/data/raw/trainig"
    output_dir = "Audio-Feature-Generation/data/features"
    temp_dir = "Audio-Feature-Generation/data/processed"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    cnn_model = resnet50(pretrained=True)
    cnn_model = torch.nn.Sequential(*list(cnn_model.children())[:-2], torch.nn.AdaptiveAvgPool2d((1, 1)))
    cnn_model.eval()

    for class_dir in os.listdir(raw_data_dir):
        class_path = os.path.join(raw_data_dir, class_dir)
        if not os.path.isdir(class_path):
            continue

        for video_file in os.listdir(class_path):
            video_path = os.path.join(class_path, video_file)
            if not video_file.endswith((".mp4", ".avi", ".mkv")):
                continue

            video_id = os.path.splitext(video_file)[0]

            try:
                frames_dir, optical_flow_dir = process_video(video_path, temp_dir)
                combined_features = process_features(frames_dir, optical_flow_dir, cnn_model)
                save_combined_features(output_dir, video_id, combined_features)
            except Exception as e:
                print(f"Error processing video {video_id}: {e}")

    print("All videos processed.")
```

해당 코드는 모든 raw video에 대해 - 각 video에서 추출된 프레임의 개수가 t라고 할 때에 - $I_R$ 은 [t, 2048] 형태의 $v_R$, $I_F$은 [t-1, 2048] 형태의 $v_F$로 반환합니다.
또한 모델에의 활용을 위해 spatial feature $v_R$과 temporal feature $v_F$를 **concatenate** 해 모든 영상에 대한 $v_C$를 .npy 파일로 저장했고, 이로써 LSTM 모델에의 피쳐 사용을 위한 기초 준비를 마쳤습니다.
그리고, spatial feature와 temporal feature를 concat 시 데이터 형태는 **[t-1, 4096]**입니다 - RGB images는 t장이지만 두 프레임 사이의 상대적 움직임을 계산한 Optical Flow images는 t-1장이기 때문입니다. RGB images의 첫 장을 드롭해 이미지 개수를 t-1개로 맞췄습니다.

모든 영상에 대해 $v_C$가 준비되었다면 LSTM 모델이 학습 시에 기대하는 데이터 형태와 일치하는지 확인이 필요합니다.
이전 단계에서 준비된 피처 벡터의 형태는 [t-1, 4096]이며, 이는 각 영상의 시퀀스 길이(t−1)와 feature 차원(4096)을 포함한 **2차원 텐서**입니다.

그러나 LSTM은 **3차원 텐서**를 input으로 학습하는 모델로, LSTM이 기대하는 입력 형태는 다음과 같습니다: **[batch, sequence length, feature dimension]**

batch: 한 번에 모델에 공급되는 시퀀스(비디오)의 개수
sequence length: 각 비디오의 프레임 시퀀스 길이 (t−1)
feature dimension: 각 프레임 또는 시퀀스의 feature 크기 (4096)

따라서, 모든 $v_C$ 데이터를 LSTM 모델에 공급하기 위해서는, 각 비디오의 피처 벡터를 3차원 텐서로 변환해야 합니다. 
이를 위해 $v_C$를 batch 단위로 묶어 [batch, t-1, 4096] 형태로 변환해야 하며, 본 포스팅에서는 실험 중인 환경의 GPU 메모리가 12GB임에 따라 Batch Size를 64로 설정했습니다.

<!-- markdownlint-capture -->
<!-- markdownlint-disable -->
> Batch Size는 일반적으로 32, 64, 128과 같은 크기가 자주 사용됩니다.
> 사이즈가 클수록 학습이 빠르고 일반화 성능이 낮아질 수 있으며, 반대로 사이즈가 작을수록 학습 속도는 느려지지만 일반화 성능이 향상될 수 있어 적절한 Batch Size를 계산하는 것이 필요합니다.
{: .prompt-info }
<!-- markdownlint-restore -->

# LSTM-based sound feature generator

이제 데이터를 LSTM의 input으로 사용할 준비를 마쳤습니다.
다음으로는 준비된 영상별 .npy 파일을 LSTM에 입력하여, last hidden state인 $x$를 출력으로 받습니다.
여기서 $x$는 LSTM 모델의 특성에 따라 **전체 비디오 시퀀스에 대한 spatial & temporal 정보**를 요약하여 나타냅니다. 

본격적으로 LSTM-based sound feature generator를 구현하기에 앞서서, 해당 모델이 구조에 대한 이해가 필요합니다.
해당 논문에서는 이 모델을 **Multi-task Learning** 방식으로 설계했습니다. 자세히는, 2개의 sub-branches에 $x$를 동시에 사용해 모델 성능을 지속적으로 개선하는 방식입니다.
첫번째 branch는 본 포스팅에서 구현 중인 **classification branch** 입니다. 즉, input 영상 정보에 대해 어떤 action class에 속할지 probability를 리스트로 리턴해, 확률이 가장 높은 클래스 라벨을 해당 비디오의 행동이라고 판단합니다.
해당 브랜치는 각 영상에 대해 존재하는 데이터 $x$와 각 영상의 action class가 매핑된 .txt 파일을 input으로 하며 함수는 아래와 같은 형태입니다.

$$
p(y) = \text{softmax}(W_p(x) + b_p)
$$

이 때, $W_p(x)$와 $b_p$는 각각 action recognition의 weights와 biases입니다.

구현 코드는 아래와 같습니다.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ClassificationBranch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationBranch, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # Fully connected layer
        self.softmax = nn.Softmax(dim=1)  # Softmax activation

    def forward(self, x):
        logits = self.fc(x)
        probabilities = self.softmax(logits)
        return probabilities

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_dim = 4096  # Dimension of the LSTM's last hidden state
    num_classes = 304  # Number of action classes in Moments in Time dataset
    batch_size = 64  # Batch size

    # Create a dummy LSTM output (last hidden state) for testing
    dummy_x = torch.randn(batch_size, input_dim)

    # Instantiate the classification branch
    model = ClassificationBranch(input_dim=input_dim, num_classes=num_classes)

    # Forward pass
    probabilities = model(dummy_x)

    print("Output probabilities shape:", probabilities.shape)  # Expected shape: [batch, num_classes]
```

이번 포스팅에서는 **LSTM-based sound feature generator**의 첫 번째 branch인 **classification branch**를 구현하고, 이를 위한 데이터 전처리와 특징 추출 과정을 다루었습니다.

먼저, 비디오 데이터를 RGB Frames와 Optical Flow Images로 분리한 뒤, CNN(ResNet50)을 사용하여 각각의 spatial feature와 temporal feature를 추출하고 이를 결합하여 LSTM 입력 데이터를 준비했습니다. 이후, LSTM의 마지막 hidden state를 활용한 classification branch를 구현해 각 비디오의 행동 클래스 확률 분포를 출력하는 과정을 다뤘습니다.

이번 포스팅에서 다룬 구현 내용은 초기 실험의 기반을 다지는 중요한 단계였습니다. 앞으로도 본 연구 과정을 지속적으로 공유하며, 연구의 완성도를 높이고 더 나은 결과를 도출할 수 있도록 노력하겠습니다. 추가 질문이나 피드백은 언제든 환영합니다 :)

# Footnote
[^footnote]: Lee, H.-C., Lin, C.-Y., Hsu, P.-C., & Hsu, W. H. (2019). Audio feature generation for missing modality problem in video action recognition. ICASSP 2019 - IEEE International Conference on Acoustics, Speech and Signal Processing, 3956–3960. https://ieeexplore.ieee.org/document/8682513?utm_source=chatgpt.com