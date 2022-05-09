---
layout: post
author: ted
read_time: true
show_date: true
title:  "Paper Review: Spatiotemporal Contrastive Video Representation Learning"
date:   2022-04-05 11:22:00 +0900
description: Single neuron perceptron that classifies elements learning quite quickly.
img: posts/20220405/2.png 
tags: [machine learning, coding, neural networks]
github: https://github.com/tensorflow/models/tree/master/official/
arxiv: https://arxiv.org/pdf/2008.03800v4.pdf
mathjax: yes
toc: yes
---

## Post Introduction

PYLER 의 첫 논문 리뷰포스트로, CVPR 2021 “Spatiotemporal Contrastive Video Representation Learning” 을 가져왔습니다. 최근 Representation Learning 분야에서 Video Level 의 연구가 활발히 진행되고 있습니다. 정답 label 이 없는 데이터셋이 주어졌을 때 self-supervised learning 단에서 pretext task를 정의하여 학습하는 방법이 있고, 발전하여 noise contrastive estimation을 기반으로 positive 와 negative pairs 의 관계를 학습하는 contrastive Learning 이 있습니다. 대표적으로 pretext task 로는 frame prediction, spatio temporal puzzling, video statistics, temporal ordering, video playback rate prediction, temporal consistency 등이 있고, constrative learning 으로는 SimCLR, MoCo, BYOL, SimSiam 등의 방식이 있습니다. 최근 Contrastive Learning 방법론을 활용해 연구가 활발히 이루어지고 있는데, 본 논문은 좋은 data augmentation 과 spatial and temporal information 의 중요성을 강조하여 downstream task에서 높은 성능을 달성합니다.

## Paper Abstract

해당 논문은 self-supervised Contrastive Video Representation Learning (CVRL) framework 를 제시하여, spatiotemporal visual representations을 학습합니다. 논문에서 핵심적인 부분인 2가지가 있는데, consistent spatial augmentation method 와 sampling-based temporal augmentation method 입니다. Consistent spatial augmentation method 를 통해 비디오 내 각 프레임들 간의 temporal consistency 를 유지한 채로 spatial augmentation을 적용하고, Sampling-based temporal augmentation method 를 통해 시간적으로 먼 클립들 사이의 invariance 가 과하게 집중되는 것을 피합니다.

![Untitled]({{ relativebase }}assets/img/posts/20220405/0.png)

Kinetics-600 데이터셋에 대해, CVRL 프레임워크를 통해 video representation 이 3D-ResNet-50 구조로 학습된 linear classifier 가 70.4% 의 accuracy 를 기록했습니다. 이는 동일한 inflated R3D-50 구조로 ImageNet supervised pre-training 한 것보다 15.7% 높고, SimCLR unsupervised pre-training 한 것보다 18.8% 높은 성능입니다. CVRL 프레임워크의 성능은 R3D-152 (2xfilters) backbone 일 때 가장 높은 72.9% accuracy 를 기록합니다. 이는 video representation learning 에서 unsupervised 기법과 supervised 기법의 성능의 차이를 줄인 것을 확인할 수 있습니다.

## Paper Introduction

단순히 비디오 내 프레임들에 대해 독립적으로 spatial augmentation을 시행했을 때에는 좋은 성능을 내기 힘듭니다. 왜냐하면 시간 축을 기준으로 분석했을 때 모션 정보를 해치기 때문이죠. 이를 해결하기 위한 방법으로 비디오 내 프레임들 간의 무작위 정도를 고정시키는 temporally consistent spatial augmentation 을 제안합니다. Temporal augmentation은 visual 정보들을 CVRL 프레임워크 내에서 sampling 하는데, 시간적으로 먼 positive clip pairs 들은 매우 다른 visual content 를 담을 수 있으므로 negative clip pairs들과 차이가 없는, 낮은 유사성으로 이어질 수 있습니다. 반면, 시간적으로 먼 클립들을 완전히 버리는 건 temporal augmentation 효과를 감소시킵니다.

![Untitled]({{ relativebase }}assets/img/posts/20220405/1.png)

결론적으로, 해당 논문은 두 개의 positive clip 들이 시간적으로 멀수록 sampling 수가 감소하는 분포를 따르는 sampling strategy 를 제안합니다. CVRL 프레임워크는 시간적으로 가까운 positive clip pairs 들을 중점적으로 학습하게 되고, 학습 중 시간적으로 먼 클립들을 상대적으로 덜 학습되게 하도록 합니다.

주 요점은 다음과 같이 3가지로 정리할 수 있겠네요.

- spatial 과 temporal 정보들을 섞는게 성능을 향상시킨다.
- 본 논문의 representation 은 각 task 별 SOTA의 성능을 보여준다.
- 본 논문의 CVRL 프레임워크는 큰 데이터셋에 유리하다.

## Methodology

### CVRL Framework

![Untitled]({{ relativebase }}assets/img/posts/20220405/2.png)

\\[
\begin{equation}
\mathcal L_i = - \log \frac{\exp(\text{sim}(z_i, z_i') / \tau }{\sum_{k=1}^{2N} 1_{[k \neq i]} \exp( \text{sim}(z_i, z_k) / \tau}
\end{equation}
\\]

Augmentation된 clip들에 feature extraction을 진행하고 contrastive loss로 InfoNCE 를 씁니다.

N 개의 raw video들이 있다고 가정했을 때 augmentation을 진행하면  $2N$개의 clips 들이 나옵니다. $i$-th input video의 augmented clip 2개의 encoded representation을 각각 $z_i$, $z_i^\prime$ 으로 표기합니다. 또한 1[.] 의 경우, encoded clip $z_i$ 에 대해 self-similarity 가 계산되는 것을 방지합니다.

\\[\operatorname{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top\mathbf{v}}{\lVert \mathbf{u} \rVert_2 \lVert \mathbf{v} \rVert_2}\\]

그리고, Cosine Similarity 를 계산합니다.

\\[\begin{equation} \mathcal L = \frac 1 N \sum^{N}_{i=1} \mathcal L_i\end{equation}\\]

InfoNCE contrastive loss 는 위의 식과 같이 N개의 비디오 수에 대해 평균을 취합니다.

### Video Encoder

Video Encoder에서의 3D-ResNet 구조는 SlowFast network 에서 ‘slow’ pathway 를 따르고 있습니다. SlowFast network 의 경우, 궁금하신 분들은 밑의 링크를 참조하시면 됩니다.

[](https://arxiv.org/pdf/1812.03982.pdf)

2개의 수정사항은 있는데, data layer 에서 temporal stride 는 2, 첫번째 convolution layer 에서 temporal kernel size 는 5, stride 는 2 라는 점입니다.

Video representation 은 2048 차원의 feature vector 이고, SimCLR 의 구조를 모방하여 multi-layer projection head 를 추가하여 128 차원의 encoded feature vector $z$를 얻습니다.

### Data Augmentation

- **Temporal Augmentation: a sampling perspective**

이전에는 비디오 내 프레임이나 클립들을 정렬하거나 비디오의 playback rates 의 변화를 주는 등의 temporal augmentation 방법들이 제안되었습니다. 그러나 직접적으로 위의 방법들을 CVRL 프레임워크에 적용한다면 temporally invariant feature들에 대해 학습하는 결과를 야기할 수 있습니다. 결국 비디오의 시간적 변화를 담고 있는 feature 들을 반영하지 못하게 됩니다. 이를 해결하기 위해 sampling strategy 를 사용하여 temporal 변화를 유도합니다.

시간적으로 먼 clip 들을 더 작은 확률로 샘플링하고 시간적으로 가까운 클립들은 그들의 feature 정보들을 가깝게 반영하면서 contrastive loss 를 계산합니다.

![Untitled]({{ relativebase }}assets/img/posts/20220405/3.png)

- **Spatial Augmentation: a temporally consistent design**

Spatial augmentation 의 기본 원리는 image-based 로 존재했던 spatial augmentation method를 video 내 프레임들 각각에 활용하는 것인데, 이것은 프레임별로 서로 다른 augmentation이 반영되기 때문에 단일 Clip 내에서 행동 정보를 상실시킬 수 있습니다. 기존의 Spatial augmentation 방법은 random cropping, color jittering, blurring 등 다양한 방법들이 존재하는데, performance 측면에서 중요하게 작용합니다. 이러한 augmentation 기법을 차용하여 논문에서는 temporal dimension 상에서 일관성을 유지하며 spatial augmentation 을 적용하도록 디자인하였습니다. 3D 비디오 encoder 는 spatiotemporal 단서들을 더 잘 활용할 수 있게 해줍니다.

![Untitled]({{ relativebase }}assets/img/posts/20220405/4.png)

## Experiments

Kinetics-400, Kinetics-600 데이터셋들로 학습을 진행했습니다. Kinetics-400 은 400개의 action class 가 담겨있는 240K training videos 와 20K validation videos 로 구성되어있고, Kinetics-600 은 600개의 action class 가 담겨있는 360K training videos 와 28K validation videos 로 구성되어 있습니다. 위의 데이터셋을 활용하여 self-supervised / supervised / semi-supervised 방법론별로 Top-1 Accuracy 수치로 비교했을 때 성능적으로 우월함으로 보여주고 있습니다.

![Linear evaluation]({{ relativebase }}assets/img/posts/20220405/5.png)

![Semi-supervised learning]({{ relativebase }}assets/img/posts/20220405/6.png)

Pre-trained 이후 downstream task에서 UCF, HMDB kinetics dataset에 대하여 action classification, action detection 성능 비교 실험을 진행하였습니다.

여기서 중요한 점은 각각 (F)low, (A)audio, (T)ext 모달리티 등이 추가된 타 모델 과는 달리, CVRL 프레임워크의 경우 (V)ision 모달리티만 있음에도 불구하고 fine-tuning 한 결과나 linear evaluation 한 결과를 비교했을 때 UCF, HMDB dataset 에 대해 Top-1 Accuracy 수치에서 큰 차이가 없음을 보여주고 있습니다.

![Untitled]({{ relativebase }}assets/img/posts/20220405/7.png)

Action Detection 에서 AVA Dataset 의 경우 211K training videos 와 57K validation videos 들이 존재합니다. 약 15분에서 30분 사이의 긴 비디오들에서 각 action labelling 이 되어있는 spatiotemporal labels 이 존재합니다.

![Untitled]({{ relativebase }}assets/img/posts/20220405/8.png)

위의 표처럼 CVRL 프레임워크의 경우 mAP 값이 다른 방법론들에 비해 가장 높은 성능 수치를 기록하고 있습니다.

## Conclusion

이번 논문을 통해 Contrastive Video Representation Learning (CVRL) framework 를 제안하며, 이는 spatial 과 temporal 단서들에 영향을 주면서 label 되지 않은 비디오들로부터 spatiotemporal representation 을 학습할 수 있도록 합니다. 저자들은 unlabeled 된 큰 비디오 세트들에 CVRL 프레임워크를 적용하고, 추가적인 모달리티를 프레임워크에 포함시키는 것을 목표로 한다고 밝히고 있습니다.
