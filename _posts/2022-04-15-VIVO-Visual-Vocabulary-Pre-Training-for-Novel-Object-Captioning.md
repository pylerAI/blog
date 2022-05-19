---
layout: post
author: cha
read_time: true
show_date: true
title:  "VIVO: Visual Vocabulary Pre-Training for Novel Object Captioning"
date:   2022-04-15 10:22:00 +0900
description: Captioning 데이터셋에 라벨링이 되어있지 않은 물체에 대해 캡션을 생성해내는 모델입니다.
img: posts/20220415/Untitled 1.png
tags: [Image Captioning, NOC, Novel object captioning]
arxiv: https://arxiv.org/pdf/2009.13682.pdf
mathjax: yes
toc: yes
---

## Post Introduction

안녕하세요 금주의 pyler 논문 리뷰 포스팅을 담당한 computer vision 개발자 차승일입니다.

오늘 소개드릴 논문은 2021년도 AAAI 학회에 개제된 “VIVO: Visual Vocabulary Pre-Training for Novel Object Captioning” 논문입니다.

Image Captioning은 Image가 입력으로 주어졌을 때 이에 대한 ‘자연어 설명’을 유추하는 분야입니다. 아래 예시와 같은 사진에 대해서는 ‘낮잠자는 강아지’와 같은 캡션 정보를 생성해내는 것을 목표로 하고 있습니다.

![example of image captioning model]({{ relativebase }}assets/img/posts/20220415/C1AD657E-CBBB-4CAE-92B5-1B135E3740B5.gif)

example of image captioning model

기존 학습 데이터셋에 등장하지 않았던 새로운 물체(novel object)를 포함한 caption을 생성해 내는 과제를 NOC(Novel Object Captioning Challenge)라고 합니다.

당연하게도 일반적인 video captioning 모델들은 학습 데이터셋에 존재하지 않는 새로운 몰체에 대해서는 caption 정보를 뽑아내지 못합니다. 그렇기 때문에 기존 모델들을 NOCAPS에 바로 적용하는데는 문제가 있습니다. 본 논문의 저자들은 이러한 문제점을 pretrain을 통해 해결하고자 했습니다. 즉 다양한 객체가 포함되어 있는 Image-tag 데이터셋에서 maksed token loss를 통해 pretrain을 진행한다면 실제 image captioning 데이터셋에 존재하지 않는 객체 정보를 포함한 캡션도 생성해 낼 수 있다고 증명하였습니다.

## Paper Abstract

NOC는 굉장히 어려운 태스크입니다. Captioning 데이터셋에 존재하는 물체 class가 적기에 특정 이미지를 잘 설명하는 캡션이 생성되기 힘들기 때문입니다. 본 논문은 이 태스크에 효과적인 방법론으로 Image-tag pre-training을 제시하고 있습니다. Image-tag pre-training 이란, Captioning 데이터셋을 이용해 pre-training을 시켰던 기존의 방식과는 다르게 Image-tag 데이터셋을 이용하는 것입니다. 본 논문에서는 이러한 방식으로 pre-training을 거친 후에 COCO와 같은 image-caption 데이터셋에 학습하는 방식을 VIVO(Visaul Vocabulary pretraining)라고 소개합니다.

## Introduction

COCO나 Flickr30k과 같은 데이터셋들 덕분에 image captioning에 대한 퍼포먼스는 크게 향상되었습니다.

그렇지만 이런 데이터셋에서 학습된 모델들은 학습때 보지 못했던 새로운 object에 대해서는 대응하지 못한다는 한계점을 가지고 있습니다.

저자들은 Novel object가 포함되어 있는 이미지를 In the wild images라 정의하고 있습니다.

In the wild images에 대한 Image Captioning, 즉 NOCAPS(Novel object captioning)의 퍼포먼스를 개선하기 위해 NOCAPS 벤치마크 모델들이 개발되어가고 있었습니다.

VIVO의 학습에 사용되는 데이터셋은 2가지로 이루어져 있습니다.

1. pre-training에 사용하는 Open Image 데이터셋(바운딩 박스와 이미지 레벨 태그를 포함)
2. fine-tuning에 사용하는 COCO 데이터셋(이미지-캡션 pair을 포함)

VIVO의 Test에는 fine-tuning 데이터셋에는 없거나 잘 등장하지 않는 400여개의 오브젝트를 포함하고 있는 이미지들(Open Images에서 추출된)을 이용합니다.

그렇다면 본 논문 이전의 연구들에서는 NOCAPS를 어떻게 다루고 있었을까요?

기존의 연구들은 먼저 In the wild images에 대해 Novel Objects는 빈 공간으로 비워둔 캡션을 생성하고, 이후에 detection모델이 추출한 label을 채우는 방식을 적용하고 있었습니다.

하지만 이와 같은 방식으로는 이미지와 텍스트 사이의 관계가 완전히 탐색되지 않습니다. detect된 label 사이의 관계가 이미지와는 달리 정보로 주어지지 않기 때문입니다.

유형별로 조금 더 자세히 Novel object captioning 모델들을 알아봅시다.

![Untitled]({{ relativebase }}assets/img/posts/20220415/Untitled.png)

먼저 Deep Compositional Captioner & Novel Object Captioner 처음에는 Novel Objects들에 대해 빈 공간으로 캡션을 생성하고, 추후에 Novel Object에 대해 Detect된 label과 이 label이 들어간 Text Data(위키피디아 등등에서 수집한 데이터셋)를 참고하여 캡션을 생성합니다.

![Untitled]({{ relativebase }}assets/img/posts/20220415/Untitled 1.png)

Neural Baby Talk & Decoupled Novel Object Captioner은 위의 경우와 동일하게 Novel object에 대해 빈 공간으로 임시 캡션을 만듭니다. 그 다음 detection모델로 추출한 label들을 RNN으로 순서를 부여해 임시 캡션의 빈 공간에 채워 넣습니다.

NOCAP도 VLP(Vision and Language Pre-training)의 일종입니다. 실제로 보통의 train방식보다 pre-training을 미리 진행하고 fine-tuning을 하면 성능이 훨씬 좋다는 점이 여러 실험들로 증명되었습니다. 하지만 대부분의 VLP 모델들은 understanding 태스크들(image-text retrieval, visual question answering)에 치중되어 있습니다.  OSCAR 등의 기존 image captioning모델들은 image-caption 데이터셋에서 pre-train을 하기 때문에 NOCAPS에 사용될 수 없습니다. caption 데이터셋에서 본 적이 없는 object label 에 대한 정보를 가지고 있지 않기 때문입니다. 따라서 저자들은 이러한 문제점들을 image-tag pre-training을 통해 해결하려고 시도하였습니다.

## Methodology

### VIVO

![Untitled]({{ relativebase }}assets/img/posts/20220415/Untitled 2.png)

VIVO는 pre-training에서 image-tag pairs에 대해 weakly supervised learning을 통해 학습합니다. fine-tuning에 사용하는 image-caption pairs와 비교해, pre-training에서 사용하는 image-tag pairs는 훨씬 많은 visual objects에 대한 정보를 담고 있습니다. pre-train단계에서는 tag를 이용해 각각의 헝가리안 매칭 로스를 사용해 image regions에 label을 대응하는 방법을 학습합니다. fine-tuning단계에서는 Detected된 objects에 대해, 이 novel object label의 caption을 만들어내는 방법을 학습합니다. VIVO는 novel object에 대해 zero-shot generalization을 가능하게 합니다. Zero-shot generalization이란, images in the wild(fine-tuning시 이용하는 데이터셋에는 존재하지 않는 오브젝트 클래스를 담고 있는 이미지)에 대한 캡션을 생성하는 것을 의미합니다.

### Implementation Details

Object detector은 Open Images라는 데이터셋에서 학습된 Updown이라는 모델을 사용했습니다. Open Images는 상당히 방대한 데이터셋으로 여러 object class를 담고 있기 때문에 pre-training에서 image-tag 데이터셋에 이용하기 적합하다고 판단됩니다. 또한 pre-training과 fine-tuning에서 사용하는 transformer는 BERT-base를 통해 initialized되었습니다. 다만 image region features의 벡터를 word embeddings과 같은 사이즈로 만들 필요가 있기 때문에 transformer에 linear layer가 추가되었다는 차이점이 있습니다.

![Untitled]({{ relativebase }}assets/img/posts/20220415/Untitled 3.png)

또한 pre-training과 fine-tuning에서, GT값이 사용되었습니다.

### VIVO Pre-training

![Untitled]({{ relativebase }}assets/img/posts/20220415/Untitled 4.png)

약 6.4k개의 클래스를 포함하고 있는 image-level tags 데이터셋(Open Images training set)을 이용해 학습합니다. caption과 달리 image tag들은 순서가 정해져 있지 않습니다. 따라서 pre-training에서 사용되는 mask tag prediction을 할 때 transformer에 bi-directional attention mask를 사용합니다. caption data없이 image region과 textual tags 사이의 joint representation을 학습합니다.

### Pre-train의 objective

image-level tag와 image regions가 주어졌을 때 임의로 가린 mask token을 예측하는 것 입니다. 학습 세트는 아래와 같이 표기될 수 있습니다(N개의 이미지와 해당되는 태그).

$$
\mathbb{D}=\\{\mathrm{I}\_i,\ \mathrm{G}\_i\\}^N_{i=1}
$$

$\mathrm{G}_i$는 이미지 $\mathrm{I}_i$에 대해 image-level tags 의 집합입니다.

$$
\begin{aligned}
\mathrm{G}\_i=\\{g_{ij}\\}^{L\_i}_{j=1}
\end{aligned}
$$

$L_i$는 image에 나와있는 visual objects에 대한 textual labels, 즉 tags입니다.

### Pre-train의 Multi-layer Transformer

![Untitled]({{ relativebase }}assets/img/posts/20220415/Untitled 7.png)

vision과 language domains의 joint representation을 학습합니다. input은 $\{\mathrm{V},\mathrm{T}\}$ ($\mathrm{V}$ : image region feature, $\mathrm{T}$ : tag tokens). $\mathrm{V}$는 기존의 object detection 모델(Updown)을 이용하여 추출합니다. 몇개의 토큰중 15 %를 랜덤하게 마스킹하여 masked token loss로 traing합니다.

### Hungarian matching loss

여러개의 token이 마스킹 되었을 때, tag는 caption과 달리 순서가 정해져 있지 않았기 때문에 예측된 여러 token과 GT값의 token을 매칭시켜 loss를 구하는게 매우 중요합니다. 본 논문은 token매칭에 흔히 이용하는 헝가리안 매칭 로스를 이용해 token들 사이의 매칭을 합니다.

- $\tilde{\mathrm{T}}$ : $M$개의 마스킹된 토큰들의 집합
$t_m$ : Vocabulary에 존재하는 임의의 토큰(마스킹된 토큰에 해당하는), 즉 GT값

$$
\begin{aligned}
\tilde{\mathrm{T}}=\\{t_m\\}^M_{m=1}
\end{aligned}
$$

- $\mathrm{p}_i$ : i번째 마스킹된 토크에 대해 classification 된 tag
$\tilde{\mathrm{T}}$ 가 순서가 정해져있지 않은 집합이기 때문에 $\tilde{\mathrm{T}}$와 $\mathrm{P}$를 loss가 최소가 되도록 매칭하는 one-to-one mapping이 필요합니다.

$$
\begin{aligned}
L(\tilde{\mathrm{T}},\mathrm{P},\alpha)=\sum\_{i=1}^M(-\text{log}(\mathrm{p}\_i(t\_{\alpha_{(i)}})))
\end{aligned}
$$

- $\alpha$ : 임의의 one-to-one mapping을 위해 정해진 $\mathrm{P}$의 순열
$\alpha(i)$ : $i$번째 prediction과 매칭된 target token의 index
- loss가 minimize 되는 $\alpha$를 $\hat{\alpha}$이라고 정의합니다.

$$
\begin{aligned}
\tilde{\alpha}=\underset{\alpha}{\operatorname{argmin}}\sum^M\_{i=1}C(\mathrm{p}\_i,t\_{\alpha(i)})
\end{aligned}
$$

- 또한 보통의 헝가리안 로스가 사용하는 -log 대신 본 논문은 $C(p_i,t_m)$을 사용하는데 이는 x=0일때 -log는 무한대로 발산하기 때문에, bound를 주기 위해서 입니다.

$$
\begin{aligned}
C(p_i,t_m)=1-p_i(t_m)
\end{aligned}
$$

### Fine-tuning

![Untitled]({{ relativebase }}assets/img/posts/20220415/Untitled 12.png)

caption과 tag가 존재하는 데이터셋에 Fine-tuning을 진행합니다(e.g COCO). detection model을 통해 태그를 뽑아낼 수 있습니다. 주어진 image regions과 tag에 대해 모델은 몇개의 token들이 마스킹 되었을 때 conditional caption sentence를 예측합니다. Input feature는 $\{\mathrm{V, T, C}\}$ (Image region features, a set of tags, caption). $\{\mathrm{V,T}\}$는 Pre-training과 같은 방식으로 설계됩니다.

### Randomly mask out some of tokens in fine tuning

![Untitled]({{ relativebase }}assets/img/posts/20220415/Untitled 13.png)

caption의 token 중 15%를  random하게 masking 합니다. 이 masked out된 token들을 cross-entropy loss를 이용해 예측하는 과정을 학습합니다. caption은 tag들의 집합과는 달리, 순서가 있기 때문에 uni-directional attention mask를 사용합니다.

### Inference

![Untitled]({{ relativebase }}assets/img/posts/20220415/Untitled 14.png)

image region features($\mathrm{V}$)와 detected된 tags($\mathrm{T}$)를 추출합니다. EOS(End of sentence) token이 나올 때까지 한번에 한 token씩 만들어내며 caption을 만들어냅니다. Masking되어있는 caption은 OSCAR이라는 모델을 base로 뽑아냅니다. 또한 이 모델은 auto-regressive 즉, 이전의 토큰을 additional input으로 사용하는 구조입니다.

## Experiments

Result

![Untitled]({{ relativebase }}assets/img/posts/20220415/Untitled 15.png)

UpDown과 OSCAR과 같은 기존 모델들은 in-domain 즉, training set에 나오는 object에 대해 caption을 생성할 때에는 훌륭한 성능을 보여줍니다. 하지만 near, out of domain에서는 성능이 매우 떨어지는데 SCST나 CBS를 사용하면 성능이 Updown은  16.1%p,  그리고 out ot damain에서는 무려 40.4%p, OSCAR은 15.5%p, 32.3%p로 매우 향상되는 것을 볼 수 있습니다. 특히 CBS는 특정한 단어들이 캡션에 등장하도록 강제하는 역할을 하기 때문에 성능에 매우 큰 영향을 미쳤다고 생각합니다. Updown과 OSCAR은 기본적으로 학습시 이용하는 COCO의 80여개의 object class에 대한 caption만 잘 생성해낼 수 있기 때문에 optimization에 매우 의존될 수 밖에 없는 구조이기 때문이라 생각합니다. 하지만 VIVO는 in-domain은 물론 near, out of domain에서 기존 모델들에 비해 압도적인 성능을 보여줍니다. VIVO는 기본 성능이 매우 뛰어나기 때문에 SCST나 CBS와 같은 optimization을 사용하여도 크게 성능향상(4.6%p, 16.4%p)이 없었다고 본 논문은 설명하고 있습니다. CBS를 사용한 것과 기존 VIVO를 비교했을 때 성능차이가 얼마 없었다는 것은, 특정 단어가 캡션에 등장하도록 강제하지 않아도 이미 그 단어들을 충분히 잘 생성해내고 있었다고 생각할 수 있을 것 같습니다. 심지어 VIVO는 인간의 결과를 뛰어넘은 놀라운 성능을 보여줬다고 합니다.  하지만 인간과 비교한 실험에 대해서는 상세한 설명이 기술되어 있지 않아 아쉬움이 남습니다. 아마도 인간도 in-domain, near-domian, out-of-domain으로 나누어 점수를 매긴 것으로 보아, VIVO와 동일한 조건으로 pre-training, fine-tuning시 이용하는 open images, COCO에 대한 정보를 주고 테스트를 진행했을 것으로 생각됩니다. 하지만 training parameter을 저장할 수 있는 컴퓨터와 달리 단기간에 노출된 정보에 대해 기억하는 능력이 한정적인 인간과 이러한 방식으로 비교하는 것이 과연 fair한 comparison인지에 대한 의문이 생깁니다.

Ablation study

![Untitled]({{ relativebase }}assets/img/posts/20220415/Untitled 16.png)

또한 본 논문은 pre-training에서 image-tag를 사용하지 않을지(즉, OSCAR만 사용할지) , 500개의 tag만을 사용할지, 6.4K개의 tag를 사용할지를 설정하여 비교 실험을 진행 하였습니다. 물론 500개의 tag를 사용한 것보다 6.4K개의 tag를 사용한 것이 성능이 좋았지만 그 차이보다 태그를 사용하지 않은 경우와 500개를 사용한 차이가 더 컸습니다. 본 논문은 이를 pre-training에 사용하는 tag의 수를 많게 하는것도 중요하지만, tag를 사용하여 pre-training을 하는 것 자체가 더 성능에 유의미한 영향을 미친다고 설명합니다.

![Untitled]({{ relativebase }}assets/img/posts/20220415/Untitled 17.png)

pre-training시 토큰을 하나만 마스킹할지, 헝가리안 매칭을 적용할지에 대한 실험 결과도 있었는데, 확실히 헝가리안 매칭을 사용한 것이 최적의 성능이 나옴을 알 수 있습니다. 태그는 집합의 형태로 저장되기 때문에 마스킹된 tag들의 GT값과 추론값을 하나하나 매칭시켜주는 것이 매우 중요하기 때문에 이런 결과가 나왔다고 생각합니다.

## Conclusion

저자는 이번 논문을 통해 VIVO(Visaul Vocabulary pretraining)를 제안하며, 이는 기존의 Image Captioning 모델들과 달리 pre-training에서 captioning data를 사용하지 않고 image-tag data만을 이용해 joint representation을 학습합니다. 이를 통해 captioning data에서 등장하지 않는 class에 대해 압도적인 성능으로 caption을 생성해낼 수 있는 모델을 만들어 낼 수 있었습니다.
