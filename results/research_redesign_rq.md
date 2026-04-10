# WBC Synthesis Research Redesign and Canonical RQ

## 한 줄 문제정의

어떤 `generation policy`가 실제 WBC 분류 일반화에 도움이 되는 `synthetic pool`을 만들며, 그 효용은 어떤 `hard class / held-out domain`에서 재현되는가?

---

## Summary

다음 사이클의 메인 축은 `평가설계 재정의`가 아니라 `생성파이프라인 복귀`다. 다만 이것은 초기 `00~07`로 단순 회귀하는 것이 아니다. `LODO/selective subset` 단계에서 얻은 교훈을 제약조건으로 고정한 상태에서, 생성 정책을 다시 메인 조작 변수로 삼는 복귀다.

핵심 원칙은 네 가지다.

1. 생성 연구의 1차 목표는 “더 그럴듯한 이미지”가 아니라 “downstream utility가 있는 synthetic pool”이다.
2. 다음 사이클의 직접 조작 변수는 `subset builder`가 아니라 `generation policy`다.
3. `LODO/selective subset`은 폐기하지 않고, 새 생성 정책의 utility를 검증하는 보조 평가 프레임으로 유지한다.
4. `boundary-aware V2`는 메인 노선이 아니라, boundary 가설을 밀어본 시도와 그 한계를 정리한 보조 브랜치로 남긴다.

---

## 선행연구 기준선과 novelty frame

이번 사이클은 `좋아 보이는 실험 계획`만으로는 충분하지 않다. 최소한 아래 네 갈래의 legacy와 비교해, 우리가 어디를 반복하고 어디를 넘어서는지 서술 가능해야 한다.

### A. WBC synthetic augmentation 계열

대표적으로 다음 축이 이미 존재한다.

- GAN 기반 leukocyte synthesis와 classifier augmentation
  예: `SyntheticCellGAN`류 연구는 morphologically realistic한 synthetic leukocyte를 만들고 classifier training에 쓰는 방향을 보였다.
- blood cell diffusion / diffusion-hybrid augmentation
  최근 blood cell augmentation 논문들은 diffusion 또는 diffusion-hybrid generator를 class imbalance 완화와 accuracy 개선에 사용한다.

이 갈래의 공통점은 대체로 다음과 같다.

- 목적이 `dataset expansion` 또는 `class imbalance 완화`
- 평가가 in-domain classification 또는 일반 accuracy 중심
- unseen-domain utility를 메인 주장으로 두지 않는 경우가 많다

대표 예시 링크:

- [Automatic generation of artificial images of leukocytes and leukemic cells using generative adversarial networks (SyntheticCellGAN)](https://www.sciencedirect.com/science/article/pii/S0169260722006952)
- [A method for expanding the training set of white blood cell images](https://pubmed.ncbi.nlm.nih.gov/36406333/)
- [Diffusion-based Wasserstein generative adversarial network for blood cell image augmentation](https://www.sciencedirect.com/science/article/abs/pii/S0952197624003798)
- [A deep learning approach for white blood cells image generation and classification using SRGAN and VGG19](https://www.sciencedirect.com/science/article/pii/S2772503024000495)

### B. WBC domain shift / domain generalization 계열

대표적으로 다음 축이 존재한다.

- WBC under domain shift benchmark 계열
- 최근 WBC DG/DA 방법
  예: `DaNet`은 synthetic augmentation과 cross-domain feature alignment를 결합해 WBC cross-domain generalization을 다룬다

이 갈래의 공통점은 다음과 같다.

- 핵심이 `feature alignment`, `mixup`, `domain adaptation/generalization`
- synthetic generation policy 그 자체보다 representation learning이 중심
- 어떤 synthetic pool이 utility를 만드는지보다는 alignment strategy가 중심

대표 예시 링크:

- [DaNet: Domain-adaptive white blood cell classification through synthetic augmentation and cross-domain feature alignment](https://www.sciencedirect.com/science/article/pii/S2590005625000438)
- [WBCAtt: A White Blood Cell Dataset Annotated with Detailed Morphological Attributes](https://arxiv.org/abs/2306.13531)

### C. 일반 medical imaging DG augmentation 계열

대표적으로 다음 축이 있다.

- `Domain Generalization for Medical Imaging` (NeurIPS 2020)
- episodic training with task augmentation (2021)
- style augmentation / dual normalization (CVPR 2022)
- complex style transformation 계열 (CVPRW 2024)

이 갈래의 공통점은 다음과 같다.

- image-level style augmentation 또는 latent/feature regularization이 중심
- segmentation task가 많고, classification이어도 `source-domain diversity expansion`이 주요 아이디어
- `cell-level generative policy`와 `downstream selective utility`를 직접 연결하지는 않는다

대표 예시 링크:

- [Domain Generalization for Medical Imaging](https://proceedings.nips.cc/paper/2020/file/201d7288b4c18a679e48b31c72c30ded-Paper.pdf)
- [Domain Generalization for Medical Imaging Classification with Linear-Dependency Regularization](https://arxiv.org/abs/2009.12829)
- [Intra-Source Style Augmentation for Improved Domain Generalization](https://openaccess.thecvf.com/content/WACV2023/papers/Li_Intra-Source_Style_Augmentation_for_Improved_Domain_Generalization_WACV_2023_paper.pdf)
- [CDDSA: Contrastive domain disentanglement and style augmentation for generalizable medical image segmentation](https://www.sciencedirect.com/science/article/pii/S1361841523001640)
- [Generative feature style augmentation for domain generalization in medical image segmentation](https://www.sciencedirect.com/science/article/abs/pii/S0031320325000767)

### D. 우리 코드베이스의 기존 legacy

우리 내부 legacy는 이미 세 갈래가 있다.

1. `00~07`
   single-domain synthetic augmentation
2. `33~37`
   utility-aware subset selection
3. `41~48`
   boundary-aware generation objective

따라서 우리의 novelty framing은 외부 선행뿐 아니라 `내부 legacy 대비 초과분`도 함께 설명해야 한다.

### 현재 시점의 잠정 novelty claim

현 시점에서 가장 안전한 framing은 다음이다.

> To our best knowledge, multi-domain WBC setting에서 `generation policy` 자체를 class-aware / target-domain-aware / context-aware하게 설계하고, 그 결과를 `LODO selective utility` 기준으로 검증하는 프레임은 기존의 단순 dataset expansion, generic filtering, feature alignment 중심 접근과 다른 문제설정이다.

여기서 중요한 점은 두 가지다.

1. `synthetic image를 쓴다`는 사실 자체는 novel claim이 아니다.
2. `domain generalization을 다룬다`는 사실 자체도 novel claim이 아니다.

우리가 주장할 수 있는 후보는 아래처럼 더 좁아야 한다.

- novelty 후보 1:
  `WBC domain generalization을 generation policy design problem으로 재정의`
- novelty 후보 2:
  `hard-class / held-out-domain failure mode를 기준으로 class-aware synthetic pool을 설계`
- novelty 후보 3:
  `selection only가 아니라 policy + selection의 결합을 canonical unit으로 다룸`

반대로 아래 주장은 현재 상태에서는 피한다.

- “세계 최초 diffusion-based WBC augmentation”
- “최초의 WBC domain generalization”
- “최초의 boundary-aware medical augmentation”

이런 표현은 focused literature matrix 없이 방어가 어렵다.

---

## Reference Review Gate

이번 사이클에서는 실험보다 먼저 아래 문헌 검토 산출물을 만들어야 한다.

### 필수 레퍼런스 매트릭스

최소 12~20편을 아래 열로 정리한다.

- task
- modality / dataset
- WBC 여부
- synthetic 사용 여부
- generator 종류
  GAN / diffusion / style transfer / classical augmentation / none
- domain shift setting 여부
- main claim
- evaluation setting
  in-domain / cross-domain / LODO / DA / DG
- 우리가 넘어서야 할 한계

### 필수 비교군 축

문헌 검토는 아래 네 bucket을 모두 포함해야 한다.

1. WBC classification + augmentation
2. WBC under domain shift / DG / DA
3. medical image DG + style/augmentation
4. synthetic medical augmentation + downstream utility evaluation

### 문헌 검토의 역할

이 review는 단순 참고문헌 목록이 아니라, 연구 설계의 gate로 사용한다.

- 어떤 novelty 표현이 가능한지 결정
- 어떤 baseline을 꼭 재현하거나 비교해야 하는지 결정
- 어떤 실험은 이미 알려진 축의 반복에 불과한지 판별

즉 이 단계가 완료되기 전에는 강한 novelty 문장을 최종 문서나 논문 초록에 넣지 않는다.

---

## Canonical RQ

### Main RQ

`RQ0. 어떤 generation policy가 hard WBC domain-generalization setting에서 실제로 유용한 synthetic pool을 만드는가?`

이 질문의 핵심은 “얼마나 realistic한가”가 아니라, “실제 held-out domain 성능에 도움이 되는가”다.

### Sub RQ

`RQ1. strength와 context preservation 수준은 downstream utility에 어떤 trade-off를 만드는가?`

- 입력 보존이 너무 강하면 variation이 부족하다.
- variation이 너무 강하면 class identity가 무너진다.
- 따라서 다음 사이클의 1차 생성 변수는 `strength`와 `context preservation level`이다.

`RQ2. class-specific generation policy는 one-size-fits-all policy보다 hard class rescue에 유리한가?`

- `monocyte`, `eosinophil`은 계속 hard class로 나타났다.
- 클래스별로 safe operating region이 다를 수 있다.

`RQ3. target-domain-conditioned generation은 generic cross-domain generation보다 utility가 높은가?`

- domain token, target-domain prompting, context branch는 이 질문 아래에서만 평가한다.
- 목표는 `도메인별 failure mode`를 직접 겨냥한 생성 정책이 generic pool보다 유리한지 보는 것이다.

---

## 메인 가설과 반증 조건

### 메인 가설

`generation policy`를 잘 설계하면, subset builder를 크게 바꾸지 않아도 hard domain에서 유의미한 downstream gain을 주는 synthetic pool을 만들 수 있다.

즉 이번 사이클의 주장은 다음 문장으로 압축한다.

> utility는 선택(selection)만의 문제가 아니라, 생성 정책(policy)의 문제이기도 하다.

### 반증 조건

아래 중 하나가 반복되면 현재 사이클의 메인 가설은 약화된다.

1. generation-side 지표가 좋아져도 `LODO` utility가 반복적으로 개선되지 않는다.
2. class-specific policy가 global policy보다 일관되게 낫지 않다.
3. target-domain-conditioned generation이 generic cross-domain generation보다 재현 가능하게 낫지 않다.

---

## 이번 사이클의 비목표

아래는 이번 사이클에서 메인 라인으로 하지 않는다.

1. `subset builder`만 계속 복잡하게 만드는 것
2. backbone sweep를 메인 기여로 삼는 것
3. easy split 점수를 메인 성과로 내세우는 것
4. boundary-near score 자체를 논문의 핵심 기여로 밀어붙이는 것
5. synthetic quantity 확대만으로 성과를 내려고 하는 것

---

## Benchmark 정책

### 메인 benchmark

`LODO selective utility benchmark`

이 benchmark를 메인으로 고정한다. 이유는 다음과 같다.

- 이미 baseline 난이도가 확인되어 있다.
- domain별 failure mode가 뚜렷하다.
- synthetic utility가 실제로 드러난 적이 있다.

이번 사이클에서 메인 검증 대상 held-out domain은 우선 다음 두 곳에 집중한다.

- `Raabin`
  monocyte rescue sensitivity가 높다.
- `AMC`
  eosinophil rescue sensitivity가 높다.

`PBC`와 `MLL23`는 보조 확인 대상으로 남긴다.

### 보조 benchmark 1

`generation-side diagnostic panel`

이 패널은 생성 정책을 빠르게 거르기 위한 비최종 평가다.

- CNN correctness / confidence
- 필요 시 VGG16 민감도 평가
- context-preservation 계열 지표
  `ssim`, `cell_ssim`, `background_ssim`, `region_gap`

이 패널의 목적은 “좋아 보이는 생성 정책 후보를 shortlist”하는 것이다. 메인 주장용 benchmark가 아니다.

### 보조 benchmark 2

`hard-class stress setting`

low-data나 class-imbalance는 메인 benchmark가 아니라 보조 검증으로 둔다.

- monocyte/eosinophil fraction 축소
- selected synthetic policy의 rescue 여부 확인

이 세팅은 generation policy가 “전체 평균”이 아니라 hard-class rescue에 실제로 쓰이는지 보는 보조 증거로만 사용한다.

---

## 생성파이프라인 메인 실험축

이번 사이클의 직접 조작 변수는 아래 4개로 제한한다.

### 1. Strength axis

후보:

- low
- medium
- high

실행 시에는 class별 safe zone을 허용한다. 모든 클래스에 동일한 strength grid를 강제하지 않는다.

### 2. Class-specific policy axis

후보:

- global shared policy
- `monocyte` 전용 policy
- `eosinophil` 전용 policy
- hard-class-only shared policy

핵심 질문은 `class별 generation operating region`이 필요한가이다.

### 3. Context preservation axis

후보:

- 기존 `processed_multidomain` 계열
- contextual branch
- context-preserving crop / larger canvas 계열

핵심 질문은 더 넓은 smear context가 실제로 utility 있는 variation을 만드는가이다.

### 4. Target-domain-conditioned prompting axis

후보:

- generic cross-domain prompting
- target-domain-conditioned prompting
- explicit domain token / style token prompting

핵심 질문은 target-domain failure mode를 직접 겨냥한 conditioning이 utility를 높이는가이다.

---

## Success Criteria

### Generation-side 성공 기준

generation 후보를 shortlist하기 위한 기준은 다음과 같이 분리한다.

1. class identity가 유지되어야 한다.
2. variation이 존재해야 한다.
3. hard class에서 safe operating region이 재현되어야 한다.

이 단계에서는 아래 형태의 판단만 한다.

- `identity preservation`
- `context/background variation`
- `class-specific stability`

이 지표는 후보 탈락용이며, 최종 주장용 성과가 아니다.

### Downstream 성공 기준

최종 성공 기준은 `LODO` 기준으로 둔다.

1. `Raabin` 또는 `AMC` 중 최소 1개에서 재현 가능한 macro-F1 상승
2. target hard class F1가 명확히 개선
3. 이미 강한 클래스의 붕괴가 제한적일 것

메인 주장으로 채택하려면 다음 중 2개 이상을 만족해야 한다.

- held-out macro-F1 개선
- hard class F1 개선
- 동일 policy가 seed/재실행에서 재현

---

## Stop Rule

다음 중 하나가 성립하면, 생성 정책 메인 축을 잠시 멈추고 다시 `subset/benchmark` 축으로 복귀한다.

1. 서로 다른 두 generation revision이 연속으로 generation-side 지표는 개선하지만 LODO utility를 개선하지 못함
2. class-specific policy 이점이 재현되지 않음
3. target-domain-conditioned generation이 generic policy 대비 안정적 우위를 만들지 못함

복귀 시에는 아래 방향으로 전환한다.

- subset builder 단순화 또는 재정의
- benchmark sensitivity 재점검
- domain별 failure analysis 강화

---

## Boundary-aware V2의 위치

`boundary-aware V2`는 이번 사이클의 메인 노선이 아니다. 그러나 아래 이유로 중요한 보조 자산이다.

1. generation objective를 다시 생각하게 만들었다.
2. cell/background 분리, contextual preprocessing, mask 분석 도구를 남겼다.
3. easy sample bias를 문제로 명시했다.

따라서 이후 문서와 발표에서는 다음처럼 정리한다.

- `메인 노선`
  generation policy redesign for downstream utility
- `보조 브랜치`
  boundary-aware generation hypothesis and lessons learned

즉 `47/48`은 canonical endpoint가 아니라, 한 번 boundary 가설을 강하게 밀어본 시도로 위치를 고정한다.

---

## 논문 메시지 후보

### 현재 자산으로 만들 수 있는 메시지

1. `synthetic utility depends on policy, not only quantity`
2. `domain-generalization용 synthetic pool은 generation policy와 subset policy가 함께 결정한다`
3. `hard-class rescue requires class-aware generation policy`
4. `WBC domain shift 문제에서 synthetic augmentation의 novelty는 생성량이 아니라 policy design과 evaluation protocol의 결합에서 나온다`

### 지금 상태로는 주장하면 안 되는 메시지

1. `boundary-aware V2가 기존 selective/LODO 성과를 이미 대체했다`
2. `하나의 universal generation policy가 모든 domain에서 통한다`
3. `realism or boundary score 자체가 downstream utility를 자동 보장한다`

---

## Immediate Next Batch

다음 배치는 “문서 작성 후 바로 실행 가능한 연구 재부팅 패키지” 기준으로 고정한다.

### Batch 0. Literature matrix and novelty audit

- 최소 12~20편 reference matrix 작성
- bucket별 legacy map 작성
- 우리 주장 후보와 방어 불가능한 주장 분리

목적:
실험에 들어가기 전에 `무엇이 실제 novelty 후보인지`를 잠근다.

### Batch 1. Hard-class generation policy shortlist

- 대상 클래스: `monocyte`, `eosinophil`
- 비교 축: strength x class-specific policy
- 출력: generation-side diagnostic panel

목적:
새 generation policy 후보를 2개 이하로 줄인다.

### Batch 2. Context preservation ablation

- 비교 축: 기존 branch vs contextual branch
- 대상: Batch 1에서 남은 class-policy 조합
- 출력: identity/variation trade-off 비교

목적:
context-preserving preprocessing이 실제 utility 후보를 만드는지 확인한다.

### Batch 3. Selective LODO validation

- held-out 우선순위: `Raabin`, `AMC`
- subset builder는 고정하거나 최소 수정만 허용
- 목적: 새 generation policy가 실제 downstream utility를 만드는지 확인

이 세 배치의 순서를 지킨다.

1. 먼저 literature / novelty gate를 통과하고
2. 그 다음 generation 후보를 줄이고
3. context branch의 필요성을 확인하고
4. 마지막으로 LODO utility를 검증한다.

---

## 결론

다음 사이클의 canonical story는 다음 문장으로 고정한다.

> 우리는 더 많은 synthetic image를 만드는 것이 아니라, hard WBC generalization에 실제로 유용한 synthetic pool을 만드는 generation policy를 찾는다.

이 문장의 실행 의미는 분명하다.

- 메인 조작 변수는 `generation policy`
- 메인 검증 기준은 `LODO utility`
- novelty 검증 기준은 `reference-grounded distinction from legacy`
- `subset/benchmark`은 보조 프레임
- `boundary-aware V2`는 보조 브랜치

따라서 다음 실험은 subset taxonomy를 더 늘리는 것이 아니라, `어떤 generation policy가 utility를 만든다고 말할 수 있는가`를 다시 묻는 방향으로 설계해야 한다.
