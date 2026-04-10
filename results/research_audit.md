# WBC Synthesis Research Audit

## Executive Summary

이 프로젝트의 출발점은 `README.md`와 `scripts/legacy/phase_00_07_initial_pipeline/00~07`에 반영된 것처럼, `single-domain PBC` 기반의 `SDXL-LoRA synthetic augmentation`이 기본 분류 성능과 robustness를 실제로 높일 수 있는지를 검증하는 것이었다. 그러나 실제 코드베이스는 그 이후 최소 세 번 질문이 바뀌었다.

1. `00~07`
   합성 증강 일반론: synthetic augmentation이 성능을 올리는가?
2. `08~17`
   멀티도메인 일반화: 도메인 갭이 얼마나 크며, 멀티도메인 학습이 이를 흡수하는가?
3. `33~37`
   utility-aware selection: synthetic 전체가 아니라 어떤 subset이 어떤 held-out domain에 유용한가?
4. `41~48`
   boundary-aware generation: easy sample이 아니라 boundary-near sample을 만들 수 있는가?

따라서 현재 코드베이스에는 하나의 연구가 직선적으로 발전한 흔적보다, 서로 다른 질문을 가진 연구 브랜치가 겹쳐 있는 상태가 더 강하다. 특히 `33~37`과 `41~48`은 연결은 되어 있지만 동일한 연구 질문의 단순한 다음 단계로 보기 어렵다. 전자는 `downstream utility of selected synth`가 중심이고, 후자는 `generation objective redesign`가 중심이다.

현재 시점의 감사 결론은 아래와 같다.

- `00~07`은 출발점으로서 의미가 있지만, 현재 코드와 결과를 대표하는 canonical story는 아니다.
- `08~17`은 연구가 `single-domain augmentation`에서 `domain generalization`으로 이동한 분기점이다.
- `33~37`은 현재까지 가장 명확한 downstream-positive signal과 논문 메시지 후보를 만든 핵심 phase다.
- `41~48`은 중요한 실패/교훈과 새 생성 가설을 남겼지만, 기존 selective/LODO 성과를 대체하는 메인 노선으로 정착하지는 못했다.

---

## 연구 타임라인 표

| Phase | 시기/축 | 대표 스크립트 | 당시 질문 | 대표 산출물 | 현재 평가 |
|---|---|---|---|---|---|
| P1 | 초기 단일도메인 합성 증강 | `00~07` | SDXL-LoRA 합성 증강이 기본 성능과 robustness를 올리는가? | `README.md`, `results/baseline/`, `results/ablation/` | 출발점으로 보존 |
| P2 | 멀티도메인/도메인갭 인식 | `08~17` | 도메인 갭이 실제로 존재하며, 멀티도메인 학습이 이를 줄일 수 있는가? | `results/cross_domain/`, `results/crossdomain_eval/` | 메인 전환점으로 보존 |
| P3 | 대량 생성 + selective subset | `33~37` | 어떤 synthetic subset이 어떤 held-out domain에 유용한가? | `results/diverse_generation/`, `results/selective_synth/`, `results/lodo_baseline/`, `results/paper_draft_image_heavy.md` | 핵심 성과 축으로 유지 |
| P4 | boundary-aware V2 | `41~48`, `60~61` | boundary-near sample을 생성해 downstream utility를 높일 수 있는가? | `results/boundary_selective_synth/`, `results/boundary_v2_*` | 보조 근거 및 실패/교훈으로 보존 |

---

## 질문 이동 표

| Phase | 명시적 질문 | 실제로 바뀐 질문 | 이동 원인 |
|---|---|---|---|
| P1 | synthetic augmentation이 성능을 올리는가? | 어떤 증강 조합이 baseline보다 나은가? | 초기 proof-of-concept, 단일 도메인 PBC 중심 |
| P2 | cross-domain에서 baseline이 얼마나 무너지는가? | 멀티도메인 학습으로 domain gap을 흡수할 수 있는가? | 도메인 갭 존재를 확인하고 단일도메인 설정의 한계를 인지 |
| P3 | synthetic가 일반적으로 도움이 되는가? | 어떤 subset이 어떤 held-out domain failure mode를 구조적으로 보완하는가? | 쉬운 split 포화, LODO 필요, all-synth 실패 |
| P4 | selective subset을 더 잘 고를 수 있는가? | easy sample이 아니라 boundary-near sample을 직접 생성할 수 있는가? | subset selection만으로는 생성 목표가 바뀌지 않는다는 문제의식 |

---

## Phase 1. 초기 단일도메인 합성 증강 단계

### 당시의 명시적 연구 질문

`README.md`와 `scripts/legacy/phase_00_07_initial_pipeline/00~07`의 구조는 매우 일관적이다.

- 데이터는 `PBC_dataset_normal_DIB_224` 단일 소스다.
- 클래스별 LoRA를 학습하고(`02_train_lora.py`)
- img2img 생성(`03_generate.py`) 후
- 품질 필터링(`04_filter_generated.py`)을 거쳐
- CNN 성능 및 robustness(`05~06`)를 비교한다.

즉 핵심 질문은 다음이었다.

> `SDXL-LoRA synthetic augmentation`이 기본 WBC 분류 성능과 corruption robustness를 실제로 높이는가?

### 사용한 benchmark와 성공 기준

- train/val/test split 기반 일반 분류 성능
- corruption robustness
- ablation A1~A4

문서상 성공 기준은 다음과 같이 비교적 단순했다.

- `Val/Test macro-F1` 상승
- corruption에서 `mean F1 drop` 감소

### 대표 스크립트/결과물

- `scripts/legacy/phase_00_07_initial_pipeline/00_download_data.py`
- `scripts/legacy/phase_00_07_initial_pipeline/01_prepare_data.py`
- `scripts/legacy/phase_00_07_initial_pipeline/02_train_lora.py`
- `scripts/legacy/phase_00_07_initial_pipeline/03_generate.py`
- `scripts/legacy/phase_00_07_initial_pipeline/04_filter_generated.py`
- `scripts/legacy/phase_00_07_initial_pipeline/05_train_cnn.py`
- `scripts/legacy/phase_00_07_initial_pipeline/06_robustness_eval.py`
- `scripts/legacy/phase_00_07_initial_pipeline/07_ablation.py`

### 실제 얻은 결론

이 phase는 프로젝트의 출발점 역할은 분명했지만, 이후 코드와 문서가 보여주듯 메인 결론을 고정하지 못했다.

- synthetic augmentation 일반론 자체는 쉽게 성립하지 않았다.
- naive synthetic augmentation은 이후 문서들에서 반복적으로 `baseline을 안정적으로 넘지 못했다`는 교훈으로 다시 요약된다.
- `07_ablation.py`도 일부 실험이 문서 수준에서만 정의되어 있고, 실제 후속 파이프라인은 이 초기 질문을 그대로 확장하지 않았다.

### 다음 phase로 넘어가게 만든 문제

- 단일 도메인 기준에서는 “진짜 일반화”를 보기 어렵다.
- PBC 기반 설정만으로는 도메인 일반화나 domain shift를 다룰 수 없다.
- synthetic augmentation의 효용 여부보다 먼저 `도메인 갭 자체`를 측정해야 한다는 필요가 커졌다.

### 지금 시점에서의 평가

- 계속 가져갈 것
  초기 문제정의, baseline augmentation 파이프라인, robustness라는 문제의식
- 보조 근거로 남길 것
  class-wise LoRA, filter-on/off, multiplier 등 초기 synthetic 실험축
- 중단할 것
  `single-domain split` 결과를 메인 일반화 증거로 사용하는 해석

---

## Phase 2. 멀티도메인/도메인갭 인식 단계

### 당시의 명시적 연구 질문

`scripts/legacy/phase_08_17_domain_gap_multidomain/08~17`에서 질문이 명확히 바뀐다.

- `08_domain_gap_viz.py`: 도메인 분포 차이 시각화
- `09_cross_domain_baseline.py`: 단일도메인 baseline의 cross-domain 성능 측정
- `10_multidomain_lora_train.py`: 멀티도메인 LoRA
- `16_multidomain_cnn_train.py`: 멀티도메인 5-class CNN
- `17_crossdomain_eval.py`: held-out evaluation

핵심 질문은 다음으로 이동했다.

> 도메인 갭은 실제로 존재하는가, 그리고 멀티도메인 학습이 이를 얼마나 흡수하는가?

### 사용한 benchmark와 성공 기준

- domain gap visualization
- cross-domain held-out evaluation
- multidomain classifier 성능

이 시기에는 synthetic의 직접 효용보다 먼저 `real-data multidomain learning`의 baseline이 중요했다.

### 대표 스크립트/결과물

- `scripts/legacy/phase_08_17_domain_gap_multidomain/08_domain_gap_viz.py`
- `scripts/legacy/phase_08_17_domain_gap_multidomain/09_cross_domain_baseline.py`
- `scripts/legacy/phase_08_17_domain_gap_multidomain/10_multidomain_lora_train.py`
- `scripts/legacy/phase_08_17_domain_gap_multidomain/16_multidomain_cnn_train.py`
- `scripts/legacy/phase_08_17_domain_gap_multidomain/17_crossdomain_eval.py`
- `results/crossdomain_eval/report.md`

### 실제 얻은 결론

이 phase는 중요한 전환점을 만들었지만, 동시에 새로운 문제도 만들었다.

1. 도메인 갭은 실제로 존재했다.
2. 그러나 멀티도메인 학습이 너무 잘 작동하는 구간이 생겼다.
3. `results/crossdomain_eval/report.md` 기준으로 easy split 성능은 너무 높아졌고, 연구 질문이 “synthetic가 도움이 되는가?”에서 “이 benchmark가 synthetic utility를 드러낼 만큼 어려운가?”로 다시 이동했다.

즉 이 단계는 성공이면서도, 이후 benchmark reset의 직접 원인이었다.

### 다음 phase로 넘어가게 만든 문제

- easy split 포화
- EfficientNet 중심 평가 포화
- synthetic gain을 검출하기 어려운 benchmark

### 지금 시점에서의 평가

- 계속 가져갈 것
  `domain gap exists`라는 문제 설정, multidomain data integration
- 보조 근거로 남길 것
  easy held-out / valsplit 결과
- 중단할 것
  이 시기의 높은 F1를 synthetic utility의 핵심 증거로 해석하는 것

---

## Phase 3. 대량 생성 + utility-aware subset 단계

### 당시의 명시적 연구 질문

이 phase에서 연구 질문은 가장 실용적인 형태로 재정의된다.

- `33_diverse_generate.py`: 대량 synthetic pool 생성
- `34_lodo_train.py`: benchmark reset
- `35_diverse_subset_builder.py`: utility-aware subset 설계
- `36_selective_synth_train.py`: subset 주입 학습
- `37_make_paper_figures.py`: 논문용 figure 정리

핵심 질문은 다음이다.

> synthetic 전체가 아니라, 어떤 subset이 어떤 held-out domain failure mode에 유용한가?

### 사용한 benchmark와 성공 기준

- 메인 benchmark: `LODO`
- 보조 세팅: low-data, hard-class imbalance 가능성
- held-out domain별 macro-F1와 class-wise F1

이 phase의 중요한 특징은 `benchmark reset`을 먼저 했다는 점이다. `results/lodo_baseline/efficientnet_b0/summary.md`는 평균 Macro-F1가 `0.5826`으로, synthetic utility를 볼 수 있는 난이도를 확보했다.

### 대표 스크립트/결과물

- `scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py`
- `scripts/legacy/phase_33_40_selective_synth_lodo/34_lodo_train.py`
- `scripts/legacy/phase_33_40_selective_synth_lodo/35_diverse_subset_builder.py`
- `scripts/legacy/phase_33_40_selective_synth_lodo/36_selective_synth_train.py`
- `scripts/legacy/phase_33_40_selective_synth_lodo/37_make_paper_figures.py`
- `results/selective_synth/summary.md`
- `results/lodo_followup_plan.md`
- `results/paper_draft_image_heavy.md`

### 실제 얻은 결론

이 phase는 현재까지 가장 명확한 downstream-positive signal을 남겼다.

1. `all synth`나 generic clean filtering은 일관된 답이 아니었다.
2. best subset은 held-out domain마다 달랐다.
3. `AMC`, `Raabin`, `PBC`, `MLL23`는 서로 다른 synthetic policy를 필요로 했다.
4. synthetic augmentation은 `quantity maximization`이 아니라 `utility-aware, domain-specific subset selection` 문제라는 강한 메시지가 형성되었다.

특히 문서상 가장 논문에 가까운 메시지는 이 phase에서 완성되었다.

### 다음 phase로 넘어가게 만든 문제

이 phase가 실패해서가 아니라, 새로운 불만족이 생겼기 때문에 다음 단계로 이동했다.

- subset builder는 생성된 pool을 고르는 도구일 뿐, 생성 목표 자체를 바꾸지 못한다.
- high-confidence / correct 중심 필터는 결국 easy sample을 선호한다.
- hard class rescue는 보였지만, 생성 정책이 downstream objective에 직접 맞춰져 있지는 않았다.

### 지금 시점에서의 평가

- 계속 가져갈 것
  `LODO`, utility-aware evaluation, domain-specific failure mode 분석
- 보조 근거로 남길 것
  subset taxonomy 자체, domain-best subset map
- 중단할 것
  subset builder만 계속 정교화하면 본질적 생성 문제가 해결될 것이라는 기대

---

## Phase 4. boundary-aware V2 단계

### 당시의 명시적 연구 질문

`results/boundary_aware_generation_redesign.md`는 문제의식을 명확히 적고 있다.

- easy, high-confidence sample을 더 잘 고르는 것은 support-vector-like augmentation 목표와 다르다.
- 중앙 세포는 보존하고 배경/도메인 variation을 더 강하게 주면서
- 분류기 경계 근처 샘플을 만들고 싶다는 요구가 생겼다.

그래서 질문은 다음으로 다시 이동한다.

> easy sample이 아니라 boundary-near synthetic sample을 직접 만들 수 있는가?

### 사용한 benchmark와 성공 기준

- generation-side metrics
  `cell_ssim`, `background_ssim`, `region_gap`, `target_margin`, `near_boundary_rate`
- boundary subset manifest
- boundary LODO wrapper
- acceptance review

### 대표 스크립트/결과물

- `scripts/legacy/phase_41_61_boundary_v2/41_boundary_aware_variation_review.py`
- `scripts/legacy/phase_41_61_boundary_v2/42_preprocess_contextual_multidomain.py`
- `scripts/legacy/phase_41_61_boundary_v2/43_build_contextual_masks.py`
- `scripts/legacy/phase_41_61_boundary_v2/44_contextual_lora_train.py`
- `scripts/legacy/phase_41_61_boundary_v2/45_background_aware_generate.py`
- `scripts/legacy/phase_41_61_boundary_v2/46_boundary_subset_builder.py`
- `scripts/legacy/phase_41_61_boundary_v2/47_boundary_lodo_train.py`
- `scripts/legacy/phase_41_61_boundary_v2/48_boundary_acceptance_review.py`
- `scripts/legacy/phase_41_61_boundary_v2/60_build_hybrid_manifest.py`
- `scripts/legacy/phase_41_61_boundary_v2/61_slice_manifest.py`
- `results/boundary_v2_implementation.md`
- `results/boundary_selective_synth/summary.md`
- `results/boundary_v2_acceptance/review.md`

### 실제 얻은 결론

이 phase는 생성 연구로서는 유의미한 자산을 남겼다.

1. preprocessing, mask, contextual LoRA, background-aware generation이라는 새 생성 가설을 실제 코드로 옮겼다.
2. `monocyte` 중심으로 low-margin 샘플 일부를 수확하는 데는 성공했다.
3. `B2`, `H1`, `H2`, `H3` 같은 하이브리드 manifest가 생겼고, selective synth 위에 재주입하는 구조도 만들었다.

그러나 메인 노선으로 정착했다고 보기 어려운 이유도 분명하다.

1. 실제 downstream-positive signal은 아직 `33~37`만큼 설득력 있게 정리되지 않았다.
2. `47_boundary_lodo_train.py`와 `48_boundary_acceptance_review.py`는 최신 하이브리드 실험 전체를 canonical summary에 안정적으로 반영하지 못한다.
3. 질문의 중심이 `downstream utility`에서 `generation principle and boundary scoring`으로 이동했다.

즉 이 phase는 “기존 selective/LODO 성과의 자연스러운 다음 장”이라기보다, 별도의 생성 가설 브랜치를 강하게 민 결과에 가깝다.

### 다음 phase로 넘어가게 만든 문제

아직 다음 phase는 공식적으로 정리되지 않았다. 하지만 현재 상태의 문제는 명확하다.

- boundary-aware V2는 아직 메인 논문 메시지가 아니다.
- 최신 브랜치가 기존 strongest downstream story를 대체하지도 못했다.
- 결과적으로 코드베이스의 `canonical question`이 사라졌다.

### 지금 시점에서의 평가

- 계속 가져갈 것
  contextual preprocessing, mask-based analysis, generation objective를 다시 생각한 시도
- 보조 근거로 남길 것
  boundary-aware V2 전체, hybrid manifest 실험, monocyte-focused harvest 결과
- 중단할 것
  `47/48`을 현재 프로젝트의 자동적 canonical endpoint로 간주하는 해석

---

## 방향 상실 진단

### 1. 질문이 한 번이 아니라 여러 번 바뀌었다

- `00~07`은 합성 증강의 일반적 효용
- `08~17`은 도메인 갭과 멀티도메인 일반화
- `33~37`은 utility-aware subset selection
- `41~48`은 boundary-aware generation objective

질문이 바뀐 것 자체는 문제가 아니다. 문제는 각 전환이 문서, 코드, 결과에 동시에 canonical하게 반영되지 않았다는 점이다.

### 2. strongest downstream story와 latest generation branch가 분리되었다

현재까지 가장 선명한 downstream story는 `LODO + selective subset`이다. 반면 최신 코드 투자량은 `boundary-aware V2`에 더 많이 들어갔다. 이 둘이 완전히 이어지지 않으면서, “우리는 무엇을 증명하려는가?”가 흐려졌다.

### 3. README와 현재 코드베이스의 story가 다르다

`README.md`는 초기 `00~07`을 프로젝트 전체 story처럼 설명한다. 하지만 실제 연구의 중심 자산은 `33~37`과 일부 `41~48`에 있다. 지금 상태에서 외부인이 README만 읽으면 현재 프로젝트의 핵심을 오해하게 된다.

---

## 괴리 표

| 영역 | 문서상 주장 | 코드상 구현 | 실제 최신 상태의 괴리 |
|---|---|---|---|
| 프로젝트 메인 스토리 | `README.md`는 `00~07`의 단일도메인 synthetic augmentation을 메인처럼 설명 | 실제 핵심 자산은 `33~37`과 `41~48`에 분산 | 문서와 현재 연구 중심이 다름 |
| benchmark 인식 | 초기 문서는 val/test 및 robustness 중심 | `34_lodo_train.py` 이후 LODO가 사실상 canonical benchmark가 됨 | canonical benchmark가 README에 반영되지 않음 |
| 논문 메시지 | `paper_draft_image_heavy.md`는 domain-specific subset selection을 강하게 주장 | `41~48`은 boundary-aware generation으로 새 질문을 던짐 | 한 저장소 안에 최소 2개의 논문 메시지가 공존 |
| 최신 결과 요약 | `48_boundary_acceptance_review.py`는 boundary branch의 canonical review처럼 보임 | 실제 최신 hybrid manifest 전체를 포괄적으로 요약하지 않음 | latest branch조차 내부 summary 체계가 완전히 닫히지 않음 |
| 연구 축의 해석 | `47/48`은 selective/LODO의 자연스러운 다음 단계처럼 보일 수 있음 | 실제로는 generation objective 자체를 재설계한 별도 브랜치 | 같은 연구의 연장선으로 보기 어렵다 |

---

## 선행연구 / legacy 관점에서 본 현재 위치

이 감사서는 내부 코드 히스토리만 정리하는 문서로 끝나면 부족하다. 이후 RQ는 반드시 “기존 선행연구와 legacy 대비 무엇이 다른가”를 기준으로 세워져야 하므로, 현재 위치를 외부 축에서도 진단해야 한다.

### 1. WBC synthetic augmentation legacy와의 관계

WBC synthetic augmentation 계열의 대표 legacy는 대체로 다음 질문을 다뤘다.

- synthetic image를 만들 수 있는가
- class imbalance를 완화할 수 있는가
- in-domain accuracy를 올릴 수 있는가

이 기준에서 보면:

- `00~07`은 외부 legacy와 가장 유사한 내부 브랜치다.
- `33~37`은 여기서 한 단계 나아가 `which synthetic subset helps which held-out domain`이라는 utility 문제로 이동했다.
- `41~48`은 다시 `어떤 생성 objective가 utility에 더 맞는 sample을 만드는가`로 이동했지만, 아직 그 novelty를 외부 legacy 대비 정리하지는 못했다.

즉 우리 저장소는 이미 단순 `dataset expansion` 프레임을 넘어가려 했지만, 그 차별점이 공식 문서에서 방어 가능한 형태로 정리되지는 않았다.

### 2. WBC domain shift / DG legacy와의 관계

WBC under domain shift 계열의 legacy는 보통 다음에 집중한다.

- feature alignment
- domain adaptation / domain generalization
- representation learning

이 기준에서 보면:

- `08~17`은 domain gap과 multidomain baseline을 구축하는 단계로 의미가 있다.
- `33~37`은 이 benchmark 위에 synthetic utility를 얹으며, alignment가 아니라 `selection under held-out failure mode`를 중심 질문으로 삼았다.
- 따라서 내부 strongest story는 “DG를 한다”가 아니라 “DG setting에서 synthetic utility를 selection 문제로 다시 묻는다”에 있다.

이 점은 이후 RQ를 설계할 때 중요하다. `domain generalization을 다룬다`는 사실만으로는 novelty가 되지 않는다.

### 3. 일반 medical image augmentation / DG legacy와의 관계

일반 medical image DG/augmentation legacy는 대체로 다음 형태다.

- style augmentation
- source-domain diversification
- latent/feature regularization

이 기준에서 보면:

- `41~48`의 contextual branch는 외형적으로는 style/context augmentation 계열과 닿아 있다.
- 하지만 현재 문서에는 이것이 기존 style augmentation 계열과 어떻게 다른지, 또는 실제로 더 강한 downstream utility를 주는지에 대한 비교 프레임이 없다.

즉 `boundary-aware V2`는 내부적으로는 큰 투자였지만, 외부 선행 대비 어디가 새로운지 아직 말로 잠겨 있지 않다.

### 4. 내부 legacy 대비 초과분

외부 선행보다 먼저, 내부 legacy 대비 무엇이 달라졌는지도 명시해야 한다.

| 내부 legacy | 이미 한 것 | 아직 없는 것 |
|---|---|---|
| `00~07` | synthetic augmentation 기본 파이프라인 | unseen-domain utility framing |
| `33~37` | selective utility, domain-best subset map | generation policy 자체를 novelty 축으로 정리 |
| `41~48` | generation objective redesign, contextual branch, mask 기반 분석 | 외부 legacy 대비 novelty 문장과 재현 가능한 downstream 우위 |

이 표가 의미하는 바는 명확하다.

- `33~37`의 초과분은 `selection + LODO utility`
- `41~48`의 초과분 후보는 `generation policy redesign`
- 하지만 후자는 아직 `후보`일 뿐, 문헌 검토와 baseline 비교 없이는 `claim`이 아니다

---

## 감사가 드러낸 핵심 공백

현재 프로젝트가 방향성을 잃은 이유는 단지 질문이 여러 번 바뀌었기 때문만은 아니다. 아래 공백들이 동시에 존재하기 때문이다.

### 1. reference-grounded novelty 문장이 없다

우리는 내부적으로는 “질문이 바뀌었다”는 것을 알고 있지만, 외부 선행연구 대비

- 어디를 반복하고
- 어디서 갈라지고
- 무엇을 넘어서려는지

를 공식 문장으로 정리해 둔 적이 없다.

### 2. strongest downstream story와 strongest novelty candidate가 다르다

- strongest downstream story: `33~37`
- strongest novelty candidate: `41~48`

이 둘이 다르기 때문에, 논문 메시지를 잡을 때 내부적으로도 흔들리게 된다.

### 3. literature matrix가 없다

현재 저장소에는 planning memo와 draft는 많지만, 실제로 다음을 한 표에서 비교한 문서는 없다.

- WBC augmentation legacy
- WBC DG/DA legacy
- medical imaging DG augmentation legacy
- 우리 내부 브랜치

이 matrix가 없으면 `To our best knowledge`류 표현은 방어가 어렵다.

---

## RQ 설계를 위한 감사 산출

이 감사서가 이후 RQ 문서에 강제해야 하는 조건은 아래와 같다.

### 1. RQ는 반드시 legacy baseline을 포함해야 한다

다음 RQ는 단순히 “무엇을 실험할까”가 아니라,

- 어떤 legacy family를 넘어서려는가
- 그 family의 대표 한계는 무엇인가

를 함께 적어야 한다.

### 2. novelty는 문제 재정의 수준에서 먼저 서술해야 한다

현재 상태에서 가장 안전한 novelty 축은 아래 둘 중 하나다.

- `selection problem`으로의 재정의
- `generation policy design problem`으로의 재정의

반대로 아래는 바로 claim하면 위험하다.

- “최초의 WBC diffusion augmentation”
- “최초의 WBC DG”
- “최초의 boundary-aware medical generation”

### 3. Batch 0은 literature review여야 한다

다음 실험 배치보다 먼저 필요한 것은 focused reference matrix다. 감사 결과상, 이 step 없이 곧바로 새 generation policy를 밀면 다시 내부 논리와 외부 novelty 서술이 분리될 가능성이 높다.

### 4. 이후 문서는 strongest story와 strongest candidate를 분리해서 써야 한다

다음 RQ 문서와 논문 프레임에서는 반드시 아래를 구분해야 한다.

- `현재 strongest evidence`
- `다음 strongest novelty candidate`

현재 감사 기준으로는:

- strongest evidence = `33~37`
- strongest novelty candidate = `41~48` 또는 그 후속 generation redesign

---

## 남길 자산 / 버릴 자산 / 보류 자산

### 남길 자산

- `LODO` benchmark와 held-out failure mode 분석
- domain gap 인식과 multidomain real baseline
- class-specific generation policy라는 아이디어
- contextual preprocessing / mask / cell-vs-background 분리 시도
- hard class가 일관되게 `monocyte`, `eosinophil`이라는 교훈

### 버릴 자산

- 쉬운 split에서의 높은 점수를 메인 성과로 해석하는 방식
- synthetic 전체를 많이 넣으면 좋아질 것이라는 기본 가정
- subset builder만 정교화하면 생성 목표도 자동으로 좋아질 것이라는 기대
- boundary-aware V2를 아직 검증되지 않은 상태에서 canonical endpoint로 취급하는 해석

### 보류 자산

- `boundary-aware V2` 전체
- `H1/H2/H3` hybrid manifest
- target-margin / region-gap 기반 scoring
- class-specific safe-zone generation policy

보류 자산은 실패가 아니라, 현재 story 안에서의 위치가 아직 확정되지 않은 자산으로 본다.

---

## 결론

이 프로젝트의 핵심 문제는 “실험이 부족하다”가 아니라 “canonical question이 사라졌다”는 데 있다. 지금 필요한 것은 새로운 실험을 바로 더하는 것이 아니라, 아래 세 문장을 먼저 고정하는 일이다.

1. 우리는 무엇을 증명하려는가?
2. 어떤 결과를 메인 증거로 채택할 것인가?
3. 어떤 브랜치는 메인 노선이 아니라 보조 근거 또는 실패/교훈으로 둘 것인가?

현재 감사 결론 기준으로는 다음이 가장 타당하다.

- 메인 성과 축: `33~37`의 `LODO + utility-aware subset` 단계
- 메인 재시작 방향: `generation policy`를 다시 주 변수로 삼되, `LODO/selective`에서 배운 utility 기준을 제약조건으로 유지
- 보조 브랜치: `41~48`의 boundary-aware V2

즉, 앞으로의 리셋은 `00~07`로의 단순 회귀가 아니라, `P3의 downstream 교훈을 안고 generation 쪽으로 다시 돌아가는 것`이어야 한다.

추가로, 이 리셋은 반드시 `선행연구 대비의 위치`를 먼저 잠근 뒤 진행해야 한다. 즉 다음 단계는 “실험 착수”가 아니라 아래 순서여야 한다.

1. literature / legacy matrix 작성
2. strongest evidence와 strongest novelty candidate 분리
3. 그 다음에 canonical RQ 확정
4. 마지막으로 generation policy 실험 시작
