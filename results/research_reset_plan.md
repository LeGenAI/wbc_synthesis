# WBC Synthesis Research Reset Plan

**작성일:** 2026-03-06  
**목적:** 지금까지의 시행착오를 정리하고, 다음 논문/실험을 위한 재설계안을 확정한다.

---

## 1. 현재까지의 핵심 교훈

### 1-1. 이미 확인된 사실

1. 초기 img2img 생성은 너무 보수적이었다.  
   `strength=0.35` 중심 실험에서는 SSIM이 매우 높고 입력 보존 비율이 과도하게 컸다.

2. `strength`가 가장 강한 제어 변수였다.  
   prompt variation이나 저강도 IP-Adapter보다 `strength` 변화가 구조 변화와 CNN 반응을 훨씬 크게 좌우했다.

3. EfficientNet-B0는 생성 품질 차이를 구분하기에 너무 강했다.  
   이후 VGG16을 도입하고 나서야 조건 간 차이가 보이기 시작했다.

4. prompt diversification은 현재 설정에서 효과가 거의 없었다.  
   고정 프롬프트와 랜덤/순환 템플릿 간 차이가 미미했다.

5. naive synthetic augmentation은 downstream 성능을 올리지 못했다.  
   `real+synth`와 `real+synth_filtered` 모두 `real_only`를 넘지 못했다.

6. 현재 cross-domain 평가 프로토콜은 너무 쉽다.  
   멀티도메인 CNN이 이미 높은 macro-F1를 달성하고 있어 synthetic gain을 입증하기 어렵다.

7. hard class는 일관되게 `monocyte`, `eosinophil`이었다.  
   반면 `lymphocyte`, `neutrophil`은 이미 매우 강해 추가 개선 여지가 작다.

8. 이제 병목은 생성 인프라가 아니라 평가 설계다.  
   `scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py` 수준의 대량 생성은 가능하지만, 그 데이터를 어디에 어떻게 써야 유의미한지가 아직 정리되지 않았다.

### 1-2. 더 이상 메인 라인으로 밀지 않을 것

1. `strength=0.35` 고정 상태에서의 추가 prompt sweep
2. low-IP / adaptive-IP 미세 조정
3. EfficientNet 단독 품질 평가
4. 쉬운 `valsplit` 결과를 메인 성과로 사용하는 것
5. synthetic 전량 투입 방식

---

## 2. 다음 연구의 메인 질문

이번 사이클의 중심 질문은 아래 하나로 고정한다.

> **어떤 synthetic WBC 이미지가, 어떤 어려운 학습 설정에서, 실제 일반화 성능에 도움이 되는가?**

이 질문을 세 개의 하위 질문으로 분해한다.

1. synthetic data는 **언제** 도움이 되는가?  
   - normal setting이 아니라 low-data, class imbalance, unseen-domain에서 도움되는지 본다.

2. synthetic data는 **어떤 조건일 때** 도움이 되는가?  
   - `strength`, class, domain, confidence, filtering 기준별 utility를 본다.

3. synthetic data는 **누구에게** 도움이 되는가?  
   - 전체 평균이 아니라 hard class(`monocyte`, `eosinophil`)와 unseen-domain 성능 개선 여부를 본다.

---

## 3. 새 실험축

### 3-1. Evaluation Reset

기존의 쉬운 benchmark를 버리고 아래 세 가지를 메인 benchmark로 삼는다.

| Benchmark | 설명 | 왜 필요한가 |
|---|---|---|
| `LODO` | 4개 도메인 중 1개를 완전 hold-out | 진짜 domain generalization 확인 |
| `Low-data` | real train 10%, 25%, 50%만 사용 | synthetic의 가치가 커질 환경 확인 |
| `Hard-class imbalance` | monocyte/eosinophil 비율만 인위적으로 축소 | hard class rescue 효과 확인 |

### 3-2. Synthetic Subset Selection

`data/generated_diverse/` 전체를 그대로 넣지 않고, 아래 subset들을 비교한다.

| Subset ID | 구성 |
|---|---|
| `S0` | no synth (`real_only`) |
| `S1` | all synth |
| `S2` | `cnn_correct == True` |
| `S3` | high-confidence only |
| `S4` | `ds025` only |
| `S5` | `ds035` only |
| `S6` | `ds045` only |
| `S7` | monocyte + eosinophil only |
| `S8` | weakest-domain targeted only |
| `S9` | curriculum (`ds025 -> ds035 -> ds045`) |

### 3-3. Model Axis

모델 축은 최소화한다. 너무 많은 backbone 비교는 논문 메시지를 흐린다.

| Model ID | 설명 | 목적 |
|---|---|---|
| `M1` | EfficientNet-B0 multidomain | 기존 강한 기준선 |
| `M2` | VGG16 frozen features | synthetic utility 민감도 측정 |
| `M3` | VGG16 full fine-tune | synthetic를 흡수할 capacity 확인 |

---

## 4. 메인 실험 설계

### Experiment A. Benchmark Reset

**질문:** synthetic 없이도 현재 모델이 실제 어려운 설정에서 얼마나 버티는가?

**설계**

1. `LODO`: train on 3 domains, test on held-out 1 domain
2. `Low-data`: train fraction 10%, 25%, 50%, 100%
3. `Hard-class imbalance`: monocyte/eosinophil train count를 25% 또는 10%로 축소

**출력**

1. domain-wise macro-F1
2. class-wise F1
3. confusion matrix
4. calibration metrics

**목표**

기존 `valsplit`보다 명확히 더 어려운 평가 세팅을 확보한다.  
이 실험은 이후 모든 synthetic augmentation 결과의 기준선이 된다.

### Experiment B. Utility-Aware Synthetic Selection

**질문:** synthetic를 어떻게 골라야 성능이 오르는가?

**설계**

1. `S0`~`S9` subset 비교
2. benchmark는 `LODO`, `Low-data`, `Hard-class imbalance` 우선
3. 모델은 우선 `M2`, `M3`에 집중

**핵심 비교**

1. all synth vs filtered synth
2. low-strength vs high-strength
3. hard-class targeted vs global augmentation
4. curriculum vs static subset

**목표**

`all synth`가 아니라 특정 subset만 유의미하다는 걸 보이는 것.  
논문 메시지는 "utility-aware selection > naive augmentation"으로 잡는다.

### Experiment C. Hard-Class Rescue

**질문:** monocyte/eosinophil에 synthetic가 실제로 도움이 되는가?

**설계**

1. 두 클래스만 synthetic 주입
2. 두 클래스만 비율을 줄인 학습 세팅에서 비교
3. 필요하면 해당 클래스에 대해 `ds045` 비율을 높인 추가 subset 구성

**주요 지표**

1. monocyte recall / F1
2. eosinophil recall / F1
3. 전체 macro-F1 손상 여부

**목표**

전체 평균이 아니라 취약 클래스 구조 개선을 보여주는 보조 실험으로 사용한다.

---

## 5. Success Criteria

### 5-1. 메인 논문용 최소 기준

아래 중 2개 이상을 만족해야 메인 스토리로 밀 수 있다.

1. `LODO`에서 synthetic selection variant가 `real_only`보다 macro-F1 `+1.0%p` 이상 개선
2. monocyte 또는 eosinophil F1가 `+3.0%p` 이상 개선
3. calibration(ECE/NLL)이 `real_only`보다 개선
4. `all synth`는 실패하지만 `selected synth`는 성공하는 패턴이 재현

### 5-2. 실패로 간주할 조건

1. 모든 subset이 `real_only`와 유의미한 차이가 없음
2. hard-class 개선이 전체 성능 손상으로 상쇄됨
3. 개선이 특정 seed에만 나타나고 재현되지 않음

---

## 6. 필요한 코드 작업

### 6-1. 새로 추가할 스크립트

| 예정 파일 | 역할 |
|---|---|
| `scripts/legacy/phase_33_40_selective_synth_lodo/34_lodo_train.py` | leave-one-domain-out 학습 실행기 |
| `scripts/legacy/phase_33_40_selective_synth_lodo/35_diverse_subset_builder.py` | `generated_diverse`에서 subset (`S1`~`S9`) 생성 |
| `scripts/legacy/phase_33_40_selective_synth_lodo/36_selective_synth_train.py` | selective synthetic training runner |
| `scripts/legacy/phase_33_40_selective_synth_lodo/37_calibration_eval.py` | ECE, NLL, Brier, risk-coverage 평가 |

### 6-2. 수정할 기존 스크립트

| 파일 | 수정 방향 |
|---|---|
| `scripts/legacy/phase_08_17_domain_gap_multidomain/17_crossdomain_eval.py` | `LODO` 모드 추가 또는 분리 |
| `scripts/legacy/phase_18_32_generation_ablation/32_synth_aug_train.py` | `generated_diverse` 입력 + subset selector 지원 |
| `scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py` | subset metadata export 강화 |

### 6-3. 산출 디렉토리 계획

| 디렉토리 | 용도 |
|---|---|
| `results/lodo_baseline/` | benchmark reset 결과 |
| `results/selective_synth/` | subset별 학습 결과 |
| `results/calibration_eval/` | calibration 결과 |
| `results/paper_figures/` | 논문용 figure/table 저장 |

---

## 7. 논문용 Figure / Table 구성

### Figure Plan

1. **Figure 1**: 전체 연구 프레임워크  
   real data -> synthetic generation -> subset selection -> hard benchmark evaluation

2. **Figure 2**: 기존 시행착오 요약  
   `strength`, prompt, IP-Adapter, evaluator 변경이 어떤 결론으로 이어졌는지 다이어그램화

3. **Figure 3**: `LODO` 성능 비교  
   `real_only` vs `all synth` vs `selected synth`

4. **Figure 4**: hard-class 성능 변화  
   monocyte/eosinophil F1 bar chart

5. **Figure 5**: synthetic subset utility map  
   subset별 macro-F1 / class-F1 / ECE heatmap

### Table Plan

1. **Table 1**: benchmark 정의
2. **Table 2**: synthetic subset 정의
3. **Table 3**: main results (`LODO`, low-data, imbalance)
4. **Table 4**: hard-class rescue results
5. **Table 5**: calibration results

---

## 8. 3주 실행 로드맵

### Week 1. Benchmark Reset

1. `LODO` split 구현
2. `Low-data` split 구현
3. `Hard-class imbalance` split 구현
4. `real_only` 기준선 재측정

**Go/No-Go 기준**

새 benchmark에서 기존보다 충분히 어려운 분포 차이가 보여야 한다.

### Week 2. Synthetic Subset Builder + Selective Training

1. `generated_diverse` metadata 파싱
2. `S1`~`S9` subset 생성
3. `M2` 또는 `M3`로 selective training 실행
4. 1차 결과 표 작성

**Go/No-Go 기준**

`all synth`와 `selected synth` 간 차이가 보여야 한다.

### Week 3. Hard-Class Rescue + Figure Draft

1. monocyte/eosinophil 집중 실험
2. calibration 결과 추가
3. main tables 확정
4. figure draft 작성

**최종 판정**

메인 메시지를 아래 셋 중 하나로 고정한다.

1. `utility-aware selection`
2. `domain generalization under scarcity`
3. `hard-class rescue`

---

## 9. 최종 권고

이번 사이클의 가장 중요한 태도는 "생성을 더 복잡하게"가 아니라 "평가를 더 어렵게"다.

지금까지의 시행착오는 대부분 생성기 한계보다 **너무 쉬운 benchmark**, **너무 둔한 evaluator**, **너무 무차별적인 synthetic 주입**에서 나왔다.  
다음 실험은 반드시 아래 순서를 지켜야 한다.

1. benchmark를 먼저 바꾼다
2. synthetic subset을 선별한다
3. hard class에서 utility를 본다
4. 마지막에만 생성 파라미터를 다시 건드린다

이 순서를 지키면, 이번에는 "이미지를 만들 수 있다"가 아니라  
"어떤 synthetic data가 실제 WBC generalization에 유효한지"를 논문 메시지로 만들 수 있다.
