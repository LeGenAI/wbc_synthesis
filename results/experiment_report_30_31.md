# WBC 합성 실험 종합 리포트: Script 30 / 30b / 31

**작성일**: 2026-02-25
**목적**: EfficientNet-B0의 포화 문제 해결을 위한 VGG16 전환 + 프롬프트 다양화 효과 검증

---

## 배경 — Script 28 결과의 문제점

Script 28 (ip_scale sweep: 0.0 → 0.20) 완료 후 발견:

| 문제 | 원인 | 영향 |
|------|------|------|
| 전 조건 CNN acc = 99% | EfficientNet-B0 F1=0.9836로 너무 강력 | IP-Adapter 효과 탐지 불가 |
| Inter SSIM = 0.46 (조건 무관) | strength=0.35로 입력 65% 그대로 보존 | 다양성 변화 측정 불가 |

→ **방향 전환**: ① VGG16으로 더 민감한 평가 분류기 학습, ② 프롬프트 다양화로 다양성 증가 시도

---

## Script 30: VGG16 멀티도메인 분류기 학습

### 설계 요점

| 항목 | EfficientNet-B0 (Script 16) | VGG16 (Script 30) |
|------|:---:|:---:|
| 파라미터 수 | 4.0M | 134.3M |
| 학습 방식 | 전체 fine-tune | features 동결 + classifier만 |
| 학습 시간 | ~30분 | ~80분 (MPS) |
| 출력 모델 | `models/multidomain_cnn.pt` | `models/multidomain_cnn_vgg16.pt` |

### 학습 결과

| 지표 | EfficientNet-B0 | VGG16 |
|------|:---:|:---:|
| **val macro-F1** | 0.9836 | **0.8846** |
| val acc | ~98% | 92.6% |
| Best epoch | - | 28/30 |
| 도메인별 acc | PBC=~100% | PBC=91.2%, Raabin=92.4%, MLL23=91.4%, AMC=91.0% |

> VGG16 F1=0.8846 < EfficientNet F1=0.9836 → VGG16이 더 민감한 평가 도구로서 적합

---

## Script 30b: VGG16으로 Script 28 이미지 재평가 (1500장)

### 목적
EfficientNet에서 전부 99%였던 생성 이미지를 VGG16으로 다시 평가 → 민감도 차이 확인

### 조건별 비교

| 조건 | 설명 | EfficientNet acc | VGG16 acc | Δ |
|------|------|:---:|:---:|:---:|
| **A** | 기준선 (ip=0.0) | 99.0% | 90.0% | -9.0%p |
| **B1** | ip=0.05 | 99.0% | 90.0% | -9.0%p |
| **B2** | ip=0.10 | 99.3% | 89.7% | -9.7%p |
| **B3** | ip=0.15 | 99.3% | 89.3% | -10.0%p |
| **C** | ip=0.20 | 99.3% | 89.3% | -10.0%p |

### 클래스별 VGG16 acc (조건별)

| 클래스 | A | B1 | B2 | B3 | C |
|--------|:---:|:---:|:---:|:---:|:---:|
| basophil | 🟩98% | 🟩98% | 🟩98% | 🟩98% | 🟩98% |
| eosinophil | 🟨85% | 🟨85% | 🟨85% | 🟨83% | 🟨85% |
| lymphocyte | 🟩100% | 🟩100% | 🟩100% | 🟩100% | 🟩100% |
| monocyte | 🟨70% | 🟨70% | 🟨68% | 🟨68% | 🟨68% |
| neutrophil | 🟩97% | 🟩97% | 🟩97% | 🟩97% | 🟩95% |

> 🟩 ≥90%: 정상 | 🟨 67~90%: 주의 | 🟥 <67%: 불량

### 해석
- **VGG16이 실제로 더 민감**: EfficientNet 99% → VGG16 89~90% (약 9~10%p 낮음)
- 그러나 **IP-Adapter 조건간 차이가 여전히 적음** (0.7~1.0%p 수준): ip_scale 0.05→0.20 변화가 VGG16에서도 구별 불가
- 취약 클래스: **monocyte (70%)**, eosinophil (83~85%) → 생성 품질 개선 대상

---

## Script 31: 프롬프트 다양화 실험 (900장)

### 4가지 프롬프트 템플릿

| # | 관점 | 핵심 키워드 |
|---|------|------------|
| 0 | 기존 (기준선) | `microscopy image`, `peripheral blood smear`, `sharp focus` |
| 1 | 배율 강조 | `100x oil immersion microscopy`, `leukocyte`, `high-resolution` |
| 2 | 병리학 | `clinical hematology`, `bright-field microscopy`, `nuclear morphology` |
| 3 | 세포학 | `cytology preparation`, `granulocyte`, `blood film`, `clinical diagnostic` |

### 3가지 비교 조건

| 조건 | 전략 | CNN acc | SSIM | Inter SSIM |
|------|------|:---:|:---:|:---:|
| **A** | 고정 프롬프트 (기준선) | **87.7%** | 0.9854 | 0.3717 |
| **B** | 매 생성마다 랜덤 템플릿 | 87.3% | 0.9861 | 0.3713 |
| **C** | inp_idx 순환 (재현성 보장) | **87.7%** | 0.9861 | 0.3712 |

### 클래스별 CNN acc (VGG16 기준)

| 클래스 | A (고정) | B (랜덤) | C (순환) |
|--------|:---:|:---:|:---:|
| basophil | 🟩93% | 🟩92% | 🟩93% |
| eosinophil | 🟨83% | 🟨78% | 🟨80% |
| lymphocyte | 🟩100% | 🟩100% | 🟩100% |
| monocyte | 🟨70% | 🟨72% | 🟨70% |
| neutrophil | 🟩92% | 🟩95% | 🟩95% |

### 해석

**프롬프트 다양화의 효과 없음 (예상과 다름):**

1. **CNN acc**: A ≈ B ≈ C (±0.4%p) → 프롬프트 변형이 클래스 보존율에 영향 없음
2. **Inter SSIM**: A=0.3717 vs B=0.3713 vs C=0.3712 → **차이 0.0005 이내** → 다양성 불변
3. **SSIM**: 모두 0.985~0.986 → 입력 보존율도 동일

**근본 원인 (재확인):**
```
출력 다양성 ≈ 입력 이미지 다양성(65%) × seed 다양성(35%)
                        ↑
              strength=0.35가 입력을 65% 그대로 보존
              → 프롬프트 변형은 나머지 35%에만 영향
              → 다양성 기여가 미미함
```

---

## 전체 실험 흐름 요약

```
Script 25 (다중입력×seed)
→ Inter SSIM=0.46, CNN acc=99% (EfficientNet 포화)

Script 28 (IP-Adapter sweep ip=0.05~0.20)
→ Inter SSIM≈0.46 (변화 없음), CNN acc=99% (여전히 포화)

Script 30 (VGG16 분류기)
→ F1=0.8846 (EfficientNet 0.9836보다 낮아 더 민감)

Script 30b (VGG16으로 재평가)
→ VGG16 acc=89~90% (EfficientNet 99%에서 하락 확인)
→ 그러나 조건간 차이 <1%p → IP-Adapter 효과 여전히 미미

Script 31 (프롬프트 다양화)
→ Inter SSIM=0.371 (조건 무관, Script 28과 비교시 오히려 낮음)
→ CNN acc=87.3~87.7% (조건 무관)
→ 프롬프트 변형은 다양성에 기여하지 않음
```

---

## 취약 클래스 분석

VGG16 기준 일관되게 낮은 두 클래스:

| 클래스 | Script 30b (A 기준) | Script 31 (A 기준) | 개선 방향 |
|--------|:---:|:---:|----------|
| **monocyte** | 70% | 70% | LoRA 추가 학습 또는 strength 조정 |
| **eosinophil** | 85% | 83% | 참조 이미지 품질 개선 필요 |
| basophil | 98% | 93% | 양호 |
| lymphocyte | 100% | 100% | 양호 |
| neutrophil | 97% | 92% | 양호 |

---

## 결론 및 다음 단계 제언

### 현재까지 파악된 핵심 사실

| # | 발견 | 시사점 |
|---|------|--------|
| 1 | `strength=0.35`에서 입력 65% 보존 → IP-Adapter/프롬프트로 다양성 추가 불가 | **strength 자체를 높이거나 완전 text2img 전환** 필요 |
| 2 | EfficientNet-B0 acc=99% → VGG16 acc=89% → 생성 품질 차이 10%p 발견 | 평가 분류기로 **VGG16 사용 권장** |
| 3 | 조건간 VGG16 acc 차이 <1%p → IP-Adapter(저강도)는 품질 손상도 없지만 다양성도 없음 | IP-Adapter **효과 없음** |
| 4 | monocyte 지속 70%, eosinophil 83% | 해당 클래스 **LoRA 재학습** 또는 학습 데이터 보강 |

### 다음 실험 방향 (우선순위 순)

1. **strength 범위 재탐색** (0.50~0.75): Inter SSIM 감소 vs CNN acc 유지 임계점 확인
2. **monocyte/eosinophil LoRA 강화**: steps 증가(400→800) 또는 full finetune 재학습
3. **Text-to-Image 생성 비교**: img2img 의존성 없이 순수 프롬프트 기반 생성의 다양성 측정
