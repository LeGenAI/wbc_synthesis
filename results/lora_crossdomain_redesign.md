# LoRA Cross-Domain Redesign Review

## 목적

현재 WBC 생성 파이프라인을 `안전한 img2img 재합성기`에서 `cross-domain 5-class WBC generator`로 재정의하기 위한 코드 관점 리뷰와 실행계획을 정리한다.

## 핵심 진단

### 1. 현재 생성기는 멀티도메인이지, 진짜 cross-domain 생성기는 아니다

기존 대량 생성 경로에서는 참조 이미지의 도메인과 프롬프트의 도메인 설명이 동일하게 묶여 있었다. 즉 `PBC reference -> PBC prompt`, `AMC reference -> AMC prompt` 구조였고, `PBC reference -> AMC style` 같은 교차 조합은 생성되지 않았다.

이 설계는 도메인별 재합성에는 유리하지만, 논문 메시지인 domain generalization과 morphology-style disentanglement에는 직접적으로 대응하지 못한다.

### 2. 도메인 conditioning 신호가 약하다

학습 캡션의 도메인 정보는 장비명, 국가명, 염색법 문자열 위주인데, 현재 LoRA 학습에서는 text encoder를 고정하고 UNet LoRA만 학습한다. 따라서 드문 도메인 토큰의 의미를 충분히 정교하게 적응시키기 어렵다.

### 3. 클래스 내부 morphology variation을 충분히 쓰지 못한다

현재 클래스별 LoRA는 도메인당 최대 200장을 균등 샘플링하고, 동일 도메인에 속한 모든 이미지에 거의 동일한 캡션을 반복 부여한다. 이는 구현은 단순하지만, monocyte와 neutrophil처럼 intra-class variation이 큰 클래스의 세부 형태를 충분히 반영하기 어렵다.

### 4. 학습 모니터링이 약하다

기존 학습은 모든 클래스에 `400 step`, `rank 8`, `lr 5e-5`를 고정 적용하고, 중간 checkpoint가 사실상 없었으며, validation image도 남기지 않았다. 이 구조에서는 클래스별로 과소학습과 과적합을 구분하기 어렵다.

### 5. 생성 템플릿 일부가 생물학적으로 부정확하다

`cytology` 템플릿이 lymphocyte와 monocyte에도 `granulocyte` 표현을 사용하고 있었다. 이는 생성 단계에서 잘못된 텍스트 prior를 주는 문제였다.

## 이번에 반영한 즉시 수정

### A. 생성 스크립트에 cross-domain mode 추가

[33_diverse_generate.py](/Users/imds/Desktop/wbc_synthesis/scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py)에 다음 모드를 추가했다.

- `same_domain`: 기존 동작 유지
- `cross_only`: 참조 도메인과 다른 도메인 스타일만 사용
- `all_pairs`: same-domain과 cross-domain을 모두 생성

이제 `reference domain`과 `prompt target domain`을 분리해서 진짜 교차 도메인 조합을 만들 수 있다.

### B. cytology 템플릿 수정

클래스별 세포 용어를 분리해 lymphocyte와 monocyte에 잘못된 `granulocyte` 표현이 들어가지 않도록 수정했다.

### C. LoRA 학습 기본값을 더 보수적으로 수정

[10_multidomain_lora_train.py](/Users/imds/Desktop/wbc_synthesis/scripts/legacy/phase_08_17_domain_gap_multidomain/10_multidomain_lora_train.py)에 다음 변경을 적용했다.

- 기본 `checkpointing_steps`: `9999 -> 100`
- 기본 `center_crop`: 활성화
- validation prompt / validation image / validation epoch 옵션 추가

즉, 앞으로는 최소한 중간 checkpoint와 validation image를 남기면서 학습 상태를 볼 수 있다.

## 다음 우선순위

### Priority 1. cross-domain 생성기 성능 확인

먼저 생성기가 실제로 `ref_domain != target_domain`에서도 클래스 정체성을 유지하는지 확인해야 한다.

권장 실험:

```bash
python scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py --class_name monocyte --n_per_domain 2 --n_seeds 1 --cross_domain_mode cross_only
python scripts/legacy/phase_33_40_selective_synth_lodo/33_diverse_generate.py --class_name eosinophil --n_per_domain 2 --n_seeds 1 --cross_domain_mode cross_only
```

여기서 확인할 것:

- CNN class correctness
- target-domain style 변화 여부
- SSIM이 과도하게 높아 복사 수준으로 남는지 여부
- hard class에서 morphology collapse가 일어나는지 여부

### Priority 2. LoRA 재학습의 건강도 개선

다음 배치부터는 학습을 아래처럼 다시 가져가는 것이 좋다.

- `center_crop on` 유지
- `checkpointing_steps=100`
- `enable_default_validation` 또는 명시적 `validation_prompt`
- 클래스별 학습 스텝을 동일값으로 고정하지 말고 `400/800/1200` 비교

권장 1차 재학습 배치:

```bash
python scripts/legacy/phase_08_17_domain_gap_multidomain/10_multidomain_lora_train.py --class_name monocyte --steps 800 --enable_default_validation
python scripts/legacy/phase_08_17_domain_gap_multidomain/10_multidomain_lora_train.py --class_name eosinophil --steps 800 --enable_default_validation
```

monocyte와 eosinophil을 먼저 추천하는 이유는 downstream에서 가장 취약했고, 생성기의 구조적 한계가 가장 쉽게 드러나는 클래스이기 때문이다.

### Priority 3. 도메인 conditioning 자체 강화

현재 구조에서 다음 병목은 text conditioning strength다. 다음 단계에서는 두 방향 중 하나가 필요하다.

1. text encoder LoRA를 제한적으로 켜는 실험
2. 도메인 설명을 긴 자연어 문장 대신 짧은 제어 토큰으로 바꾸는 실험

예시:

- `<pbc_style>`
- `<raabin_style>`
- `<mll23_style>`
- `<amc_style>`

이렇게 바꾸면 UNet만으로도 스타일 신호를 더 일관되게 받을 수 있다.

### Priority 4. per-image morphology captioning

현 구조는 클래스당 morphology 문장이 고정이다. 장기적으로는 이미지별 attribute를 추출해 캡션을 세분화하는 편이 좋다.

예:

- nucleus round vs folded
- granule dense vs sparse
- cytoplasm thin vs abundant
- nucleus eccentric vs centered

이 단계는 가장 효과가 클 수 있지만 구현 비용이 높아, 지금은 후순위다.

## 추천 실행 순서

1. `cross_only` 소규모 smoke test
2. monocyte/eosinophil LoRA 재학습 (`steps=800`, validation on)
3. 재학습 LoRA로 `cross_only` 재생성
4. 기존 evaluator로 quality screening
5. selective augmentation에 투입해 utility 확인

## 판단 기준

아래 3개가 동시에 만족되면 재설계가 성공적이라고 볼 수 있다.

- cross-domain 생성에서도 class correctness가 유지된다
- same-domain 대비 스타일 변화가 분명해진다
- downstream selective augmentation에서 hard-class rescue가 더 커진다

## 보류한 항목

아래 항목은 중요하지만 아직 코드에 바로 반영하지 않았다.

- prior preservation 사용
- text encoder LoRA 기본 활성화
- per-image morphology auto-caption
- class별 adaptive rank
- synthetic utility를 generation-time loss와 직접 연결하는 weighting

이 항목들은 위의 Priority 1~2 결과를 본 뒤 2차 리팩터링에서 다루는 편이 안전하다.
