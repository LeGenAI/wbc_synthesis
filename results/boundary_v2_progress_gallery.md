# Boundary V2 Progress Gallery

완료된 V2 생성 결과를 육안으로 바로 확인할 수 있게 정리한 갤러리다.

## Monocyte

### Baseline V2
- Ref: `AMC -> MLL23`
- Input: `domain_e_amc_monocyte_000228`
- Setting: `bg=0.65`, `refine=0.20`
- Cell SSIM: `0.9986`
- Background SSIM: `0.9771`
- Target margin: `0.3192`
- Entropy: `0.9098`

Input  
![monocyte baseline input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/monocyte/domain_e_amc_monocyte_000228.jpg)

Generated  
![monocyte baseline output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/monocyte/ref_domain_e_amc/to_domain_c_mll23/bg065/rf020/0000_s0.png)

### Aggressive V2
- Ref: `AMC -> MLL23`
- Input: `domain_e_amc_monocyte_000759`
- Setting: `bg=0.85`, `refine=0.00`
- Cell SSIM: `0.9992`
- Background SSIM: `0.9207`
- Target margin: `0.3989`
- Entropy: `1.0415`

Input  
![monocyte aggressive input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/monocyte/domain_e_amc_monocyte_000759.jpg)

Generated  
![monocyte aggressive output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/monocyte/ref_domain_e_amc/to_domain_c_mll23/bg085/rf000/0000_s0.png)

### Ultra V2
- Ref: `AMC -> MLL23`
- Input: `domain_e_amc_monocyte_000759`
- Setting: `bg=0.95`, `refine=0.00`
- Cell SSIM: `0.9990`
- Background SSIM: `0.8794`
- Target margin: `0.0557`
- Entropy: `1.0448`
- Note: 현재까지 첫 `near-boundary` 샘플

Input  
![monocyte ultra input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/monocyte/domain_e_amc_monocyte_000759.jpg)

Generated  
![monocyte ultra output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/monocyte/ref_domain_e_amc/to_domain_c_mll23/bg095/rf000/0000_s0.png)

## Eosinophil

### Aggressive Low-Margin Sample
- Ref: `AMC -> PBC`
- Input: `domain_e_amc_eosinophil_000759`
- Setting: `bg=0.85`, `refine=0.00`
- Cell SSIM: `0.9995`
- Background SSIM: `0.9468`
- Target margin: `0.5696`
- Entropy: `0.8538`

Input  
![eosinophil low-margin input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/eosinophil/domain_e_amc_eosinophil_000759.jpg)

Generated  
![eosinophil low-margin output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/eosinophil/ref_domain_e_amc/to_domain_a_pbc/bg085/rf000/0000_s0.png)

### Ultra Low-Margin Sample
- Ref: `AMC -> MLL23`
- Input: `domain_e_amc_eosinophil_000759`
- Setting: `bg=0.95`, `refine=0.00`
- Cell SSIM: `0.9995`
- Background SSIM: `0.9205`
- Target margin: `0.3795`
- Entropy: `0.8748`

Input  
![eosinophil ultra input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/eosinophil/domain_e_amc_eosinophil_000759.jpg)

Generated  
![eosinophil ultra output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/eosinophil/ref_domain_e_amc/to_domain_c_mll23/bg095/rf000/0000_s0.png)

## Quick Read

- `monocyte`는 `bg=0.95`에서 배경 변화가 가장 크게 나타났고, 수치상으로도 첫 low-margin 샘플이 나왔다.
- `eosinophil`도 `bg=0.95`에서 더 강한 변형이 가능해졌지만, 아직 `near-boundary` 샘플은 나오지 않았다.
- 이전 결과는 보존되어 있다.
  - [monocyte baseline archive](/Users/imds/Desktop/wbc_synthesis/results/boundary_v2_generation_archive/monocyte_probe_bg065_rf020_n2_20260310_111504)
  - [monocyte aggressive archive](/Users/imds/Desktop/wbc_synthesis/results/boundary_v2_generation_archive/monocyte_probe_bg075_bg085_rf000_n1_20260310_221235)
  - [eosinophil aggressive archive](/Users/imds/Desktop/wbc_synthesis/results/boundary_v2_generation_archive/eosinophil_probe_bg065_rf020_n2_20260310_120006)
