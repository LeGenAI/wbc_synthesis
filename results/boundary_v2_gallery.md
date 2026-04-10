# Boundary V2 Gallery

완료된 `boundary_v2_generation` probe에서 background SSIM이 가장 낮았던 샘플들을 기준으로 정리한 입력-출력 갤러리다.

## Monocyte

### Sample M1
- Ref: `AMC -> PBC`
- Cell SSIM: `0.9987`
- Background SSIM: `0.9728`
- Target margin: `0.9096`
- Entropy: `0.3567`

Input  
![M1 input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/monocyte/domain_e_amc_monocyte_000142.jpg)

Generated  
![M1 output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/monocyte/ref_domain_e_amc/to_domain_a_pbc/bg065/rf020/0001_s0.png)

### Sample M2
- Ref: `AMC -> MLL23`
- Cell SSIM: `0.9987`
- Background SSIM: `0.9732`
- Target margin: `0.8989`
- Entropy: `0.3884`

Input  
![M2 input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/monocyte/domain_e_amc_monocyte_000142.jpg)

Generated  
![M2 output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/monocyte/ref_domain_e_amc/to_domain_c_mll23/bg065/rf020/0001_s0.png)

### Sample M3
- Ref: `AMC -> Raabin`
- Cell SSIM: `0.9987`
- Background SSIM: `0.9752`
- Target margin: `0.9010`
- Entropy: `0.3822`

Input  
![M3 input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/monocyte/domain_e_amc_monocyte_000142.jpg)

Generated  
![M3 output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/monocyte/ref_domain_e_amc/to_domain_b_raabin/bg065/rf020/0001_s0.png)

### Sample M4
- Ref: `AMC -> MLL23`
- Cell SSIM: `0.9986`
- Background SSIM: `0.9771`
- Target margin: `0.3192`
- Entropy: `0.9098`

Input  
![M4 input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/monocyte/domain_e_amc_monocyte_000228.jpg)

Generated  
![M4 output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/monocyte/ref_domain_e_amc/to_domain_c_mll23/bg065/rf020/0000_s0.png)

## Eosinophil

### Sample E1
- Ref: `AMC -> MLL23`
- Cell SSIM: `0.9983`
- Background SSIM: `0.9659`
- Target margin: `0.8634`
- Entropy: `0.4669`

Input  
![E1 input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/eosinophil/domain_e_amc_eosinophil_000142.jpg)

Generated  
![E1 output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/eosinophil/ref_domain_e_amc/to_domain_c_mll23/bg065/rf020/0000_s0.png)

### Sample E2
- Ref: `AMC -> PBC`
- Cell SSIM: `0.9983`
- Background SSIM: `0.9660`
- Target margin: `0.8603`
- Entropy: `0.4713`

Input  
![E2 input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/eosinophil/domain_e_amc_eosinophil_000142.jpg)

Generated  
![E2 output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/eosinophil/ref_domain_e_amc/to_domain_a_pbc/bg065/rf020/0000_s0.png)

### Sample E3
- Ref: `AMC -> PBC`
- Cell SSIM: `0.9980`
- Background SSIM: `0.9664`
- Target margin: `0.5297`
- Entropy: `0.9206`

Input  
![E3 input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/eosinophil/domain_e_amc_eosinophil_000754.jpg)

Generated  
![E3 output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/eosinophil/ref_domain_e_amc/to_domain_a_pbc/bg065/rf020/0001_s0.png)

### Sample E4
- Ref: `AMC -> Raabin`
- Cell SSIM: `0.9983`
- Background SSIM: `0.9690`
- Target margin: `0.8022`
- Entropy: `0.5541`

Input  
![E4 input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/eosinophil/domain_e_amc_eosinophil_000142.jpg)

Generated  
![E4 output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/eosinophil/ref_domain_e_amc/to_domain_b_raabin/bg065/rf020/0000_s0.png)

## Quick Read

- 두 클래스 모두 배경 변화는 있긴 하지만 `background SSIM`이 여전히 `0.96~0.98`대라 육안상 큰 차이가 나지 않는다.
- `cell SSIM`은 거의 `0.998`로 고정이라 중심 세포는 지나치게 보존되고 있다.
- `M4`, `E3`처럼 margin이 상대적으로 낮은 샘플도 여전히 결정경계 근처라고 부르기엔 부족하다.
