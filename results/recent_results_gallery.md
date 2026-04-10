# Recent Result Gallery

최근 결과 중 논문에 바로 넣을 만한 downstream 비교와, boundary-aware V2에서 실제로 살아남은 ranked 샘플 6장을 한 파일로 정리했다.

## Downstream Snapshot

| Setting | Held-out | Synth Train | Macro-F1 | Acc | Monocyte F1 | Eosinophil F1 | Note |
|---|---|---:|---:|---:|---:|---:|---|
| `S7` | Raabin | `792` | `0.7655` | `0.8994` | `0.7566` | `0.6700` | current best |
| `H1 = S7 + B2` | Raabin | `794` | `0.6975` | `0.8330` | `0.7931` | `0.5980` | monocyte stronger, overall lower |
| `H2 = S7 + Raabin boundary 4` | Raabin | `792` | `0.6136` | `0.7711` | `0.5951` | `0.5101` | below `S7` |
| `H3 = S7 + Raabin boundary top2` | Raabin | `792` | `0.6136` | `0.7711` | `0.5951` | `0.5101` | same as `H2` |
| `S7` | AMC | `792` | `0.7231` | `0.8754` | `0.5758` | `0.8438` | current best |
| `B2` | AMC | `6` | `0.5894` | `0.8432` | `0.6209` | `0.3850` | exploratory, unstable |

## Boundary Pool Snapshot

- Source images: `1073`
- Eligible low-margin images: `18`
- Ranked images: `6`
- Incorrect near-miss: `8`
- Current pattern: `MLL23 -> Raabin` monocyte 축이 가장 생산적

## Ranked Boundary Samples (`B2`)

### 1. MLL23 -> Raabin
- Margin: `0.0636`
- Background SSIM: `0.8593`
- Cell SSIM: `0.9985`
- Background strength: `0.95`

Input  
![ranked-1-input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_c_mll23/monocyte/domain_c_mll23_monocyte_000396.jpg)

Generated  
![ranked-1-output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/monocyte/ref_domain_c_mll23/to_domain_b_raabin/bg095/rf000/0011_s0.png)

### 2. MLL23 -> Raabin
- Margin: `0.1870`
- Background SSIM: `0.8421`
- Cell SSIM: `0.9985`
- Background strength: `0.95`

Input  
![ranked-2-input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_c_mll23/monocyte/domain_c_mll23_monocyte_000383.jpg)

Generated  
![ranked-2-output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2_runs/monocyte_mll23_raabin_bg095_n64_s8_20260313/monocyte/ref_domain_c_mll23/to_domain_b_raabin/bg095/rf000/0013_s4.png)

### 3. MLL23 -> Raabin
- Margin: `0.1471`
- Background SSIM: `0.8711`
- Cell SSIM: `0.9992`
- Background strength: `0.95`

Input  
![ranked-3-input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_c_mll23/monocyte/domain_c_mll23_monocyte_000396.jpg)

Generated  
![ranked-3-output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2_runs/monocyte_mll23_raabin_bg095_n64_s8_20260313/monocyte/ref_domain_c_mll23/to_domain_b_raabin/bg095/rf000/0038_s0.png)

### 4. MLL23 -> Raabin
- Margin: `0.0829`
- Background SSIM: `0.8669`
- Cell SSIM: `0.9993`
- Background strength: `0.95`

Input  
![ranked-4-input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_c_mll23/monocyte/domain_c_mll23_monocyte_001881.jpg)

Generated  
![ranked-4-output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2_runs/monocyte_mll23_raabin_bg095_n64_s8_20260313/monocyte/ref_domain_c_mll23/to_domain_b_raabin/bg095/rf000/0044_s4.png)

### 5. AMC -> PBC
- Margin: `0.2582`
- Background SSIM: `0.8557`
- Cell SSIM: `0.9992`
- Background strength: `0.95`

Input  
![ranked-5-input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/monocyte/domain_e_amc_monocyte_000733.jpg)

Generated  
![ranked-5-output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/monocyte/ref_domain_e_amc/to_domain_a_pbc/bg095/rf000/0000_s0.png)

### 6. AMC -> MLL23
- Margin: `0.0557`
- Background SSIM: `0.8794`
- Cell SSIM: `0.9990`
- Background strength: `0.95`

Input  
![ranked-6-input](/Users/imds/Desktop/wbc_synthesis/data/processed_contextual_multidomain/domain_e_amc/monocyte/domain_e_amc_monocyte_000759.jpg)

Generated  
![ranked-6-output](/Users/imds/Desktop/wbc_synthesis/data/generated_boundary_v2/monocyte/ref_domain_e_amc/to_domain_c_mll23/bg095/rf000/0000_s0.png)

## Quick Read

- 최근 주력 augmentation policy는 여전히 `S7 / S3` 계열이다.
- boundary-aware V2는 원하는 방향의 변형을 만들었고, 실제 ranked low-margin 샘플 6장까지 확보했다.
- 다만 downstream에서는 `B2` 단독보다 기존 `S7`이 더 안정적이었다.
- 현재 boundary branch의 가장 강한 메시지는 `useful exploratory generator for low-margin morphology-preserving variants`에 가깝다.
