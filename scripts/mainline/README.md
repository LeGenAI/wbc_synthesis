# Mainline Pipeline Scaffold

이 디렉토리는 앞으로의 메인 연구 라인을 위한 canonical scaffold다. 구조는 최신 논문 제출용 보충자료의 방법 섹션 순서를 따른다.

## Supplementary-aligned stages

1. `data/01_prepare_multidomain_dataset.py`
   source-domain 정규화, split 통제, manifest 생성
2. `generation/02_train_generation_policy.py`
   generation backbone 및 policy 학습
3. `generation/03_generate_synthetic_pool.py`
   synthetic pool 생성
4. `scoring/04_score_synthetic_pool.py`
   preservation / diversity / utility-proxy scoring
5. `benchmark/05_train_lodo_utility_benchmark.py`
   leakage-safe LODO utility benchmark
6. `reporting/06_make_submission_package.py`
   표, 그림, appendix, supplementary artifacts 묶기

## Working rule

- 새 실험은 numbered legacy 흐름을 잇지 않는다.
- 새 스크립트는 위 stage에만 추가한다.
- novelty 주장은 반드시 `reference/reference_matrix.md`와 함께 읽히도록 설계한다.
