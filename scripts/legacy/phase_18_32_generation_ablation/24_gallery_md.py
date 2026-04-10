"""
Script 24: Few-Shot IP-Adapter 실험 이미지 갤러리 마크다운 생성
=============================================================
summary.json + images/ 디렉토리를 읽어, 각 cls×domain 조합에 대해
실제 이미지를 인라인 embed한 갤러리 마크다운(gallery.md)을 생성한다.

출력: results/fewshot_diversity_eval/gallery.md

Usage:
    python3 scripts/legacy/phase_18_32_generation_ablation/24_gallery_md.py
"""

import json
from pathlib import Path

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "results" / "fewshot_diversity_eval"
IMG_DIR = DATA / "images"
SUMMARY = DATA / "summary.json"
OUT     = DATA / "gallery.md"

# ── 실험 상수 ─────────────────────────────────────────────────────────────────
CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
DOMAINS = ["PBC", "Raabin", "MLL23", "AMC"]

# dir_name → (ip_scale, n_ref, label)
CONFIGS = [
    ("scale000_ref0",  0.0,  0,  "**Baseline** (ip=0.0)"),
    ("scale020_ref1",  0.2,  1,  "ip=0.2 · ref=1"),
    ("scale020_ref2",  0.2,  2,  "ip=0.2 · ref=2"),
    ("scale020_ref4",  0.2,  4,  "ip=0.2 · ref=4"),
    ("scale035_ref1",  0.35, 1,  "ip=0.35 · ref=1"),
    ("scale035_ref2",  0.35, 2,  "ip=0.35 · ref=2"),
    ("scale035_ref4",  0.35, 4,  "ip=0.35 · ref=4"),
    ("scale050_ref1",  0.5,  1,  "ip=0.5 · ref=1"),
    ("scale050_ref2",  0.5,  2,  "ip=0.5 · ref=2"),
    ("scale050_ref4",  0.5,  4,  "ip=0.5 · ref=4"),
]

N_SEEDS = 3

# ── 뱃지 함수 ─────────────────────────────────────────────────────────────────
def acc_badge(cnn_acc: float) -> str:
    """CNN accuracy → 색깔 뱃지"""
    pct = f"{cnn_acc:.0%}"
    if cnn_acc >= 0.90:
        return f"🟩 {pct}"
    if cnn_acc >= 0.67:
        return f"🟨 {pct}"
    return f"🟥 {pct}"


def ssim_badge(ssim: float, baseline_ssim: float) -> str:
    """SSIM → 다양성 뱃지 (낮을수록 다양)"""
    delta = ssim - baseline_ssim
    d_str = f"{delta:+.4f}"
    s_str = f"{ssim:.4f}"
    if delta < -0.03:
        return f"🔵 {s_str} (Δ{d_str})"  # 다양성 유의미하게 증가
    if delta < -0.01:
        return f"🟡 {s_str} (Δ{d_str})"  # 약간 증가
    return f"⚪ {s_str} (Δ{d_str})"       # 거의 변화 없음


# ── 데이터 조회 헬퍼 ───────────────────────────────────────────────────────────
def lookup(data: list, cls: str, domain: str):
    """summary.json data 배열에서 cls×domain 항목 반환"""
    # input_domain 컬럼은 "domain_a_pbc" / "domain_b_raabin" / "domain_c_mll23" / "domain_d_amc" 형태
    domain_key_map = {
        "PBC":    "domain_a_pbc",
        "Raabin": "domain_b_raabin",
        "MLL23":  "domain_c_mll23",
        "AMC":    "domain_d_amc",
    }
    target_domain = domain_key_map.get(domain, domain.lower())
    for entry in data:
        if entry.get("class") == cls and entry.get("input_domain") == target_domain:
            return entry
    # fallback: 대소문자 무시 substring 매칭
    for entry in data:
        if entry.get("class") == cls and domain.lower() in entry.get("input_domain", "").lower():
            return entry
    return None


def best_config(entry: dict):
    """CNN acc ≥ 90% 조건에서 SSIM이 가장 낮은 조건 반환 (IP-Adapter only)"""
    candidates = [c for c in entry["conditions"] if c["ip_scale"] > 0.0 and c["cnn_acc"] >= 0.90]
    if not candidates:
        # 90% 미달 시 baseline 권장
        return entry["conditions"][0]  # baseline
    return min(candidates, key=lambda c: c["ssim_mean"])


# ── 마크다운 앵커 헬퍼 ─────────────────────────────────────────────────────────
def anchor(cls: str, dom: str) -> str:
    """GitHub-style 마크다운 앵커 ID"""
    return f"{cls.lower()}-{dom.lower()}"


# ── 갤러리 마크다운 생성 ───────────────────────────────────────────────────────
def make_gallery(data: list) -> str:
    lines: list[str] = []

    # ── 헤더 ──────────────────────────────────────────────────────────────────
    lines += [
        "# WBC Few-Shot IP-Adapter 실험 — 이미지 갤러리",
        "",
        "> **Strength:** 0.35 고정  ",
        "> **IP_SCALES:** 0.0 · 0.2 · 0.35 · 0.5  ",
        "> **N_REF_LIST:** 1 · 2 · 4  ",
        "> **N_SEEDS:** 3  ",
        "> **평가:** CNN acc (EfficientNet-B0) · SSIM (vs 원본 입력)  ",
        "",
        "뱃지 범례:",
        "- CNN acc: 🟩 ≥90% · 🟨 ≥67% · 🟥 <67%",
        "- SSIM (다양성↑=낮을수록 좋음): 🔵 Δ<−0.03 · 🟡 Δ<−0.01 · ⚪ 거의 변화 없음",
        "",
        "---",
        "",
    ]

    # ── 목차 ──────────────────────────────────────────────────────────────────
    lines.append("## 목차")
    lines.append("")
    for cls in CLASSES:
        lines.append(f"### {cls}")
        for dom in DOMAINS:
            lines.append(f"- [{dom}](#{anchor(cls, dom)})")
        lines.append("")
    lines.append("---")
    lines.append("")

    # ── 클래스 × 도메인 섹션 ──────────────────────────────────────────────────
    for cls in CLASSES:
        lines.append(f"# {cls.upper()}")
        lines.append("")

        for dom in DOMAINS:
            # 앵커 (목차 링크 대상)
            lines.append(f'<a id="{anchor(cls, dom)}"></a>')
            lines.append("")
            lines.append(f"## {cls} × {dom}")
            lines.append("")

            # summary.json 항목 조회
            entry = lookup(data, cls, dom)
            if entry is None:
                lines.append(f"> ⚠️ 데이터 없음: {cls} × {dom}")
                lines.append("")
                continue

            ref_pool_size = entry.get("ref_pool_size", "N/A")
            conditions    = entry.get("conditions", [])
            baseline_ssim = conditions[0]["ssim_mean"] if conditions else None

            # ── 원본 입력 이미지 ───────────────────────────────────────────────
            input_rel = f"images/{cls}/{dom}/input.png"
            input_abs = DATA / input_rel
            lines.append("### 원본 입력 이미지")
            lines.append("")
            if input_abs.exists():
                lines.append(f"| 원본 입력 | 참조 풀 크기 |")
                lines.append(f"| :---: | :---: |")
                lines.append(f"| ![input]({input_rel}) | {ref_pool_size:,} 장 |")
            else:
                lines.append(f"> ⚠️ input.png 없음: `{input_rel}`")
            lines.append("")

            # ── 실험 조건 비교 테이블 ──────────────────────────────────────────
            lines.append("### 생성 이미지 비교")
            lines.append("")

            # 테이블 헤더
            seed_headers = " | ".join(f"seed {i}" for i in range(N_SEEDS))
            lines.append(f"| 설정 | {seed_headers} | CNN acc | SSIM (다양성) |")
            sep = " | ".join([":---"] + [":---:"] * N_SEEDS + [":---:", ":---:"])
            lines.append(f"| {sep} |")

            for dir_name, ip_scale, n_ref, label in CONFIGS:
                # conditions에서 매칭 항목 찾기
                cond = next(
                    (c for c in conditions
                     if abs(c["ip_scale"] - ip_scale) < 1e-4 and c["n_ref"] == n_ref),
                    None
                )

                # 이미지 셀 생성
                img_cells: list[str] = []
                for s in range(N_SEEDS):
                    img_path_rel = f"images/{cls}/{dom}/{dir_name}/seed0{s}.png"
                    img_path_abs = DATA / img_path_rel
                    if img_path_abs.exists():
                        img_cells.append(f"![s{s}]({img_path_rel})")
                    else:
                        img_cells.append("—")

                imgs_str = " | ".join(img_cells)

                if cond:
                    b_acc  = acc_badge(cond["cnn_acc"])
                    b_ssim = ssim_badge(cond["ssim_mean"], baseline_ssim) if baseline_ssim else f"{cond['ssim_mean']:.4f}"
                else:
                    b_acc  = "—"
                    b_ssim = "—"

                lines.append(f"| {label} | {imgs_str} | {b_acc} | {b_ssim} |")

            lines.append("")

            # ── per-seed 상세 (접기/펼치기) ────────────────────────────────────
            lines.append("<details>")
            lines.append("<summary>📋 Per-seed 상세 결과 (펼치기)</summary>")
            lines.append("")
            lines.append("| 설정 | seed | SSIM | 예측 클래스 | conf | 정답 |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
            for dir_name, ip_scale, n_ref, label in CONFIGS:
                cond = next(
                    (c for c in conditions
                     if abs(c["ip_scale"] - ip_scale) < 1e-4 and c["n_ref"] == n_ref),
                    None
                )
                if cond is None:
                    continue
                for ps in cond.get("per_seed", []):
                    seed_idx = ps.get("seed", "?")
                    ssim_val = f"{ps.get('ssim', 0):.4f}"
                    pred     = ps.get("pred", "—")
                    conf_val = f"{ps.get('conf', 0):.3f}"
                    correct  = "✅" if ps.get("correct") else "❌"
                    lines.append(
                        f"| {label} | {seed_idx} | {ssim_val} | {pred} | {conf_val} | {correct} |"
                    )
            lines.append("")
            lines.append("</details>")
            lines.append("")

            # ── 이 조합의 권장 설정 ────────────────────────────────────────────
            best = best_config(entry)
            if best:
                if best["ip_scale"] == 0.0:
                    rec_note = "⚠️ IP-Adapter 적용 시 CNN accuracy 미달 → **baseline만 권장**"
                else:
                    rec_note = (
                        f"✅ **ip={best['ip_scale']}, n_ref={best['n_ref']}**  "
                        f"CNN {acc_badge(best['cnn_acc'])} · SSIM {best['ssim_mean']:.4f}"
                    )
                lines.append(f"> 🏆 권장 설정: {rec_note}")
                lines.append("")

            lines.append("---")
            lines.append("")

    # ── 전체 요약 테이블 ───────────────────────────────────────────────────────
    lines.append("# 전체 권장 설정 요약")
    lines.append("")
    lines.append("| 클래스 | PBC | Raabin | MLL23 | AMC |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for cls in CLASSES:
        row_cells: list[str] = []
        for dom in DOMAINS:
            entry = lookup(data, cls, dom)
            if entry is None:
                row_cells.append("—")
                continue
            best = best_config(entry)
            if best is None:
                row_cells.append("—")
            elif best["ip_scale"] == 0.0:
                row_cells.append("baseline ⚠️")
            else:
                row_cells.append(f"ip={best['ip_scale']} ✅")
        lines.append(f"| **{cls}** | {' | '.join(row_cells)} |")

    lines.append("")
    lines.append("> 🟩 = CNN ≥ 90% 달성 · ⚠️ = IP-Adapter 비권장(baseline 사용)  ")
    lines.append("> 모든 ip_scale / n_ref 조합에서 SSIM 수치는 동일하게 수렴 → **n_ref=1** 이 가장 효율적  ")
    lines.append("")

    # ── SSIM 개요 (클래스별) ────────────────────────────────────────────────────
    lines.append("# SSIM 개요 (Baseline → IP-Adapter 전환 효과)")
    lines.append("")
    lines.append("| 클래스 | 도메인 | Baseline SSIM | IP-Adapter SSIM | ΔSSIM |")
    lines.append("| :--- | :--- | :---: | :---: | :---: |")

    for cls in CLASSES:
        for dom in DOMAINS:
            entry = lookup(data, cls, dom)
            if entry is None:
                continue
            conds = entry.get("conditions", [])
            base_cond = next((c for c in conds if c["ip_scale"] == 0.0), None)
            # n_ref=1, ip=0.2 조건을 대표로 선택
            ip_cond = next(
                (c for c in conds if abs(c["ip_scale"] - 0.2) < 1e-4 and c["n_ref"] == 1),
                None
            )
            if base_cond is None:
                continue
            base_s = base_cond["ssim_mean"]
            if ip_cond:
                ip_s   = ip_cond["ssim_mean"]
                delta  = ip_s - base_s
                d_str  = f"{delta:+.4f}"
                color  = "🔵" if delta < -0.03 else ("🟡" if delta < -0.01 else "⚪")
                lines.append(
                    f"| {cls} | {dom} | {base_s:.4f} | {ip_s:.4f} | {color} {d_str} |"
                )
            else:
                lines.append(f"| {cls} | {dom} | {base_s:.4f} | — | — |")

    lines.append("")

    return "\n".join(lines)


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print(f"▶  summary.json 로드: {SUMMARY}")
    if not SUMMARY.exists():
        raise FileNotFoundError(f"summary.json 없음: {SUMMARY}")

    raw = json.loads(SUMMARY.read_text(encoding="utf-8"))
    # summary.json 루트 키 탐색 (data or results)
    data_list = raw.get("data") or raw.get("results") or []
    if not data_list:
        raise ValueError(f"summary.json에 'data' 또는 'results' 키가 없음. 키 목록: {list(raw.keys())}")

    print(f"   총 항목 수: {len(data_list)}")

    md_text = make_gallery(data_list)
    OUT.write_text(md_text, encoding="utf-8")
    print(f"✅  갤러리 마크다운 저장 완료: {OUT}")
    print(f"   파일 크기: {OUT.stat().st_size / 1024:.1f} KB")

    # 간단 검증: 섹션 수 체크
    section_count = md_text.count("## ") - 1  # 목차 내 ## 제외 근사
    print(f"   예상 섹션 수 (cls×dom): {len(CLASSES) * len(DOMAINS)} → 실제 '##' 카운트 ≈ {section_count}")


if __name__ == "__main__":
    main()
