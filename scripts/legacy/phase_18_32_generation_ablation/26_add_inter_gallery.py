"""
Script 26: gallery.md에 인터 관점(Inter-Input) 섹션 추가
==========================================================

기존 gallery.md (인트라 관점)에 인터 관점 섹션을 추가하여 재생성.

인트라 관점 (기존):
  행(row) = 입력 이미지  |  열(col) = seed 0~4
  → 같은 입력에서 seed 랜덤성이 얼마나 작용하는지

인터 관점 (신규 추가):
  행(row) = seed 번호  |  열(col) = 입력 0~4
  → 같은 seed로 5개 다른 입력이 각각 얼마나 다른 결과를 만드는지

사용법:
    python3 scripts/legacy/phase_18_32_generation_ablation/26_add_inter_gallery.py
"""

import json
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "diversity_reeval"
SUMMARY = OUT_DIR / "summary.json"
OUT_MD  = OUT_DIR / "gallery.md"

CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
DOMAINS = ["PBC", "Raabin", "MLL23", "AMC"]

STRENGTH = 0.35


# ── 뱃지 함수 ─────────────────────────────────────────────────────────────────
def acc_badge(acc: float) -> str:
    if acc >= 0.90: return f"🟩 {acc:.0%}"
    if acc >= 0.67: return f"🟨 {acc:.0%}"
    return f"🟥 {acc:.0%}"


# ── 인트라 테이블 생성 (기존과 동일) ──────────────────────────────────────────
def make_intra_table(cls: str, dom: str, entry: dict, n_inputs: int, n_seeds: int) -> str:
    lines = []
    lines.append("### 🔵 인트라 관점: 같은 입력 → seed별 변이")
    lines.append("")
    lines.append("> 한 행에서 가로로 보면: 같은 입력 이미지를 seed만 바꿔 생성한 결과 (seed 다양성 효과)")
    lines.append("")

    seed_headers = " | ".join(f"seed {s}" for s in range(n_seeds))
    lines.append(f"| 입력 이미지 | {seed_headers} | CNN acc | SSIM |")
    sep_parts = [":---:"] + [":---:"] * n_seeds + [":---:", ":---:"]
    lines.append("| " + " | ".join(sep_parts) + " |")

    for inp_data in entry["inputs"]:
        inp_idx  = inp_data["inp_idx"]
        inp_rel  = f"images/{cls}/{dom}/input_{inp_idx:02d}/input.png"
        inp_cell = f"![inp{inp_idx}]({inp_rel})"

        seed_cells = []
        for s in range(n_seeds):
            sd = next((x for x in inp_data["seeds"] if x["seed_offset"] == s), None)
            if sd:
                img_rel = f"images/{cls}/{dom}/input_{inp_idx:02d}/seed_{s:02d}.png"
                ok = "✅" if sd["correct"] else "❌"
                seed_cells.append(f"![s{s}]({img_rel}){ok}")
            else:
                seed_cells.append("—")

        acc_val  = inp_data["cnn_acc"]
        ssim_val = inp_data["ssim_mean"]
        lines.append(
            f"| {inp_cell} | "
            + " | ".join(seed_cells)
            + f" | {acc_badge(acc_val)} | {ssim_val:.4f} |"
        )

    return "\n".join(lines)


# ── 인터 테이블 생성 (신규) ───────────────────────────────────────────────────
def make_inter_table(cls: str, dom: str, entry: dict, n_inputs: int, n_seeds: int) -> str:
    lines = []
    lines.append("### 🟠 인터 관점: 같은 seed → 입력별 결과 비교")
    lines.append("")
    lines.append("> 한 행에서 가로로 보면: 같은 seed를 사용했지만 **입력 이미지가 다를 때** 얼마나 다른 결과가 나오는지 (입력 다양성 효과)")
    lines.append("")

    # 헤더: | seed | 입력 0 | 입력 1 | 입력 2 | 입력 3 | 입력 4 |
    inp_headers = " | ".join(f"입력 {i}" for i in range(n_inputs))
    lines.append(f"| seed | {inp_headers} |")
    sep = " | ".join([":---:"] * (n_inputs + 1))
    lines.append(f"| {sep} |")

    # 첫 번째 행: 원본 입력 이미지 나열
    orig_cells = ["**원본 입력**"]
    for inp_idx in range(n_inputs):
        inp_rel = f"images/{cls}/{dom}/input_{inp_idx:02d}/input.png"
        orig_cells.append(f"![orig{inp_idx}]({inp_rel})")
    lines.append("| " + " | ".join(orig_cells) + " |")

    for s in range(n_seeds):
        row_cells = [f"**seed {s}**"]
        for inp_idx in range(n_inputs):
            # 해당 inp_idx의 inp_data 찾기
            inp_data = next(
                (x for x in entry["inputs"] if x["inp_idx"] == inp_idx), None
            )
            if inp_data is None:
                row_cells.append("—")
                continue

            img_rel = f"images/{cls}/{dom}/input_{inp_idx:02d}/seed_{s:02d}.png"
            sd = next((x for x in inp_data["seeds"] if x["seed_offset"] == s), None)
            ok = "✅" if (sd and sd["correct"]) else "❌"
            row_cells.append(f"![i{inp_idx}s{s}]({img_rel}){ok}")

        lines.append("| " + " | ".join(row_cells) + " |")

    return "\n".join(lines)


# ── 전체 갤러리 생성 ──────────────────────────────────────────────────────────
def make_gallery(all_results: list, n_inputs: int, n_seeds: int) -> str:
    lines = []

    # ── 헤더 ──
    lines += [
        "# WBC 다양성 재실험 갤러리 (다중 입력 × 다중 seed)",
        "",
        f"> **Strength:** {STRENGTH} 고정  ",
        f"> **N_INPUTS:** {n_inputs} (cls×dom당 입력 이미지 수)  ",
        f"> **N_SEEDS:** {n_seeds} (입력당 seed 수)  ",
        f"> **총 생성:** {5 * 4 * n_inputs * n_seeds}장  ",
        "",
        "뱃지: 🟩 CNN ≥ 90% · 🟨 ≥ 67% · 🟥 < 67%",
        "",
        "---",
        "",
        "## 두 가지 관점",
        "",
        "| 관점 | 행(row) | 열(col) | 무엇을 보는가 |",
        "| :--- | :---: | :---: | :--- |",
        "| 🔵 **인트라(Intra)** | 입력 이미지 | seed 0~4 | 같은 입력에서 seed 변화로 얼마나 달라지는가 |",
        "| 🟠 **인터(Inter)** | seed 번호 | 입력 0~4 | 같은 seed에서 입력이 달라지면 얼마나 달라지는가 |",
        "",
        f"> **핵심 발견:** Inter-input SSIM(≈0.46) ≪ Intra-input SSIM(≈0.88)  ",
        f"> → **입력 이미지 다양성**이 seed보다 출력 다양성에 훨씬 큰 영향을 미침  ",
        "",
        "---",
        "",
    ]

    # ── 목차 ──
    lines.append("## 목차")
    lines.append("")
    for cls in CLASSES:
        lines.append(f"### {cls}")
        for dom in DOMAINS:
            lines.append(f"- [{cls} × {dom}](#{cls.lower()}-{dom.lower()})")
        lines.append("")
    lines.append("---")
    lines.append("")

    # ── 클래스 × 도메인 섹션 ──
    for cls in CLASSES:
        lines.append(f"# {cls.upper()}")
        lines.append("")

        for dom in DOMAINS:
            lines.append(f'<a id="{cls.lower()}-{dom.lower()}"></a>')
            lines.append("")
            lines.append(f"## {cls} × {dom}")
            lines.append("")

            entry = next(
                (r for r in all_results if r["cls"] == cls and r["dom"] == dom), None
            )
            if entry is None:
                lines.append("> ⚠️ 데이터 없음")
                lines.append("")
                continue

            n_inp = entry.get("n_inputs", n_inputs)

            # 지표 요약 박스
            intra = entry.get("intra_input_ssim_mean")
            inter = entry.get("inter_input_ssim_mean")
            lines.append(f"> **CNN acc:** {acc_badge(entry['overall_cnn_acc'])} · "
                         f"**SSIM vs 입력:** {entry['overall_ssim_mean']:.4f}  ")
            if intra is not None and inter is not None:
                lines.append(f"> 🔵 Intra SSIM: **{intra:.4f}** · "
                             f"🟠 Inter SSIM: **{inter:.4f}**  ")
                if inter < intra:
                    ratio = intra / inter
                    lines.append(
                        f"> ✅ Inter({inter:.4f}) ≪ Intra({intra:.4f}) "
                        f"→ 입력 다양성 효과가 {ratio:.1f}배 더 큼  "
                    )
            lines.append("")

            # 인트라 테이블
            lines.append(make_intra_table(cls, dom, entry, n_inp, n_seeds))
            lines.append("")

            # 인터 테이블
            lines.append(make_inter_table(cls, dom, entry, n_inp, n_seeds))
            lines.append("")

            lines.append("---")
            lines.append("")

    # ── 전체 요약 테이블 ──
    lines.append("# 전체 결과 요약")
    lines.append("")
    lines.append("| 클래스 | 도메인 | CNN acc | SSIM mean | 🔵 Intra SSIM | 🟠 Inter SSIM | 비율(Intra/Inter) |")
    lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for r in all_results:
        intra = r.get("intra_input_ssim_mean")
        inter = r.get("inter_input_ssim_mean")
        intra_s = f"{intra:.4f}" if intra else "—"
        inter_s = f"{inter:.4f}" if inter else "—"
        ratio_s = f"{intra/inter:.2f}×" if (intra and inter and inter > 0) else "—"
        lines.append(
            f"| {r['cls']} | {r['dom']} | {acc_badge(r['overall_cnn_acc'])} "
            f"| {r['overall_ssim_mean']:.4f} | {intra_s} | {inter_s} | {ratio_s} |"
        )
    lines.append("")

    # 전체 평균
    all_intra = [r["intra_input_ssim_mean"] for r in all_results if r.get("intra_input_ssim_mean")]
    all_inter = [r["inter_input_ssim_mean"] for r in all_results if r.get("inter_input_ssim_mean")]
    all_acc   = [r["overall_cnn_acc"] for r in all_results]
    all_ssim  = [r["overall_ssim_mean"] for r in all_results]

    if all_intra and all_inter:
        avg_intra = sum(all_intra) / len(all_intra)
        avg_inter = sum(all_inter) / len(all_inter)
        avg_acc   = sum(all_acc) / len(all_acc)
        avg_ssim  = sum(all_ssim) / len(all_ssim)
        lines += [
            "",
            "## 전체 평균",
            "",
            f"| 지표 | 값 |",
            f"| :--- | :---: |",
            f"| CNN accuracy | {acc_badge(avg_acc)} |",
            f"| SSIM vs 입력 | {avg_ssim:.4f} |",
            f"| 🔵 Intra-input SSIM | {avg_intra:.4f} |",
            f"| 🟠 Inter-input SSIM | {avg_inter:.4f} |",
            f"| Intra / Inter 비율 | **{avg_intra/avg_inter:.2f}×** |",
            "",
            f"> → 입력 이미지가 달라지면 생성 결과의 차이(Inter SSIM={avg_inter:.4f})가  ",
            f">   같은 입력에서 seed만 바꿀 때(Intra SSIM={avg_intra:.4f})보다  ",
            f">   **{avg_intra/avg_inter:.1f}배** 더 크게 나타남.  ",
            f"> → **다양한 합성 데이터를 원한다면 입력 이미지 자체를 다양하게 사용해야 한다.**  ",
        ]

    return "\n".join(lines)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print("Script 26: 인터 관점 갤러리 추가")
    print(f"  summary.json: {SUMMARY}")
    print(f"  출력: {OUT_MD}")

    if not SUMMARY.exists():
        print(f"  ❌ summary.json 없음: {SUMMARY}")
        return

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    n_inputs = summary.get("n_inputs", 5)
    n_seeds  = summary.get("n_seeds",  5)
    results  = summary.get("results", [])

    print(f"  N_INPUTS={n_inputs}, N_SEEDS={n_seeds}, 조합수={len(results)}")

    md = make_gallery(results, n_inputs, n_seeds)
    OUT_MD.write_text(md, encoding="utf-8")

    n_chars = len(md)
    n_lines = md.count("\n")
    print(f"\n  ✅ gallery.md 저장 완료")
    print(f"     {n_lines}행, {n_chars:,}자")
    print(f"  경로: {OUT_MD}")

    # 섹션 수 확인
    n_sections = md.count("## 인트라 관점")
    n_inter    = md.count("## 인터 관점")
    print(f"\n  인트라 섹션: {n_sections}개")
    print(f"  인터 섹션:   {n_inter}개")


if __name__ == "__main__":
    main()
