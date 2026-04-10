"""
Script 23: Few-Shot IP-Adapter 다양화 실험
==========================================
기존 img2img(단일 이미지 + strength)의 다양성 한계를 극복하기 위해
IP-Adapter를 통해 같은 클래스의 다른 도메인 이미지를 스타일 참조로 주입.

핵심 원리:
  - strength=0.35 고정 → 세포 형태(핵, 과립) 원본 보존
  - IP-Adapter → 다른 도메인의 배경색/염색 스타일 Cross-Attention 주입
  - 목표: 다양성 증가(SSIM↓) + CNN accuracy 유지(≥90%)

실험 설계:
  - 5클래스 × 4도메인(입력) × 4 ip_scales × 3 n_ref 크기 × 3 seeds
  - ip_scale=0.0은 기존 베이스라인 (n_ref=0)
  - 총 600장 생성

출력:
  results/fewshot_diversity_eval/
    summary.json    — 전체 매트릭스
    report.md       — ip_scale × n_ref 히트맵 + 권장 설정
    images/         — 대표 샘플
"""

import importlib.util
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# ── WBCRouter 동적 임포트 ────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "router_inference", ROOT / "scripts" / "15_router_inference.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
WBCRouter       = mod.WBCRouter
ssim_pair       = mod.ssim_pair
MULTI_CLASSES   = mod.MULTI_CLASSES
DOMAINS         = mod.DOMAINS
NEGATIVE_PROMPT = mod.NEGATIVE_PROMPT

# ── 상수 ────────────────────────────────────────────────────────────────
ROUTER_CKPT = ROOT / "models" / "dual_head_router.pt"
CNN_CKPT    = ROOT / "models" / "multidomain_cnn.pt"
DATA_DIR    = ROOT / "data" / "processed_multidomain"
OUT_DIR     = ROOT / "results" / "fewshot_diversity_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 실험 파라미터
IP_SCALES   = [0.0, 0.2, 0.35, 0.5]   # 0.0 = 기존 베이스라인
N_REF_LIST  = [1, 2, 4]               # 참조 이미지 수 (ip_scale>0 전용)
N_SEEDS     = 3
STRENGTH    = 0.35
SEED_BASE   = 42

IP_ADAPTER_REPO      = "h94/IP-Adapter"
IP_ADAPTER_SUBFOLDER = "sdxl_models"
IP_ADAPTER_WEIGHT    = "ip-adapter_sdxl.bin"

DOMAIN_SHORT = {
    "domain_a_pbc":    "PBC",
    "domain_b_raabin": "Raabin",
    "domain_c_mll23":  "MLL23",
    "domain_e_amc":    "AMC",
}
CLASS_IDX  = {c: i for i, c in enumerate(MULTI_CLASSES)}
DOMAIN_IDX = {d: i for i, d in enumerate(DOMAINS)}

# ── CNN 유틸 ────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def load_cnn(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    base = models.efficientnet_b0(weights=None)
    base.classifier[1] = nn.Linear(base.classifier[1].in_features, 5)
    base.load_state_dict(ckpt["model_state_dict"])
    return base.eval().to(device)

@torch.no_grad()
def cnn_classify(cnn, img, device):
    x = TRANSFORM(img).unsqueeze(0).to(device)
    probs = F.softmax(cnn(x), dim=1).squeeze(0).cpu()
    idx   = probs.argmax().item()
    return {"pred_class": MULTI_CLASSES[idx], "conf": round(probs[idx].item(), 4)}


# ── FewShotSampler ───────────────────────────────────────────────────────
class FewShotSampler:
    """
    같은 클래스 × 다른 도메인에서 참조 이미지를 랜덤 샘플링.
    입력 이미지의 도메인은 자동으로 제외.
    """
    def __init__(self, data_dir: Path, seed: int = SEED_BASE):
        self.rng = random.Random(seed)
        self._pool: dict[str, dict[str, list[Path]]] = {}
        for dom in DOMAINS:
            self._pool[dom] = {}
            for cls in MULTI_CLASSES:
                d = data_dir / dom / cls
                if d.exists():
                    files = sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))
                    self._pool[dom][cls] = files
                else:
                    self._pool[dom][cls] = []

    def sample(self, cls_name: str, exclude_domain: str, n: int) -> list[Image.Image]:
        """다른 도메인에서 동일 클래스 이미지 n장 랜덤 샘플링."""
        combined = []
        for dom in DOMAINS:
            if dom != exclude_domain:
                combined.extend(self._pool[dom][cls_name])
        if not combined:
            return []
        chosen = self.rng.choices(combined, k=n)
        return [Image.open(p).convert("RGB") for p in chosen]

    def pool_size(self, cls_name: str, exclude_domain: str) -> int:
        return sum(
            len(self._pool[d][cls_name])
            for d in DOMAINS if d != exclude_domain
        )


# ── IP-Adapter 로딩 ──────────────────────────────────────────────────────
def load_ip_adapter_on_pipe(pipe):
    """
    파이프라인에 IP-Adapter 장착.
    첫 실행 시 h94/IP-Adapter에서 자동 다운로드 (~100MB).
    주의: enable_attention_slicing()과 충돌 → 로드 전에 비활성화 필요.
    """
    print(f"  IP-Adapter 로드 중 ({IP_ADAPTER_REPO}/{IP_ADAPTER_WEIGHT})...")
    # attention_slicing이 활성화된 경우 IP-Adapter AttnProcessor 초기화 충돌 방지
    pipe.disable_attention_slicing()
    pipe.load_ip_adapter(
        IP_ADAPTER_REPO,
        subfolder=IP_ADAPTER_SUBFOLDER,
        weight_name=IP_ADAPTER_WEIGHT,
    )
    print("  IP-Adapter 로드 완료.")


# ── 생성 함수 ────────────────────────────────────────────────────────────
def generate_with_ip_adapter(
    pipe,
    device,
    input_img: Image.Image,
    ref_imgs: list[Image.Image],
    prompt: str,
    strength: float,
    ip_scale: float,
    seed: int,
) -> Image.Image:
    """
    IP-Adapter scale 적용 img2img 생성.
    ip_scale=0.0 → 기존 베이스라인 (참조 이미지 무시)
    ip_scale>0.0 → 참조 이미지 스타일 주입
    """
    pipe.set_ip_adapter_scale(ip_scale)

    ref = input_img.convert("RGB").resize((512, 512))
    gen = torch.Generator(device).manual_seed(seed)

    kwargs = dict(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        image=ref,
        strength=strength,
        guidance_scale=6.0,
        num_inference_steps=25,
        generator=gen,
    )

    if ref_imgs:
        # diffusers 0.36: IP-Adapter가 장착된 경우 ip_adapter_image 항상 필요.
        # 여러 참조 이미지를 1개 IP-Adapter에: [[img1, img2, ...]] 중첩 리스트
        # 단일 참조: 그대로 단일 이미지
        kwargs["ip_adapter_image"] = [ref_imgs] if len(ref_imgs) > 1 else ref_imgs[0]

    with torch.no_grad():
        result = pipe(**kwargs).images[0]
    return result


# ── 보고서 생성 ──────────────────────────────────────────────────────────
def make_report(raw_data: list) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Few-Shot IP-Adapter 다양화 실험 결과",
        "",
        f"> **생성 일시:** {ts}  ",
        f"> **Strength:** {STRENGTH} (고정)  ",
        f"> **IP_SCALES:** {IP_SCALES}  ",
        f"> **N_REF_LIST:** {N_REF_LIST}  ",
        f"> **N_SEEDS:** {N_SEEDS}  ",
        "",
        "---",
        "",
        "## CNN Accuracy vs SSIM Trade-off 요약",
        "",
    ]

    # ── 전체 평균 테이블 (ip_scale × n_ref) ─────────────────────────────
    lines += [
        "### 전체 클래스·도메인 평균",
        "",
        "| ip_scale | n_ref | CNN acc | SSIM | Δ SSIM (vs base) |",
        "|----------|-------|---------|------|-----------------|",
    ]

    # 기준: ip_scale=0.0 baseline
    baseline_ssim = None
    agg = {}  # (ip_scale, n_ref) → {cnn_accs, ssims}
    for entry in raw_data:
        for cond in entry["conditions"]:
            key = (cond["ip_scale"], cond["n_ref"])
            if key not in agg:
                agg[key] = {"cnn_accs": [], "ssims": []}
            agg[key]["cnn_accs"].append(cond["cnn_acc"])
            agg[key]["ssims"].append(cond["ssim_mean"])

    baseline_ssim = float(np.mean(agg.get((0.0, 0), {}).get("ssims", [0.98])))

    for ip_scale in IP_SCALES:
        if ip_scale == 0.0:
            key = (0.0, 0)
            if key in agg:
                cnn_m = float(np.mean(agg[key]["cnn_accs"]))
                ssim_m = float(np.mean(agg[key]["ssims"]))
                lines.append(
                    f"| {ip_scale} | — (baseline) | {cnn_m*100:.1f}% | {ssim_m:.4f} | — |"
                )
        else:
            for n_ref in N_REF_LIST:
                key = (ip_scale, n_ref)
                if key in agg:
                    cnn_m  = float(np.mean(agg[key]["cnn_accs"]))
                    ssim_m = float(np.mean(agg[key]["ssims"]))
                    delta  = ssim_m - baseline_ssim
                    emoji  = "🟩" if cnn_m >= 0.9 else ("🟨" if cnn_m >= 0.8 else "🟥")
                    lines.append(
                        f"| {ip_scale} | {n_ref} | {emoji} {cnn_m*100:.1f}% | {ssim_m:.4f} | {delta:+.4f} |"
                    )
    lines.append("")

    # ── 클래스별 세부 테이블 ─────────────────────────────────────────────
    lines += ["---", "", "## 클래스별 상세 결과", ""]

    for cls_name in MULTI_CLASSES:
        cls_entries = [e for e in raw_data if e["class"] == cls_name]
        if not cls_entries:
            continue

        lines += [
            f"### {cls_name}",
            "",
            "| 입력 도메인 | ip_scale | n_ref | CNN acc | SSIM |",
            "|------------|----------|-------|---------|------|",
        ]

        for entry in cls_entries:
            dom_short = DOMAIN_SHORT[entry["input_domain"]]
            for cond in entry["conditions"]:
                n_ref_str = "—" if cond["n_ref"] == 0 else str(cond["n_ref"])
                emoji = "🟩" if cond["cnn_acc"] >= 0.9 else ("🟨" if cond["cnn_acc"] >= 0.8 else "🟥")
                lines.append(
                    f"| {dom_short} | {cond['ip_scale']} | {n_ref_str} "
                    f"| {emoji} {cond['cnn_acc']*100:.0f}% | {cond['ssim_mean']:.4f} |"
                )
        lines.append("")

    # ── 권장 설정 요약 ───────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## 권장 설정 (CNN acc ≥ 90% 조건에서 SSIM 최소화)",
        "> baseline 대비 SSIM이 가장 낮은(=다양성 최대) ip_scale × n_ref 조합",
        "",
        "| 클래스 | 권장 ip_scale | 권장 n_ref | SSIM | CNN acc |",
        "|--------|--------------|------------|------|---------|",
    ]

    for cls_name in MULTI_CLASSES:
        cls_entries = [e for e in raw_data if e["class"] == cls_name]
        best_ssim, best_cond, best_entry = 999.0, None, None
        for entry in cls_entries:
            for cond in entry["conditions"]:
                if cond["cnn_acc"] >= 0.9 and cond["ssim_mean"] < best_ssim:
                    best_ssim = cond["ssim_mean"]
                    best_cond = cond
                    best_entry = entry
        if best_cond:
            lines.append(
                f"| **{cls_name}** | {best_cond['ip_scale']} | {best_cond['n_ref']} "
                f"| {best_cond['ssim_mean']:.4f} | {best_cond['cnn_acc']*100:.0f}% |"
            )
        else:
            lines.append(f"| **{cls_name}** | — | — | N/A | <90% |")

    lines += ["", ""]
    return "\n".join(lines)


# ── 메인 ────────────────────────────────────────────────────────────────
def main():
    # 총 생성 수 계산
    # ip_scale=0.0 → n_ref=0 고정 (1 조합)
    # ip_scale>0   → N_REF_LIST 각각 (3 조합)
    n_scale_configs = 1 + len([s for s in IP_SCALES if s > 0.0]) * len(N_REF_LIST)
    n_cls_dom = len(MULTI_CLASSES) * len(DOMAINS)
    n_total   = n_cls_dom * n_scale_configs * N_SEEDS

    print("=" * 65)
    print("Script 23: Few-Shot IP-Adapter 다양화 실험")
    print(f"  IP_SCALES:  {IP_SCALES}")
    print(f"  N_REF_LIST: {N_REF_LIST}")
    print(f"  N_SEEDS:    {N_SEEDS}")
    print(f"  Strength:   {STRENGTH} (고정)")
    print(f"  총 생성:    {n_total}장")
    print("=" * 65)

    device = get_device()
    print(f"\n디바이스: {device}")

    print("[1/5] CNN 로드...")
    cnn = load_cnn(CNN_CKPT, device)

    print("[2/5] WBCRouter 초기화 (파이프라인 포함)...")
    router = WBCRouter(router_ckpt=ROUTER_CKPT, cnn_ckpt=CNN_CKPT, device=None)

    print("[3/5] 파이프라인 워밍업...")
    # 첫 번째 도메인/클래스에서 파이프라인 로딩
    _dummy_path = sorted((DATA_DIR / "domain_a_pbc" / "basophil").glob("*.jpg"))[0]
    _dummy_img  = Image.open(_dummy_path).convert("RGB")
    router.route(_dummy_img, generate=True, seed=99)
    pipe = router.pipe

    print("[4/5] IP-Adapter 장착...")
    load_ip_adapter_on_pipe(pipe)

    print("[5/5] FewShotSampler 초기화...")
    sampler = FewShotSampler(DATA_DIR, seed=SEED_BASE)
    rng_input = random.Random(SEED_BASE)

    # ── 실험 루프 ────────────────────────────────────────────────────────
    raw_data   = []
    total_done = 0

    print(f"\n생성 루프 시작 (총 {n_total}장)...")

    for cls_name in MULTI_CLASSES:
        for input_domain in DOMAINS:
            img_dir = DATA_DIR / input_domain / cls_name
            if not img_dir.exists():
                print(f"  ⚠️ {input_domain}/{cls_name} 없음, skip")
                continue

            # 입력 이미지 고정 선택
            files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
            input_path = rng_input.choice(files)
            input_img  = Image.open(input_path).convert("RGB")

            # 프롬프트
            dom_idx = DOMAIN_IDX.get(input_domain, 0)
            cls_idx = CLASS_IDX[cls_name]
            prompt  = mod.build_class_domain_prompt(cls_idx, dom_idx)

            pool_sz = sampler.pool_size(cls_name, exclude_domain=input_domain)
            print(f"\n  ── {cls_name} × {DOMAIN_SHORT[input_domain]} "
                  f"({input_path.name}, 참조풀={pool_sz}장) ──")

            # 입력 이미지 저장
            img_save_dir = OUT_DIR / "images" / cls_name / DOMAIN_SHORT[input_domain]
            img_save_dir.mkdir(parents=True, exist_ok=True)
            input_img.save(img_save_dir / "input.png")

            entry = {
                "class":        cls_name,
                "input_domain": input_domain,
                "input_path":   str(input_path),
                "ref_pool_size": pool_sz,
                "conditions":   [],
            }

            # baseline용 참조 이미지 (ip_scale=0.0에서도 필요, 영향도=0으로 설정)
            _baseline_refs = sampler.sample(cls_name, exclude_domain=input_domain, n=1)

            for ip_scale in IP_SCALES:
                n_ref_configs = [0] if ip_scale == 0.0 else N_REF_LIST

                for n_ref in n_ref_configs:
                    # 참조 이미지 샘플링
                    # ip_scale=0.0이면 scale=0이므로 실질적 영향 없음; dummy ref 1장 전달
                    if n_ref > 0:
                        ref_imgs = sampler.sample(cls_name, exclude_domain=input_domain, n=n_ref)
                    else:
                        ref_imgs = _baseline_refs  # dummy (scale=0.0으로 무효화)

                    gen_results = []
                    label = f"ip={ip_scale} ref={n_ref}"
                    print(f"    [{label}] ", end="", flush=True)

                    for seed_offset in range(N_SEEDS):
                        seed = SEED_BASE + seed_offset

                        gen_img = generate_with_ip_adapter(
                            pipe, device, input_img, ref_imgs,
                            prompt, STRENGTH, ip_scale, seed
                        )
                        print("·", end="", flush=True)

                        # 저장
                        save_dir = img_save_dir / f"scale{int(ip_scale*100):03d}_ref{n_ref}"
                        save_dir.mkdir(exist_ok=True)
                        gen_img.save(save_dir / f"seed{seed_offset:02d}.png")

                        # 평가
                        ssim_val = ssim_pair(gen_img, input_img)
                        cnn_res  = cnn_classify(cnn, gen_img, device)

                        gen_results.append({
                            "seed":     seed_offset,
                            "ssim":     round(ssim_val, 4),
                            "pred":     cnn_res["pred_class"],
                            "conf":     cnn_res["conf"],
                            "correct":  cnn_res["pred_class"] == cls_name,
                        })
                        total_done += 1

                    # 조건별 집계
                    ssims   = [r["ssim"] for r in gen_results]
                    acc     = sum(r["correct"] for r in gen_results) / N_SEEDS
                    conf_m  = float(np.mean([r["conf"] for r in gen_results]))
                    pred_dist = {}
                    for r in gen_results:
                        pred_dist[r["pred"]] = pred_dist.get(r["pred"], 0) + 1

                    print(f" [{total_done}/{n_total}]")
                    print(f"      SSIM={np.mean(ssims):.4f}  "
                          f"CNN={int(acc*N_SEEDS)}/{N_SEEDS} ({acc*100:.0f}%)  "
                          f"conf={conf_m:.4f}  {pred_dist}")

                    entry["conditions"].append({
                        "ip_scale":  ip_scale,
                        "n_ref":     n_ref,
                        "ssim_mean": round(float(np.mean(ssims)), 4),
                        "ssim_std":  round(float(np.std(ssims)),  4),
                        "cnn_acc":   round(acc, 4),
                        "cnn_conf":  round(conf_m, 4),
                        "pred_dist": pred_dist,
                        "per_seed":  gen_results,
                    })

            raw_data.append(entry)

    # ── 저장 ─────────────────────────────────────────────────────────────
    print("\n결과 저장...")
    json_path = OUT_DIR / "summary.json"
    json_path.write_text(json.dumps({
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ip_scales":  IP_SCALES,
        "n_ref_list": N_REF_LIST,
        "n_seeds":    N_SEEDS,
        "strength":   STRENGTH,
        "data":       raw_data,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → {json_path}")

    md_path = OUT_DIR / "report.md"
    md_path.write_text(make_report(raw_data), encoding="utf-8")
    print(f"  → {md_path}")

    # ── 터미널 요약 ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("📊 전체 평균 CNN acc / SSIM (ip_scale × n_ref)")
    print("=" * 65)

    agg = {}
    for entry in raw_data:
        for cond in entry["conditions"]:
            key = (cond["ip_scale"], cond["n_ref"])
            if key not in agg:
                agg[key] = {"cnn": [], "ssim": []}
            agg[key]["cnn"].append(cond["cnn_acc"])
            agg[key]["ssim"].append(cond["ssim_mean"])

    base_ssim = float(np.mean(agg.get((0.0, 0), {}).get("ssim", [0.98])))
    base_cnn  = float(np.mean(agg.get((0.0, 0), {}).get("cnn",  [1.0])))
    print(f"  Baseline (ip=0.0):  CNN={base_cnn*100:.1f}%  SSIM={base_ssim:.4f}")
    print()

    for ip_scale in IP_SCALES:
        if ip_scale == 0.0:
            continue
        for n_ref in N_REF_LIST:
            key = (ip_scale, n_ref)
            if key in agg:
                cnn_m  = float(np.mean(agg[key]["cnn"]))
                ssim_m = float(np.mean(agg[key]["ssim"]))
                delta  = ssim_m - base_ssim
                flag   = "✅" if cnn_m >= 0.9 else ("⚠️" if cnn_m >= 0.8 else "❌")
                print(f"  {flag} ip={ip_scale} ref={n_ref}:  "
                      f"CNN={cnn_m*100:.1f}%  SSIM={ssim_m:.4f}  Δ={delta:+.4f}")

    print(f"\n✅ 완료! → {OUT_DIR}")


if __name__ == "__main__":
    main()
