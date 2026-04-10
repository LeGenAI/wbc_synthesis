"""
Script 08: Domain Gap Visualization
=====================================
4개 도메인 WBC 데이터셋의 분포 차이를 시각화:

  1. EfficientNet-B0 임베딩 (1280-d) 추출
  2. t-SNE 2D 투영 → 도메인 분리 시각화
  3. RGB 채널 히스토그램 → 색상 분포 차이 시각화
  4. 도메인 간 Fréchet Distance 행렬 출력

출력:
  results/domain_gap/tsne_domains.png
  results/domain_gap/rgb_histograms.png

Usage:
    python scripts/legacy/phase_08_17_domain_gap_multidomain/08_domain_gap_viz.py
    python scripts/legacy/phase_08_17_domain_gap_multidomain/08_domain_gap_viz.py --n_per_class 50   # 빠른 테스트
    python scripts/legacy/phase_08_17_domain_gap_multidomain/08_domain_gap_viz.py --n_per_class 200  # 기본
"""

import argparse
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 경로 설정 ─────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed_multidomain"
CKPT     = ROOT / "models" / "baseline_cnn.pt"
OUT_DIR  = ROOT / "results" / "domain_gap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 도메인 / 클래스 메타데이터 ────────────────────────────────────────
DOMAINS = [
    "domain_a_pbc",
    "domain_b_raabin",
    "domain_c_mll23",
    "domain_e_amc",
]
DOMAIN_LABELS = {
    "domain_a_pbc":    "PBC (Spain, MGG)",
    "domain_b_raabin": "Raabin (Iran, Giemsa)",
    "domain_c_mll23":  "MLL23 (Germany, Pappenheim)",
    "domain_e_amc":    "AMC (Korea, Romanowsky)",
}
DOMAIN_COLORS = {
    "domain_a_pbc":    "#E63946",
    "domain_b_raabin": "#2A9D8F",
    "domain_c_mll23":  "#F4A261",
    "domain_e_amc":    "#6A4C93",
}
CLASSES = ["basophil", "eosinophil", "lymphocyte", "monocyte", "neutrophil"]
CLASS_MARKERS = {
    "basophil":   "o",
    "eosinophil": "s",
    "lymphocyte": "^",
    "monocyte":   "D",
    "neutrophil": "P",
}
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# ── 디바이스 ──────────────────────────────────────────────────────────
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── FeatureExtractor (test_generation_similarity.py 동일 구현) ────────
class FeatureExtractor(nn.Module):
    def __init__(self, ckpt_path: Path, n_classes: int, device):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        base.classifier[1] = nn.Linear(base.classifier[1].in_features, n_classes)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        base.load_state_dict(state)
        self.features   = nn.Sequential(*list(base.children())[:-1])
        self.classifier = base.classifier
        self.eval()

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return feat.flatten(1)  # (B, 1280)


def build_transform(size: int = 224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


# ── frechet_distance (test_generation_similarity.py 동일 구현) ────────
def frechet_distance(mu1, sig1, mu2, sig2) -> float:
    from scipy.linalg import sqrtm
    eps  = 1e-6
    sig1 = sig1 + eps * np.eye(sig1.shape[0])
    sig2 = sig2 + eps * np.eye(sig2.shape[0])
    diff = mu1 - mu2
    covmean = sqrtm(sig1 @ sig2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig1 + sig2 - 2 * covmean))


# ── 이미지 경로 샘플링 ────────────────────────────────────────────────
def sample_image_paths(
    data_dir: Path,
    n_per_class: int,
    seed: int = 42,
) -> dict:
    """각 (domain, class) 조합에서 최대 n_per_class 장 경로 반환."""
    rng  = random.Random(seed)
    result = {}
    print("\n  [샘플 현황]")
    print(f"  {'도메인':<28} {'클래스':<14} {'사용':>6} / {'전체':>6}")
    print("  " + "-" * 58)
    for domain in DOMAINS:
        for cls in CLASSES:
            cls_dir = data_dir / domain / cls
            if not cls_dir.exists():
                warnings.warn(f"디렉토리 없음: {cls_dir}")
                result[(domain, cls)] = []
                continue
            paths = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
            if len(paths) > n_per_class:
                paths = rng.sample(paths, n_per_class)
            result[(domain, cls)] = paths
            tag = "⚠" if len(paths) < n_per_class else " "
            print(f"  {tag}{DOMAIN_LABELS[domain]:<28} {cls:<14} {len(paths):>6} / "
                  f"{len([p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTS]):>6}")
    print()
    return result


# ── 배치 임베딩 추출 ──────────────────────────────────────────────────
def extract_all_embeddings_batched(
    path_dict: dict,
    ckpt_path: Path,
    device: torch.device,
    batch_size: int = 64,
) -> tuple:
    """
    반환:
        embeddings   : np.ndarray (N, 1280)
        domain_idx   : np.ndarray (N,)  int
        class_idx    : np.ndarray (N,)  int
    """
    print(f"  FeatureExtractor 로드 중 (device={device})...")
    model = FeatureExtractor(ckpt_path, n_classes=8, device=device).to(device)
    tf    = build_transform()

    all_embeds, all_domain_idx, all_class_idx = [], [], []
    domain2idx = {d: i for i, d in enumerate(DOMAINS)}
    class2idx  = {c: i for i, c in enumerate(CLASSES)}

    total_paths = [(p, d, c)
                   for (d, c), paths in path_dict.items()
                   for p in paths]
    random.shuffle(total_paths)  # 배치 내 다양성

    batch_imgs, batch_domain, batch_class = [], [], []

    def flush_batch():
        if not batch_imgs:
            return
        t = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            emb = model.embed(t).cpu().numpy()
        all_embeds.append(emb)
        all_domain_idx.extend(batch_domain)
        all_class_idx.extend(batch_class)
        batch_imgs.clear(); batch_domain.clear(); batch_class.clear()

    for path, domain, cls in tqdm(total_paths, desc="  임베딩 추출", ncols=80):
        try:
            img = Image.open(path).convert("RGB")
            batch_imgs.append(tf(img))
            batch_domain.append(domain2idx[domain])
            batch_class.append(class2idx[cls])
        except Exception as e:
            warnings.warn(f"로드 실패 ({path.name}): {e}")
            continue
        if len(batch_imgs) >= batch_size:
            flush_batch()
    flush_batch()

    embeddings = np.vstack(all_embeds).astype(np.float32)
    domain_idx = np.array(all_domain_idx, dtype=np.int32)
    class_idx  = np.array(all_class_idx,  dtype=np.int32)
    print(f"  임베딩 추출 완료: {embeddings.shape} (N×1280)")
    return embeddings, domain_idx, class_idx


# ── t-SNE ──────────────────────────────────────────────────────────────
def run_tsne(
    embeddings: np.ndarray,
    perplexity: int = 40,
    n_iter: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    from sklearn.manifold import TSNE
    print(f"  t-SNE 실행 중 (perplexity={perplexity}, n_iter={n_iter})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=seed,
        init="pca",
        n_jobs=-1,
    )
    return tsne.fit_transform(embeddings).astype(np.float32)


def plot_tsne(
    tsne_2d: np.ndarray,
    domain_idx: np.ndarray,
    class_idx: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    domain_list = DOMAINS
    class_list  = CLASSES

    for di, domain in enumerate(domain_list):
        color = DOMAIN_COLORS[domain]
        for ci, cls in enumerate(class_list):
            mask = (domain_idx == di) & (class_idx == ci)
            if not mask.any():
                continue
            ax.scatter(
                tsne_2d[mask, 0], tsne_2d[mask, 1],
                c=color,
                marker=CLASS_MARKERS[cls],
                s=18, alpha=0.55, linewidths=0,
            )

    # 범례 1: 도메인 (색상)
    domain_patches = [
        mpatches.Patch(color=DOMAIN_COLORS[d], label=DOMAIN_LABELS[d])
        for d in domain_list
    ]
    legend1 = ax.legend(
        handles=domain_patches, title="Domain",
        loc="upper left", framealpha=0.3,
        labelcolor="white", facecolor="#0f0f23",
        title_fontsize=10, fontsize=8,
    )
    legend1.get_title().set_color("white")
    ax.add_artist(legend1)

    # 범례 2: 클래스 (마커)
    class_handles = [
        plt.scatter([], [], c="white", marker=CLASS_MARKERS[c], s=40, label=c)
        for c in class_list
    ]
    legend2 = ax.legend(
        handles=class_handles, title="Cell Type",
        loc="upper right", framealpha=0.3,
        labelcolor="white", facecolor="#0f0f23",
        title_fontsize=10, fontsize=8,
    )
    legend2.get_title().set_color("white")

    ax.set_title("t-SNE: EfficientNet-B0 Embeddings — 4 WBC Domains",
                 color="white", fontsize=14, pad=12)
    ax.set_xlabel("t-SNE dim 1", color="grey")
    ax.set_ylabel("t-SNE dim 2", color="grey")
    ax.tick_params(colors="grey")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ t-SNE 저장: {out_path}")


# ── RGB 히스토그램 ───────────────────────────────────────────────────
def compute_rgb_histograms(
    path_dict: dict,
    n_bins: int = 64,
) -> dict:
    """반환: {domain: {R/G/B: ndarray(n_bins)}}"""
    hist_data = {d: {"R": np.zeros(n_bins), "G": np.zeros(n_bins), "B": np.zeros(n_bins)}
                 for d in DOMAINS}
    bin_edges = np.linspace(0, 256, n_bins + 1)

    for (domain, cls), paths in tqdm(path_dict.items(), desc="  RGB 히스토그램", ncols=80):
        for p in paths:
            try:
                arr = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
            except Exception:
                continue
            for ch, name in enumerate(["R", "G", "B"]):
                h, _ = np.histogram(arr[:, :, ch].ravel(), bins=bin_edges)
                hist_data[domain][name] += h

    # 정규화 (확률 밀도)
    for domain in DOMAINS:
        for ch in ["R", "G", "B"]:
            total = hist_data[domain][ch].sum()
            if total > 0:
                hist_data[domain][ch] = hist_data[domain][ch] / total

    return hist_data


def plot_rgb_histograms(hist_data: dict, out_path: Path) -> None:
    channels = ["R", "G", "B"]
    ch_colors = ["#ff4444", "#44cc44", "#4488ff"]
    n_bins    = len(next(iter(hist_data.values()))["R"])
    x         = np.linspace(0, 255, n_bins)

    fig, axes = plt.subplots(4, 3, figsize=(15, 12), dpi=150)
    fig.patch.set_facecolor("#111827")

    for row, domain in enumerate(DOMAINS):
        domain_color = DOMAIN_COLORS[domain]
        for col, (ch, ch_color) in enumerate(zip(channels, ch_colors)):
            ax = axes[row, col]
            ax.set_facecolor("#1f2937")

            y = hist_data[domain][ch]
            ax.fill_between(x, y, alpha=0.4, color=ch_color)
            ax.plot(x, y, color=ch_color, linewidth=1.2)

            # 도메인 강조 테두리
            for spine in ax.spines.values():
                spine.set_edgecolor(domain_color)
                spine.set_linewidth(2)

            if row == 0:
                ax.set_title(f"Channel {ch}", color="white", fontsize=11)
            if col == 0:
                short = DOMAIN_LABELS[domain].split("(")[0].strip()
                ax.set_ylabel(short, color=domain_color, fontsize=9)
            ax.tick_params(colors="grey", labelsize=7)
            ax.set_xlim(0, 255)
            ax.set_ylim(bottom=0)

    fig.suptitle("RGB Channel Distributions — 4 WBC Domains",
                 color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✅ RGB 히스토그램 저장: {out_path}")


# ── 도메인 간 FD 행렬 출력 ───────────────────────────────────────────
def print_domain_fd_matrix(
    embeddings: np.ndarray,
    domain_idx: np.ndarray,
) -> None:
    print("\n  === 도메인 간 Fréchet Distance 행렬 (낮을수록 유사) ===\n")
    n_domains = len(DOMAINS)
    mu_list, sig_list = [], []
    for di in range(n_domains):
        mask = domain_idx == di
        emb  = embeddings[mask]
        mu_list.append(emb.mean(axis=0))
        sig_list.append(np.cov(emb, rowvar=False) if len(emb) > 1 else np.eye(emb.shape[1]))

    # 헤더
    labels = [DOMAIN_LABELS[d].split("(")[0].strip()[:16] for d in DOMAINS]
    col_w  = 12
    header = f"{'':22}" + "".join(f"{l:>{col_w}}" for l in labels)
    print("  " + header)
    print("  " + "-" * len(header))

    for i in range(n_domains):
        row_label = labels[i]
        row = f"  {row_label:<22}"
        for j in range(n_domains):
            if i == j:
                row += f"{'—':>{col_w}}"
            else:
                fd = frechet_distance(mu_list[i], sig_list[i], mu_list[j], sig_list[j])
                row += f"{fd:>{col_w}.1f}"
        print(row)
    print()


# ── argparse ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Domain gap visualization for WBC datasets")
    p.add_argument("--n_per_class", type=int, default=200,
                   help="도메인당 클래스당 샘플 수 (기본: 200)")
    p.add_argument("--batch_size",  type=int, default=64,
                   help="임베딩 추출 배치 크기")
    p.add_argument("--perplexity",  type=int, default=40,
                   help="t-SNE perplexity")
    p.add_argument("--n_iter",      type=int, default=1000,
                   help="t-SNE 반복 수")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Script 08 — Domain Gap Visualization")
    print(f"  device={device}, n_per_class={args.n_per_class}")
    print(f"{'='*60}")

    # 1) 이미지 경로 샘플링
    path_dict = sample_image_paths(DATA_DIR, args.n_per_class, args.seed)

    # 2) 임베딩 추출
    embeddings, domain_idx, class_idx = extract_all_embeddings_batched(
        path_dict, CKPT, device, args.batch_size
    )

    # 3) t-SNE
    tsne_2d = run_tsne(embeddings, args.perplexity, args.n_iter, args.seed)

    # 4) t-SNE 플롯
    plot_tsne(tsne_2d, domain_idx, class_idx, OUT_DIR / "tsne_domains.png")

    # 5) RGB 히스토그램
    print("\n  RGB 히스토그램 계산 중...")
    hist_data = compute_rgb_histograms(path_dict)
    plot_rgb_histograms(hist_data, OUT_DIR / "rgb_histograms.png")

    # 6) FD 행렬
    print_domain_fd_matrix(embeddings, domain_idx)

    print(f"\n  결과 저장 위치: {OUT_DIR}")
    print("  Done.\n")


if __name__ == "__main__":
    main()
