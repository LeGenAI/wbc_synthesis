#!/bin/bash
# Sequential WBC multi-domain downloader
# 각 파일을 완전히 받은 후에만 다음으로 진행 (0바이트 버그 방지)
set -e

MLL23_DIR="/Users/imds/Desktop/wbc_synthesis/data/raw/multi_domain/mll23"
RAABIN_DIR="/Users/imds/Desktop/wbc_synthesis/data/raw/multi_domain/raabin"
AMC_DIR="/Users/imds/Desktop/wbc_synthesis/data/raw/multi_domain/amc_korea"
ZENODO="https://zenodo.org/api/records/14277609/files"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

download_and_extract() {
    local URL="$1"
    local DEST_FILE="$2"
    local EXTRACT_DIR="$3"

    # 이미 추출됐으면 스킵
    if [ -d "$EXTRACT_DIR" ] && [ "$(find "$EXTRACT_DIR" -type f | wc -l)" -gt 10 ]; then
        log "SKIP (already extracted): $(basename $EXTRACT_DIR)"
        return 0
    fi

    # 파일이 없거나 0바이트면 다운로드
    if [ ! -s "$DEST_FILE" ]; then
        log "Downloading: $(basename $DEST_FILE)"
        rm -f "$DEST_FILE"
        wget -c --progress=dot:giga -O "$DEST_FILE" "$URL"
        log "Download complete: $(basename $DEST_FILE) ($(du -sh $DEST_FILE | cut -f1))"
    else
        log "Already downloaded: $(basename $DEST_FILE) ($(du -sh $DEST_FILE | cut -f1))"
    fi

    # 압축 해제
    log "Extracting: $(basename $DEST_FILE) → $(basename $EXTRACT_DIR)/"
    EXT="${DEST_FILE##*.}"
    if [ "$EXT" = "zip" ]; then
        python3 -c "
import zipfile, pathlib
with zipfile.ZipFile('$DEST_FILE') as zf:
    zf.extractall('$(dirname $DEST_FILE)')
print('  zip extracted OK')
"
    else
        unar -o "$(dirname $DEST_FILE)" -D "$DEST_FILE"
    fi
    log "Extracted: $(basename $EXTRACT_DIR) — $(find $EXTRACT_DIR -type f | wc -l | tr -d ' ') files"
}

# ── MLL23 남은 클래스 ──────────────────────────────────────────────
log "=== MLL23 (Germany / Zenodo) ==="
mkdir -p "$MLL23_DIR"

download_and_extract \
    "$ZENODO/monocyte.zip/content" \
    "$MLL23_DIR/monocyte.zip" \
    "$MLL23_DIR/monocyte"

download_and_extract \
    "$ZENODO/basophil.zip/content" \
    "$MLL23_DIR/basophil.zip" \
    "$MLL23_DIR/basophil"

# ── Raabin-WBC ────────────────────────────────────────────────────
log "=== Raabin-WBC (Iran) ==="
mkdir -p "$RAABIN_DIR"
RAABIN_BASE="http://dl.raabindata.com/WBC/Cropped_double_labeled"

download_and_extract \
    "$RAABIN_BASE/Train.rar" \
    "$RAABIN_DIR/Train.rar" \
    "$RAABIN_DIR/Train"

download_and_extract \
    "$RAABIN_BASE/TestA.rar" \
    "$RAABIN_DIR/TestA.rar" \
    "$RAABIN_DIR/TestA"

download_and_extract \
    "$RAABIN_BASE/TestB.zip" \
    "$RAABIN_DIR/TestB.zip" \
    "$RAABIN_DIR/TestB"

# ── AMC Korea ─────────────────────────────────────────────────────
log "=== AMC Multi-Focus (South Korea / Figshare) ==="
mkdir -p "$AMC_DIR"
# Figshare redirects to S3 — wget handles redirect automatically
download_and_extract \
    "https://ndownloader.figshare.com/files/48650791" \
    "$AMC_DIR/multi-focus-wbc-dataset.zip" \
    "$AMC_DIR/multi-focus-wbc-dataset"

# ── 최종 현황 요약 ────────────────────────────────────────────────
log "=== 다운로드 완료 — 최종 현황 ==="
for DOMAIN in mll23 raabin amc_korea; do
    DIR="/Users/imds/Desktop/wbc_synthesis/data/raw/multi_domain/$DOMAIN"
    N=$(find "$DIR" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.tif" -o -name "*.bmp" \) 2>/dev/null | wc -l | tr -d ' ')
    SIZE=$(du -sh "$DIR" 2>/dev/null | cut -f1)
    log "  [$DOMAIN] images=$N  size=$SIZE"
done
