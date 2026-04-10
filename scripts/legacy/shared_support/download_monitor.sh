#!/bin/bash
# Download progress monitor — run this in a separate terminal
# Usage: bash scripts/legacy/shared_support/download_monitor.sh

ROOT="/Users/imds/Desktop/wbc_synthesis/data/raw/multi_domain"
DATASETS=("mll23" "raabin" "amc_korea")

echo "=== Multi-Domain WBC Download Monitor ==="
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "=== Multi-Domain WBC Download Monitor === $(date '+%H:%M:%S')"
    echo ""

    TOTAL_SIZE=0
    for ds in "${DATASETS[@]}"; do
        DIR="$ROOT/$ds"
        if [ -d "$DIR" ]; then
            # Count files
            N_ZIP=$(find "$DIR" -maxdepth 2 -name "*.zip" -o -name "*.rar" 2>/dev/null | wc -l | tr -d ' ')
            N_IMG=$(find "$DIR" -name "*.jpg" -o -name "*.png" -o -name "*.tif" -o -name "*.bmp" 2>/dev/null | wc -l | tr -d ' ')
            SIZE=$(du -sh "$DIR" 2>/dev/null | cut -f1)

            echo "  [$ds]"
            echo "    Archives downloaded : $N_ZIP"
            echo "    Images extracted    : $N_IMG"
            echo "    Disk usage          : $SIZE"

            # Show per-class if extracted
            for cls in neutrophil lymphocyte eosinophil monocyte basophil; do
                CLS_DIR="$DIR/$cls"
                if [ -d "$CLS_DIR" ]; then
                    N=$(find "$CLS_DIR" -name "*.jpg" -o -name "*.png" -o -name "*.tif" 2>/dev/null | wc -l | tr -d ' ')
                    printf "      %-15s %5s images\n" "$cls" "$N"
                fi
            done
            echo ""
        else
            echo "  [$ds] not started yet"
            echo ""
        fi
    done

    # Disk space
    AVAIL=$(df -h /Users/imds/Desktop | awk 'NR==2 {print $4}')
    echo "  Disk available: $AVAIL"
    echo ""

    # Active wget processes
    WGET_COUNT=$(pgrep -c wget 2>/dev/null || echo 0)
    if [ "$WGET_COUNT" -gt 0 ]; then
        echo "  Active downloads: $WGET_COUNT wget process(es) running"
    else
        echo "  Active downloads: none"
    fi

    sleep 15
done
