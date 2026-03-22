#!/usr/bin/env bash
# =============================================================================
# Meneame Media Bias — Data Processing Pipeline
# =============================================================================
# Runs all pipeline steps sequentially:
#   Step 1: Ingest JSON files to Parquet
#   Step 2: Filter and subsample
#   Step 3: Bias labeling with franfj/fdtd_media_bias_E
#   Step 4: Extract interaction features
#
# Usage:
#   ./run_pipeline.sh                     # Run all steps
#   ./run_pipeline.sh --from 2            # Resume from step 2
#   ./run_pipeline.sh --only 3            # Run only step 3
#
# Requirements:
#   pip install polars torch transformers
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"
INPUT_DIR="/Users/franfj/Desktop/AI Factory/meneame/dataset/_test/"

# Parse arguments
FROM_STEP=1
ONLY_STEP=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --from)
            FROM_STEP="$2"
            shift 2
            ;;
        --only)
            ONLY_STEP="$2"
            shift 2
            ;;
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--from N] [--only N] [--input-dir /path/to/jsons]"
            echo ""
            echo "Options:"
            echo "  --from N        Start from step N (1-4)"
            echo "  --only N        Run only step N (1-4)"
            echo "  --input-dir     Path to directory with JSON files"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine which steps to run
should_run() {
    local step=$1
    if [[ $ONLY_STEP -gt 0 ]]; then
        [[ $step -eq $ONLY_STEP ]]
    else
        [[ $step -ge $FROM_STEP ]]
    fi
}

echo "=============================================="
echo " Meneame Media Bias — Data Pipeline"
echo "=============================================="
echo " Script dir: ${SCRIPT_DIR}"
echo " Data dir:   ${DATA_DIR}"
echo " Input dir:  ${INPUT_DIR}"
echo " Started:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

mkdir -p "${DATA_DIR}"

# ---- Step 1: Ingest ----
if should_run 1; then
    echo ""
    echo ">>> Step 1/4: Ingest JSON files to Parquet"
    echo "----------------------------------------------"
    python3 "${SCRIPT_DIR}/01_ingest_dataset.py" \
        --input-dir "${INPUT_DIR}" \
        --output-dir "${DATA_DIR}" \
        --batch-size 10000
    echo ">>> Step 1 complete."
fi

# ---- Step 2: Filter and subsample ----
if should_run 2; then
    echo ""
    echo ">>> Step 2/4: Filter and subsample"
    echo "----------------------------------------------"
    if [[ ! -f "${DATA_DIR}/articles_raw.parquet" ]]; then
        echo "ERROR: articles_raw.parquet not found. Run step 1 first."
        exit 1
    fi
    python3 "${SCRIPT_DIR}/02_filter_subsample.py" \
        --data-dir "${DATA_DIR}" \
        --target-size 15000 \
        --seed 42
    echo ">>> Step 2 complete."
fi

# ---- Step 3: Bias labeling ----
if should_run 3; then
    echo ""
    echo ">>> Step 3/4: Bias labeling (franfj/fdtd_media_bias_E)"
    echo "----------------------------------------------"
    if [[ ! -f "${DATA_DIR}/articles_subsample.parquet" ]]; then
        echo "ERROR: articles_subsample.parquet not found. Run step 2 first."
        exit 1
    fi
    python3 "${SCRIPT_DIR}/03_bias_labeling.py" \
        --data-dir "${DATA_DIR}" \
        --batch-size 32 \
        --device cpu
    echo ">>> Step 3 complete."
fi

# ---- Step 4: Interaction features ----
if should_run 4; then
    echo ""
    echo ">>> Step 4/4: Extract interaction features"
    echo "----------------------------------------------"
    if [[ ! -f "${DATA_DIR}/articles_labeled.parquet" ]]; then
        echo "ERROR: articles_labeled.parquet not found. Run step 3 first."
        exit 1
    fi
    if [[ ! -f "${DATA_DIR}/comments_raw.parquet" ]]; then
        echo "ERROR: comments_raw.parquet not found. Run step 1 first."
        exit 1
    fi
    python3 "${SCRIPT_DIR}/04_interaction_features.py" \
        --data-dir "${DATA_DIR}"
    echo ">>> Step 4 complete."
fi

echo ""
echo "=============================================="
echo " Pipeline finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo " Output files:"
for f in "${DATA_DIR}"/*.parquet; do
    if [[ -f "$f" ]]; then
        size=$(du -h "$f" | cut -f1)
        echo "   ${f##*/}  (${size})"
    fi
done
echo "=============================================="
