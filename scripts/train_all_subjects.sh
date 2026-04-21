#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BRAIN_BACKBONE="${BRAIN_BACKBONE:-Shared_Temporal_Attention_Encoder}"
VISION_BACKBONE="${VISION_BACKBONE:-RN50}"
ADAPTOR_BACKBONE="${ADAPTOR_BACKBONE:-ShrinkAdapter}"
SEED="${SEED:-0}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
EPOCH="${EPOCH:-150}"
LR="${LR:-1e-4}"
LAMBDA_IMG2IMG="${LAMBDA_IMG2IMG:-0.0}"
LAMBDA_IMG2EEG="${LAMBDA_IMG2EEG:-0.0}"
LAMBDA_MMD="${LAMBDA_MMD:-0.0}"
LAMBDA_MAP="${LAMBDA_MAP:-0.0}"
EXP_SETTING="${EXP_SETTING:-intra-subject}"
DEVICE="${DEVICE:-cuda:6}"
CONFIG="${CONFIG:-configs/ats.yaml}"

SUBJECTS=()
for subject_idx in $(seq 1 10); do
    printf -v subject_name "sub-%02d" "$subject_idx"
    SUBJECTS+=("$subject_name")
done

EXP_NAME="${EXP_SETTING}_ubp_${BRAIN_BACKBONE}_${VISION_BACKBONE}_${ADAPTOR_BACKBONE}"
RUN_NAME="seed${SEED}_batch_size${TRAIN_BATCH_SIZE}_epoch${EPOCH}_img2eeg_${LAMBDA_IMG2EEG}_img2img_${LAMBDA_IMG2IMG}_mmd_${LAMBDA_MMD}_map_${LAMBDA_MAP}"

cd "$REPO_ROOT"

for SUBJECT in "${SUBJECTS[@]}"; do
    echo "==== Training ${SUBJECT} ===="
    python main.py \
        --config "$CONFIG" \
        --subjects "$SUBJECT" \
        --seed "$SEED" \
        --exp_setting "$EXP_SETTING" \
        --brain_backbone "$BRAIN_BACKBONE" \
        --vision_backbone "$VISION_BACKBONE" \
        --adaptor_backbone "$ADAPTOR_BACKBONE" \
        --epoch "$EPOCH" \
        --lr "$LR" \
        --device "$DEVICE" \
        --train_batch_size "$TRAIN_BATCH_SIZE" \
        --lambda_img2img "$LAMBDA_IMG2IMG" \
        --lambda_img2eeg "$LAMBDA_IMG2EEG" \
        --lambda_mmd "$LAMBDA_MMD" \
        --lambda_map "$LAMBDA_MAP"
done

echo "==== Averaging test metrics across ${#SUBJECTS[@]} subjects ===="
python "$SCRIPT_DIR/average_test_results.py" \
    --repo-root "$REPO_ROOT" \
    --config "$CONFIG" \
    --exp-name "$EXP_NAME" \
    --run-name "$RUN_NAME" \
    --subjects "${SUBJECTS[@]}"
