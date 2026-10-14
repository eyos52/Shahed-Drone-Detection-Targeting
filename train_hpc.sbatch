#!/usr/bin/env bash
#SBATCH --job-name=shahed-yolov7
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Submit from a login node AFTER setup_hpc.sh has finished:
#     sbatch ~/Shahed_Drone_Detection/train_hpc.sbatch
#
# Runs three stages in one allocation so you do not queue twice:
#   1. finetune           tools/train.py
#   2. dump predictions   tools/test.py --out
#   3. confusion matrix   confusion_matrix_report.py
set -euo pipefail

# keep these in sync with whatever you passed to setup_hpc.sh
CODE=${CODE_DIR:-$HOME/Shahed_Drone_Detection}
DATA=${DATA_DIR:-$SCRATCH/shahed}
VENV=${VENV_DIR:-$DATA/venv}
# Stable name (NOT job-id suffixed) so `--resume auto` below can pick up where a
# previous job stopped. Matters on a 4-hour partition: resubmitting continues
# instead of restarting.
RUNS=${RUNS_DIR:-$DATA/runs}/${RUN_NAME:-yolov7_20ep}
SRC=${SRC_DIR:-$DATA/edgeai-tensorlab}/edgeai-mmdetection
PY=$VENV/bin/python
CFG=$SRC/configs_edgeailite/yolov7/yolov7_l_lite_aod4shahed.py

# Colab's kernel exports an MPLBACKEND the venv's matplotlib rejects, and
# mmdet.utils imports matplotlib -- harmless to set defensively here too.
export MPLBACKEND=Agg
# TI's tidl_meta_arch_mmdet_pb2.py predates protobuf 4; the newer onnx that
# onnxscript pulls needs protobuf >= 5. Both coexist only under pure-python.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

mkdir -p "$RUNS"
cp "$CODE/yolov7_l_lite_aod4shahed.py" "$CFG"

echo "=== node: $(hostname)  job: $SLURM_JOB_ID ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "cpus: ${SLURM_CPUS_PER_TASK:-?}   runs: $RUNS"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cd "$SRC"

echo; echo "################ 1/3  finetune ################"
# Pass --resume ONLY when a checkpoint actually exists. TI's tools/train.py does:
#     if args.resume == 'auto': cfg.resume = True; cfg.load_from = None
# so on a fresh work-dir --resume auto NULLS the pretrained COCO checkpoint and
# trains from random init (symptom: zero "size mismatch" lines and loss stuck
# near 17 instead of dropping to ~6 within 100 iters).
RESUME=()
if [ -f "$RUNS/last_checkpoint" ] || compgen -G "$RUNS/epoch_*.pth" >/dev/null 2>&1; then
    RESUME=(--resume auto)
    echo "  existing checkpoint found -> RESUMING"
else
    echo "  no checkpoint -> fresh finetune from $(basename "$(grep -o "load_from = '[^']*'" "$CFG" | head -1 | cut -d"'" -f2)")"
fi
"$PY" tools/train.py "$CFG" --work-dir "$RUNS" "${RESUME[@]}"

# newest checkpoint: prefer best_*, else the last epoch
BEST=$(ls -t "$RUNS"/best_*.pth 2>/dev/null | head -1 || true)
CKPT=${BEST:-$(ls -t "$RUNS"/epoch_*.pth | head -1)}
echo; echo "using checkpoint: $CKPT"

echo; echo "################ 2/3  dump test predictions ################"
"$PY" tools/test.py "$CFG" "$CKPT" \
    --out "$RUNS/test_preds.pkl" --work-dir "$RUNS/test"

echo; echo "################ 3/3  confusion matrix ################"
"$PY" "$CODE/confusion_matrix_report.py" "$CFG" "$RUNS/test_preds.pkl" \
    --repo "$SRC" --score-thr 0.3 --iou 0.5 | tee "$RUNS/confusion_matrix.txt"

echo; echo "DONE. artifacts in $RUNS"
