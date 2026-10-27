#!/usr/bin/env bash
# One-time setup on an SMU M3 LOGIN node (m3login0N). Do NOT sbatch this --
# it needs internet, and compute nodes usually have none.
#
#   bash ~/Shahed_Drone_Detection/setup_hpc.sh
#
# Layout:
#   code   ~/Shahed_Drone_Detection           (uploaded via Open OnDemand)
#   venv   $WORK/venvs/shahed                 python 3.10.12 + torch/mmcv/mmdet
#   data   $SCRATCH/shahed                    8.6 GB zip + 1.8 GB extract
#   runs   $WORK/shahed_runs                  checkpoints (scratch may be purged)
set -euo pipefail

# Every location is overridable, because $WORK on M3 has a quota that a torch
# venv (~8 GB) will blow through. Override like:
#     DATA_DIR=$SCRATCH/shahed VENV_DIR=$SCRATCH/shahed/venv \
#     SRC_DIR=$SCRATCH/shahed/edgeai-tensorlab bash setup_hpc.sh
CODE=${CODE_DIR:-$HOME/Shahed_Drone_Detection}
DATA=${DATA_DIR:-$SCRATCH/shahed}
VENV=${VENV_DIR:-$DATA/venv}
RUNS=${RUNS_DIR:-$DATA/runs}
SRC=${SRC_DIR:-$DATA/edgeai-tensorlab}
PY=$VENV/bin/python
# uv's cache defaults to ~/.cache and can be several GB; keep it off $HOME
export UV_CACHE_DIR=${UV_CACHE_DIR:-$DATA/.uv-cache}

mkdir -p "$DATA" "$RUNS" "$UV_CACHE_DIR" "$(dirname "$VENV")"

echo "############ 0/7  space preflight"
printf '  %-8s %s\n' CODE "$CODE" DATA "$DATA" VENV "$VENV" RUNS "$RUNS" SRC "$SRC"
for d in "$DATA" "$(dirname "$VENV")"; do
  [ -w "$d" ] || { echo "NOT WRITABLE: $d"; exit 1; }
done
# needs roughly 20 GB peak: venv ~8, zip 8.6, extract 1.8, ckpt 0.4, runs ~1
avail=$(df -Pk "$DATA" | awk 'NR==2{print int($4/1024/1024)}')
echo "  available at DATA: ${avail} GiB (need ~20 GiB peak)"
lfs quota -h -u "$USER" "$DATA" 2>/dev/null | head -3 || true

echo "############ 1/7  python 3.10 venv"
# NOT `python3 -m venv`: M3's system python3 has no working ensurepip, so that
# fails with "ensurepip ... returned non-zero exit status 1". uv fetches a
# prebuilt standalone CPython 3.10 (still cp310, so the mmcv cp310 wheel fits)
# and seeds pip itself. No root, no modules.
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -x "$PY" ]; then
  uv python install 3.10
  uv venv "$VENV" --python 3.10 --seed
fi
"$PY" -V

echo "############ 2/7  torch 2.4.0 + cu121"
# cu121 deliberately, NOT the cu124 in TI's setup.sh: OpenMMLab publishes no
# mmcv wheel index for cu124/torch2.4.0 (404), which forces a source build.
uv pip install --python "$PY" -q ninja cython "numpy==1.23.0"
uv pip install --python "$PY" -q torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "############ 3/7  mmengine + prebuilt mmcv 2.2.0"
# edgeai-mmdetection asserts mmcv >=2.0.0rc4,<=2.2.0 and mmengine >=0.7.1,<1.0.0
uv pip install --python "$PY" -q "mmengine>=0.7.1,<1.0.0"
uv pip install --python "$PY" -q mmcv==2.2.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html

echo "############ 4/7  TI repos (sparse: the monorepo is ~680 MB)"
if [ ! -d "$SRC" ]; then
  git clone --filter=blob:none --sparse -q \
      https://github.com/TexasInstruments/edgeai-tensorlab.git "$SRC"
  git -C "$SRC" sparse-checkout set edgeai-mmdetection edgeai-modeloptimization
fi
# deps TI's setup.sh omits but edgeai_torchmodelopt imports eagerly
uv pip install --python "$PY" -q protobuf onnx onnxscript pycocotools
"$VENV/bin/pip" install -q --no-build-isolation -e "$SRC/edgeai-mmdetection"
"$VENV/bin/pip" install -q --no-build-isolation -e "$SRC/edgeai-modeloptimization/torchmodelopt"

# tools/train.py imports save_model_proto from mmdeploy, which TI never installs.
# Their own line 22 is the commented-out original pointing at mmdet.utils, which
# still has that function with an identical signature; build_model_from_cfg is
# imported but never called. So delegate one, stub the other.
SP=$("$PY" -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SP/mmdeploy/utils"
echo "__version__ = '0.0.0+shim'" > "$SP/mmdeploy/__init__.py"
cat > "$SP/mmdeploy/utils/__init__.py" <<'PYEOF'
def save_model_proto(*args, **kwargs):
    from mmdet.utils.save_model import save_model_proto as _f
    return _f(*args, **kwargs)

def build_model_from_cfg(*args, **kwargs):
    raise NotImplementedError('mmdeploy shim: build_model_from_cfg unused by train.py')
PYEOF

echo "############ 5/7  download DataSets.zip (8.63 GB) + TI checkpoint"
ZIP=$DATA/DataSets.zip
EXPECT=8631372495
BOX="https://smu.app.box.com/index.php?rm=box_download_shared_file&shared_name=l97rp84nqgh2sv2aluk51vpe9tev42rk&file_id=f_2493962699598"
if [ ! -f "$ZIP" ] || [ "$(stat -c%s "$ZIP")" != "$EXPECT" ]; then
  curl -L --retry 3 -o "$ZIP" "$BOX"
fi
got=$(stat -c%s "$ZIP")
[ "$got" = "$EXPECT" ] || { echo "SIZE MISMATCH: $got != $EXPECT"; exit 1; }
echo "  zip OK ($got bytes)"

CKPT=$DATA/yolov7_l_coco_lite_640x640.pth
if [ ! -f "$CKPT" ]; then
  curl -L -o "$CKPT" \
    "http://software-dl.ti.com/jacinto7/esd/modelzoo/11_02_00/models/vision/detection/coco/edgeai-mmdet/yolov7_l_coco_lite_640x640_20250109_checkpoint.pth"
fi
echo "  checkpoint $(stat -c%s "$CKPT") bytes"

echo "############ 6/7  selective extract (74,526 of 250,883 entries)"
# skips __MACOSX, shahed_dataset_1/{clean,clean_jpg,mask,txt} (5.19 GB of
# duplication), TFRecords (1.07 GB), nested archives, old 1-class aod4, VOC, OBB
if [ ! -d "$DATA/ds/DataSets/combined_dataset" ]; then
  "$PY" - <<PYEOF
import zipfile, pathlib, time
OUT = pathlib.Path("$DATA/ds")
KEEP = ("DataSets/AOD 4/Images/",
        "DataSets/AOD 4/Annotations/YOLOv8 format/",
        "DataSets/AOD 4/Annotations/COCO Annotation format/",
        "DataSets/shahed/", "DataSets/Shahed-136_youtube_batch2/",
        "DataSets/shahed_dataset_1/prepped/", "DataSets/combined_dataset/")
def keep(n):
    if n.startswith("__MACOSX") or "/._" in n or n.endswith("/"): return False
    if "__pycache__" in n or n.rsplit("/",1)[-1] == ".DS_Store": return False
    return n.startswith(KEEP)
t0 = time.time()
with zipfile.ZipFile("$ZIP") as z:
    names = z.namelist(); members = [n for n in names if keep(n)]
    print(f"  extracting {len(members):,} of {len(names):,}")
    z.extractall(OUT, members=members)
print(f"  done in {time.time()-t0:.1f}s")
PYEOF
fi

echo "############ 7/7  build the 5-class dataset"
export SHAHED_ROOT=$DATA
"$PY" "$CODE/build_dataset.py"

echo
echo "================ verify ================"
"$PY" - <<'PYEOF'
import torch, mmcv, mmengine, mmdet
print("torch   ", torch.__version__, "| cuda build:", torch.version.cuda)
print("mmcv    ", mmcv.__version__)
print("mmengine", mmengine.__version__)
print("mmdet   ", mmdet.__version__)   # importing mmdet runs TI's version asserts
PYEOF
echo
echo "venv : $VENV"
echo "data : $DATA/tidl_dataset"
echo "runs : $RUNS"
echo "SETUP_OK  ->  next:  sbatch ~/Shahed_Drone_Detection/train_hpc.sbatch"
