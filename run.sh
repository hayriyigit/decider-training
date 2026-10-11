#!/usr/bin/env bash
# One LoRA round of decider on a fresh GPU server, end to end:
#   the training data from the Hub, the Python environments (uv), a causal-conv1d build for this GPU, the starting model's
#   score on the test split, the LoRA training + export + evaluation + push (scripts/decider_lora.py), and the logs and the
#   starting model's score uploaded next to the trained model (training/), so the results can be read from the Hub.
#
#   export HF_TOKEN=hf_...              # read access to the data and the starting model, write access to the target repo
#   ./run.sh configs/decider_judge3.yaml
#   PUSH=0 ./run.sh configs/decider_judge3.yaml   # no uploads from this server: everything is packed under results/ instead
# When the round finishes, bench.sh scores the starting and the trained model on system-one-bench-tr (BENCH=0 skips it).
#
# Resumable: run the same command again after a crash or a time limit; finished steps are skipped.
set -uo pipefail
cd "$(dirname "$0")"
CONFIG=${1:-configs/decider_judge3.yaml}
DATA_REPO=${DATA_REPO:-hayriyigit/decider-judge-ttl-mix}
: "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
export HF_TOKEN
. ./env.sh                                                 # every cache, download and build inside this folder

# The GPU: one card, chosen with CUDA_VISIBLE_DEVICES. Training needs ~40-60 GB of its memory; a card an inference engine
# (vLLM, SGLang, TGI...) has reserved would run both out of memory, so the script stops if the card is not free enough.
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one free GPU, e.g. export CUDA_VISIBLE_DEVICES=3 (nvidia-smi shows which)}"
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID      # the same numbering as nvidia-smi
GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1)
FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "GPU $GPU: ${FREE} MiB free" | tee -a logs/run_gpu.log
[ "${FREE:-0}" -ge "${MIN_FREE_MIB:-70000}" ] || { echo "GPU $GPU has only ${FREE} MiB free (needs ${MIN_FREE_MIB:-70000}); pick another card or set MIN_FREE_MIB"; exit 1; }
name=$(basename "$CONFIG" .yaml)
cfg() { grep -E "^  $1:" "$CONFIG" | head -1 | awk '{print $2}'; }
MODEL=$(cfg repo); REVISION=$(cfg revision); TARGET=$(grep -E "push_to_hub:" "$CONFIG" | sed -E 's/.*repo: *([^,} ]+).*/\1/')
TRAIN=$(cfg train); TEST=$(cfg test)
echo "config $CONFIG: $MODEL@$REVISION -> $TARGET; data $TRAIN, $TEST" | tee -a logs/$name.log

# 1. uv, a pinned version in .uv-bin (the shell profile is not edited): --torch-backend needs a recent uv
UV_VERSION=0.12.17
[ "$("$UV_INSTALL_DIR/uv" --version 2>/dev/null | awk '{print $2}')" = "$UV_VERSION" ] || \
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_UNMANAGED_INSTALL="$UV_INSTALL_DIR" sh
# torch built for the CUDA this machine's driver supports (PyPI's default build may need a newer driver: torch then sees no
# GPU and Triton fails with "0 active drivers"); environments built with another backend are rebuilt
export UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-auto}
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
STAMP="$UV_CACHE_DIR/torch_backend"
if [ "$(cat "$STAMP" 2>/dev/null)" != "$UV_TORCH_BACKEND $DRIVER" ]; then
  rm -rf "$UV_CACHE_DIR"/environments-v2/decider-* logs/causal_conv1d.log
  mkdir -p "$UV_CACHE_DIR"; echo "$UV_TORCH_BACKEND $DRIVER" > "$STAMP"
fi
echo "driver $DRIVER, torch backend $UV_TORCH_BACKEND, $(uv --version)" | tee -a logs/run_gpu.log

# 2. the data
if [ ! -s "$TRAIN" ] || [ ! -s "$TEST" ]; then
  uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download
snapshot_download('$DATA_REPO', repo_type='dataset', local_dir='.', allow_patterns=['$TRAIN', '$TEST'])" || exit 1
fi

# 3. the environments, and causal-conv1d built for this GPU (~1.7x faster training; optional: training runs without it)
ARCH=$(nvidia-smi -i "$GPU" --query-gpu=compute_cap --format=csv,noheader | head -1)
export TORCH_CUDA_ARCH_LIST=$ARCH MAX_JOBS=${MAX_JOBS:-8}           # the build uses CPU only; 8 jobs leave the other services room
echo "GPU: $(nvidia-smi -i "$GPU" --query-gpu=name,memory.total --format=csv,noheader | head -1), compute capability $ARCH" | tee -a logs/$name.log
for s in decider_eval decider_lora; do uv run --script scripts/$s.py --help > /dev/null 2>&1; done
for env in "$UV_CACHE_DIR"/environments-v2/decider-*/bin/python; do         # torch must see the GPU before anything else runs
  $env -c "import torch; ok = torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU', ok, '$env'); raise SystemExit(0 if ok else 1)" \
    2>&1 | tee -a logs/run_gpu.log
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "torch does not see the GPU (driver $DRIVER); send logs/run_gpu.log"; exit 1; }
done
if [ ! -s logs/causal_conv1d.log ]; then
  for env in "$UV_CACHE_DIR"/environments-v2/decider-*/bin/python; do                # only this folder's environments
    uv pip install --python $env setuptools wheel ninja packaging > /dev/null 2>&1
    nice -n 10 uv pip install --python $env --no-build-isolation causal-conv1d 2>&1 | tail -3
    $env -c "import causal_conv1d; print('causal_conv1d', causal_conv1d.__version__, '$env')" 2>/dev/null \
      || echo "causal-conv1d not built for $env: training runs without it (slower)"
  done 2>&1 | tee logs/causal_conv1d.log
fi

# 4. the starting model on the test split
START=${name}_eval_start.json
[ -s "$START" ] || uv run --script scripts/decider_eval.py --data "$TEST" --model "$MODEL" --revision "$REVISION" --out "$START" 2>&1 \
  | grep --line-buffered -vE "Fetching|Loading weights" | tee logs/${name}_eval_start.log

# 5. LoRA: items, train, export (merge + calibration), evaluate, push (PUSH=0: a copy of the config without the push)
RUN_CONFIG=$CONFIG
if [ "${PUSH:-1}" = 0 ]; then
  RUN_CONFIG=${CONFIG%.yaml}.nopush.yaml
  sed -E 's/^(  push_to_hub:).*/\1 null/' "$CONFIG" > "$RUN_CONFIG"
fi
uv run --script scripts/decider_lora.py --config "$RUN_CONFIG" 2>&1 | grep --line-buffered -vE "Fetching|Loading weights|Writing model" \
  | tee -a logs/${name}_lora.log
code=${PIPESTATUS[0]}
echo "EXIT $code" | tee -a logs/${name}_lora.log

# 7. after a finished round: the starting and the trained model on system-one-bench-tr (bench.sh; BENCH=0 skips it)
bench() {
  [ "$code" = 0 ] && [ "${BENCH:-1}" = 1 ] && [ -d "$WORK/model" ] || return 0
  ./bench.sh "$MODEL@$REVISION" "$WORK/model" 2>&1 | tee -a logs/bench_run.log
}

# 6. the records next to the model: training/ in the target repo, or (PUSH=0) results/ on this server
WORK=$(grep -E "^workdir:" "$CONFIG" | awk '{print $2}')
if [ "${PUSH:-1}" = 0 ]; then
  OUT=results/$name; mkdir -p "$OUT/training"
  cp -f "$CONFIG" "$START" "$WORK/train_log.jsonl" "$WORK/eval_report.json" logs/*.log "$OUT/training/" 2>/dev/null
  [ -d "$WORK/model" ] && cp -f "$WORK/eval_report.json" "$WORK/model/" 2>/dev/null
  tar -czf "results/${name}_reports.tar.gz" -C "$OUT" training                       # small: the numbers and the logs
  [ -d "$WORK/model" ] && tar -cf "results/${name}_model.tar" --exclude=__pycache__ -C "$WORK" model   # ~8 GB: the merged model
  ls -lh results/ | tee -a logs/${name}_lora.log
  echo "results: $(pwd)/results (reports: ${name}_reports.tar.gz, model: ${name}_model.tar)"
  bench
  exit $code
fi
uv run --with huggingface_hub python -c "
from pathlib import Path
from huggingface_hub import HfApi
api, files = HfApi(), [Path('$CONFIG'), Path('$START'), Path('$WORK/train_log.jsonl'), *Path('logs').glob('*.log')]
files = [f for f in files if f.exists()]
api.create_repo('$TARGET', private=True, exist_ok=True)
for f in files:
    api.upload_file(path_or_fileobj=str(f), path_in_repo=f'training/{f.name}', repo_id='$TARGET', commit_message=f'training record: {f.name}')
print('records uploaded to https://huggingface.co/$TARGET/tree/main/training:', [f.name for f in files])
" 2>&1 | tail -1
bench
exit $code
