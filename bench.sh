#!/usr/bin/env bash
# decider models on system-one-bench-tr (github.com/UlkuTuncerKucuktas/system-one-bench-tr): the bench's built dev and test
# items from the Hub (hayriyigit/s1bench-tr-data, built with `s1bench build`), every model's answers in the bench's results
# format (scripts/s1bench_decider.py), packed into results/s1bench_predictions.tar.gz. Scoring (`s1bench score`) needs no GPU
# and is done elsewhere from that file.
#
#   export HF_TOKEN=hf_... CUDA_VISIBLE_DEVICES=5 MIN_FREE_MIB=20000
#   ./bench.sh hayriyigit/decider-4b-tr-judge-ttl@8b69f8a8dead233b95cc82e960a507a53a68af2c judge3_run/model
# A model is a Hub repo (optionally @revision) or a local folder; `MAIN=1` runs only the bench's 26 area tasks.
# Resumable: tasks already answered are skipped. Do not `git pull` while run.sh or this script is running.
set -uo pipefail
cd "$(dirname "$0")"
[ $# -ge 1 ] || { echo "usage: ./bench.sh <model>[@revision] ..."; exit 1; }
: "${HF_TOKEN:?export HF_TOKEN=hf_... first}"
export HF_TOKEN UV_TORCH_BACKEND=${UV_TORCH_BACKEND:-auto}
. ./env.sh                                                 # as run.sh: every cache inside this folder; one free GPU
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to one free GPU, e.g. export CUDA_VISIBLE_DEVICES=5}"
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID
GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d, -f1)
FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
[ "${FREE:-0}" -ge "${MIN_FREE_MIB:-20000}" ] || { echo "GPU $GPU has only ${FREE} MiB free (needs ${MIN_FREE_MIB:-20000})"; exit 1; }
UV_VERSION=0.12.17
[ "$("$UV_INSTALL_DIR/uv" --version 2>/dev/null | awk '{print $2}')" = "$UV_VERSION" ] || \
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_UNMANAGED_INSTALL="$UV_INSTALL_DIR" sh
mkdir -p logs bench results

# the bench's items: dev and test of every task
[ -s bench/data/xcopa_tr/test.jsonl ] || uv run --with huggingface_hub python -c "
from huggingface_hub import snapshot_download
snapshot_download('hayriyigit/s1bench-tr-data', repo_type='dataset', local_dir='bench', allow_patterns=['data/*/dev.jsonl', 'data/*/test.jsonl'])" || exit 1
TASKS=()
[ "${MAIN:-0}" = 1 ] && TASKS=(--tasks massive_intent_tr xsid_tr news_topic_tr buyuksinema_tr musteri_yorumlari_tr biltweetnews_tr tweet_emotion_tr \
  offenseval_tr sms_spam_tr hateday_tr xfact_teyit_tr checkthat_tr trclaim19_tr seahorse_tr webfaq_tr jev_noul_tr jev_noul_unseen_tr touche_tr \
  nli_tr stsb_tr belebele_tr xcopa_tr turblimp_tr include_tr global_mmlu_tr turkishmmlu_tr)

for spec in "$@"; do
  model=${spec%@*}; rev=$([ "$spec" != "$model" ] && echo "--revision ${spec#*@}")
  name=$(basename "${model%/model}")                         # judge3_run/model -> judge3_run
  echo "=== $name ($spec)" | tee -a logs/bench.log
  uv run --script scripts/s1bench_decider.py --bench bench --model "$model" $rev --name "$name" ${TASKS[@]+"${TASKS[@]}"} 2>&1 \
    | grep --line-buffered -vE "Fetching|Loading weights" | tee -a logs/bench.log
done
cp logs/bench.log bench/results/ 2>/dev/null
tar -czf results/s1bench_predictions.tar.gz -C bench results
ls -lh results/s1bench_predictions.tar.gz
