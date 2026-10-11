# Sourced by run.sh and bench.sh. Everything they install, download, compile or cache stays inside this folder: on a
# server whose root disk is small and whose work disk holds this repo, nothing grows outside it. Other Python environments
# and inference engines on the machine are left as they are. Deleting the folder removes it all.
W=$PWD
export UV_CACHE_DIR=$W/.uv-cache UV_PYTHON_INSTALL_DIR=$W/.uv-python UV_TOOL_DIR=$W/.uv-tools UV_PYTHON_BIN_DIR=$W/.uv-bin
export UV_INSTALL_DIR=$W/.uv-bin UV_NO_MODIFY_PATH=1 UV_NO_CACHE=0 UV_LINK_MODE=hardlink
export HF_HOME=$W/.hf-home                                 # models, datasets, xet chunks
export XDG_CACHE_HOME=$W/.cache XDG_DATA_HOME=$W/.local/share XDG_STATE_HOME=$W/.local/state
export TMPDIR=$W/.tmp TMP=$W/.tmp TEMP=$W/.tmp             # nvcc and pip build trees, tarballs in flight
export TRITON_HOME=$W/.triton TRITON_CACHE_DIR=$W/.triton/cache   # fla's and torch's compiled kernels
export TORCH_HOME=$W/.cache/torch TORCH_EXTENSIONS_DIR=$W/.cache/torch_extensions TORCHINDUCTOR_CACHE_DIR=$W/.cache/torchinductor
export CUDA_CACHE_PATH=$W/.nv/ComputeCache                 # the driver's PTX JIT cache (~/.nv by default)
export PYTHONPYCACHEPREFIX=$W/.cache/pycache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$W/.uv-bin:$PATH:/usr/local/cuda/bin
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" logs
echo "work folder $W: $(df -h "$W" | awk 'NR==2 {print $4 " free on " $6}')"
