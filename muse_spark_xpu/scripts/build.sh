#!/bin/bash
# Build upstream vllm-xpu-kernels main + vllm main against the frameworks module torch.
set -eo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"
# Pinned: a rerun must not silently move to a newer main (2026-10-21 build).
VLLM_REF=${VLLM_REF:-e8ae84bf7e992c63570905fce3073fd7a7de2412}
KERNELS_REF=${KERNELS_REF:-19c6ddca8f6ef7a0f1de1bce1bc4dea9505cf8d4}
export MAX_JOBS=${MAX_JOBS:-32} CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS:-32}
cd ${MU}/src
[ -d vllm ] || git clone -q https://github.com/vllm-project/vllm.git vllm
[ -d kernels ] || git clone -q https://github.com/vllm-project/vllm-xpu-kernels.git kernels
git -C vllm fetch -q origin main; git -C vllm checkout -q ${VLLM_REF:-origin/main}
git -C kernels fetch -q origin main; git -C kernels checkout -q ${KERNELS_REF:-origin/main}
git -C kernels submodule update --init --recursive -q
echo "vllm $(git -C vllm rev-parse HEAD)  kernels $(git -C kernels rev-parse HEAD)" | tee ${MU}/logs/commits.txt
python -c 'import torch;print("torch",torch.__version__)'

# deps: everything except torch-family / triton / kernels (keep the module's)
FILTER='^\s*(torch|torchvision|torchaudio|torchcodec|triton|triton-xpu|vllm[_-]xpu[_-]kernels|pre-commit)\s*([=<>!~@;]|$)'
grep -viE "$FILTER" kernels/requirements.txt | grep -v '^--' > ${TMPDIR}/kreq.txt || true
# --no-deps: the module torch is "2.13.0a0" (pre-release), which pip treats as NOT
# satisfying torch>=2.13.0 (auto-round-lib), so resolving pulls a CUDA torch 2.14.1 +
# nvidia-* into the venv and shadows the module torch.
pip install -q --no-deps -r ${TMPDIR}/kreq.txt
pip install -q psutil pybind11 "setuptools>=78.1.1,<81" setuptools_scm ninja cmake

export VLLM_TARGET_DEVICE=xpu VLLM_XPU_AOT_DEVICES=pvc VLLM_XPU_XE2_AOT_DEVICES=pvc VLLM_XPU_ENABLE_XE3P=OFF
if [ -z "${SKIP_KERNELS}" ]; then
  cd ${MU}/src/kernels; rm -rf build dist *.egg-info
  time pip wheel . -w ${MU}/wheels --no-deps --no-build-isolation
  pip install --force-reinstall --no-deps ${MU}/wheels/vllm_xpu_kernels-*.whl
  cd ${MU}/src
fi
python -c "import torch,vllm_xpu_kernels;print('kernels OK',vllm_xpu_kernels.__file__)"

cd ${MU}/src/vllm
for f in requirements/common.txt requirements/xpu.txt; do grep -viE "$FILTER" $f | grep -vE '^\s*(-r|--)' ; done > ${TMPDIR}/vreq.txt
pip install -q --no-deps -r ${TMPDIR}/vreq.txt
python -c "import torch; assert torch.__file__.startswith('/opt/aurora'), torch.__file__"
rm -rf build dist *.egg-info
time pip wheel . -w ${MU}/wheels --no-deps --no-build-isolation
pip install --force-reinstall --no-deps ${MU}/wheels/vllm-*.whl
cd ${MU}
python -c "import torch,vllm,transformers;print('vllm',vllm.__version__,'tf',transformers.__version__,'torch',torch.__version__)"
check_venv_clean
echo BUILD_DONE
