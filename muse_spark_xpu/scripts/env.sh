# Source me. Upstream vLLM main on the Aurora frameworks/2026.1.0 module (oneapi/release/2026.1.0).
# venv layers on the module's python (--system-site-packages): torch/oneCCL/MPI come from the module.
export KIT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export MU=${MU:-/lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/muse_work}
mkdir -p ${MU}/tmp ${MU}/logs ${MU}/src ${MU}/wheels
export TMPDIR=${MU}/tmp            # UAN /tmp is shared across users
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128 HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=${HTTP_PROXY} https_proxy=${HTTPS_PROXY}
export HF_HOME=${HF_HOME:-/lus/flare/projects/datasets/model-weights}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}
module load frameworks/2026.1.0 >/dev/null 2>&1     # never pipe: conda activate dies in a subshell
export CMAKE_PREFIX_PATH="${CMPROOT}:${CMAKE_PREFIX_PATH}"
[ -d ${MU}/venv ] || python -m venv --system-site-packages ${MU}/venv
source ${MU}/venv/bin/activate
ulimit -c 0

# The module owns torch and every Intel runtime. A pip copy in the venv shadows it:
# CUDA torch drags nvidia-*, XPU torch wheels drag intel-sycl-rt/oneccl/mkl/... (the
# pt215 STRIP_INTEL list). Called at the end of build.sh / build_spark.sh.
check_venv_clean() {
    local bad
    bad=$(pip list --local --format=freeze 2>/dev/null | cut -d= -f1 | grep -i -E \
      '^(torch|torchvision|torchaudio|triton|nvidia-.*|tcmlib|intel-.*|impi-rt|umf|tbb|oneccl.*|mkl|dpcpp-cpp-rt|onemkl-.*)$' || true)
    if [ -n "${bad}" ]; then echo "VENV NOT CLEAN (shadows the module): ${bad}" | tr '\n' ' '; echo; return 1; fi
    python -c "import torch; assert torch.__file__.startswith('/opt/aurora'), torch.__file__"
    echo "venv clean: torch + Intel runtimes come from the module"
}
