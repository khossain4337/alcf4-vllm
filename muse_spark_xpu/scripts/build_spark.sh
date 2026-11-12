#!/bin/bash
# venv_spark = venv's packages + the same kernels wheel + vLLM main e8ae84bf7 with
# patches/muse_spark_pr61879_xpu_on_vllm_main_e8ae84bf7.patch applied
# (draft PR #61879 merged onto that main + the 15-line XPU port). Run build.sh first.
set -eo pipefail
source "$(dirname "$(readlink -f "$0")")/env.sh"      # activates ${MU}/venv
pip freeze --local | grep -v -E '^(vllm|vllm[_-]xpu[_-]kernels)[=@ ]' > ${TMPDIR}/freeze.txt
deactivate
[ -d ${MU}/venv_spark ] || python -m venv --system-site-packages ${MU}/venv_spark
source ${MU}/venv_spark/bin/activate
pip install -q --no-deps -r ${TMPDIR}/freeze.txt
pip install -q --no-deps ${MU}/wheels/vllm_xpu_kernels-*.whl
cd ${MU}/src
[ -d vllm_spark ] || git clone -q https://github.com/vllm-project/vllm.git vllm_spark
cd vllm_spark
git fetch -q origin main; git checkout -q -B spark_xpu e8ae84bf7e992c63570905fce3073fd7a7de2412
git apply --index ${KIT}/patches/muse_spark_pr61879_xpu_on_vllm_main_e8ae84bf7.patch
git apply --index ${KIT}/patches/pp_one_batch_e8ae84bf7.patch
git -c user.name=muse-kit -c user.email=muse-kit@local commit -q -m "Muse Spark: PR #61879 + XPU port" || true
git log -1 --format='spark_xpu %H %s' | tee ${MU}/logs/commits_spark.txt
export VLLM_TARGET_DEVICE=xpu
rm -rf build dist *.egg-info
mkdir -p ${MU}/wheels_spark
pip wheel . -w ${MU}/wheels_spark --no-deps --no-build-isolation
pip install --force-reinstall --no-deps ${MU}/wheels_spark/vllm-*.whl
cd ${MU}
check_venv_clean
python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
echo BUILD_SPARK_DONE
