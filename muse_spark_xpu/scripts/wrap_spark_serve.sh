#!/bin/bash
# One per node under `mpiexec -ppn 1` (pbs_muse_opt.sh). Node 0 serves HTTP and runs
# smoke + bench; the others run --headless until node 0 writes ${OUT}/.done.
set -u
NODE_RANK=${PALS_RANKID:?}
exec > ${OUT}/node${NODE_RANK}.log 2>&1
command -v numactl >/dev/null || { echo "numactl missing"; exit 1; }

PORT=8000
NAME=Muse-Spark
SUM=${OUT}/summary.txt READY=${OUT}/.ready DONE=${OUT}/.done
PERSIST=${MU}/jit_cache/${CACHE_TAG:-spark_e8ae84bf7_tp8_pp${NNODES}_eager}/node${NODE_RANK}

# Stack env, as in pbs_muse.sh
export VLLM_TARGET_DEVICE=xpu VLLM_WORKER_MULTIPROC_METHOD=spawn
export ZE_FLAT_DEVICE_HIERARCHY=FLAT ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
# GPUs 0,1 (socket 0) and 3,4 (socket 1), to match the NUMA/CPU lists below.
export ZE_AFFINITY_MASK=0,1,2,3,6,7,8,9
export XPU_CCL_BACKEND=xccl FI_PROVIDER="cxi,tcp;ofi_rxm" CCL_ATL_TRANSPORT=ofi CCL_PROCESS_LAUNCHER=none
export GLOO_SOCKET_IFNAME=hsn0 TP_SOCKET_IFNAME=hsn0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export no_proxy="localhost,127.0.0.1,::1,.hsn.cm.aurora.alcf.anl.gov" NO_PROXY="localhost,127.0.0.1,::1,.hsn.cm.aurora.alcf.anl.gov"
unset ONEAPI_DEVICE_SELECTOR WORLD_SIZE RANK LOCAL_RANK
# Servesh's
export VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=true PYTHONNOUSERSITE=1
export RAYON_NUM_THREADS=16 OMP_NUM_THREADS=8 KMP_BLOCKTIME=1
export PYTHONWARNINGS="ignore:Warning only once for all operators"

# Node-local caches. The IPC path must stay short (sun_path ~108 bytes).
L=/tmp/mu_${UID}_${NODE_RANK}
export VLLM_RPC_BASE_PATH=/tmp/vr_${UID}_${NODE_RANK}
export VLLM_CACHE_ROOT=$L/vllm TRITON_CACHE_DIR=$L/triton NEO_CACHE_DIR=$L/neo TORCHINDUCTOR_CACHE_DIR=$L/inductor
export NEO_CACHE_PERSISTENT=1 TMPDIR=$L/tmp
rm -rf $L $VLLM_RPC_BASE_PATH
mkdir -p $VLLM_RPC_BASE_PATH $VLLM_CACHE_ROOT $TRITON_CACHE_DIR $NEO_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $TMPDIR
[ -d $PERSIST ] && rsync -a $PERSIST/ $L/ && echo "restored JIT caches from $PERSIST"

PID=
cleanup() {
  if [ -n "$PID" ] && kill -TERM $PID 2>/dev/null; then
    timeout 45 tail --pid=$PID -f /dev/null || kill -KILL $PID
  fi
  # Save only after a successful start, so a failed run can't clobber good caches.
  [ -f $READY ] && mkdir -p $PERSIST && rsync -a --delete --exclude tmp $L/ $PERSIST/ && echo "saved JIT caches"
  rm -rf $VLLM_RPC_BASE_PATH
}
trap cleanup EXIT

ARGS=(
  ${MODEL} --served-model-name ${NAME}
  --distributed-executor-backend mp --nnodes ${NNODES} --node-rank ${NODE_RANK}
  --master-addr ${MASTER_ADDR} --master-port ${MASTER_PORT}
  --tensor-parallel-size 8 --pipeline-parallel-size ${NNODES}
  --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 2048
  --gpu-memory-utilization 0.90 --enforce-eager --no-enable-prefix-caching
  --limit-mm-per-prompt '{"image":1,"audio":0,"video":0}' --seed 0
  --safetensors-load-strategy lazy
  # DRAM nodes 0/1, not the CPU-HBM nodes 2/3: HBM is 64 GB and its DRAM fallback can hang.
  --numa-bind --numa-bind-nodes 0 0 0 0 1 1 1 1
  --numa-bind-cpus 2-9,106-113 10-17,114-121 18-25,122-129 26-33,130-137
                   54-61,158-165 62-69,166-173 70-77,174-181 78-85,182-189
)
if [ ${NODE_RANK} -eq 0 ]; then ARGS+=(--host 0.0.0.0 --port ${PORT}); else ARGS+=(--headless); fi

echo "host=$(hostname) node_rank=${NODE_RANK} model=${MODEL}"
printf 'vllm serve'; printf ' %q' "${ARGS[@]}"; echo
vllm serve "${ARGS[@]}" &
PID=$!

if [ ${NODE_RANK} -ne 0 ]; then
  until [ -f $DONE ] || ! kill -0 $PID 2>/dev/null; do sleep 10; done
  exit 0
fi

t0=$SECONDS
until curl -fs localhost:${PORT}/v1/models | grep -q ${NAME}; do
  kill -0 $PID 2>/dev/null || { echo "FAIL: server died before ready" | tee -a $SUM; touch $DONE; exit 1; }
  sleep 10
done
touch $READY
echo "READY after $((SECONDS - t0)) s" | tee -a $SUM

t1=$SECONDS
curl -s localhost:${PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "'${NAME}'", "temperature": 0, "max_tokens": 512,
  "chat_template_kwargs": {"reasoning_effort": "low"},
  "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}]}' > ${OUT}/smoke.json
python -c 'import json,sys; r=json.load(open(sys.argv[1])); print("SMOKE", r["usage"], repr(r["choices"][0]["message"]["content"][-200:]))' \
  ${OUT}/smoke.json >> $SUM 2>&1 || head -c 500 ${OUT}/smoke.json >> $SUM
echo "SMOKE took $((SECONDS - t1)) s" >> $SUM

for C in ${BENCH_CONC//,/ }; do
  timeout 1800 vllm bench serve --base-url http://127.0.0.1:${PORT} --model ${NAME} --tokenizer ${MODEL} \
    --dataset-name random --random-input-len 1000 --random-output-len 1000 --ignore-eos \
    --num-prompts ${C} --max-concurrency ${C} --seed 0 > ${OUT}/bench_c${C}.log 2>&1
  rc=$?
  echo "BENCH conc=${C} rc=${rc} $(grep -E 'Output token throughput|Mean TTFT|Mean TPOT' ${OUT}/bench_c${C}.log | tr -s ' ' | tr '\n' ';')" >> $SUM
done
touch $DONE
