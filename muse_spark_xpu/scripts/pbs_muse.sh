#!/bin/bash

#PBS -S /bin/bash
#PBS -j oe

# =============================================================================
# MUSE ON UPSTREAM vLLM main -- frameworks/2026.1.0 module (oneapi 2026.1.0,
# torch 2.13) + a venv with vllm main / vllm-xpu-kernels main built from source
# (muse_upstream/build.sh).
#
#   -v MODEL_KEY=glimmer -> 1 node, TP=4, online bring-up (run_new_model_bringup.sh)
#                           on the venv's plain vllm main: Glimmer is native there,
#                           so NO patch_muse_glimmer_support.sh.
#   -v MODEL_KEY=spark   -> 4 nodes, TP=8 PP=4 = 32 ranks, offline smoke via
#                           mpiexec + external_launcher, on the spark_xpu branch
#                           (PR #61879 merged onto the same main + XPU dispatch).
#
# SPARK SIZING: 1644 GB bf16. TP must divide 128 heads / 16 KV heads, so the
# in-node options are 1/2/4/8 (12 does not divide, 16 exceeds the node). TP=8
# x PP=4 puts ~51 GB of weights on each tile against ~58 GB usable at util
# 0.90. 3 nodes would force TP=4 x PP=9, and pipeline DEPTH is what costs on
# this stack. TP never crosses a node (it page-faults).
# =============================================================================

cd $PBS_O_WORKDIR/

MODEL_KEY="${MODEL_KEY:?set -v MODEL_KEY=glimmer|glimmer_tp2|glimmer_tp4|spark5|spark5_bench|spark4x12_bench|spark6_bench}"
WALLTIME="${WALLTIME:-02:00:00}"   # carried into the dirty-node requeue
ATTEMPT="${ATTEMPT:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
RESIDUE_LIMIT_MIB="${RESIDUE_LIMIT_MIB:-700}"
ROOT=/lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4
KIT="${KIT:-${ROOT}/alcf4-vllm/muse_spark_xpu}"   # PBS runs a spooled copy, so no $0
SELF="${KIT}/scripts/pbs_muse.sh"
MU="${MU:-${ROOT}/muse_work}"       # must match env.sh (venvs live here)
HF_HOME="${HF_HOME:-/lus/flare/projects/datasets/model-weights}"
HUB="${HF_HUB_CACHE:-${HF_HOME}/hub}"
PBS_ACCOUNT="${PBS_ACCOUNT:-datascience}"
SC=${KIT}/scripts
SPARK_REPO=${HUB}/models--meta-models--Muse-Spark-Open-816B-A42B
SPARK=${SPARK_REPO}/snapshots/$(cat ${SPARK_REPO}/refs/main)

case "${MODEL_KEY}" in
  # glimmer_tp2 / glimmer_tp4: the WW36 1-node reporting ladder (ISL=OSL=1000, online,
  # 1 run/point), so the upstream numbers line up with the August Frost+patch tables.
  glimmer_tp2|glimmer_tp4)
    MODEL=${HUB}/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68
    MODEL_TAG=MuseGlimmer30B_upstream
    VENV=${MU}/venv
    NODES=1; PP=1
    case "${MODEL_KEY}" in
      glimmer_tp2) TP=2; MASK=0,1 ;;
      glimmer_tp4) TP=4; MASK=0,1,2,3 ;;
    esac
    PROBE_CONC_DEFAULT="1,16,32,48,64,96,128,256"
    ;;
  # Job 8954982: TP=2 conc 256 died with UR_RESULT_ERROR_OUT_OF_RESOURCES in the sampler's
  # top-k/top-p. Upstream allots 22.58 GiB of KV at TP=2 vs 20.29 in Aug. Hypothesis: too
  # little headroom left for sampler temporaries. 0.85 returns ~3 GiB. Conc 128 is the
  # cross-check point (981.81 tok/s at 0.90).
  glimmer_tp2_u85)
    MODEL=${HUB}/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68
    MODEL_TAG=MuseGlimmer30B_upstream_u85
    VENV=${MU}/venv
    NODES=1; PP=1; TP=2; MASK=0,1
    PROBE_CONC_DEFAULT="128,256"
    export GPU_MEM_UTIL=0.85
    ;;
  glimmer)
    MODEL=${HUB}/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68
    MODEL_TAG=MuseGlimmer30B_upstream
    VENV=${MU}/venv
    TP=4; PP=1; NODES=1; MASK=0,1,2,3
    ;;
  spark)
    MODEL=${SPARK}
    MODEL_TAG=MuseSpark816B
    VENV=${MU}/venv_spark
    TP=8; PP=4; NODES=4; MASK=0,1,2,3,4,5,6,7
    # 62 layers. Stage 0 also carries the vision tower and embeddings, the last
    # stage the 202k-vocab lm_head, so the two middle stages take the extra layer.
    export VLLM_PP_LAYER_PARTITION=15,16,16,15
    ;;
  spark5)
    # Job 8949364 (4 nodes): weights 51.36 GiB/rank on 15-layer stages, 54.41 on 16-layer
    # (~3.05 GiB/layer/rank), KV -3.67 GiB after profiling -> "No available memory for the
    # cache blocks". util 0.95 only buys ~3 GiB. PP=5 caps a stage at 13 layers (~40 GiB).
    MODEL=${SPARK}
    MODEL_TAG=MuseSpark816B
    VENV=${MU}/venv_spark
    TP=8; PP=5; NODES=5; MASK=0,1,2,3,4,5,6,7
    export VLLM_PP_LAYER_PARTITION=12,13,13,12,12
    ;;
  # ---- Spark throughput (offline bench_metrics.py, ISL=OSL=1000) -------------------
  # Offline because multi-node online `mp` is ~11x slow here. NOT comparable with the
  # online Glimmer tables (offline reads higher: whole batch submitted at once).
  # Per-submodule decode attribution on the best layout (spark_profile.py + spark_timing.py).
  spark5_profile)
    MODEL=${SPARK}
    MODEL_TAG=MuseSpark816B
    VENV=${MU}/venv_spark
    PROFILE=1
    TP=8; PP=5; NODES=5; MASK=0,1,2,3,4,5,6,7
    export VLLM_PP_LAYER_PARTITION=12,13,13,12,12
    ;;
  # Follow-ups to the decode attribution (job 8958314), each ONE change vs spark5_bench:
  #   spark5_onebatch: max_concurrent_batches=1 (whole batch per step, no PP micro-batches)
  #   spark5_graph   : XPUGraph FULL_DECODE_ONLY instead of eager (the 405B-proven setting)
  spark5_onebatch|spark5_graph)
    MODEL=${SPARK}
    MODEL_TAG=MuseSpark816B_${MODEL_KEY#spark5_}
    VENV=${MU}/venv_spark
    BENCH=1
    TP=8; PP=5; NODES=5; MASK=0,1,2,3,4,5,6,7
    export VLLM_PP_LAYER_PARTITION=12,13,13,12,12
    case "${MODEL_KEY}" in
      spark5_onebatch) BENCH_PY=${SC}/bench_one_batch.py ;;
      # CCL_OP_SYNC=1 breaks multi-rank XPUGraph capture; 26.181.0 defaults to async, set anyway.
      spark5_graph)    EAGER_ARGS='--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}'
                       export CCL_OP_SYNC=0 ;;
    esac
    ;;
  spark5_bench|spark4x12_bench|spark6_bench)
    MODEL=${SPARK}
    MODEL_TAG=MuseSpark816B
    VENV=${MU}/venv_spark
    BENCH=1
    case "${MODEL_KEY}" in
      # The smoke-tested layout: 40 of 60 tiles, 587k KV tokens on the tightest stage.
      spark5_bench)    TP=8; PP=5;  NODES=5; MASK=0,1,2,3,4,5,6,7
                       export VLLM_PP_LAYER_PARTITION=12,13,13,12,12 ;;
      # Every tile of 4 nodes: TP=4 (128/4 heads, 16/4 KV heads) x PP=12, ~32 GiB of
      # weights/rank. Trades TP width for 12-deep pipeline -- depth is what cost DeepSeek.
      spark4x12_bench) TP=4; PP=12; NODES=4; MASK=0,1,2,3,4,5,6,7,8,9,10,11
                       export VLLM_PP_LAYER_PARTITION=5,5,5,5,6,6,5,5,5,5,5,5 ;;
      # One more stage than spark5: ~33 GiB/rank, roughly 2x the KV, shallower than 4x12.
      spark6_bench)    TP=8; PP=6;  NODES=6; MASK=0,1,2,3,4,5,6,7
                       export VLLM_PP_LAYER_PARTITION=10,11,11,10,10,10 ;;
    esac
    ;;
  *) echo "unknown MODEL_KEY=${MODEL_KEY}"; exit 1 ;;
esac

# 25.190.0/support/tools/xpu-smi/1.2.42 is GONE on the 26.181.0 image: it read blank on every
# tile and the gate passed blind (jobs 8949363/8949364). This is the module xpu-smi/1.2.43.
# Subshell: the module's grpc/protobuf LD_LIBRARY_PATH must not leak into the run.
XPUSMI_BASE=$(module load xpu-smi >/dev/null 2>&1; echo "${XPU_SMI_ROOT}")
XPUSMI_ENV="LD_LIBRARY_PATH=${XPUSMI_BASE}/lib64:${XPUSMI_BASE}/lib"
read_tile () {  # $1=device $2=tile -> MiB on stdout
    env -u ZE_AFFINITY_MASK ${XPUSMI_ENV} ${XPUSMI_BASE}/bin/xpu-smi dump -d "$1" -t "$2" -m 18 -i 1 -n 1 \
        2>/dev/null | tail -1 | awk -F, '{print $NF}' | tr -d ' '
}

echo "=========================================="
echo "MUSE/UPSTREAM ${MODEL_TAG}  TP=${TP} PP=${PP} nodes=${NODES}"
echo "Host    : $(hostname -f)"
echo "Job     : ${PBS_JOBID}"
echo "Attempt : ${ATTEMPT} of ${MAX_ATTEMPTS} (residue limit ${RESIDUE_LIMIT_MIB} MiB/tile)"
echo "=========================================="

# ---- clean-node gate (head node; WIKI_aurora_clean_node_selection.md) --------
WORST=0; WORST_WHERE=none
for d in 0 1 2 3 4 5; do for t in 0 1; do
    v="$(read_tile $d $t)"
    printf "  dev%s tile%s: %10s MiB\n" "$d" "$t" "${v:-?}"
    if [ -n "$v" ] && awk -v a="$v" -v b="$WORST" 'BEGIN{exit !(a>b)}'; then
        WORST="$v"; WORST_WHERE="dev${d} tile${t}"
    fi
done; done
echo "Worst tile: ${WORST_WHERE} at ${WORST} MiB (limit ${RESIDUE_LIMIT_MIB})"
if [ "${WORST_WHERE}" == "none" ]; then
    echo "GATE BLIND: xpu-smi returned no reading for any tile (${XPUSMI_BASE}). Refusing to run."
    exit 1
fi
if awk -v a="$WORST" -v b="$RESIDUE_LIMIT_MIB" 'BEGIN{exit !(a>b)}'; then
    echo "DIRTY NODE: ${WORST_WHERE} holds ${WORST} MiB."
    if [ "${ATTEMPT}" -lt "${MAX_ATTEMPTS}" ]; then
        qsub -A ${PBS_ACCOUNT} -q capacity -l select=${NODES}:ncpus=208 -l walltime=${WALLTIME} \
             -l filesystems=home:flare -l place=scatter -k doe -j oe -o ${ROOT}/outdir_aurora -N "mu-${MODEL_KEY}" \
             -v MODEL_KEY=${MODEL_KEY},ATTEMPT=$(( ATTEMPT + 1 )),MAX_ATTEMPTS=${MAX_ATTEMPTS},RESIDUE_LIMIT_MIB=${RESIDUE_LIMIT_MIB},WALLTIME=${WALLTIME} \
             "${SELF}" || echo "WARNING: requeue failed. Resubmit by hand."
    fi
    exit 0
fi
echo "Head node CLEAN. Proceeding."

# ---- environment ---------------------------------------------------------------
ulimit -c 0
module load frameworks/2026.1.0 >/dev/null 2>&1     # never pipe: conda activate dies in a subshell
source ${VENV}/bin/activate

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128 HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=${HTTP_PROXY} https_proxy=${HTTPS_PROXY}
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"

unset ONEAPI_DEVICE_SELECTOR
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export ZE_AFFINITY_MASK=${MASK}
export XPU_CCL_BACKEND=xccl
export FI_PROVIDER="cxi,tcp;ofi_rxm"
export CCL_ATL_TRANSPORT=ofi
export CCL_PROCESS_LAUNCHER=none        # module sets pmix; neither mp nor the wrapper uses PMI
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export GLOO_SOCKET_IFNAME=hsn0 TP_SOCKET_IFNAME=hsn0

export HF_HOME HF_HUB_CACHE=${HUB}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

export VLLM_CACHE_ROOT=${MU}/.vllm_cache;   mkdir -p "${VLLM_CACHE_ROOT}"
export TRITON_CACHE_DIR=${MU}/.triton_cache; mkdir -p "${TRITON_CACHE_DIR}"
export TMPDIR=${MU}/.tmp_${PBS_JOBID%%.*};   mkdir -p "${TMPDIR}"

echo "python: $(which python)"
python -c 'import torch,vllm,vllm_xpu_kernels,transformers;print("torch",torch.__version__,"| vllm",vllm.__version__,"| transformers",transformers.__version__)'

RC=0
if [[ "${MODEL_KEY}" == glimmer* ]]; then
    export MODEL MODEL_TAG TP PP
    export EXPERT_PARALLEL=0 MAX_MODEL_LEN=8192 MM_SMOKE=1
    export BASE_DIR=${MU}/results
    export RUN_SWEEP="${RUN_SWEEP:-1}" PROBE_CONC="${PROBE_CONC:-${PROBE_CONC_DEFAULT:-16,32}}"
    bash ${SC}/run_new_model_bringup.sh
    RC=$?
else
    NNODES=$(wc -l < $PBS_NODEFILE)
    HEAD_NODE=$(head -n1 $PBS_NODEFILE | sed 's/\..*//')
    export MASTER_ADDR=$(getent hosts "${HEAD_NODE}.hsn.cm.aurora.alcf.anl.gov" | awk 'NR==1{print $1}')
    [ -z "${MASTER_ADDR}" ] && export MASTER_ADDR=$(head -n1 $PBS_NODEFILE | sed 's/hostmgmt/hsn/')
    export TP PP
    RANKS=$(( TP * PP )); RPN=$(( RANKS / NNODES ))
    OUT=${MU}/results/$(date '+%Y-%m-%d_%H%M%S')_j${PBS_JOBID%%.*}_spark_tp${TP}_pp${PP}$([[ -n "${BENCH:-}" ]] && echo _bench)$([[ "${MODEL_KEY}" == spark5_onebatch || "${MODEL_KEY}" == spark5_graph ]] && echo _${MODEL_KEY#spark5_})$([[ -n "${PROFILE:-}" ]] && echo _profile)
    mkdir -p ${OUT}
    echo "nodes=${NNODES} ranks=${RANKS} ppn=${RPN} MASTER_ADDR=${MASTER_ADDR} partition=${VLLM_PP_LAYER_PARTITION}"
    echo "log: ${OUT}/smoke.log"
    # Same CPU / CCL-worker binding as the DeepSeek runs (run_bench_sweep_v024.sh).
    eval "$(${SC}/set_cpu-and-ccl-worker-affinity_aurora.sh ${RPN})"
    export CPU_BIND CCL_WORKER_AFFINITY
    echo "CPU_BIND=${CPU_BIND} CCL_WORKER_AFFINITY=${CCL_WORKER_AFFINITY}"
    if [[ -n "${PROFILE:-}" ]]; then
        echo "log: ${OUT}/profile.log"
        mpiexec --cpu-bind ${CPU_BIND} -n ${RANKS} -ppn ${RPN} -env PALS_WORLD_SIZE=${RANKS} \
            ${SC}/wrap.sh python -u ${SC}/spark_profile.py --model ${MODEL} --tp ${TP} --pp ${PP} \
            --concurrencies 1,16,64 --decode-tokens 33 > ${OUT}/profile.log 2>&1
        RC=$?
        grep -E '^PROFILE' ${OUT}/profile.log
        grep -E '^TIMING' ${OUT}/profile.log | sort -u > ${OUT}/timing.txt
        echo "TIMING lines: $(wc -l < ${OUT}/timing.txt) -> ${OUT}/timing.txt"
        grep -q 'PROFILE DONE' ${OUT}/profile.log || { echo "PROFILE DID NOT FINISH (rc=${RC})";
            grep -nE "Error|ERROR|Segmentation|ur_die|out of memory|No available memory" ${OUT}/profile.log | head -30; }
        echo "MUSE RC=${RC} for ${MODEL_TAG}"
        exit 0
    fi
    if [[ -n "${BENCH:-}" ]]; then
        # Same engine settings as the passing smoke (eager, util 0.90, image-capable
        # encoder budget) so the KV numbers describe a deployable multimodal config.
        # max_num_seqs 128 = the top of the ladder; KV holds >=587k tokens vs 256k needed.
        BENCH_CONC="${BENCH_CONC:-1,16,32,64,128}"
        echo "log: ${OUT}/bench.log  conc=${BENCH_CONC}"
        mpiexec --cpu-bind ${CPU_BIND} -n ${RANKS} -ppn ${RPN} -env PALS_WORLD_SIZE=${RANKS} \
            ${SC}/wrap.sh python -u ${BENCH_PY:-${SC}/bench_metrics.py} \
            --model ${MODEL} --tensor-parallel-size ${TP} --pipeline-parallel-size ${PP} \
            --distributed-executor-backend external_launcher \
            --max-model-len 4096 --max-num-seqs 128 --max-num-batched-tokens 2048 \
            --gpu-memory-utilization 0.90 ${EAGER_ARGS:---enforce-eager} --no-enable-prefix-caching \
            --limit-mm-per-prompt '{"image":1,"audio":0,"video":0}' --seed 0 \
            --concurrencies ${BENCH_CONC} --input-len 1000 --output-len 1000 --warmup 1 \
            > ${OUT}/bench.log 2>&1
        RC=$?
        grep -m1 -E 'ONE_BATCH|cudagraph_mode' ${OUT}/bench.log | cut -c1-200
        grep -E 'Available KV cache memory|XPU KV cache size' ${OUT}/bench.log | sed 's/^.*\] //' | sort | uniq -c
        grep -E '^RESULT|SWEEP COMPLETE' ${OUT}/bench.log | tee ${OUT}/summary.txt
        if ! grep -q 'SWEEP COMPLETE' ${OUT}/bench.log; then
            echo "BENCH DID NOT FINISH (rc=${RC}). Diagnostic extract:"
            grep -nE "NotImplementedError|No module named|Error|ERROR|Segmentation|out of memory|ur_die|page fault|No available memory" \
                ${OUT}/bench.log | grep -v 'Traceback (most recent' | head -40
        fi
        echo "MUSE RC=${RC} for ${MODEL_TAG}"
        exit 0
    fi
    mpiexec --cpu-bind ${CPU_BIND} -n ${RANKS} -ppn ${RPN} -env PALS_WORLD_SIZE=${RANKS} \
        ${SC}/wrap.sh python -u ${SC}/spark_smoke.py --model ${MODEL} --tp ${TP} --pp ${PP} \
        > ${OUT}/smoke.log 2>&1
    RC=$?
    grep -E '^SMOKE' ${OUT}/smoke.log
    if ! grep -q 'SMOKE DONE' ${OUT}/smoke.log; then
        echo "SMOKE DID NOT FINISH (rc=${RC}). Diagnostic extract:"
        grep -nE "NotImplementedError|No module named|Traceback|Error|ERROR|Segmentation|out of memory|ur_die|page fault" \
            ${OUT}/smoke.log | grep -v 'Traceback (most recent' | head -40
    fi
fi

echo ""
echo "---- post-run residue (head node) ----"
for d in 0 1 2 3 4 5; do for t in 0 1; do
    printf "  dev%s tile%s: %10s MiB\n" "$d" "$t" "$(read_tile $d $t)"
done; done
echo "MUSE RC=${RC} for ${MODEL_TAG}"
exit 0
