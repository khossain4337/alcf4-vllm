#!/bin/bash
# Run body; submit via ../qsub_spark_online_*.sh. Online multi-node `vllm serve` (mp) for Muse Spark, Servesh-style: one wrapper per node.
# Serves on the first SERVE_NODES nodes (PP = SERVE_NODES, TP = 8); extra nodes are clients.
#   MODE=bench (default): smoke + vllm bench serve, then exit
#   MODE=serve          : agent flags, write endpoint.txt, serve until walltime

ROOT=/lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4
export KIT=${KIT:-${ROOT}/alcf4-vllm/muse_spark_xpu} MU=${MU:-${ROOT}/muse_work}
export HF_HOME=/lus/flare/projects/datasets/model-weights
export HF_HUB_CACHE=${HF_HOME}/hub HF_DATASETS_CACHE=${HF_HOME} HF_MODULES_CACHE=${HF_HOME}
REPO=${HF_HUB_CACHE}/models--meta-models--Muse-Spark-Open-816B-A42B
export MODEL=${REPO}/snapshots/$(cat ${REPO}/refs/main)
export BENCH_CONC=${BENCH_CONC:-1,16,64} MODE=${MODE:-bench}

module load frameworks/2026.1.0 >/dev/null 2>&1     # never pipe: conda activate dies in a subshell
source ${MU}/venv_spark/bin/activate
if [ "${VLLM_PP_ONE_BATCH:-}" = 1 ]; then
  grep -q VLLM_PP_ONE_BATCH $(python -c 'import vllm,os; print(os.path.dirname(vllm.__file__))')/config/vllm.py \
    || { echo "VLLM_PP_ONE_BATCH=1 but venv_spark lacks patches/pp_one_batch_e8ae84bf7.patch"; exit 1; }
fi

export NNODES=${SERVE_NODES:-5}
case ${NNODES} in
  5) P=12,13,13,12,12 ;;
  6) P=10,11,11,10,10,10 ;;
  *) echo "no layer partition for ${NNODES} nodes"; exit 1 ;;
esac
export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION:-$P}
HEAD=$(head -n1 $PBS_NODEFILE | sed 's/\..*//')
export MASTER_ADDR=$(getent hosts "${HEAD}.hsn.cm.aurora.alcf.anl.gov" | awk 'NR==1{print $1}')
export MASTER_PORT=29500
RUN_TAG=spark_online_${MODE}_tp8_pp${NNODES}$([ "${VLLM_PP_ONE_BATCH:-}" = 1 ] && echo _onebatch)${TAG:+_${TAG}}
export OUT=${ROOT}/RUNS/${RUN_TAG}_$(date '+%Y-%m-%d_%H%M%S')_j${PBS_JOBID%%.*}
mkdir -p ${OUT}
head -n ${NNODES} $PBS_NODEFILE > ${OUT}/hosts
echo "mode=${MODE} serve_nodes=${NNODES} partition=${VLLM_PP_LAYER_PARTITION} master=${MASTER_ADDR}"
echo "client nodes: $(tail -n +$((NNODES + 1)) $PBS_NODEFILE | sed 's/\..*//' | tr '\n' ' ')"
echo "out=${OUT}"

# --cpu-bind none: vLLM's --numa-bind places the workers; mpiexec binding traps them on one core.
mpiexec --hostfile ${OUT}/hosts -n ${NNODES} -ppn 1 --cpu-bind none bash ${KIT}/scripts/${WRAP:-wrap_spark_serve.sh} < /dev/null &
MPID=$!
if [ ${MODE} = serve ]; then
  until [ -f ${OUT}/endpoint.txt ] || ! kill -0 $MPID 2>/dev/null; do sleep 30; done
  CLIENT=$(sed -n "$((NNODES + 1))p" $PBS_NODEFILE)
  if [ -f ${OUT}/endpoint.txt ] && [ -n "${CLIENT}" ]; then
    mpiexec --hosts ${CLIENT} -n 1 -ppn 1 bash ${KIT}/scripts/client_check.sh
    cat ${OUT}/client_check.txt >> ${OUT}/summary.txt
  fi
  [ "${KEEP_SERVING:-1}" = 1 ] || touch ${OUT}/.done
fi
wait $MPID
echo "mpiexec rc=$?"
cat ${OUT}/summary.txt
