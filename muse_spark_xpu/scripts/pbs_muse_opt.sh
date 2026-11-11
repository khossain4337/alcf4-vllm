#!/bin/bash
#PBS -l select=5
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -q capacity
#PBS -A datascience
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -o /lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/outdir_aurora
#PBS -N spark_serve
# Online multi-node `vllm serve` (mp) for Muse Spark, Servesh-style: one wrapper per node.
# Serves on the first SERVE_NODES nodes (PP = SERVE_NODES, TP = 8); extra nodes are clients.
#   -v MODE=bench (default): smoke + vllm bench serve, then exit
#   -v MODE=serve          : agent flags, write endpoint.txt, serve until walltime

ROOT=/lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4
export KIT=${KIT:-${ROOT}/alcf4-vllm/muse_spark_xpu} MU=${MU:-${ROOT}/muse_work}
export HF_HOME=/lus/flare/projects/datasets/model-weights
export HF_HUB_CACHE=${HF_HOME}/hub HF_DATASETS_CACHE=${HF_HOME} HF_MODULES_CACHE=${HF_HOME}
REPO=${HF_HUB_CACHE}/models--meta-models--Muse-Spark-Open-816B-A42B
export MODEL=${REPO}/snapshots/$(cat ${REPO}/refs/main)
export BENCH_CONC=${BENCH_CONC:-1,16,64} MODE=${MODE:-bench}

module load frameworks/2026.1.0 >/dev/null 2>&1     # never pipe: conda activate dies in a subshell
source ${MU}/venv_spark/bin/activate

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
export OUT=${MU}/results/$(date '+%Y-%m-%d_%H%M%S')_j${PBS_JOBID%%.*}_spark_${MODE}_tp8_pp${NNODES}
mkdir -p ${OUT}
head -n ${NNODES} $PBS_NODEFILE > ${OUT}/hosts
echo "mode=${MODE} serve_nodes=${NNODES} partition=${VLLM_PP_LAYER_PARTITION} master=${MASTER_ADDR}"
echo "client nodes: $(tail -n +$((NNODES + 1)) $PBS_NODEFILE | sed 's/\..*//' | tr '\n' ' ')"
echo "out=${OUT}"

# --cpu-bind none: vLLM's --numa-bind places the workers; mpiexec binding traps them on one core.
mpiexec --hostfile ${OUT}/hosts -n ${NNODES} -ppn 1 --cpu-bind none bash ${KIT}/scripts/wrap_spark_serve.sh
echo "mpiexec rc=$?"
cat ${OUT}/summary.txt
