#!/bin/bash
#PBS -N spark5_onebatch
#PBS -A datascience
#PBS -q capacity
#PBS -l select=5
#PBS -l place=scatter
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -o /lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/RUNS/outdir_aurora
#PBS -e /lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/RUNS/errordir_aurora
# Kit offline bench, TP8 x PP5, one batch per step. Target ~180 tok/s at conc 64-128.

export MODEL_KEY=spark5_onebatch
export VLLM_USE_V2_MODEL_RUNNER=0   # V1 runner, explicit (already the offline default)
export WALLTIME=03:00:00   # = walltime above; reused by the dirty-node requeue
export TAG=                # optional label, appended to the run dir name
bash /lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/alcf4-vllm/muse_spark_xpu/scripts/pbs_muse.sh
