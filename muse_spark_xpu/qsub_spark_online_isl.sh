#!/bin/bash
#PBS -N spark_online_isl
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
# Online bench, V1 + one batch, one server: ISL = OSL = 1000, then 5192, each at conc 1/16/64.

export MODE=bench SERVE_NODES=5 WRAP=wrap_spark_serve_isl.sh
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_PP_ONE_BATCH=1
export MAX_MODEL_LEN=131072 BENCH_ISL="1000 5192" BENCH_CONC=1,16,64
export BENCH_TIMEOUT=3600  # per ISL x conc
export TAG=isl
bash /lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/alcf4-vllm/muse_spark_xpu/scripts/pbs_muse_opt.sh
