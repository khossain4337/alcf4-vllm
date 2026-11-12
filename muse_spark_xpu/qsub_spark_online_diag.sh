#!/bin/bash
#PBS -N spark_online_diag
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
# Why is online bs 16 ~4x slower than offline? Same one-batch server twice: V2 runner, then V1
# (V1 + PP also turns async scheduling off). Each runs conc 16 greedy, then server-default sampling.

K=/lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/alcf4-vllm/muse_spark_xpu
export MODE=bench SERVE_NODES=5 BENCH_CONC=16 BENCH_TEMPS="0 default"
export VLLM_PP_ONE_BATCH=1 WRAP=wrap_spark_serve_diag.sh

TAG=diag_v2 bash $K/scripts/pbs_muse_opt.sh
sleep 60   # let ports 8000/29500 and the workers clear
TAG=diag_v1 VLLM_USE_V2_MODEL_RUNNER=0 bash $K/scripts/pbs_muse_opt.sh
