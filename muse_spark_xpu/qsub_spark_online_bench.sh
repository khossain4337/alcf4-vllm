#!/bin/bash
#PBS -N spark_online_bench
#PBS -A datascience
#PBS -q capacity
#PBS -l select=5
#PBS -l place=scatter
#PBS -l walltime=02:00:00
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -o /lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/RUNS/outdir_aurora
#PBS -e /lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/RUNS/errordir_aurora
# Online `vllm serve` mp, TP8 x PP5: smoke chat + vllm bench serve, then exit.

export MODE=bench
export SERVE_NODES=5       # PP; 6 also needs select=6
export BENCH_CONC=1,16,64
export TAG=                # optional label, appended to the run dir name
bash /lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/alcf4-vllm/muse_spark_xpu/scripts/pbs_muse_opt.sh
