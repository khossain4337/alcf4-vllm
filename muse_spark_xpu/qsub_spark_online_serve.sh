#!/bin/bash
#PBS -N spark_online_serve
#PBS -A datascience
#PBS -q capacity
#PBS -l select=6
#PBS -l place=scatter
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -o /lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/RUNS/outdir_aurora
#PBS -e /lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/RUNS/errordir_aurora
# Online serve, V1 + one batch, TP8 x PP5 on nodes 1-5. Node 6 runs client_check.sh (USER_GUIDE.md section 4)
# once endpoint.txt is up; results in client_check.txt and summary.txt.

export MODE=serve SERVE_NODES=5
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_PP_ONE_BATCH=1
export MAX_MODEL_LEN=786432    # 768k; KV 1.16M tokens. 1048576 (config max) would leave ~10% headroom
export LONG_CTX="131072 524288"  # needle probes from the client, ~prompt tokens
export KEEP_SERVING=0             # 1 = keep serving until walltime after the check
export TAG=len1m
bash /lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/alcf4-vllm/muse_spark_xpu/scripts/pbs_muse_opt.sh
