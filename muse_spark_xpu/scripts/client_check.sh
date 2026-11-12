#!/bin/bash
# Runs on the client node once the server is up (pbs_muse_opt.sh, MODE=serve): USER_GUIDE.md section 4, as checks.
set -u
exec > ${OUT}/client_check.log 2>&1
source ${OUT}/endpoint.txt
R=${OUT}/client_check.txt
echo "CLIENT $(hostname -s) -> ${OPENAI_BASE_URL}" > $R

curl -fs ${OPENAI_BASE_URL}/models | grep -q ${OPENAI_MODEL} && echo "MODELS PASS" >> $R || echo "MODELS FAIL" >> $R

vllm bench serve --base-url ${OPENAI_BASE_URL%/v1} --model ${OPENAI_MODEL} --tokenizer ${MODEL} \
  --dataset-name random --random-input-len 1000 --random-output-len 256 --ignore-eos \
  --num-prompts 16 --max-concurrency 16 --seed 0 --temperature 0 > ${OUT}/client_bench.log 2>&1
echo "BENCH conc=16 isl=1000 osl=256 rc=$? $(grep -E 'Output token throughput|Mean TTFT|Mean TPOT' ${OUT}/client_bench.log | tr -s ' ' | tr '\n' ';')" >> $R

python ${KIT}/scripts/client_check.py >> $R 2>&1
