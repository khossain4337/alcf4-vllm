#!/bin/bash
# =============================================================================
# NEW-MODEL BRING-UP  (Muse-Glimmer-30B, DeepSeek-V4-Flash-0731)
#
# Deliberately "out of the box": no --kernel-config, no VLLM_PP_LAYER_PARTITION,
# no pinned KV. The DeepSeek-V3 tuning in WW34 / the parallelism study was earned
# empirically for THAT checkpoint; carrying it over would confound a first run.
# The only two things inherited are the ones that are stack-level facts:
#   - omit --kernel-config so moe_backend stays "auto"   (WW34: 4-5x on XPU FP8)
#   - ZE_AFFINITY_MASK trimmed to the rank count          (parallelism study:
#     multiproc_executor asserts when local_world_size < visible tiles)
#
# STAGE 1  smoke: does the server come up and generate coherent text at all?
# STAGE 2  numbers: vllm bench sweep serve (only if stage 1 passed).
#
# Stage 1 exists because for a first run "it produced no tokens" and "it
# produced garbage" are different bugs, and a bench harness reports neither --
# it reports throughput for --ignore-eos random tokens either way.
# =============================================================================
set +e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${MODEL:?MODEL must be set}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL}")}"
TP="${TP:-8}"
PP="${PP:-1}"
EXPERT_PARALLEL="${EXPERT_PARALLEL:-1}"     # 0 to drop --enable-expert-parallel
ISL="${ISL:-1000}"
OSL="${OSL:-1000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
PROBE_CONC="${PROBE_CONC:-16,32}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-3600}"
PORT="${PORT:-8000}"
NUM_RUNS="${NUM_RUNS:-1}"
RUN_SWEEP="${RUN_SWEEP:-1}"
EXTRA_SERVE_ARGS="${EXTRA_SERVE_ARGS:-}"

RUN_TIMESTAMP=$(date '+%Y-%m-%d_%H%M%S')
BASE_DIR="${BASE_DIR:-${MU:-$HOME}/results}"
JOB_TAG="${PBS_JOBID%%.*}"; JOB_TAG="${JOB_TAG:-$$}"
LOG_DIR="${BASE_DIR}/${RUN_TIMESTAMP}_j${JOB_TAG}/bringup_${MODEL_TAG}_tp${TP}_pp${PP}"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "NEW-MODEL BRING-UP: ${MODEL_TAG}"
echo "  TP=${TP} PP=${PP} EP=${EXPERT_PARALLEL} mask=${ZE_AFFINITY_MASK:-unset}"
echo "  MODEL   = ${MODEL}"
echo "  LOG_DIR = ${LOG_DIR}"
echo "  vllm    = $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo N/A)"
echo "=========================================="

# ---- what does the checkpoint say it is, and do we support it? ---------------
# Cheap, and it turns "the job died 20 minutes in" into "the arch is not in the
# registry", which is a different report.
# NOTE the brace group: with `python - <<PY | tee ...` bash attaches the
# heredoc to the LAST command in the pipeline (tee), so tee echoes the Python
# source and python gets an empty stdin. Wrapping fixes the redirection target.
{ python - "${MODEL}" <<'PY'
import sys, traceback
path = sys.argv[1]
try:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    archs = getattr(cfg, "architectures", None) or []
    print("model_type    :", cfg.model_type)
    print("architectures :", archs)
    from vllm.model_executor.models.registry import ModelRegistry
    supported = set(ModelRegistry.get_supported_archs())
    missing = [a for a in archs if a not in supported]
    for a in archs:
        print(f"registry[{a}] :", "SUPPORTED" if a in supported else "*** NOT IN REGISTRY ***")
    # Muse Glimmer lives in site-packages as a hand-applied file port
    # (patch_muse_glimmer_support.sh), so ANY reinstall of the vllm wheel
    # silently un-supports it. Without this gate the job proceeds and dies
    # ~20 min later inside the engine, which reads like a model bug.
    if missing:
        print("ARCH_CHECK_FAILED", ",".join(missing))
except Exception:
    traceback.print_exc()
    print("ARCH_CHECK_FAILED probe-error")
PY
} 2>&1 | tee "${LOG_DIR}/arch_check.log"

if grep -q "ARCH_CHECK_FAILED" "${LOG_DIR}/arch_check.log" 2>/dev/null; then
    echo ""
    echo "=========================================="
    echo "ABORTING: architecture not in the vLLM registry."
    echo "  If this is Muse Glimmer, the file port is missing or was overwritten"
    echo "  by a wheel reinstall. Re-apply it and resubmit:"
    echo "    bash ${SCRIPT_DIR}/patch_muse_glimmer_support.sh"
    echo "  logs: ${LOG_DIR}/arch_check.log"
    echo "=========================================="
    exit 2
fi

# ---- shared engine flags ----------------------------------------------------
SERVE_ARGS=(
    --host 127.0.0.1
    --port "${PORT}"
    --tensor-parallel-size "${TP}"
    --pipeline-parallel-size "${PP}"
    --distributed-executor-backend mp
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --enforce-eager
    --no-enable-prefix-caching
)
[[ "${EXPERT_PARALLEL}" == "1" ]] && SERVE_ARGS+=(--enable-expert-parallel)
# shellcheck disable=SC2206
[[ -n "${EXTRA_SERVE_ARGS}" ]] && SERVE_ARGS+=(${EXTRA_SERVE_ARGS})

# =============================================================================
# STAGE 1 -- smoke: launch, wait for /health, ask for real tokens
# =============================================================================
echo ""
echo "---- STAGE 1: smoke test ----"
SRV_LOG="${LOG_DIR}/stage1_server.log"
PROBE_LOG="${LOG_DIR}/stage1_probe.log"

vllm serve "${MODEL}" "${SERVE_ARGS[@]}" > "${SRV_LOG}" 2>&1 &
SRV_PID=$!
echo "server pid=${SRV_PID} log=${SRV_LOG}"

DEADLINE=$(( $(date +%s) + SERVER_READY_TIMEOUT ))
# READY is published via a FLAG FILE, not a variable. `{ ... } | tee` runs the
# brace group in a SUBSHELL, so a `READY=1` assigned inside is lost to the
# parent. That cost job 8787762 a whole allocation: the server reached
# /health=200 and logged "SERVER READY", then the parent still saw READY=0,
# skipped every generation request and reported SMOKE FAILED on a server that
# was in fact healthy. (The same latent bug is in run_serve_sweep_probe.sh's
# stage 2.)
READY_FLAG="${LOG_DIR}/.server_ready"
rm -f "${READY_FLAG}"
{
    while [[ $(date +%s) -lt ${DEADLINE} ]]; do
        if ! kill -0 ${SRV_PID} 2>/dev/null; then
            echo "$(date '+%H:%M:%S') SERVER PROCESS EXITED before ready"
            break
        fi
        LISTEN=$(ss -ltn 2>/dev/null | grep -c ":${PORT}")
        HCODE=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:${PORT}/health" 2>/dev/null)
        echo "$(date '+%H:%M:%S') listening=${LISTEN} /health=${HCODE:-none}"
        if [[ "${HCODE}" == "200" ]]; then
            echo "$(date '+%H:%M:%S') SERVER READY"; : > "${READY_FLAG}"; break
        fi
        sleep 20
    done
} 2>&1 | tee "${PROBE_LOG}"

READY=0
[[ -f "${READY_FLAG}" ]] && READY=1

SMOKE_OK=0
if [[ ${READY} -eq 1 ]]; then
    # Greedy, so the output is reproducible and a wrong answer is a real signal.
    curl -s --max-time 300 "http://127.0.0.1:${PORT}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\": \"${MODEL}\", \"prompt\": \"The capital of France is\", \"max_tokens\": 24, \"temperature\": 0}" \
        2>&1 | tee "${LOG_DIR}/stage1_completion.json"
    echo ""
    curl -s --max-time 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"Name three prime numbers.\"}], \"max_tokens\": 48, \"temperature\": 0}" \
        2>&1 | tee "${LOG_DIR}/stage1_chat.json"
    echo ""
    if grep -q '"text"\|"content"' "${LOG_DIR}/stage1_completion.json" "${LOG_DIR}/stage1_chat.json" 2>/dev/null; then
        SMOKE_OK=1
        echo "STAGE 1 PASS: server generated tokens."
    else
        echo "STAGE 1 FAIL: server was ready but returned no text."
    fi

    # ---- image smoke (VLMs only) --------------------------------------------
    # A text-only smoke test on an image-text-to-text model proves half the
    # model. The vision tower, the adapter and the projection are exactly the
    # parts whose config we had to back-fill, so they are the parts most worth
    # exercising. A synthetic image is enough: we are testing the plumbing, not
    # the model's eyesight, so the pass criterion is "described the colour",
    # which a blind-but-wired-up model would still fail.
    if [[ "${MM_SMOKE:-0}" == "1" ]]; then
        echo ""
        echo "---- image smoke ----"
        IMG_PY="${LOG_DIR}/make_image.py"
        cat > "${IMG_PY}" <<'PYIMG'
import base64, io, sys
from PIL import Image
# Solid red 224x224 -- unambiguous, and 224 is the classic ViT input size.
img = Image.new("RGB", (224, 224), (220, 20, 20))
buf = io.BytesIO(); img.save(buf, format="PNG")
sys.stdout.write(base64.b64encode(buf.getvalue()).decode())
PYIMG
        B64=$(python "${IMG_PY}")
        if [[ -z "${B64}" ]]; then
            echo "IMAGE SMOKE SKIP: could not build the test image."
        else
            python - "${LOG_DIR}/stage1_image_req.json" "${MODEL}" "${B64}" <<'PYREQ'
import json, sys
out, model, b64 = sys.argv[1], sys.argv[2], sys.argv[3]
json.dump({
    "model": model,
    "messages": [{"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": "data:image/png;base64," + b64}},
        {"type": "text",
         "text": "What single colour fills this image? Answer with one word."},
    ]}],
    # 256, not 16. Muse Glimmer speaks the ATEM protocol: it emits a `to=self`
    # reasoning channel BEFORE the user-facing answer (see chat_template.jinja),
    # and this build has no ATEM parser to strip it. At 16 tokens job 8787833
    # stopped on finish_reason="length" while still restating the question, so
    # the run scored PARTIAL on a vision path that was in fact working.
    "max_tokens": 256,
    "temperature": 0,
}, open(out, "w"))
PYREQ
            curl -s --max-time 300 "http://127.0.0.1:${PORT}/v1/chat/completions" \
                -H 'Content-Type: application/json' \
                --data-binary "@${LOG_DIR}/stage1_image_req.json" \
                2>&1 | tee "${LOG_DIR}/stage1_image.json"
            echo ""
            # Report the prompt-token count: it is the objective evidence that
            # the patches reached the model, independent of what the model said.
            # A 224x224 image at patch 14 with merge 2 is 8*8 = 64 image tokens,
            # so an image request should run ~64 tokens longer than the text one
            # (measured: 134 vs 61 in job 8787833).
            python - "${LOG_DIR}/stage1_image.json" "${LOG_DIR}/stage1_chat.json" <<'PYCHK'
import json, sys
def usage(p):
    try:
        return json.load(open(p)).get("usage", {}) or {}
    except Exception:
        return {}
img, txt = usage(sys.argv[1]), usage(sys.argv[2])
pi, pt = img.get("prompt_tokens"), txt.get("prompt_tokens")
print(f"  image prompt_tokens={pi} vs text-only={pt}", end="")
if isinstance(pi, int) and isinstance(pt, int):
    print(f" -> +{pi - pt} tokens attributable to the image")
else:
    print()
PYCHK
            # Look anywhere in the reply, not just at its start: the answer
            # trails the reasoning channel rather than leading it.
            if grep -qi 'red' "${LOG_DIR}/stage1_image.json" 2>/dev/null; then
                echo "IMAGE SMOKE PASS: vision path answered with the right colour."
            elif grep -q '"content"' "${LOG_DIR}/stage1_image.json" 2>/dev/null; then
                echo "IMAGE SMOKE PARTIAL: vision path ran but did not say 'red'."
                echo "  -> check the +tokens line above: if the image DID add ~64"
                echo "     tokens the plumbing is fine and this is the missing"
                echo "     ATEM reasoning parser, not the config back-fill."
            else
                echo "IMAGE SMOKE FAIL: no content returned. See stage1_image.json."
            fi
        fi
    fi
else
    echo "STAGE 1 FAIL: server never became ready. Diagnostic extract:"
    grep -nE "not supported|NotImplementedError|no module named|Unsupported|Traceback|ERROR|CCL|xccl|ze_|Segmentation|No available memory|Free memory on device|out of memory" \
        "${SRV_LOG}" | tail -40 | tee "${LOG_DIR}/stage1_diagnosis.txt"
fi

kill -TERM ${SRV_PID} 2>/dev/null
wait ${SRV_PID} 2>/dev/null
sleep 20

if [[ ${SMOKE_OK} -ne 1 ]]; then
    echo ""
    echo "=========================================="
    echo "VERDICT ${MODEL_TAG}: SMOKE FAILED -- not running the sweep."
    echo "  logs: ${LOG_DIR}"
    echo "=========================================="
    exit 1
fi

if [[ "${RUN_SWEEP}" != "1" ]]; then
    echo "RUN_SWEEP=0 -- stopping after smoke."
    exit 0
fi

# =============================================================================
# STAGE 2 -- vllm bench sweep serve (the harness the JIRA asks for; validated
# on one node in the parallelism study, job 8779118)
# =============================================================================
echo ""
echo "---- STAGE 2: vllm bench sweep serve ----"
SERVE_CMD="vllm serve ${MODEL} ${SERVE_ARGS[*]}"
BENCH_CMD="vllm bench serve --model ${MODEL} --base-url http://127.0.0.1:${PORT} \
--dataset-name random --random-input-len ${ISL} --random-output-len ${OSL} \
--ignore-eos --seed 0 --percentile-metrics ttft,tpot,itl,e2el"

BENCH_PARAMS="${LOG_DIR}/bench_params.json"
python - "$BENCH_PARAMS" "$PROBE_CONC" <<'PY'
import json, sys
out, conc = sys.argv[1], [int(x) for x in sys.argv[2].split(",") if x]
recs = [{"max_concurrency": c, "num_prompts": 2 * c} for c in conc]
json.dump(recs, open(out, "w"), indent=2)
print("bench_params:", recs)
PY

S2_LOG="${LOG_DIR}/stage2_sweep_serve.log"
vllm bench sweep serve \
    --serve-cmd "${SERVE_CMD}" \
    --bench-cmd "${BENCH_CMD}" \
    --bench-params "${BENCH_PARAMS}" \
    --num-runs "${NUM_RUNS}" \
    --server-ready-timeout "${SERVER_READY_TIMEOUT}" \
    --show-stdout \
    --output-dir "${LOG_DIR}/sweep_out" \
    --experiment-name "${MODEL_TAG}_tp${TP}_pp${PP}" \
    2>&1 | tee "${S2_LOG}"
S2_RC=${PIPESTATUS[0]}

echo ""
echo "=========================================="
echo "VERDICT ${MODEL_TAG}: smoke PASS, sweep exit ${S2_RC}"
grep -hE "Successful requests|Failed requests|Request throughput|Output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT" \
    "${S2_LOG}" 2>/dev/null | tee "${LOG_DIR}/summary.txt"
echo "  logs: ${LOG_DIR}"
echo "=========================================="
exit ${S2_RC}
