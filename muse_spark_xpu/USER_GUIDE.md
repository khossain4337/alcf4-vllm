# Muse Spark on Aurora — user guide (draft)

Muse Spark (816B MoE, BF16) runs on 5 Aurora nodes: TP8 inside a node, PP5 across nodes.
You get an OpenAI-compatible `vllm serve` endpoint. No build needed.

> Draft. Marked **(unverified)** = not yet run end to end on Aurora.

## 1. What you need

| Item | Value |
|---|---|
| Nodes | 5 for the server, +1 if you want a client node in the same job |
| Software | `module load frameworks/2026.1.0` + our venv `<root>/muse_work/venv_spark` |
| Weights | `/lus/flare/projects/datasets/model-weights` (`HF_HOME`), model `meta-models/Muse-Spark-Open-816B-A42B` |
| Access | Read access to the venv and the weights **(unverified: ask us)** |

`<root>` = `/lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4`.

## 2. Two settings that matter

```bash
export VLLM_USE_V2_MODEL_RUNNER=0   # V1 runner. V2 is ~4x slower here at 16 concurrent requests.
export VLLM_PP_ONE_BATCH=1          # one batch through the pipeline, no micro-batching
```

| Setting | Conc 16 output (ISL = OSL = 1000) |
|---|---|
| V1 + one batch (use this) | 75.8 tok/s |
| V2 + one batch | 18.1 tok/s |
| V2 + stock micro-batching | 22.1 tok/s |

- One batch helps because PP micro-batches do not overlap on Aurora. Unset it to get stock vLLM.
- Why: `<root>/EXPLAINER_pp_one_batch.md`.
- Sampling settings do not change speed.

## 3. Start the server

Easiest: copy our job script and submit it.

```bash
K=/lus/flare/projects/datasets/softwares/testing/vllm_bench_alcf4/alcf4-vllm/muse_spark_xpu
cp $K/qsub_spark_online_serve.sh ~/my_spark.sh   # edit -A, -o, -e; KEEP_SERVING=1 to keep it up
qsub ~/my_spark.sh
```

- Nodes 1–5 serve; node 6 runs the section 4 examples as checks (`scripts/client_check.py`) **(unverified)**.
- The run dir gets `endpoint.txt`; `source` it on any node in the job.
- Start-up takes ~22–25 min (weight load ~13 min). Plan walltime for it.
- Logs: one dir per job under `<root>/RUNS/` (`ROOT` in `scripts/pbs_muse_opt.sh`); read `summary.txt` first.

Doing it by hand: `scripts/wrap_spark_serve.sh` is the reference. It runs one `vllm serve` per node
under `mpiexec -ppn 1`: node 0 with `--host 0.0.0.0 --port 8000`, the others with `--headless`.
Copy its env block and `--numa-bind` lists as they are; without them decode was ~11x slower.

## 4. Talk to it

From a node in the same job (or anywhere that can reach node 0), set:

```bash
export OPENAI_BASE_URL=http://<node0>:8000/v1 OPENAI_API_KEY=EMPTY
export no_proxy=<node0>,$no_proxy NO_PROXY=<node0>,$NO_PROXY
curl $OPENAI_BASE_URL/models
```

Chat with curl:

```bash
curl $OPENAI_BASE_URL/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Muse-Spark", "max_tokens": 512,
  "chat_template_kwargs": {"reasoning_effort": "low"},
  "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}]}'
```

Python (the `openai` package is in the venv). More examples, tools and images included: `scripts/client_check.py`.

```python
from openai import OpenAI
c = OpenAI()   # reads OPENAI_BASE_URL, OPENAI_API_KEY
r = c.chat.completions.create(
    model="Muse-Spark", max_tokens=1024,
    messages=[{"role": "user", "content": "Explain pipeline parallelism in two sentences."}],
    extra_body={"chat_template_kwargs": {"reasoning_effort": "low"}},
)
print(r.choices[0].message.content)
```

- Spark reasons before it answers. `reasoning_effort` defaults to xhigh; use `"low"` for quick tests and give `max_tokens` room.
- Images: pass an `image_url` content part (one image per request).
- Tool calls and a separate reasoning field need the serve script's `--tool-call-parser muse --reasoning-parser muse` **(unverified)**.

Benchmark:

```bash
vllm bench serve --base-url http://<node0>:8000 --model Muse-Spark --tokenizer <snapshot dir> \
  --dataset-name random --random-input-len 1000 --random-output-len 1000 --ignore-eos \
  --num-prompts 16 --max-concurrency 16 --temperature 0
```

## 5. Limits

| Item | Value |
|---|---|
| Max context | 4096 in the bench script, 65536 in serve **(unverified)**; 131072 under test |
| Concurrency | `--max-num-seqs` 128 (bench), 16 (serve) |
| Eager mode | `--enforce-eager`; no graphs on XPU yet |
| Prefix caching | Off |
