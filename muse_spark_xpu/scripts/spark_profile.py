# Decode-step attribution for Muse Spark (offline, every rank runs this under
# mpiexec + external_launcher). Same engine settings as the bench_metrics runs.
#
# For each concurrency C: prefill C prompts untimed-ish, decode OSL tokens with the
# per-submodule timers on, then print per-rank TIMING lines normalised per decode
# step. Wall time per decode step comes from two generate() calls that differ
# only in max_tokens, so prefill cancels out.
import argparse
import functools
import os
import time

from vllm import LLM, SamplingParams

import spark_timing as T

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--tp", type=int, required=True)
p.add_argument("--pp", type=int, required=True)
p.add_argument("--concurrencies", default="1,16,64")
p.add_argument("--decode-tokens", type=int, default=33)
p.add_argument("--input-len", type=int, default=1000)
args = p.parse_args()

rank = int(os.environ.get("RANK", "0"))
llm = LLM(
    model=args.model, tensor_parallel_size=args.tp, pipeline_parallel_size=args.pp,
    distributed_executor_backend="external_launcher",
    max_model_len=4096, max_num_seqs=128, max_num_batched_tokens=2048,
    gpu_memory_utilization=0.90, enforce_eager=True, enable_prefix_caching=False,
    limit_mm_per_prompt={"image": 1, "audio": 0, "video": 0}, seed=0,
)
info = llm.apply_model(T.install)
if rank == 0:
    print(f"PROFILE installed: {info}", flush=True)


def prompts(n):
    # Same prompt construction as bench_metrics.py.
    return ["The quick brown fox jumps over the lazy dog. " * (args.input_len // 10)] * n


def gen(n, max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    llm.generate(prompts(n), sp, use_tqdm=False)
    return time.perf_counter() - t0


gen(max(int(c) for c in args.concurrencies.split(",")), 4)   # warm every shape once

for c in [int(x) for x in args.concurrencies.split(",")]:
    # Untimed reference: wall(prefill + 1 token) vs wall(prefill + N tokens).
    w1 = gen(c, 1)
    wn = gen(c, args.decode_tokens)
    steps = args.decode_tokens - 1
    if rank == 0:
        print(f"PROFILE WALL conc={c} ms_per_decode_step={(wn - w1) / steps * 1000:.1f} "
              f"(untimed run; prefill+1tok {w1:.2f}s)", flush=True)
    # Timed run: same work with the synchronising timers on.
    llm.apply_model(T.reset)
    llm.apply_model(functools.partial(T.enable, on=True))
    gen(c, args.decode_tokens)
    llm.apply_model(functools.partial(T.enable, on=False))
    stats = llm.apply_model(T.dump)[0]
    layers = stats.get("decode|layer", (0, 1))[1] / max(1, steps)
    for key, (ms, calls) in sorted(stats.items()):
        bucket, name = key.split("|", 1)
        if bucket != "decode":
            continue
        print(f"TIMING rank={rank} stage={rank // args.tp} conc={c} name={name} "
              f"ms_per_step={ms / steps:.2f} calls_per_step={calls / steps:.1f}", flush=True)
    if rank % args.tp == 0:
        # Sanity check: should equal this stage's layer count from VLLM_PP_LAYER_PARTITION.
        print(f"TIMING rank={rank} stage={rank // args.tp} conc={c} name=_layers_per_step "
              f"value={layers:.1f}", flush=True)

if rank == 0:
    print("PROFILE DONE", flush=True)
