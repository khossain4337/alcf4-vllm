# SPDX-License-Identifier: Apache-2.0
"""
Offline concurrency-sweep benchmark for DeepSeek-V3 (XPU, external_launcher).

Loads the model ONCE, then for each concurrency C sweeps:
  - TTFT phase : generate C prompts with max_tokens=1   -> e2e_ttft (prefill+1tok)
  - Full phase : generate C prompts with max_tokens=OSL -> e2e_full, output tokens

Derived metrics (two-phase, robust on XPU offline path):
  TTFT_s        = e2e_ttft                       (wall time to first token, batch)
  TPOT_s        = (e2e_full - e2e_ttft)/(OSL-1)  (per-output-token decode time)
  throughput    = total_output_tokens / e2e_full (tokens/s, whole batch)

Also opportunistically reports per-request RequestOutput.metrics (real TTFT/TPOT
mean+median) when vLLM populates them.

Emits machine-parseable RESULT lines (grep 'RESULT' to build the table).
"""
import time
import statistics

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser


def create_parser():
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(model="deepseek-ai/DeepSeek-V3")
    g = parser.add_argument_group("benchmark")
    g.add_argument("--concurrencies", type=str, default="1,16,32,48",
                   help="comma-separated batch sizes to sweep")
    g.add_argument("--input-len", type=int, default=1000)
    g.add_argument("--output-len", type=int, default=1000)
    g.add_argument("--warmup", type=int, default=1,
                   help="warmup generate() calls before timing (JIT/compile)")
    return parser


def make_prompts(n, input_len):
    # Approximate input length by repetition (matches existing harness style).
    if input_len <= 50:
        base = "Hello, my name is"
    else:
        base = "The quick brown fox jumps over the lazy dog. " * (input_len // 10)
    return [base] * n


def real_metrics(outputs):
    """Pull true TTFT/TPOT from RequestOutput.metrics if vLLM populated them."""
    ttfts, tpots = [], []
    for o in outputs:
        m = getattr(o, "metrics", None)
        if not m:
            continue
        at = getattr(m, "arrival_time", None)
        ft = getattr(m, "first_token_time", None)
        lt = getattr(m, "last_token_time", None)
        ntok = len(o.outputs[0].token_ids)
        if at is not None and ft is not None:
            ttfts.append(ft - at)
        if ft is not None and lt is not None and ntok > 1:
            tpots.append((lt - ft) / (ntok - 1))
    return ttfts, tpots


def timed_generate(llm, prompts, max_tokens):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    return outs, dt


def main(args: dict):
    concurrencies = [int(x) for x in args.pop("concurrencies").split(",") if x]
    input_len = args.pop("input_len")
    output_len = args.pop("output_len")
    warmup = args.pop("warmup")

    llm = LLM(**args)

    # warmup (covers Triton JIT / first-pass kernel compile so it's not in timings).
    # NOTE: warm up ONLY at max concurrency (original phase-0 methodology). This
    # intentionally leaves the conc=1 prefill shape uncompiled at its timed run,
    # reproducing the expected PVC pattern of an inflated single-request TTFT that
    # dips at moderate concurrency and rises again at high concurrency. Kept
    # consistent with the phase-0 baseline for apples-to-apples TTFT reporting.
    if warmup > 0:
        wp = make_prompts(max(concurrencies), input_len)
        for _ in range(warmup):
            timed_generate(llm, wp, 4)

    print(f"\n{'='*70}")
    print(f"SWEEP input_len={input_len} output_len={output_len} "
          f"concurrencies={concurrencies}")
    print(f"{'='*70}")

    for C in concurrencies:
        prompts = make_prompts(C, input_len)

        # Phase A: TTFT (prefill + 1 token), batch of C
        _, e2e_ttft = timed_generate(llm, prompts, 1)

        # Phase B: full decode
        outs, e2e_full = timed_generate(llm, prompts, output_len)
        total_out = sum(len(o.outputs[0].token_ids) for o in outs)

        tpot = (e2e_full - e2e_ttft) / max(1, (output_len - 1))
        thr = total_out / e2e_full if e2e_full > 0 else 0.0

        # real per-request metrics if available
        rttft, rtpot = real_metrics(outs)
        real_str = ""
        if rttft:
            real_str += (f" real_TTFT_ms(mean/med)="
                         f"{1000*statistics.mean(rttft):.1f}/"
                         f"{1000*statistics.median(rttft):.1f}")
        if rtpot:
            real_str += (f" real_TPOT_ms(mean/med)="
                         f"{1000*statistics.mean(rtpot):.1f}/"
                         f"{1000*statistics.median(rtpot):.1f}")

        print(
            f"RESULT concurrency={C} isl={input_len} osl={output_len} "
            f"TTFT_s={e2e_ttft:.3f} TPOT_ms={1000*tpot:.2f} "
            f"throughput_tok_s={thr:.2f} total_out_tokens={total_out} "
            f"e2e_full_s={e2e_full:.2f}{real_str}",
            flush=True,
        )

    print(f"{'='*70}\nSWEEP COMPLETE\n{'='*70}", flush=True)


if __name__ == "__main__":
    parser = create_parser()
    args = vars(parser.parse_args())
    main(args)
