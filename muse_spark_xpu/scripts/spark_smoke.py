# Offline smoke for Muse Spark on Aurora: every rank runs this under mpiexec with
# --distributed-executor-backend external_launcher (multi-node online `mp` is
# ~11x slow on this stack, so offline is the only usable multi-node path).
#
# Pass criteria are deliberately checkable without reading the text:
#   text  : the two greedy answers contain "Paris" and a prime number
#   image : a solid-red 224x224 image is answered with "red"; prompt_tokens must
#           grow vs the text-only prompt, which proves the vision tower ran
import argparse
import io
import os
import time

from PIL import Image

from vllm import LLM, SamplingParams

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--tp", type=int, default=int(os.environ.get("TP", 8)))
p.add_argument("--pp", type=int, default=int(os.environ.get("PP", 1)))
p.add_argument("--max-model-len", type=int, default=8192)
p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
p.add_argument("--max-tokens", type=int, default=512)
p.add_argument("--no-image", action="store_true")
args = p.parse_args()

rank0 = int(os.environ.get("RANK", "0")) == 0
t0 = time.time()
llm = LLM(
    model=args.model,
    tensor_parallel_size=args.tp,
    pipeline_parallel_size=args.pp,
    distributed_executor_backend="external_launcher",
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    enforce_eager=True,
    enable_prefix_caching=False,
    max_num_seqs=16,
    # Image only: audio/video dummies would inflate the encoder profiling budget.
    limit_mm_per_prompt={"image": 0 if args.no_image else 1, "audio": 0, "video": 0},
    seed=0,
)
if rank0:
    print(f"SMOKE engine up in {time.time() - t0:.0f} s", flush=True)

# Greedy, so a wrong answer is a real signal. Spark reasons before answering
# (reasoning_effort defaults to xhigh), hence the generous max_tokens and "low".
sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
kw = {"chat_template_kwargs": {"reasoning_effort": "low"}}

text_msgs = [
    [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
    [{"role": "user", "content": "Name three prime numbers."}],
]
t1 = time.time()
outs = llm.chat(text_msgs, sp, **kw)
if rank0:
    dt = time.time() - t1
    ntok = sum(len(o.outputs[0].token_ids) for o in outs)
    for m, o in zip(text_msgs, outs):
        print(f"SMOKE Q: {m[0]['content']}\nSMOKE prompt_tokens={len(o.prompt_token_ids)}"
              f" finish={o.outputs[0].finish_reason}\nSMOKE A: {o.outputs[0].text!r}\n", flush=True)
    ok = "paris" in outs[0].outputs[0].text.lower() and any(
        n in outs[1].outputs[0].text for n in ("2", "3", "5", "7", "11"))
    print(f"SMOKE TEXT {'PASS' if ok else 'FAIL'}  ({ntok} tokens in {dt:.1f} s)", flush=True)

if not args.no_image:
    img = Image.new("RGB", (224, 224), (220, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    import base64
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    q = "What single colour fills this image? Answer with one word."
    img_msgs = [[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": q}]}]]
    ref = llm.chat([[{"role": "user", "content": q}]], SamplingParams(max_tokens=1), **kw)
    out = llm.chat(img_msgs, sp, **kw)[0]
    if rank0:
        pi, pt = len(out.prompt_token_ids), len(ref[0].prompt_token_ids)
        print(f"SMOKE image prompt_tokens={pi} vs text-only={pt} -> +{pi - pt} from the image")
        print(f"SMOKE A: {out.outputs[0].text!r}")
        verdict = ("PASS" if "red" in out.outputs[0].text.lower()
                   else "PARTIAL" if pi > pt else "FAIL")
        print(f"SMOKE IMAGE {verdict}", flush=True)

if rank0:
    print("SMOKE DONE", flush=True)
