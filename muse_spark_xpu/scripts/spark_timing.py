# Per-submodule wall-clock attribution for Muse Spark on XPU, without the torch
# profiler (PTI hangs/crashes outside PP<=2 on earlier stacks). Every timed region
# is bracketed by torch.xpu.synchronize(), so times are INCLUSIVE and serialised:
# good for "where does a decode step go", not for absolute overlap-aware numbers.
#
# Lives in an importable module so LLM.apply_model can pickle the installer by
# reference; with external_launcher the engine runs in-process on every rank.
import time
from collections import defaultdict

import torch

STATS = defaultdict(lambda: [0.0, 0])   # (bucket, name) -> [seconds, calls]
ENABLED = [False]
DECODE_MAX_TOKENS = 128                 # token rows <= this => decode step


def _bucket(args):
    for a in args:
        if isinstance(a, torch.Tensor) and a.dim() >= 1:
            return "decode" if a.shape[0] <= DECODE_MAX_TOKENS else "prefill"
    return "other"


def _timed(name, fn):
    def wrapper(*args, **kwargs):
        if not ENABLED[0]:
            return fn(*args, **kwargs)
        b = _bucket(args) if name != "allreduce" else _bucket(args)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        torch.xpu.synchronize()
        s = STATS[(b, name)]
        s[0] += time.perf_counter() - t0
        s[1] += 1
        return out
    return wrapper


# Submodule attribute names on MuseSpark1xDecoderLayer / Attention / MoE (amd/model.py).
MODULE_ATTRS = {
    "self_attn": "attn_block", "feed_forward": "moe_block",
    "qkv_proj": "attn.qkv_proj", "o_proj": "attn.o_proj", "attn": "attn.core",
    "router": "moe.router", "pooler": "moe.pooler", "pre_expert_norm": "moe.pre_norm",
    "experts": "moe.experts", "unpooler": "moe.unpooler",
}


def install(model):
    """apply_model target: wrap submodule forwards + the MoE / all-reduce internals."""
    n = 0
    layers = 0
    for mname, mod in model.named_modules():
        leaf = mname.rsplit(".", 1)[-1]
        if leaf in MODULE_ATTRS and ".layers." in f".{mname}":
            mod.forward = _timed(MODULE_ATTRS[leaf], mod.forward)
            n += 1
        if type(mod).__name__ == "MuseSpark1xDecoderLayer":
            mod.forward = _timed("layer", mod.forward)
            layers += 1
    # Class-level patches (shared by every layer): expert GEMMs, Muse finalize, TP all-reduce.
    from vllm.models.muse_spark.amd import muse_moe_runner as r
    r.MuseSparkTritonExperts.apply = _timed("moe.triton_experts", r.MuseSparkTritonExperts.apply)
    r.MuseSparkTopKWeightAndReduce.apply = _timed("moe.finalize", r.MuseSparkTopKWeightAndReduce.apply)
    from vllm.distributed.parallel_state import GroupCoordinator
    GroupCoordinator.all_reduce = _timed("allreduce", GroupCoordinator.all_reduce)
    return {"wrapped_modules": n, "decoder_layers": layers}


def reset(_model=None):
    STATS.clear()
    return True


def enable(_model=None, on=True):
    ENABLED[0] = on
    return on


def dump(_model=None):
    return {f"{b}|{k}": (round(v[0] * 1000, 3), v[1]) for (b, k), v in STATS.items()}
