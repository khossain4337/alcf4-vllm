# bench_metrics.py with ONE batch in flight instead of pp_size micro-batches.
#
# VllmConfig.max_concurrent_batches returns pp_size (config/vllm.py:583), so EngineCore
# builds a batch queue and the scheduler splits each step into PP micro-batches. Under
# external_launcher they do not overlap across stages (job 8958314: conc 16 step = 958 ms
# = ~5x the conc 1 step at PP5). With 1, EngineCore uses plain step(): the whole batch
# goes through all stages once per step.
#
# Patched before LLM() is built, in every rank's process (the engine runs in-process
# under external_launcher), so the scheduler and model-runner buffers agree.
import os
import runpy
import sys

from vllm.config.vllm import VllmConfig

VllmConfig.max_concurrent_batches = property(lambda self: 1)
print("ONE_BATCH: VllmConfig.max_concurrent_batches forced to 1", flush=True)

sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_metrics.py")
runpy.run_path(sys.argv[0], run_name="__main__")
