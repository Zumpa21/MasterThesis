# Fault-isolating inference runner.
#
# Runs the model in a separate, persistent worker subprocess instead of in-process.
# If a task causes the OS/cgroup to SIGKILL the worker (OOM), the parent detects
# the dead process, logs that task as failed, respawns a fresh worker (reloading
# the model once), and continues with the next task -- instead of losing the
# whole run like an in-process crash does.
#
# Every task still runs with IDENTICAL settings (same min_prob, same aug_score_opts,
# same augmentation) as the original inference_run -- nothing about the computation
# itself is changed. Only the failure mode changes: one bad task fails on its own
# instead of taking down the other 399.

import time
import queue
import multiprocessing as mp
from tqdm import tqdm

mp.set_start_method('spawn', force=True)  # required for CUDA-safe subprocesses


def _worker_main(model_path, adapter_path, dataset, fmt_opts, infer_params, cache_path, task_q, result_q, ready_q):
    # runs inside the subprocess: load model + build the same diskcache-backed
    # inference call the original script used, then serve tasks one at a time
    #
    # unsloth must be imported FIRST, before anything that pulls in transformers/peft
    # (model_tools.py imports peft/transformers at its own module top-level) -- otherwise
    # unsloth can't apply its patches and silently falls back to a slower path.
    from unsloth import FastLanguageModel
    from model_tools import load_unsloth_4bit
    from inference_tools import infer_task

    model, tokenizer = load_unsloth_4bit(model_path)
    if adapter_path is not None:
        # LoRA adapter kept separate (NOT merged): merging into 4-bit weights rounds small updates away
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"[worker] adapter loaded from {adapter_path}: {list(model.peft_config)}", flush=True)
    else:
        print(f"[worker] NO adapter, plain model from {model_path}", flush=True)
    FastLanguageModel.for_inference(model)

    cache = None
    if cache_path is not None:
        from diskcache import Cache
        cache = Cache(cache_path).memoize(typed=True, ignore=set(['model_tok', 'guess']))

    ready_q.put('ready')

    while True:
        keys = task_q.get()
        if keys is None:  # sentinel: shut down cleanly
            return
        try:
            res = infer_task(
                keys=keys, dataset=dataset, fmt_opts=fmt_opts,
                model_tok=(model, tokenizer), cache=cache, print_func=lambda *a, **kw: None,
                **infer_params,
            )
            result_q.put(('ok', res))
        except Exception as e:
            import traceback
            result_q.put(('error', f'{repr(e)}\n{traceback.format_exc()}'))
        finally:
            # FORCE CLEANUP: Prevents recursive DFS memory leaks between tasks
            import gc, torch
            gc.collect()
            torch.cuda.empty_cache()


class TaskWorker:
    def __init__(self, model_path, dataset, fmt_opts, infer_params, cache_path=None, adapter_path=None):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.dataset = dataset
        self.fmt_opts = fmt_opts
        self.infer_params = infer_params  # e.g. dict(min_prob=0.1, aug_score_opts=..., max_new_tokens=...)
        self.cache_path = cache_path
        self.task_q = None
        self.result_q = None
        self.ready_q = None
        self.proc = None
        self._spawn()

    def _spawn(self):
        self.task_q = mp.Queue()
        self.result_q = mp.Queue()
        self.ready_q = mp.Queue()
        self.proc = mp.Process(
            target=_worker_main,
            args=(self.model_path, self.adapter_path, self.dataset, self.fmt_opts, self.infer_params,
                  self.cache_path, self.task_q, self.result_q, self.ready_q),
            daemon=True,
        )
        self.proc.start()
        self.ready_q.get()  # block until model finished loading in the worker

    def run_task(self, keys, timeout=None):
        """Returns (success: bool, result_or_None).
        If `timeout` (seconds) elapses with no result and the worker is still
        alive, it's treated as HUNG (e.g. an explore() DFS tree so large it
        never OOMs, just runs forever) -- distinct from a crashed/OOM-killed
        worker, but handled the same way by the caller: skip, restart, continue.
        """
        self.task_q.put(keys)
        start = time.time()
        while True:
            if not self.proc.is_alive():
                return False, 'worker process died (likely OOM-killed)'
            if timeout is not None and (time.time() - start) > timeout:
                return False, f'task exceeded {timeout}s timeout (worker still alive but hung)'
            try:
                status, payload = self.result_q.get(timeout=2)
                return (status == 'ok'), payload
            except queue.Empty:
                continue

    def restart(self):
        if self.proc.is_alive():
            self.proc.terminate()  # SIGTERM
            self.proc.join(timeout=15)
        if self.proc.is_alive():
            # SIGTERM didn't work -- e.g. stuck deep in a CUDA/C call that
            # ignores it. Force kill so we don't hang here too.
            self.proc.kill()  # SIGKILL
            self.proc.join(timeout=15)
        self._spawn()

    def shutdown(self):
        if self.proc.is_alive():
            try:
                self.task_q.put(None)
                self.proc.join(timeout=15)
            except Exception:
                pass
        if self.proc.is_alive():
            self.proc.terminate()


def inference_run_resilient(model_path, dataset, fmt_opts, max_new_tokens=None, callback=None,
                             cache_path=None, failed_tasks_log=None, task_timeout=7200, adapter_path=None, **kwargs):
    # task_timeout (seconds, default 2h): if a single task's DFS search hasn't
    # returned by then, its worker is treated as hung and force-killed+respawned.
    # Your observed per-task pace has ranged ~5-65 min for a TTFT-trained model;
    # 2h gives real slow tasks room while still bounding a runaway search tree.
    # Set to None to disable (wait indefinitely, old behavior).
    if max_new_tokens is None:
        max_new_tokens = dataset.max_new_tokens(**fmt_opts)
    if 'max_tokens' in fmt_opts:
        fmt_opts = {**fmt_opts, 'max_tokens': fmt_opts['max_tokens'] - max_new_tokens, 'len_name': 'input'}

    infer_params = dict(max_new_tokens=max_new_tokens, **kwargs)  # min_prob, aug_score_opts, pass_guess, etc.
    worker = TaskWorker(model_path, dataset, fmt_opts, infer_params, cache_path=cache_path, adapter_path=adapter_path)

    results = {}
    failed = []
    try:
        with tqdm(dataset.grouped_keys().items(), desc='inference') as pbar:
            for base_key, tasks in pbar:
                results[base_key] = []
                for task_num, task in enumerate(tasks):
                    ok, payload = worker.run_task(task, timeout=task_timeout)
                    task_name = f'{base_key}_{task_num}'

                    if not ok:
                        pbar.write(f"!!! task '{task_name}' FAILED ({payload}) -- skipping, respawning worker")
                        failed.append(task_name)
                        if failed_tasks_log:
                            with open(failed_tasks_log, 'a') as f:
                                f.write(f"{task_name}\t{payload}\n")
                        worker.restart()
                        results[base_key].append([])  # counts as 0 candidates, same shape as a normal empty result
                        continue

                    res = payload
                    if callback is not None:
                        callback(res, name=task_name, value=1/len(tasks), print_func=pbar.write)
                    results[base_key].append([dict(output=g['output'], correct=g.get('correct')) for g in res])
    finally:
        worker.shutdown()

    if failed_tasks_log and failed:
        print(f"{len(failed)} task(s) failed and were skipped -- see {failed_tasks_log}")

    return results
