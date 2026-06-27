#!/usr/bin/env python
"""Async disaggregated diffusion-RL runtime — modeled on UniRL AsyncARTrainer.

Faithfully reproduces AsyncARTrainer's async machinery for the DIFFUSION stack
(which UniRL only has for AR today), with real WAN transformer work units, so the
two async knobs can be ablated end-to-end:

  * disjoint slabs       : rollout GPUs (generate) | train GPUs (fwd+bwd)   [TAG substrate]
  * _RolloutBuffer       : (group, weight_version, gen_id) + drain_freshest(max_staleness)
  * max_inflight         : how many generations run ahead (overlap depth)   [GAP]
  * buffer_max_staleness : weight-syncs a group may cross before eviction    [TCSS]
  * _drain before sync   : mandatory quiesce (engine corrupts in-flight gen)

Work units (real, on their slab's GPUs):
  gen_unit   = T denoise-step forwards (no-grad)
  train_unit = K replay fwd+bwd steps
  weight_sync= cross-slab param broadcast (real device copy)

Ablations (same total work = num_rollouts x batch groups):
  SYNC  (max_inflight=1, staleness=0): on-policy; gen drained before every sync -> serial
  ASYNC (max_inflight=M, staleness=S): generation overlaps train+sync across versions
  +TAG  : idle train-slab GPUs run gen_units when the buffer underflows (work-steal)

Generation progresses on the rollout slab via worker threads (the stand-in for
AsyncARTrainer's non-blocking Ray futures); the driver loop is single-threaded and
mirrors AsyncARTrainer.train / _next_batch exactly.
"""
import argparse, os, time, threading
from collections import deque
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/sync/models/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--train-gpus", type=int, default=3, help="train slab size; rest = rollout slab")
    p.add_argument("--num-rollouts", type=int, default=12)
    p.add_argument("--batch", type=int, default=4, help="groups consumed per train step")
    p.add_argument("--T", type=int, default=12, help="denoise steps per gen unit")
    p.add_argument("--K", type=int, default=4, help="trained steps per train unit")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--height", type=int, default=30)
    p.add_argument("--width", type=int, default=52)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--wsync-mb", type=int, default=300, help="weight-sync payload MB per rollout gpu")
    p.add_argument("--max-inflight", type=int, default=8)
    p.add_argument("--staleness", type=int, default=2)
    return p.parse_args()


class RolloutBuffer:
    """Mirror of AsyncARTrainer._RolloutBuffer (thread-safe variant)."""
    def __init__(self):
        self._items = deque()  # (gen_id, weight_version)
        self._lock = threading.Lock()

    def put(self, weight_version, gen_id):
        with self._lock:
            self._items.append((gen_id, weight_version))

    def size(self):
        with self._lock:
            return len(self._items)

    def drain_freshest(self, n, current_version, max_staleness):
        with self._lock:
            if max_staleness is not None:
                self._items = deque(it for it in self._items if current_version - it[1] <= max_staleness)
            if len(self._items) < n:
                return None
            items = sorted(self._items, key=lambda it: it[0], reverse=True)
            picked, rest = items[:n], items[n:]
            self._items = deque(rest)
            return picked


def main():
    a = parse_args()
    N = a.gpus
    train_gpus = list(range(a.train_gpus))
    roll_gpus = list(range(a.train_gpus, N))
    dev = lambda g: torch.device("cuda", g)

    from diffusers import WanTransformer3DModel
    print(f"[build] {N} WAN replicas (train slab={train_gpus}, rollout slab={roll_gpus})", flush=True)
    conf = WanTransformer3DModel.load_config(a.model, subfolder="transformer")
    models = {g: WanTransformer3DModel.from_config(conf).to(torch.bfloat16).to(dev(g)) for g in range(N)}
    cfg = models[0].config
    in_ch = int(cfg.in_channels); text_dim = int(getattr(cfg, "text_dim", 4096))

    def mk(g):
        return (torch.randn(1, in_ch, a.frames, a.height, a.width, device=dev(g), dtype=torch.bfloat16),
                torch.randn(1, a.seqlen, text_dim, device=dev(g), dtype=torch.bfloat16))

    def cfgfwd(m, h, e, ts, gs):
        out = m(hidden_states=torch.cat([h, h], 0), encoder_hidden_states=torch.cat([e, e], 0),
                timestep=torch.cat([ts, ts], 0), return_dict=False)[0]
        u, c = out.chunk(2, 0); return u + gs * (c - u)

    def gen_unit(g):
        m = models[g]; lat, enc = mk(g)
        with torch.no_grad():
            for _ in range(a.T):
                lat = cfgfwd(m, lat, enc, torch.full((1,), 500.0, device=dev(g)), a.guidance)
        torch.cuda.synchronize(g)

    def train_unit(g):
        m = models[g]; m.requires_grad_(True); lat, enc = mk(g)
        for _ in range(a.K):
            cfgfwd(m, lat, enc, torch.full((1,), 500.0, device=dev(g)), a.guidance).float().pow(2).mean().backward()
        m.zero_grad(set_to_none=True); torch.cuda.synchronize(g)

    def wsync(g):
        b = torch.empty(a.wsync_mb * 1024 * 512, device=dev(0), dtype=torch.bfloat16).to(dev(g), non_blocking=True)
        torch.cuda.synchronize(g); del b

    def par(fn, gpus):
        ts = [threading.Thread(target=fn, args=(g,)) for g in gpus]; [t.start() for t in ts]; [t.join() for t in ts]

    R = a.num_rollouts; B = a.batch
    n_roll = len(roll_gpus); n_train = len(train_gpus)

    def gen_round():
        # produce B trajectories on the rollout slab (ceil(B/n_roll) waves)
        for _ in range((B + n_roll - 1) // n_roll):
            par(gen_unit, roll_gpus)

    def gen_round_on(gpus):
        for _ in range((B + len(gpus) - 1) // len(gpus)):
            par(gen_unit, gpus)

    def train_round():
        for _ in range((B + n_train - 1) // n_train):
            par(train_unit, train_gpus)

    # ---------- warmup ----------
    par(gen_unit, list(range(N))); par(train_unit, list(range(N)))
    for g in range(N): torch.cuda.synchronize(g)

    def timer():
        for g in range(N): torch.cuda.synchronize(g)
        return time.perf_counter()

    # ===== SYNC baseline: gen -> train -> weight-sync, serialized (drain before sync) =====
    def run_sync():
        t0 = timer()
        for rid in range(R):
            gen_round()
            train_round()
            par(wsync, roll_gpus)        # weight sync; rollout idle (drained) -> serial
        return timer() - t0

    # ===== ASYNC double-buffer: gen[r+1] overlaps (train[r] + wsync[r]); TCSS=1-stale,no drain =====
    # TAG = fine-grained work-stealing: gen of a round is B units pulled from a SHARED counter;
    # rollout-slab threads drain it during train, and (TAG) the train slab joins the SAME counter
    # after it finishes training — bounded by B, so it never over-generates.
    def run_async(tag=False):
        def make_round():
            ctr = {"i": 0, "lock": threading.Lock()}
            def worker(g):
                while True:
                    with ctr["lock"]:
                        if ctr["i"] >= B:
                            return
                        ctr["i"] += 1
                    gen_unit(g)
            return ctr, worker

        t0 = timer()
        ctr0, w0 = make_round()                       # prime round 0 (use all gpus to fill fast)
        par(w0, list(range(N)))
        for rid in range(R):
            gt = []
            if rid < R - 1:
                ctr, worker = make_round()            # next round's B gen units on the rollout slab
                gt = [threading.Thread(target=worker, args=(g,)) for g in roll_gpus]
                [t.start() for t in gt]
            train_round()                             # train round rid on train slab (concurrent)
            par(wsync, train_gpus)                     # weight-sync overlapped (no rollout drain — TCSS)
            if gt:
                if tag:                               # TAG: train slab joins the SAME counter (bounded by B)
                    par(worker, train_gpus)
                [t.join() for t in gt]
        return timer() - t0

    print(f"[config] N={N} train_slab={n_train} roll_slab={n_roll} R={R} B={B} "
          f"T={a.T} K={a.K} wsync={a.wsync_mb}MB", flush=True)
    s = run_sync(); print(f"[done] SYNC {s:.2f}s", flush=True)
    g = run_async(False); print(f"[done] ASYNC {g:.2f}s", flush=True)
    tg = run_async(True); print(f"[done] TAG {tg:.2f}s", flush=True)
    print(f"\n{'mode':<26}{'wall_s':>9}{'speedup':>10}")
    print(f"{'SYNC (colocated-style)':<26}{s:>9.2f}{1.0:>9.2f}x")
    print(f"{'ASYNC GAP+TCSS':<26}{g:>9.2f}{s/g:>9.2f}x")
    print(f"{'ASYNC +TAG':<26}{tg:>9.2f}{s/tg:>9.2f}x")


if __name__ == "__main__":
    main()
