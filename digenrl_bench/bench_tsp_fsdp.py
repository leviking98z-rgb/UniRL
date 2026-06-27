#!/usr/bin/env python
"""DigenRL TSP under FSDP — the regime where TSP actually helps.

DigenRL's TSP claim: "in sharded setups like FSDP/ZeRO, parameters are split
across devices and must be all-gathered" per forward. UniRL's replay runs the
transformer forward once PER selected denoise step (K sequential forwards) =>
K all-gathers of the sharded params. TSP batches the K steps into ONE forward
=> ONE all-gather. This script measures that amortization under real FSDP.

Single-GPU forward is compute-bound (TSP ~1.0x); under FSDP the per-step
all-gather is a fixed overhead that batching amortizes. Tiny compute shapes
make the all-gather the relatively dominant cost so the mechanism is visible.

Launch (8 GPUs, one node):
  torchrun --nproc_per_node=8 bench_tsp_fsdp.py --frames 5 --height 30 --width 52 \
      --batch 1 --ks 1,2,4,8 --grad
"""
import argparse, os, time, contextlib
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import functools


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/.cache/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--height", type=int, default=30)
    p.add_argument("--width", type=int, default=52)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--ks", default="1,2,4,8")
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--grad", action="store_true")
    p.add_argument("--init", default="config", choices=["config", "pretrained"])
    p.add_argument("--seqlen", type=int, default=512)
    return p.parse_args()


def cfg_forward(tf, hidden, enc, timestep, guidance):
    if guidance > 1.0:
        out = tf(hidden_states=torch.cat([hidden, hidden], 0),
                 encoder_hidden_states=torch.cat([enc, enc], 0),
                 timestep=torch.cat([timestep, timestep], 0), return_dict=False)[0]
        unc, con = out.chunk(2, 0)
        return unc + guidance * (con - unc)
    return tf(hidden_states=hidden, encoder_hidden_states=enc, timestep=timestep, return_dict=False)[0]


def main():
    a = parse_args()
    lr = int(os.environ["LOCAL_RANK"]); rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(lr)
    dist.init_process_group("nccl")
    dev = torch.device("cuda", lr)
    is0 = rank == 0
    dtype = torch.bfloat16

    from diffusers import WanTransformer3DModel
    from diffusers.models.transformers.transformer_wan import WanTransformerBlock
    if is0:
        print(f"[load] {a.model}  world={world}  grad={a.grad}  init={a.init}", flush=True)
    if a.init == "config":
        # random-init same architecture (no checkpoint load -> ~5s startup; FSDP
        # all-gather cost depends on param COUNT, not values -> identical for TSP timing)
        conf = WanTransformer3DModel.load_config(a.model, subfolder="transformer")
        tf = WanTransformer3DModel.from_config(conf).to(dtype)
    else:
        tf = WanTransformer3DModel.from_pretrained(a.model, subfolder="transformer", torch_dtype=dtype)
    tf = tf.to(dtype)  # force uniform bf16 (some norms load fp32 -> FSDP flatten needs uniform dtype)
    cfg = tf.config
    in_ch = int(cfg.in_channels); text_dim = int(getattr(cfg, "text_dim", 4096))

    wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={WanTransformerBlock})
    tf = FSDP(tf, sharding_strategy=ShardingStrategy.FULL_SHARD, auto_wrap_policy=wrap,
              device_id=lr, use_orig_params=True)
    tf.requires_grad_(a.grad)
    if not a.grad:
        tf.eval()
    if is0:
        print(f"[cfg] in_ch={in_ch} text_dim={text_dim} layers={getattr(cfg,'num_layers','?')} "
              f"latent=[{a.batch},{in_ch},{a.frames},{a.height},{a.width}] sharded over {world} GPUs", flush=True)

    B = a.batch
    g = torch.Generator(device=dev).manual_seed(0)
    mkL = lambda n: torch.randn(n, in_ch, a.frames, a.height, a.width, generator=g, device=dev, dtype=dtype)
    mkE = lambda n: torch.randn(n, a.seqlen, text_dim, generator=g, device=dev, dtype=dtype)
    grad_ctx = contextlib.nullcontext() if a.grad else torch.no_grad()

    def sync():
        torch.cuda.synchronize(); dist.barrier()

    def timed(fn):
        for _ in range(a.warmup):
            fn();
            if a.grad: tf.zero_grad(set_to_none=True)
        sync(); torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            fn()
            if a.grad: tf.zero_grad(set_to_none=True)
        sync()
        return (time.perf_counter() - t0) / a.iters, torch.cuda.max_memory_allocated() / 2**30

    ks = [int(x) for x in a.ks.split(",")]
    if is0:
        print(f"\n{'K':>3} {'seq_ms':>9} {'tsp_ms':>9} {'speedup':>8} {'tsp_GB':>8}")
    for K in ks:
        latents = [mkL(B) for _ in range(K)]
        embeds = mkE(B)
        sigmas = torch.linspace(0.9, 0.1, K, device=dev, dtype=torch.float32)

        def run_seq():
            with grad_ctx:
                outs = [cfg_forward(tf, latents[k], embeds, (sigmas[k] * 1000).expand(B), a.guidance) for k in range(K)]
                o = torch.stack(outs, 1)
                if a.grad: o.float().pow(2).mean().backward()
            return o

        def run_tsp():
            with grad_ctx:
                hidden = torch.cat(latents, 0); enc = embeds.repeat(K, 1, 1)
                t = torch.cat([(sigmas[k] * 1000).expand(B) for k in range(K)], 0)
                o = cfg_forward(tf, hidden, enc, t, a.guidance)
                o = o.reshape(K, B, *o.shape[1:]).transpose(0, 1)
                if a.grad: o.float().pow(2).mean().backward()
            return o

        seq_ms, _ = timed(run_seq)
        tsp_ms, tsp_gb = timed(run_tsp)
        if is0:
            print(f"{K:>3} {seq_ms*1e3:>9.2f} {tsp_ms*1e3:>9.2f} {seq_ms/tsp_ms:>7.2f}x {tsp_gb:>8.2f}", flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
