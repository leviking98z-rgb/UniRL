#!/usr/bin/env python
"""DigenRL all-techniques primitive benchmark (FSDP).

Measures the 3 primitives that the DigenRL pipeline model needs to attribute a
per-technique speedup, on a real UniRL video transformer under FSDP:

  gen_step_ms  : one denoise-step forward, eval / no-grad (CFG)  -> GENERATOR per-step cost
  train_seq_ms : replay K steps sequentially, fwd+bwd            -> TRAINER cost (baseline)
  train_tsp_ms : replay K steps batched into 1 forward, fwd+bwd  -> TRAINER cost w/ TSP

From these + (T denoise steps, K train steps, M micro-batches) the analyzer
(`digenrl_pipeline_model.py`) computes TSP / GAP / TAG / TCSS speedups.

Emits a machine-readable `RESULT {json}` line on rank 0.

Launch (single node, 8 GPU):
  CUDA_VISIBLE_DEVICES=0-7 torchrun --nproc_per_node=8 bench_digenrl.py --ks 2,4 --grad
"""
import argparse, os, time, json, contextlib, functools
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/.cache/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--height", type=int, default=30)
    p.add_argument("--width", type=int, default=52)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--ks", default="2,4")
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--init", default="config", choices=["config", "pretrained"])
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
    torch.cuda.set_device(lr); dist.init_process_group("nccl")
    dev = torch.device("cuda", lr); is0 = rank == 0; dtype = torch.bfloat16

    from diffusers import WanTransformer3DModel
    from diffusers.models.transformers.transformer_wan import WanTransformerBlock
    if a.init == "config":
        conf = WanTransformer3DModel.load_config(a.model, subfolder="transformer")
        tf = WanTransformer3DModel.from_config(conf).to(dtype)
    else:
        tf = WanTransformer3DModel.from_pretrained(a.model, subfolder="transformer", torch_dtype=dtype).to(dtype)
    cfg = tf.config
    in_ch = int(cfg.in_channels); text_dim = int(getattr(cfg, "text_dim", 4096))
    wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={WanTransformerBlock})
    tf = FSDP(tf, sharding_strategy=ShardingStrategy.FULL_SHARD, auto_wrap_policy=wrap, device_id=lr, use_orig_params=True)

    B = a.batch
    g = torch.Generator(device=dev).manual_seed(0)
    mkL = lambda n: torch.randn(n, in_ch, a.frames, a.height, a.width, generator=g, device=dev, dtype=dtype)
    mkE = lambda n: torch.randn(n, a.seqlen, text_dim, generator=g, device=dev, dtype=dtype)

    def sync():
        torch.cuda.synchronize(); dist.barrier()

    def timed(fn, grad):
        for _ in range(a.warmup):
            fn()
            if grad: tf.zero_grad(set_to_none=True)
        sync(); t0 = time.perf_counter()
        for _ in range(a.iters):
            fn()
            if grad: tf.zero_grad(set_to_none=True)
        sync(); return (time.perf_counter() - t0) / a.iters

    embeds = mkE(B); sig = torch.tensor(0.5, device=dev)

    # --- GENERATOR: one denoise-step forward, eval/no-grad ---
    tf.eval()
    one = mkL(B)
    def gen_step():
        with torch.no_grad():
            return cfg_forward(tf, one, embeds, (sig * 1000).expand(B), a.guidance)
    gen_ms = timed(gen_step, grad=False) * 1e3

    # --- TRAINER: replay K steps, fwd+bwd, seq vs TSP ---
    tf.requires_grad_(True)
    results = {"world": world, "model": os.path.basename(a.model), "B": B,
               "shape": [B, in_ch, a.frames, a.height, a.width], "gen_step_ms": round(gen_ms, 3), "tsp": {}}
    for K in [int(x) for x in a.ks.split(",")]:
        lat = [mkL(B) for _ in range(K)]
        sigmas = torch.linspace(0.9, 0.1, K, device=dev)
        def seq():
            outs = [cfg_forward(tf, lat[k], embeds, (sigmas[k] * 1000).expand(B), a.guidance) for k in range(K)]
            torch.stack(outs, 1).float().pow(2).mean().backward()
        def tsp():
            hidden = torch.cat(lat, 0); enc = embeds.repeat(K, 1, 1)
            t = torch.cat([(sigmas[k] * 1000).expand(B) for k in range(K)], 0)
            o = cfg_forward(tf, hidden, enc, t, a.guidance)
            o.reshape(K, B, *o.shape[1:]).float().pow(2).mean().backward()
        s = timed(seq, grad=True) * 1e3
        t = timed(tsp, grad=True) * 1e3
        results["tsp"][K] = {"train_seq_ms": round(s, 3), "train_tsp_ms": round(t, 3), "speedup": round(s / t, 4)}
        if is0:
            print(f"[K={K}] gen_step={gen_ms:.2f}ms  train_seq={s:.2f}ms  train_tsp={t:.2f}ms  TSP={s/t:.3f}x", flush=True)

    if is0:
        print("RESULT " + json.dumps(results), flush=True)
    dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
