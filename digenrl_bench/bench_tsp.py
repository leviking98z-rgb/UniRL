#!/usr/bin/env python
"""DigenRL Time-Step Parallelism (TSP) micro-benchmark for UniRL.

Measures the core speedup TSP gives the diffusion-RL *replay* (train forward):
instead of running the transformer forward once per selected denoise step
(``for step_idx in target:`` in WAN21DiffusionStage.replay), TSP stacks the K
selected steps into ONE forward of batch K*B. Trajectory is fixed during replay,
so the steps are independent -> math-equivalent (we assert it).

This isolates the transformer forward+backward (the >95% cost), which is exactly
what TSP batches. Uses the real diffusers WanTransformer3DModel (the module
UniRL's WAN21 bundle wraps) and replicates UniRL's CFG-batched predict_noise.

Run on ONE GPU:
  python bench_tsp.py --model /root/.cache/Wan2.1-T2V-1.3B-Diffusers \
      --frames 21 --height 60 --width 104 --batch 1 --ks 2,4,8 --grad
"""
import argparse, time, os, contextlib
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/.cache/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--frames", type=int, default=21, help="latent temporal length T")
    p.add_argument("--height", type=int, default=60, help="latent H")
    p.add_argument("--width", type=int, default=104, help="latent W")
    p.add_argument("--batch", type=int, default=1, help="B samples per step")
    p.add_argument("--ks", default="2,4,8", help="comma list of K (selected steps)")
    p.add_argument("--guidance", type=float, default=5.0, help=">1 -> CFG (2x forward)")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--grad", action="store_true", help="time fwd+bwd (train regime), else fwd-only")
    p.add_argument("--seqlen", type=int, default=512, help="text token length")
    return p.parse_args()


DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def cfg_forward(transformer, hidden, enc, timestep, guidance):
    """Replicates UniRL WAN21DiffusionStep.predict_noise CFG batching."""
    if guidance > 1.0:
        out = transformer(
            hidden_states=torch.cat([hidden, hidden], 0),
            encoder_hidden_states=torch.cat([enc, enc], 0),
            timestep=torch.cat([timestep, timestep], 0),
            return_dict=False,
        )[0]
        unc, con = out.chunk(2, 0)
        return unc + guidance * (con - unc)
    return transformer(hidden_states=hidden, encoder_hidden_states=enc,
                        timestep=timestep, return_dict=False)[0]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    sync()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated() / 2**30
    return dt, peak


def main():
    a = parse_args()
    dtype = DT[a.dtype]
    dev = torch.device("cuda")
    ks = [int(x) for x in a.ks.split(",")]

    from diffusers import WanTransformer3DModel
    print(f"[load] {a.model}/transformer  dtype={a.dtype}", flush=True)
    tf = WanTransformer3DModel.from_pretrained(a.model, subfolder="transformer", torch_dtype=dtype).to(dev)
    cfg = tf.config
    in_ch = int(cfg.in_channels)
    text_dim = int(getattr(cfg, "text_dim", getattr(cfg, "cross_attention_dim", 4096)))
    print(f"[cfg] in_channels={in_ch} text_dim={text_dim} layers={getattr(cfg,'num_layers','?')} "
          f"latent=[B,{in_ch},{a.frames},{a.height},{a.width}] guidance={a.guidance} grad={a.grad}", flush=True)
    tf.requires_grad_(a.grad)
    if not a.grad:
        tf.eval()

    B = a.batch
    g = torch.Generator(device=dev).manual_seed(0)

    def mk_latents(n):
        return torch.randn(n, in_ch, a.frames, a.height, a.width, generator=g, device=dev, dtype=dtype)

    def mk_embeds(n):
        return torch.randn(n, a.seqlen, text_dim, generator=g, device=dev, dtype=dtype)

    grad_ctx = contextlib.nullcontext() if a.grad else torch.no_grad()

    print(f"\n{'K':>3} {'seq_ms':>9} {'tsp_ms':>9} {'speedup':>8} {'seq_GB':>8} {'tsp_GB':>8} {'maxdiff':>10}")
    results = []
    for K in ks:
        # fixed inputs for this K (same data both ways -> equivalence holds)
        latents = [mk_latents(B) for _ in range(K)]
        embeds = mk_embeds(B)                       # same prompt embeds, all steps
        sigmas = torch.linspace(0.9, 0.1, K, device=dev, dtype=torch.float32)
        ts_scale = 1000.0

        def run_seq():
            outs = []
            with grad_ctx:
                for k in range(K):
                    t = (sigmas[k] * ts_scale).expand(B)
                    o = cfg_forward(tf, latents[k], embeds, t, a.guidance)
                    outs.append(o)
                stacked = torch.stack(outs, 1)        # [B,K,...]
                if a.grad:
                    stacked.float().pow(2).mean().backward()
            return stacked

        def run_tsp():
            with grad_ctx:
                hidden = torch.cat(latents, 0)                      # [K*B,...]
                enc = embeds.repeat(K, 1, 1)                        # [K*B,L,D]
                t = torch.cat([(sigmas[k] * ts_scale).expand(B) for k in range(K)], 0)  # [K*B]
                o = cfg_forward(tf, hidden, enc, t, a.guidance)     # ONE forward
                out = o.reshape(K, B, *o.shape[1:]).transpose(0, 1)  # [B,K,...]
                if a.grad:
                    out.float().pow(2).mean().backward()
            return out

        # equivalence (fwd only, no grad, tight check)
        with torch.no_grad():
            s = run_seq() if not a.grad else None
            if a.grad:
                tf.requires_grad_(False)
                s = run_seq(); t_ = run_tsp()
                tf.requires_grad_(True)
            else:
                t_ = run_tsp()
            maxdiff = (s.float() - t_.float()).abs().max().item()

        if a.grad:
            tf.zero_grad(set_to_none=True)
        seq_ms, seq_gb = timed(lambda: (run_seq(), tf.zero_grad(set_to_none=True)) if a.grad else run_seq(), a.iters, a.warmup)
        tsp_ms, tsp_gb = timed(lambda: (run_tsp(), tf.zero_grad(set_to_none=True)) if a.grad else run_tsp(), a.iters, a.warmup)
        sp = seq_ms / tsp_ms
        print(f"{K:>3} {seq_ms*1e3:>9.2f} {tsp_ms*1e3:>9.2f} {sp:>7.2f}x {seq_gb:>8.2f} {tsp_gb:>8.2f} {maxdiff:>10.2e}", flush=True)
        results.append((K, seq_ms, tsp_ms, sp, maxdiff))

    print("\n[summary] K, seq_ms, tsp_ms, speedup, maxdiff")
    for r in results:
        print(f"  K={r[0]:>2}  {r[1]*1e3:8.2f}ms -> {r[2]*1e3:8.2f}ms  {r[3]:.2f}x  (eq {r[4]:.1e})")


if __name__ == "__main__":
    main()
