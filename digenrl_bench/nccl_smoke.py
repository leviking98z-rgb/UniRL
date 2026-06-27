import os, torch, torch.distributed as dist, time
lr = int(os.environ["LOCAL_RANK"]); rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(lr)
dist.init_process_group("nccl")
t0 = time.perf_counter()
x = torch.ones(1 << 20, device="cuda") * rank
dist.all_reduce(x)
torch.cuda.synchronize()
exp = world * (world - 1) / 2
ok = abs(x[0].item() - exp) < 1e-3
if rank == 0:
    print(f"[smoke] world={world} all_reduce sum0={x[0].item():.0f} expect={exp:.0f} OK={ok} t={(time.perf_counter()-t0)*1e3:.0f}ms", flush=True)
dist.barrier(); dist.destroy_process_group()
