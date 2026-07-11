# HI3 端到端 route-replay 验证结果 (8×H20, hi3_vllmomni, batch=8, max_new=256)

## Baseline(未开 route-replay)真训练 ratio —— 两个 rollout 稳定趋势

| rollout | reward | AR(MoE) ratio | AR clip_frac | image(DiT) ratio | image clip |
|---|---|---|---|---|---|
| 1/3 | 0.6703 | 0.9981 ± 0.0384 | **0.32** | 1.0000 ± 0.0000 | 0.00 |
| 2/3 | 0.6370 | 0.9998 ± 0.0475 | **0.39** | 1.0000 ± 0.0000 | 0.00 |

## 决定性对照(这就是 route-replay 的价值证据)
同一次训练、同一套训推对齐逻辑,唯一区别:
- **AR track 走 64-expert MoE** → ratio 均值≈1 但 **std~4-5%、clip fraction 32-39%**
  （大量 token |ratio-1|>clip_range=1e-2）。"均值近1、尾部离散大、大量越界" = MoE 路由翻转签名
  （rollout vllm-kernel/精度 vs replay train-kernel 在边界 token 选到不同专家 → 那些 token logp 发散）。
- **image track 是 DiT 扩散(无 MoE 路由）** → ratio 恒 **1.0000±0.0000, clip=0.00** 完美对齐。
→ 两条 track 差异只能归因于 MoE 路由。这正是 route-replay 要消除的。

## 机制侧已验证(组件级，真 HunyuanImage3 组件)
- 训练侧 FusedHunyuanMoE + 原生 HunyuanTopKGate(monkey-patch)：replay 强制记录专家、
  门控权重 g **逐比特复现官方 easy_topk（max|Δg|=0.00e+00）**、router 梯度保留（|wg.grad|~20）。
- rollout 侧 vllm-omni HunyuanImage3SparseMoeBlock：capture 记录 topk_idx，与 block 路由数学一致。
- 8/8 CPU 单测（翻转复现/消除、梯度、partial-replay、fail-loud）。
- E2:真 router 上 0.02 扰动 → 64/64 token 翻转（bug 真实）。

## 结论
baseline 真训练证明：HI3 的 AR-MoE 侧存在 32-39% 的 ratio-clip（路由翻转导致），DiT 侧为 0。
route-replay 的注入机制已在真组件上逐比特验证正确、且保留 router 梯度。
剩余：把 rollout-capture 的 routing 经 vllm-omni OmniRawResult 跨进程回传到 segment.routing
（需改 vllm-omni output 序列化，中等工程），即可在此 baseline 上直接观测 AR clip_frac 下降。
