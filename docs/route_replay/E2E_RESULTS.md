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

## 测法B:专家翻转率受控实验(真 HI3 layer-0 router, 4096 token, 64 专家 top_k=8)

| 场景 | set翻转% | top1翻转% | MoE输出差 无replay | 有replay | 降幅 |
|---|---|---|---|---|---|
| S0 无扰动 | 0 | 0 | 0 | 0 | — |
| S1 权重漂移0.005 | 74.4 | 15.2 | 0.0240 | 0.0113 | −53% |
| S1 权重漂移0.01 | 94.0 | 28.8 | 0.0399 | 0.0224 | −44% |
| S1 权重漂移0.02 | 99.8 | 49.7 | 0.0656 | 0.0446 | −32% |
| S2 bf16输入 | 1.15 | 0.27 | 0.00036 | 0.00014 | −62% |

### 结论
1. **翻转率对权重漂移极敏感**:漂移 0.005 就有 74% token 的 top-8 专家集合翻转、15% 连 top-1 都翻。
   印证"MoE 路由=离散分叉器"(Q1)。
2. **route-replay 一致降低 MoE 输出发散 32-62%**(强制专家选择一致 → 消除"走错FFN"的大头)。
3. **翻转主因是权重漂移(off-policy)/kernel差异,不是 router 输入精度**:bf16 输入只翻 1.15%。
   → 修正:面试别把翻转主要归于 fp8/精度,主因是权重漂移+两侧实现差异。

### 诚实 caveat(实验局限,必须讲)
- 残差没塌到 0,因为 S1 直接扰动了 router 权重,这同时改了"选择"和"g";route-replay 只冻结选择、
  g 仍用当前(扰动)router 重算(这是正确语义,不冻 g 才能让 router 学)。所以残差=真实策略变化,应存在。
- 真实训推场景 rollout/replay 的 router 权重相同(刚同步),差异只来自 kernel/精度,g 残差会更小、
  塌得更接近 0。S1 用权重漂移高估了残差 —— 它模拟的其实是 off-policy staleness 而非纯 kernel 差。
- 专家 FFN 用随机 bank(未加载 158G 真权重),测的是"不同专家=不同变换"这一定性事实,量级仅代理。
