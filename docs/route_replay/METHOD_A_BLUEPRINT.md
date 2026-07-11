# 测法A(端到端跨引擎 route-replay 对比)实现蓝图 + 可行性结论

## 目标
真训练里对比"开/关 route-replay"的 AR clip fraction / ratio std,拿到端到端曲线。
需要打通:rollout(vllm-omni AR worker,MoE routing 发生处)capture 的 topk_idx
→ 跨进程回传 → driver 的 TextSegment.routing → 训练侧 ar.replay 注入。

## 已摸清的完整链路(全部有文件:行依据)
1. **capture 已实现**:`patches/moe_route_capture.py` 在 worker 子进程 patch
   `HunyuanImage3SparseMoeBlock.forward`,每层每 decode-step 记 topk_idx 进线程本地 session。
2. **注入已实现**:`models/hunyuan_image3/ar.py` replay 检测 `segment.routing` 自动注入(offset=prompt_len)。
3. **数据结构已就位**:`TextSegment.routing` packed_field。
4. **回传通道已找到**:`collective_rpc` + `_diffrl_*` verbs 是 driver 取 worker 数据的现成通路
   (先例 `_diffrl_loaded_param_checksums`,native.py:695-711 返回 `{stage:[rank_dict]}`)。
5. **driver 侧切片可行**:`build_ar_segment`(tracks.py:338)已按 per-request 遍历,每个 request 的
   token 数从 `token_ids` 已知 → 可按长度切 routing。

## 关键有利事实(简化归属)
- AR stage 配置 `max_num_seqs: 1` + `enforce_eager: true`(hunyuan_image3_ar_recaption_rl.yaml:27,45)
  → 同一时刻只处理一个 request,**不 batch 内交织**,routing 天然按 request 串行分段。

## 剩余难点(诚实)—— 这是"全力做"要啃的
worker 侧 capture buffer 按 **layer-visit × decode-step** 累积,一次 `omni.generate([prompt0,prompt1,...])`
里多 prompt **依次**跑、buffer 首尾拼接且 prefill/decode 交织。要按 request 切开,需在 worker 侧
**感知 request 切换 + prefill/decode 边界**,把 routing 分段存进按 request_id 索引的 buffer,
再由 driver 经 `collective_rpc("_diffrl_drain_routing")` 取回。
→ 约 100+ 行,深入 vllm-omni AR 调度语义;验证要跑通完整两引擎训练(单 rollout ~5min)。

## 实现步骤(蓝图,若继续)
1. `moe_route_capture.py`:capture session 增加 per-request 分段
   (挂 request start/finish 信号,或按 vLLM scheduler 的 seq_group 边界)。
2. `ar_extension.py`(HI3ARWeightSyncExtension):加 `_diffrl_drain_routing` 方法,
   返回 `{request_id: [n_layers, n_tok, top_k]}`,drain 后清 buffer。
3. `backends/native.py` generate():`with capture 激活`(env gated)→ generate →
   `collective_rpc("_diffrl_drain_routing")` 取回 → 附到对应 OmniRawResult。
4. `utils/tracks.py` build_ar_segment / _extract_completion:读 routing → TextSegment.pack(routing=...)。
5. 真训练对比:UNIRL_MOE_ROUTE_CAPTURE=1 + UNIRL_MOE_ROUTE_REPLAY=1 开,vs 关,看 AR clip_frac。

## 可行性结论
- **driver 侧切片 + 回传通道**:可行,有先例。
- **worker 侧 per-request 分段 capture**:是真正的工程量(vllm-omni 调度语义),100+ 行 + 完整训练验证。
- **测法B 已从机制上量化证明有效**(route-replay 降 MoE 输出发散 32-62%,翻转率 vs 漂移曲线),
  测法A 是"端到端曲线"这最后一块外围证据,成本/风险高、增量有限。

## 当前状态:蓝图完成,worker 侧 per-request 分段待实现。

---

## 测法A 端到端真跑:诊断结论(2 个真实集成 bug,已精确定位)

真训练(capture+replay 开)结果:AR ratio=0.992±0.038 clip=0.38,与 baseline(clip 32-39%)
**无差异 → route-replay 未生效**。埋诊断日志后精确定位到两个根因:

### Bug 1(根本):capture 的 patched forward 从未被触发
- 日志有 `[ROUTE_CAPTURE] installed on HunyuanImage3SparseMoeBlock.forward (env_enabled=True)`(×2 worker),
  但 **"forward fired" 一次都没出现**。
- 说明 AR 生成时真正执行的 MoE forward **不是**我 monkey-patch 的那个 Python `HunyuanImage3SparseMoeBlock.forward`。
- 疑因:vllm 对 AR decode 可能用 CUDA graph 捕获 / torch.compile / 走父类 `HunYuanSparseMoeBlock` 的
  另一路径,绕过了被 patch 的 Python 方法。**这是 AR 侧 capture 的深水区**——patch 点选错了执行层。

### Bug 2:driver drain 的 collective_rpc 发错 stage
- `driver drained: routing=None n_forwards=0` + 报错
  `'DiffusionWorkerWithDiTWeightSyncExtension' object has no attribute '_diffrl_drain_routing'`。
- 我的 `collective_rpc(method="_diffrl_drain_routing", stage_ids=[0])` 打到了 **DiT worker**(它没这方法),
  说明 stage_id=0 不是 AR stage,或 stage 路由需按 modality 而非固定 0。要查 AR stage 的真实 id/路由方式。

### 结论
- 管道结构正确(install 成功、ar.replay 正确检测 None 并 fallback、best-effort 未崩训练)。
- 但 **capture 源头(patched forward 未触发)是拦路虎**:需找到 vllm AR decode 真正执行 MoE 的层
  (可能要 patch kernel 层 / 关 CUDA graph / patch 父类),这是又一轮深水集成。
- 诚实定位到此:测法A 的数据面管道已实现且组件验证,但**真跑未生效,根因是 AR MoE forward 的 patch 点
  被 vllm 执行路径绕过 + drain stage 路由错**。修复路径清楚(见上),但成本高。

---

## 测法A 第二轮(修 Bug1/Bug2 后):capture 打通,回传时序是新拦路虎

### ✅ Bug1 已修 —— capture 真的触发了
把 `moe_route_capture.install()` 从 `patches/__init__`(只覆盖外层 stage worker)移到
`BucketedIPCReceiveMixin.__new__`(每个 vllm TP worker 子进程都跑)后:
- 日志出现 **`forward fired WITH session, capturing` ×4**(pid 3292566-569,AR 的 TP worker),
  之前是 0 次 → **routing 在 AR TP worker 里被成功记录**。
- 根因确认:AR MoE forward 跑在 vllm 内层 TP-spawn worker(pid 329xxxx),外层 stage worker(328xxxx)
  的 patch 到不了它们。与 torch.compile 无关(enforce_eager=True, CompilationMode.NONE, CUDAGraph.NONE)。

### ✗ 新拦路虎 Bug3 —— 回传时序 / 两引擎 generate 分离
- `driver drained` print **0 次** → `_maybe_drain_routing` 没在 AR generate 后执行;
- 但 DiT worker 收到了 `_diffrl_drain_routing`(报 AttributeError)→ drain 是在 **DiT stage 的
  generate** 里触发的、遍历 stage 发 drain,发给 DiT worker 报错。
- 推断:HI3 两引擎里 **AR recaption 和 DiT 是分离的 generate 调用**;AR 的 routing 记在 AR worker buffer,
  但 drain 的时机/stage 对不上 AR。要在 **AR 的 generate 之后、对 AR stage** drain,而不是 DiT generate 后遍历。
- ar.replay 仍见 `segment.routing=None` → routing 没回到训练侧。

### 状态
capture(源头)完全打通;剩"在正确的时机对正确的 stage drain 并 stamp"这层回传时序。
这是测法A 的第三层集成细节 —— 每修一层冒下一层(进程层→已修;时序/stage 层→进行中)。

---

## 测法A 第三/四轮:capture 打通,drain 收窄到"跨 RPC env/进程" 最后一环
- Bug1(patch 进程层)已修:forward fired ×4 in AR TP workers。
- Bug3a(routing 多-rank 嵌套解包 + DiT 非 tuple 返回):已修(递归 _find_routing)。
- **剩余 Bug3b**:drain 的 `collective_rpc("_diffrl_drain_routing", stage_ids=[0])` 从 AR stage 拿回
  `routing=None n_forwards=0`,尽管同 run 里 capture forward 明确 fired 并写入进程-global `_GLOBAL`。
  → 疑因:drain 经 collective_rpc 执行 `drain_global()` 时,`_global_enabled()` 读到的
  `UNIRL_MOE_ROUTE_CAPTURE` env 在 RPC 执行栈里没透传(worker 的 rpc handler 起于干净 env),
  或 capture-forward 进程与 rpc-handler 进程的 `_GLOBAL` 非同一实例。需在 drain_global 内加 env/pid 信标定位。
- 端到端仍 fallback(segment.routing=None,clip 未降),但**best-effort 从未破坏训练**(每轮都正常出 ratio)。

结论:capture 源头 + 多-rank 解包已通;剩 drain 侧 env/进程一致性这一环。每轮 GPU 验证 ~10min。

---

## 测法A 终局(5-6 轮深水,推进到架构本质)

### 一路攻克(全部实测确认)
1. **Bug1 patch 进程层** ✅ — install 移到 BucketedIPCReceiveMixin.__new__(每个 vllm TP worker),
   forward fired ×4 in AR TP workers。
2. **Bug3a 多-rank/DiT 返回** ✅ — 递归 _find_routing。
3. **Bug3b drain 读不到 buffer** ✅ — drain_global 直接读 _GLOBAL(不 env-gate)+ install latch _CAPTURE_ON。
   **实测:drain_global pid=383xxxx routing=(32, 27382, 8) nfwd=1027 ×4 rank** — routing 在 worker 里完整取到!
4. **Bug3c 序列化(终极拦路虎)** — collective_rpc 把 worker 返回的 tensor/numpy **都序列化成嵌套
   Python list**(`results=list[list[list[list[...]]]]`),driver 侧 _find_routing 认不出 ndarray/tensor。

### 终极结论:collective_rpc 不是传大张量的通道
- routing = [32层, 27382 token, 8] ≈ 56MB/rank × 4 rank/rollout。
- collective_rpc 为**小控制信息**设计,会把返回值转成纯 Python 结构(丢 tensor 类型),
  且这么大的东西每 rollout 传 4 份本身是错误的传输方式。
- **正确修法**:接一个真正的 tensor 传输通道(共享内存 / 文件 / CUDA-IPC,就像权重同步那样),
  而不是 collective_rpc;且应在 worker 侧**只保留 response 段**(27382 里绝大部分是 prompt,不进 loss)
  大幅减小传输。这是又一个独立中等工程。

### 达成度
- capture(源头)→ drain(worker 侧取到完整 routing)**全链路技术打通并实测确认**(routing (32,27382,8) 真的拿到了)。
- 剩最后一环:**大张量的跨进程回传通道**(collective_rpc 不胜任)+ response 段裁剪。
- 端到端 clip 未降(routing 没到训练侧),但 best-effort 全程未破坏训练(每轮正常出 ratio)。
- **核心科学结论不受影响**:测法B 已量化 route-replay 降发散 32-62% + baseline 内置对照证明问题真实
  + 训练侧注入/capture/drain 全部逐组件实测验证。测法A 差的是"演示端到端曲线"的传输管道工程。
