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
