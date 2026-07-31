# Video track

Video is a capability view across two runtimes, not a third execution framework:
video understanding stays in the autoregressive runtime, while video and
audio-video generation stay in the diffusion runtime. This page records only
recipe-backed combinations. An adapter without a checked-in entry recipe is not
marked supported.

The engine columns below name the rollout engine. Training may still use FSDP or
another backend. “Supported” means that a recipe is checked in and its static
contracts pass; it does not imply that every checkpoint, dataset, or hardware
combination has been benchmarked. The verification cell states the available
evidence explicitly.

## Recipe-backed support

| Runtime | Model / task | Canonical entry | Trainside | SGLang | vLLM-Omni | FastVideo | Default reward | Status | Owners | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AR | Qwen3-Omni / video MCQA | [1x4](../../examples/ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x4.yaml) | — | — | [1x4](../../examples/ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x4.yaml), [1x8](../../examples/ar/qwen3_omni_video_r1_gspo_lora_vllm_omni_1x8.yaml) | — | MCExactMatch | Supported | @CjhHa1 @zzhuoxin1508 | `b149901`; GPU: unrecorded |
| AR | Qwen3-Omni / audio-in-video MCQA | [1x4](../../examples/ar/qwen3_omni_audio_video_gspo_lora_vllm_omni_1x4.yaml) | — | — | [1x4](../../examples/ar/qwen3_omni_audio_video_gspo_lora_vllm_omni_1x4.yaml), [1x8](../../examples/ar/qwen3_omni_audio_video_gspo_lora_vllm_omni_1x8.yaml) | — | MCExactMatch | Supported | @CjhHa1 @zzhuoxin1508 | `b149901`; GPU: unrecorded |
| Diffusion | WAN 2.1 / T2V | [trainside](../../examples/diffusion/wan21/wan21_t2v.yaml) | [trainside](../../examples/diffusion/wan21/wan21_t2v.yaml) | [SGLang](../../examples/diffusion/wan21/wan21_t2v_sglang.yaml) | — | Preview: [FastVideo](../../examples/diffusion/wan21/wan21_t2v_dancegrpo_fastvideo.yaml) | VideoPickScore | Supported | @celve @haonan3 @leviking98z-rgb | `b149901`; GPU: unrecorded |
| Diffusion | WAN 2.1 / I2V | [trainside](../../examples/diffusion/wan21/wan21_i2v.yaml) | [trainside](../../examples/diffusion/wan21/wan21_i2v.yaml) | — | — | — | VideoPickScore | Supported | @celve @haonan3 @leviking98z-rgb | `b149901`; GPU: unrecorded |
| Diffusion | WAN 2.2 / T2V | [trainside](../../examples/diffusion/wan22/wan22_t2v_14b.yaml) | [trainside](../../examples/diffusion/wan22/wan22_t2v_14b.yaml) | [SGLang](../../examples/diffusion/wan22/wan22_t2v_14b_sglang.yaml) | — | — | VideoPickScore | Supported | @celve @haonan3 @leviking98z-rgb | `b149901`; GPU: unrecorded |
| Diffusion | WAN 2.2 / I2V | [trainside](../../examples/diffusion/wan22/wan22_i2v.yaml) | [trainside](../../examples/diffusion/wan22/wan22_i2v.yaml) | — | — | — | VideoPickScore | Supported | @celve @haonan3 @leviking98z-rgb | `b149901`; GPU: unrecorded |
| Diffusion | WAN 2.2 / V2V editing | [trainside](../../examples/diffusion/wan22_v2v/wan22_v2v_14b.yaml) | [trainside](../../examples/diffusion/wan22_v2v/wan22_v2v_14b.yaml) | — | — | — | VideoCLIPDelta | Preview | @xshrz @celve | `b149901`; GPU: unrecorded |
| Diffusion | HunyuanVideo 1.0 / T2V | [trainside](../../examples/diffusion/hunyuan_video/hunyuan_video_t2v_trainside.yaml) | [trainside](../../examples/diffusion/hunyuan_video/hunyuan_video_t2v_trainside.yaml) | [SGLang](../../examples/diffusion/hunyuan_video/hunyuan_video_t2v_sglang.yaml) | — | — | VideoPickScore | Supported | @celve @CjhHa1 @haonan3 | `b149901`; GPU: 1x8 validated recipe note |
| Diffusion | HunyuanVideo 1.5 / T2V | [trainside](../../examples/diffusion/hunyuan_video15/hunyuan_video15_t2v_dancegrpo_trainside.yaml) | [trainside](../../examples/diffusion/hunyuan_video15/hunyuan_video15_t2v_dancegrpo_trainside.yaml) | — | [colocate](../../examples/diffusion/hunyuan_video15/hunyuan_video15_t2v_vllmomni_colocate.yaml), [separate](../../examples/diffusion/hunyuan_video15/hunyuan_video15_t2v_vllmomni_nccl_separate.yaml) | — | VideoPickScore | Supported | @celve @CjhHa1 @haonan3 | `b149901`; GPU: 1x8 H20 recipe note |
| Diffusion | LTX-2 / T2V | [trainside](../../examples/diffusion/ltx2/ltx2_t2v_trainside.yaml) | [trainside](../../examples/diffusion/ltx2/ltx2_t2v_trainside.yaml) | [SGLang](../../examples/diffusion/ltx2/ltx2_t2v_sglang_loramerge.yaml) | — | — | VideoPickScore | Supported | @KemingWu @leviking98z-rgb | `b149901`; GPU: 1x8 validated recipe note |
| Diffusion | LTX-2.3 / T2AV | [audio reward](../../examples/diffusion/ltx2/ltx2_3_t2av_audioreward_trainside.yaml) | [trainside](../../examples/diffusion/ltx2/ltx2_3_t2av_audioreward_trainside.yaml) | — | — | — | T2AVComposite | Supported | @KemingWu @leviking98z-rgb | `b149901`; GPU: 1x8 launch profile, run unrecorded |

## Reward ownership

Lightweight same-process rewards live in core. Heavy, dependency-conflicting,
non-differentiable inference lives in `unirl-reward-service`. Differentiable
VideoAlign remains self-contained under `experimental/refl` because gradients
cannot cross the service boundary.

| Reward | Runtime / owner | Canonical usage | Status | Verification |
| --- | --- | --- | --- | --- |
| VideoPickScore | Core local; @KemingWu @Zcchill @haonan3 | [WAN 2.1 T2V](../../examples/diffusion/wan21/wan21_t2v.yaml) | Core | `b149901`; GPU: unrecorded |
| VideoCLIPDelta | Core local; @KemingWu @Zcchill @haonan3 | [WAN 2.2 V2V](../../examples/diffusion/wan22_v2v/wan22_v2v_14b.yaml) | Core | `b149901`; GPU: unrecorded |
| T2AVComposite | Core local; @KemingWu @Zcchill @haonan3 | [LTX-2.3 T2AV](../../examples/diffusion/ltx2/ltx2_3_t2av_audioreward_trainside.yaml) | Core | `b149901`; GPU: run unrecorded |
| VideoAlign | Reward service; @KemingWu @haonan3 | [WAN 2.1 VideoAlign](../../examples/diffusion/wan21_t2v_videoalign_dancegrpo.yaml) | Service | `b16dc12`; GPU: service run unrecorded |
| VideoAlign ReFL | Experimental in-process; @celve @Ideny42 | [WAN 2.1 ReFL](../../experimental/refl/examples/wan21_t2v_videoalign_refl.yaml) | Experimental | `b149901`; GPU: 1x8 H20 smoke recorded in experimental README |

## Maintenance contract

`lint/check_video_support_matrix.py` enforces the parts of this page that can be
checked without GPUs:

- every canonical and engine recipe link exists;
- a recipe listed under an engine column actually selects that rollout engine;
- the canonical recipe contains the declared default reward;
- the canonical entry set cannot silently gain or lose a row;
- every row records status, owners, a verification commit, and explicit GPU evidence
  (including “unrecorded”).

GPU validation remains an evidence update, not a static-CI claim. When a run is
validated, replace “unrecorded” with the hardware shape and update the commit in
the same change.
