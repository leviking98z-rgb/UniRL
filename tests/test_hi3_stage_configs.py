from pathlib import Path

import yaml


def test_standalone_hi3_dit_does_not_wait_for_upstream_kv() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "unirl"
        / "rollout"
        / "engine"
        / "vllm_omni"
        / "stage_configs"
        / "hunyuan_image3_dit_recaption_rl.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    stages = config["stage_args"]
    assert len(stages) == 1
    assert stages[0]["stage_type"] == "diffusion"
    assert stages[0]["engine_args"]["omni_kv_config"]["need_recv_cache"] is False


def test_hi3_pickscore_uses_clip_processor_default() -> None:
    examples = Path(__file__).parents[1] / "examples" / "unified_model"
    for config_name in ("hi3_vllmomni.yaml", "hi3_vllmomni_veomni_ep.yaml"):
        config_text = (examples / config_name).read_text()
        assert (
            "processor_id: "
            "${oc.env:PICKSCORE_PROCESSOR,laion/CLIP-ViT-H-14-laion2B-s32B-b79K}"
        ) in config_text
