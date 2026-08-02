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
        assert ("processor_id: ${oc.env:PICKSCORE_PROCESSOR,laion/CLIP-ViT-H-14-laion2B-s32B-b79K}") in config_text


def test_hi3_per_track_micro_batches_preserve_ar_sequence_objective() -> None:
    examples = Path(__file__).parents[1] / "examples" / "unified_model"
    for config_name in ("hi3_vllmomni.yaml", "hi3_vllmomni_veomni_ep.yaml"):
        config = yaml.safe_load((examples / config_name).read_text())
        assert config["stack"]["ar_micro_batch_size"] == 1
        assert config["stack"]["image_micro_batch_size"] == 1
        assert config["algorithm"]["ar"]["loss_agg_mode"] == "seq-mean-token-mean"
        assert config["bundle"]["config"]["batch_replay_steps"] is True

        prompts = int(config["batch_size"])
        ar_batch = prompts * int(config["sampling"]["ar"]["samples_per_prompt"])
        image_batch = ar_batch * int(config["sampling"]["diffusion"]["samples_per_prompt"])
        train_dp = int(config["num_devices"])
        assert ar_batch % train_dp == 0
        assert image_batch % train_dp == 0


def test_bagel_per_track_micro_batches_preserve_ar_sequence_objective() -> None:
    config_path = Path(__file__).parents[1] / "examples" / "unified_model" / "bagel_trainside_unigrpo.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["stack"]["ar_micro_batch_size"] is None
    assert config["stack"]["image_micro_batch_size"] is None
    assert config["algorithm"]["ar"]["loss_agg_mode"] == "seq-mean-token-mean"
