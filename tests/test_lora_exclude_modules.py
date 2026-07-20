import json
import tempfile
import unittest
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
from peft.tuners.lora import LoraLayer
from torch import nn

from unirl.tools.export_adapter import write_adapter_config
from unirl.train.configs import LoraConfig
from unirl.train.lora import inject_lora


class _Tower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_model = _Tower()
        self.audio_tower = _Tower()


class LoraExcludeModulesTests(unittest.TestCase):
    def test_regex_excludes_frozen_tower_but_injects_text_tower(self) -> None:
        model = _ToyModel()
        model.audio_tower.requires_grad_(False)

        inject_lora(
            model,
            rank=2,
            alpha=2,
            target_modules=("q_proj",),
            exclude_modules=r".*audio_tower.*",
        )

        self.assertIsInstance(model.text_model.q_proj, LoraLayer)
        self.assertNotIsInstance(model.audio_tower.q_proj, LoraLayer)
        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(any("text_model.q_proj.lora_" in name for name in trainable))
        self.assertFalse(any("audio_tower" in name for name in trainable))

    def test_none_keeps_existing_injection_behavior(self) -> None:
        model = _ToyModel()

        inject_lora(
            model,
            rank=2,
            alpha=2,
            target_modules=("q_proj",),
            exclude_modules=None,
        )

        self.assertIsInstance(model.text_model.q_proj, LoraLayer)
        self.assertIsInstance(model.audio_tower.q_proj, LoraLayer)

    def test_config_and_export_preserve_regex_string(self) -> None:
        pattern = r".*audio_tower.*|.*visual.*"
        config = instantiate(
            OmegaConf.create(
                {
                    "_target_": "unirl.train.configs.LoraConfig",
                    "exclude_modules": pattern,
                }
            )
        )
        self.assertIsInstance(config, LoraConfig)
        self.assertEqual(config.exclude_modules, pattern)

        with tempfile.TemporaryDirectory() as output:
            write_adapter_config(
                output,
                base="example/model",
                r=2,
                lora_alpha=2,
                target_modules=["q_proj"],
                exclude_modules=config.exclude_modules,
            )
            exported = json.loads(Path(output, "adapter_config.json").read_text())

        self.assertEqual(exported["exclude_modules"], pattern)


if __name__ == "__main__":
    unittest.main()
