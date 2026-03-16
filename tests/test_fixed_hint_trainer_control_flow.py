import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = REPO_ROOT / "fixed_hint_grpo_trainer.py"


class StopAfterSingleGenerate(RuntimeError):
    pass


class UnexpectedSecondGenerate(RuntimeError):
    pass


def _install_trainer_stubs():
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = object
    torch_mod.LongTensor = object
    torch_mod.FloatTensor = object

    def _stop_tensor(*args, **kwargs):
        raise StopAfterSingleGenerate("stop after first generate call")

    torch_mod.tensor = _stop_tensor

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_mod.no_grad = lambda: _NoGrad()
    sys.modules["torch"] = torch_mod

    accelerate_mod = types.ModuleType("accelerate")
    accelerate_utils_mod = types.ModuleType("accelerate.utils")
    accelerate_utils_mod.gather_object = lambda value: value
    accelerate_utils_mod.gather = lambda value: value
    sys.modules["accelerate"] = accelerate_mod
    sys.modules["accelerate.utils"] = accelerate_utils_mod

    trl_mod = types.ModuleType("trl")
    trl_mod.GRPOTrainer = type("GRPOTrainer", (), {})
    sys.modules["trl"] = trl_mod

    trl_data_utils_mod = types.ModuleType("trl.data_utils")
    trl_data_utils_mod.is_conversational = lambda example: False
    trl_data_utils_mod.maybe_apply_chat_template = lambda example, processing_class: {"prompt": example["prompt"]}
    sys.modules["trl.data_utils"] = trl_data_utils_mod

    trl_trainer_mod = types.ModuleType("trl.trainer")
    trl_grpo_mod = types.ModuleType("trl.trainer.grpo_trainer")
    trl_grpo_mod.nanstd = lambda value: value
    trl_utils_mod = types.ModuleType("trl.trainer.utils")
    trl_utils_mod.pad = lambda *args, **kwargs: args[0]
    sys.modules["trl.trainer"] = trl_trainer_mod
    sys.modules["trl.trainer.grpo_trainer"] = trl_grpo_mod
    sys.modules["trl.trainer.utils"] = trl_utils_mod


def _load_trainer_module():
    _install_trainer_stubs()
    spec = importlib.util.spec_from_file_location("fixed_hint_grpo_trainer_under_test", TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FixedHintTrainerControlFlowTests(unittest.TestCase):
    def test_mixed_depth_batch_uses_single_generate_call(self):
        module = _load_trainer_module()
        trainer = object.__new__(module.FixedHintRuleOnlyGRPOTrainer)
        trainer.accelerator = types.SimpleNamespace(device="cpu", process_index=0)
        trainer.model = types.SimpleNamespace(training=True)
        trainer.num_generations = 2
        trainer.processing_class = object()

        generate_calls = []

        def fake_generate(prompts, images):
            generate_calls.append((list(prompts), images))
            if len(generate_calls) > 1:
                raise UnexpectedSecondGenerate("mixed-depth batch should not trigger a second _generate call")
            size = len(prompts) * trainer.num_generations
            return (
                [[100 + i] for i in range(size)],
                [[200 + i] for i in range(size)],
                len(prompts),
                None,
                {},
            )

        trainer._generate = fake_generate

        inputs = [
            {"prompt": "prompt-0", "oracle_hint_depth": 2, "oracle_hint_text": "<a_1><b_2>"},
            {"prompt": "prompt-1", "oracle_hint_depth": 0, "oracle_hint_text": ""},
            {"prompt": "prompt-2", "oracle_hint_depth": 1, "oracle_hint_text": "<c_3>"},
        ]

        with self.assertRaises(StopAfterSingleGenerate):
            trainer._generate_and_score_completions(inputs)

        self.assertEqual(
            generate_calls,
            [
                (
                    [
                        "prompt-0<a_1><b_2>",
                        "prompt-1",
                        "prompt-2<c_3>",
                    ],
                    None,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
