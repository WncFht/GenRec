import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = REPO_ROOT / "fixed_hint_grpo_trainer.py"


class _FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class _FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def max(self):
        return _FakeScalar(max(self.values))


def _install_trainer_stubs():
    if "torch" not in sys.modules:
        torch_mod = types.ModuleType("torch")
        torch_mod.Tensor = object
        torch_mod.tensor = lambda values, device=None, dtype=None: _FakeTensor(values)
        torch_mod.int32 = "int32"
        sys.modules["torch"] = torch_mod

    if "accelerate.utils" not in sys.modules:
        accelerate_mod = types.ModuleType("accelerate")
        accelerate_utils_mod = types.ModuleType("accelerate.utils")
        accelerate_utils_mod.gather_object = lambda obj: obj
        sys.modules["accelerate"] = accelerate_mod
        sys.modules["accelerate.utils"] = accelerate_utils_mod

    if "trl" not in sys.modules:
        trl_mod = types.ModuleType("trl")
        trl_mod.GRPOTrainer = object
        sys.modules["trl"] = trl_mod

    if "trl.data_utils" not in sys.modules:
        trl_data_utils_mod = types.ModuleType("trl.data_utils")
        trl_data_utils_mod.is_conversational = lambda example: False
        trl_data_utils_mod.maybe_apply_chat_template = lambda data, processing_class: data
        sys.modules["trl.data_utils"] = trl_data_utils_mod

    if "trl.trainer.grpo_trainer" not in sys.modules:
        trl_grpo_mod = types.ModuleType("trl.trainer.grpo_trainer")
        trl_grpo_mod.nanstd = lambda tensor: tensor
        sys.modules["trl.trainer.grpo_trainer"] = trl_grpo_mod

    if "trl.trainer.utils" not in sys.modules:
        trl_trainer_utils_mod = types.ModuleType("trl.trainer.utils")
        trl_trainer_utils_mod.pad = lambda values, padding_value=None, padding_side=None: values
        sys.modules["trl.trainer.utils"] = trl_trainer_utils_mod

    fixed_hint_utils_mod = sys.modules.get("fixed_hint_utils")
    if fixed_hint_utils_mod is None:
        fixed_hint_utils_mod = types.ModuleType("fixed_hint_utils")
        sys.modules["fixed_hint_utils"] = fixed_hint_utils_mod
    fixed_hint_utils_mod.build_hint_text = lambda ground_truth, hint_depth: f"<hint_{hint_depth}>"
    fixed_hint_utils_mod.build_prompt_with_hint = lambda example, formatter=None: formatter(example["prompt"])


def _load_dynamic_hint_trainer_module():
    _install_trainer_stubs()
    spec = importlib.util.spec_from_file_location("fixed_hint_grpo_trainer_under_test", TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DynamicHintCascadeDistributedSafetyTests(unittest.TestCase):
    def test_cascade_keeps_running_when_other_ranks_still_have_unresolved_groups(self):
        module = _load_dynamic_hint_trainer_module()
        trainer = object.__new__(module.DynamicHintRuleOnlyGRPOTrainer)

        call_depths = []
        gather_results = iter((_FakeTensor([0, 1]), _FakeTensor([0, 0])))

        trainer.num_generations = 2
        trainer.accelerator = types.SimpleNamespace(
            is_main_process=False,
            device="cpu",
            gather=lambda tensor: next(gather_results),
        )
        trainer._build_runtime_hinted_example = lambda example, hint_depth: {
            **example,
            "oracle_hint_depth": hint_depth,
            "oracle_hint_text": "<a_1><b_2>" if hint_depth == 0 else "",
            "oracle_hint_unsolved": False,
        }
        trainer._build_hinted_prompts = lambda inputs: [
            f"{example['prompt']}-depth{example['oracle_hint_depth']}" for example in inputs
        ]
        trainer.processing_class = types.SimpleNamespace(batch_decode=lambda batch, skip_special_tokens=True: batch)

        def _generate_dynamic_stage(prompts_text, images):
            first_prompt = prompts_text[0]
            hint_depth = int(first_prompt.rsplit("depth", 1)[1])
            call_depths.append(hint_depth)
            if hint_depth == 0:
                completions = ["<c_3><d_4>"] * len(prompts_text)
            else:
                completions = ["<c_0><d_0>"] * len(prompts_text)
            return prompts_text, completions

        trainer._generate_dynamic_stage = _generate_dynamic_stage

        inputs = [
            {"prompt": "prompt-a", "reward_model": {"ground_truth": "<a_1><b_2><c_3><d_4>"}},
            {"prompt": "prompt-b", "reward_model": {"ground_truth": "<a_1><b_2><c_3><d_4>"}},
        ]

        cascade = trainer._run_dynamic_hint_cascade(inputs, images=None, max_hint_depth=1)

        self.assertEqual(call_depths, [0, 1])
        self.assertEqual(cascade["selected_group_hint_depths"], [0])
        self.assertEqual(cascade["selected_group_rule_hits"], [True])

    def test_cascade_respects_per_example_max_hint_depth_override(self):
        module = _load_dynamic_hint_trainer_module()
        trainer = object.__new__(module.DynamicHintRuleOnlyGRPOTrainer)

        call_batches = []

        trainer.num_generations = 1
        trainer.accelerator = types.SimpleNamespace(
            is_main_process=False,
            device="cpu",
            gather=lambda tensor: tensor,
        )
        trainer._build_runtime_hinted_example = lambda example, hint_depth: {
            **example,
            "oracle_hint_depth": hint_depth,
            "oracle_hint_text": "<a_1>" if hint_depth > 0 else "",
            "oracle_hint_unsolved": False,
        }
        trainer._build_hinted_prompts = lambda inputs: [
            f"{example['prompt']}-depth{example['oracle_hint_depth']}" for example in inputs
        ]
        trainer.processing_class = types.SimpleNamespace(batch_decode=lambda batch, skip_special_tokens=True: batch)

        def _generate_dynamic_stage(prompts_text, images):
            call_batches.append(list(prompts_text))
            completions = []
            for prompt in prompts_text:
                if prompt == "prompt-b-depth1":
                    completions.append("<b_2>")
                else:
                    completions.append("<z_0>")
            return prompts_text, completions

        trainer._generate_dynamic_stage = _generate_dynamic_stage

        inputs = [
            {
                "prompt": "prompt-a",
                "reward_model": {"ground_truth": "<a_1><b_2>"},
                "dynamic_hint_max_depth_override": 0,
            },
            {
                "prompt": "prompt-b",
                "reward_model": {"ground_truth": "<a_1><b_2>"},
                "dynamic_hint_max_depth_override": 1,
            },
        ]

        cascade = trainer._run_dynamic_hint_cascade(inputs, images=None, max_hint_depth=3)

        self.assertEqual(call_batches, [["prompt-a-depth0", "prompt-b-depth0"], ["prompt-b-depth1"]])
        self.assertEqual(cascade["selected_group_hint_depths"], [0, 1])
        self.assertEqual(cascade["selected_group_rule_hits"], [False, True])


if __name__ == "__main__":
    unittest.main()
