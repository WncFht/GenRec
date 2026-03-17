import importlib.util
import sys
import types
import unittest
from collections import defaultdict
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

    def test_dynamic_hint_cascade_only_retries_unresolved_groups(self):
        module = _load_trainer_module()
        trainer = object.__new__(module.DynamicHintRuleOnlyGRPOTrainer)
        trainer.num_generations = 2
        decode_map = {
            (11, 12): "<a_1><b_2>",
            (13, 14): "<a_1><x_9>",
            (15,): "<x_9><x_9>",
            (16,): "<x_9><x_8>",
            (21, 22, 23): "<d_4><e_5>",
            (24, 25, 26, 27): "<d_4><x_9>",
        }
        trainer.processing_class = types.SimpleNamespace(
            batch_decode=lambda batch, skip_special_tokens=True: [decode_map[tuple(row)] for row in batch]
        )

        single_turn_calls = []

        def fail_generate(prompts, images):
            raise UnexpectedSecondGenerate("dynamic cascade should not call _generate for each stage")

        def fake_generate_single_turn(prompts, images):
            single_turn_calls.append(list(prompts))
            if prompts == ["prompt-0", "prompt-0", "prompt-1", "prompt-1"]:
                return (
                    [[100], [101], [102], [103]],
                    [
                        [11, 12],
                        [13, 14],
                        [15],
                        [16],
                    ],
                    None,
                    {},
                )
            if prompts == ["prompt-1<c_3>", "prompt-1<c_3>"]:
                return (
                    [[200], [201]],
                    [
                        [21, 22, 23],
                        [24, 25, 26, 27],
                    ],
                    None,
                    {},
                )
            raise AssertionError(f"Unexpected prompts: {prompts}")

        trainer._generate = fail_generate
        trainer._generate_single_turn = fake_generate_single_turn

        inputs = [
            {"prompt": "prompt-0", "reward_model": {"ground_truth": "<a_1><b_2>"}},
            {"prompt": "prompt-0", "reward_model": {"ground_truth": "<a_1><b_2>"}},
            {"prompt": "prompt-1", "reward_model": {"ground_truth": "<c_3><d_4><e_5>"}},
            {"prompt": "prompt-1", "reward_model": {"ground_truth": "<c_3><d_4><e_5>"}},
        ]

        cascade = trainer._run_dynamic_hint_cascade(inputs, images=None, max_hint_depth=3)

        self.assertEqual(
            single_turn_calls,
            [
                ["prompt-0", "prompt-0", "prompt-1", "prompt-1"],
                ["prompt-1<c_3>", "prompt-1<c_3>"],
            ],
        )
        self.assertEqual(cascade["selected_group_hint_depths"], [0, 1])
        self.assertEqual(
            cascade["selected_prompt_texts"],
            [
                "prompt-0",
                "prompt-0",
                "prompt-1<c_3>",
                "prompt-1<c_3>",
            ],
        )
        self.assertEqual(cascade["selected_inputs"][2]["oracle_hint_text"], "<c_3>")
        self.assertEqual(cascade["local_num_items_in_batch"], 11)

    def test_dynamic_hint_cascade_keeps_max_depth_outputs_when_all_stages_miss(self):
        module = _load_trainer_module()
        trainer = object.__new__(module.DynamicHintRuleOnlyGRPOTrainer)
        trainer.num_generations = 2
        decode_map = {
            (11,): "<x_9><x_8>",
            (21,): "<x_7><x_6>",
            (12,): "<x_9><x_8>",
            (22,): "<x_7><x_6>",
            (13,): "<x_9><x_8>",
            (23,): "<x_7><x_6>",
            (14,): "<x_9><x_8>",
            (24,): "<x_7><x_6>",
        }
        trainer.processing_class = types.SimpleNamespace(
            batch_decode=lambda batch, skip_special_tokens=True: [decode_map[tuple(row)] for row in batch]
        )

        single_turn_calls = []

        def fail_generate(prompts, images):
            raise UnexpectedSecondGenerate("dynamic cascade should not call _generate for each stage")

        def fake_generate_single_turn(prompts, images):
            single_turn_calls.append(list(prompts))
            stage_index = len(single_turn_calls)
            return (
                [[100 + stage_index], [200 + stage_index]],
                [
                    [10 + stage_index],
                    [20 + stage_index],
                ],
                None,
                {},
            )

        trainer._generate = fail_generate
        trainer._generate_single_turn = fake_generate_single_turn

        inputs = [
            {"prompt": "prompt-2", "reward_model": {"ground_truth": "<a_1><b_2><c_3><d_4>"}},
            {"prompt": "prompt-2", "reward_model": {"ground_truth": "<a_1><b_2><c_3><d_4>"}},
        ]

        cascade = trainer._run_dynamic_hint_cascade(inputs, images=None, max_hint_depth=3)

        self.assertEqual(
            single_turn_calls,
            [
                ["prompt-2", "prompt-2"],
                ["prompt-2<a_1>", "prompt-2<a_1>"],
                ["prompt-2<a_1><b_2>", "prompt-2<a_1><b_2>"],
                ["prompt-2<a_1><b_2><c_3>", "prompt-2<a_1><b_2><c_3>"],
            ],
        )
        self.assertEqual(cascade["selected_group_hint_depths"], [3])
        self.assertEqual(
            cascade["selected_prompt_texts"],
            ["prompt-2<a_1><b_2><c_3>", "prompt-2<a_1><b_2><c_3>"],
        )
        self.assertEqual(cascade["selected_inputs"][0]["oracle_hint_text"], "<a_1><b_2><c_3>")
        self.assertEqual(cascade["local_num_items_in_batch"], 2)

    def test_dynamic_generation_metrics_returns_scalar_tensor_like_num_items(self):
        module = _load_trainer_module()

        class FakeScalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

            def __add__(self, other):
                return FakeScalar(self.value + other.value)

        class FakeMask:
            def __init__(self, values):
                self.values = list(values)

            def __invert__(self):
                return FakeMask([not value for value in self.values])

            def float(self):
                return FakeVector([1.0 if value else 0.0 for value in self.values])

        class FakeVector:
            def __init__(self, values):
                self.values = list(values)

            def sum(self):
                return FakeScalar(sum(self.values))

            def float(self):
                return self

            def mean(self):
                return FakeScalar(sum(self.values) / len(self.values))

            def min(self):
                return FakeScalar(min(self.values))

            def max(self):
                return FakeScalar(max(self.values))

            def __getitem__(self, key):
                if isinstance(key, FakeMask):
                    return FakeVector([value for value, keep in zip(self.values, key.values) if keep])
                return self.values[key]

            def __len__(self):
                return len(self.values)

        def fake_tensor(values, device=None):
            if values and isinstance(values[0], bool):
                return FakeMask(values)
            return FakeVector(values)

        module.torch = types.SimpleNamespace(
            tensor=fake_tensor,
            zeros=lambda size, device=None: FakeVector([0] * size),
        )

        trainer = object.__new__(module.DynamicHintRuleOnlyGRPOTrainer)
        trainer.accelerator = types.SimpleNamespace(device="cpu", gather=lambda value: value)
        trainer.state = types.SimpleNamespace(num_input_tokens_seen=0)
        trainer._metrics = {"train": defaultdict(list)}
        trainer.eos_token_id = 99
        trainer.pad_token_id = 0

        num_items = trainer._log_selected_batch_generation_metrics(
            "train",
            prompt_ids_list=[[1, 2], [3]],
            completion_ids_list=[[4, 5, 99], [6, 99]],
        )

        self.assertTrue(hasattr(num_items, "item"))
        self.assertEqual(num_items.item(), 5)


if __name__ == "__main__":
    unittest.main()
