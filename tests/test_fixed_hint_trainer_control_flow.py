import importlib.util
import io
import re
import sys
import types
import unittest
from collections import defaultdict
from contextlib import redirect_stdout
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

    class _GRPOTrainer:
        def _compute_loss(self, model, inputs):
            return None

        def _get_per_token_logps_and_entropies(
            self,
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size=None,
            compute_entropy=False,
            pixel_values=None,
            image_grid_thw=None,
            num_images=None,
            pixel_attention_mask=None,
            image_sizes=None,
            token_type_ids=None,
        ):
            self._super_logps_call = {
                "model": model,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "logits_to_keep": logits_to_keep,
                "batch_size": batch_size,
                "compute_entropy": compute_entropy,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "num_images": num_images,
                "pixel_attention_mask": pixel_attention_mask,
                "image_sizes": image_sizes,
                "token_type_ids": token_type_ids,
            }
            return "super-logps", "super-entropies"

    trl_mod.GRPOTrainer = _GRPOTrainer
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

    fixed_hint_utils_mod = types.ModuleType("fixed_hint_utils")
    fixed_hint_utils_mod.build_hint_text = lambda ground_truth, hint_depth: "".join(
        re.findall(r"<[^>]+>", ground_truth)[: max(hint_depth, 0)]
    )
    fixed_hint_utils_mod.build_prompt_with_hint = lambda example, formatter: (
        f"{formatter(example['prompt'])}{example.get('oracle_hint_text', '')}"
    )
    sys.modules["fixed_hint_utils"] = fixed_hint_utils_mod


def _load_trainer_module():
    _install_trainer_stubs()
    spec = importlib.util.spec_from_file_location("fixed_hint_grpo_trainer_under_test", TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FixedHintTrainerControlFlowTests(unittest.TestCase):
    def test_hint_ce_shift_mask_marks_only_suffix_hint_predictions(self):
        module = _load_trainer_module()

        mask = module.build_prompt_hint_shift_mask(
            prompt_lengths=[5, 4, 3],
            hint_token_counts=[2, 0, 1],
        )

        self.assertEqual(
            mask,
            [
                [0, 0, 1, 1],
                [0, 0, 0],
                [0, 1],
            ],
        )

    def test_hint_ce_shift_mask_is_empty_for_single_token_prompt(self):
        module = _load_trainer_module()

        mask = module.build_prompt_hint_shift_mask(
            prompt_lengths=[1],
            hint_token_counts=[0],
        )

        self.assertEqual(mask, [[]])

    def test_hint_ce_loss_uses_cached_prompt_logits_without_second_model_call(self):
        module = _load_trainer_module()

        class FakeScalar:
            def __init__(self, value):
                self.value = float(value)

            def item(self):
                return self.value

            def nanmean(self):
                return self

            def sum(self):
                return self

            def detach(self):
                return self

            def clamp(self, min=0.0):
                return FakeScalar(max(self.value, min))

            def __truediv__(self, other):
                other_value = other.value if isinstance(other, FakeScalar) else other
                return FakeScalar(self.value / other_value)

            def __mul__(self, other):
                other_value = other.value if isinstance(other, FakeScalar) else other
                return FakeScalar(self.value * other_value)

        class FakeVector:
            def __init__(self, values):
                self.values = [float(v) for v in values]

            def float(self):
                return self

            def sum(self):
                return FakeScalar(sum(self.values))

            def view_as(self, other):
                return self

            def detach(self):
                return self

            def nanmean(self):
                return FakeScalar(sum(self.values) / len(self.values))

            def __mul__(self, other):
                return FakeVector([a * b for a, b in zip(self.values, other.values)])

        class FakePromptIds:
            def new_zeros(self, shape, dtype=None):
                return FakeScalar(0.0)

            def size(self, dim):
                return 3

            def __getitem__(self, key):
                return self

            def reshape(self, *shape):
                return self

        class FakePromptShiftLogits:
            def size(self, dim):
                return 5

            def reshape(self, *shape):
                return self

        cross_entropy_calls = []

        def fake_cross_entropy(logits, labels, reduction="none"):
            cross_entropy_calls.append((logits, labels, reduction))
            return FakeVector([0.25, 0.75])

        module.torch = types.SimpleNamespace(
            float32="float32",
            isclose=lambda a, b: a.value == b.value,
            cat=lambda tensors, dim=1: "concatenated",
            nn=types.SimpleNamespace(functional=types.SimpleNamespace(cross_entropy=fake_cross_entropy)),
        )

        trainer = object.__new__(module.FixedHintRuleOnlyGRPOTrainer)
        trainer.model = types.SimpleNamespace(training=True)
        trainer.accelerator = types.SimpleNamespace(gather=lambda value: value)
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer.current_gradient_accumulation_steps = 2
        trainer._cached_prompt_shift_logits = FakePromptShiftLogits()

        class FailModel:
            def __call__(self, **kwargs):
                raise AssertionError("hint CE should reuse cached prompt logits instead of calling model again")

        loss = trainer._compute_prompt_hint_ce_loss(
            FailModel(),
            {
                "prompt_hint_ce_mask": FakeVector([1.0, 1.0]),
                "prompt_ids": FakePromptIds(),
                "prompt_mask": "prompt-mask",
                "completion_ids": "completion-ids",
                "completion_mask": "completion-mask",
            },
        )

        self.assertEqual(len(cross_entropy_calls), 1)
        self.assertIs(cross_entropy_calls[0][0], trainer._cached_prompt_shift_logits)
        self.assertEqual(trainer._metrics["train"]["hint_ce/loss"], [0.5])
        self.assertEqual(trainer._metrics["train"]["hint_ce/token_count"], [2.0])
        self.assertEqual(loss.item(), 0.25)

    def test_hint_ce_loss_raises_when_prompt_logits_cache_is_missing(self):
        module = _load_trainer_module()

        class FakeScalar:
            def __init__(self, value):
                self.value = float(value)

            def item(self):
                return self.value

            def nanmean(self):
                return self

            def sum(self):
                return self

            def detach(self):
                return self

            def clamp(self, min=0.0):
                return FakeScalar(max(self.value, min))

        class FakeVector:
            def __init__(self, values):
                self.values = [float(v) for v in values]

            def float(self):
                return self

            def sum(self):
                return FakeScalar(sum(self.values))

            def detach(self):
                return self

        class FakePromptIds:
            def new_zeros(self, shape, dtype=None):
                return FakeScalar(0.0)

        module.torch = types.SimpleNamespace(
            float32="float32",
            isclose=lambda a, b: a.value == b.value,
            cat=lambda tensors, dim=1: "concatenated",
        )

        trainer = object.__new__(module.FixedHintRuleOnlyGRPOTrainer)
        trainer.model = types.SimpleNamespace(training=True)
        trainer.accelerator = types.SimpleNamespace(gather=lambda value: value)
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer._cached_prompt_shift_logits = None

        class FailClosedModel:
            def __init__(self):
                self.calls = 0

            def __call__(self, **kwargs):
                self.calls += 1
                raise AssertionError("model should not be called when prompt logits cache is missing")

        model = FailClosedModel()
        with self.assertRaisesRegex(RuntimeError, "prompt logits cache"):
            trainer._compute_prompt_hint_ce_loss(
                model,
                {
                    "prompt_hint_ce_mask": FakeVector([1.0, 1.0]),
                    "prompt_ids": FakePromptIds(),
                    "prompt_mask": "prompt-mask",
                    "completion_ids": "completion-ids",
                    "completion_mask": "completion-mask",
                },
            )

        self.assertEqual(model.calls, 0)

    def test_compute_loss_records_rl_base_and_weighted_hint_ce_metrics(self):
        module = _load_trainer_module()

        trainer = object.__new__(module.FixedHintRuleOnlyGRPOTrainer)
        trainer.model = types.SimpleNamespace(training=True)
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer.hint_ce_loss_coef = 0.2
        trainer._cached_prompt_shift_logits = None

        original_super_compute_loss = module.GRPOTrainer._compute_loss
        original_hint_ce = module.FixedHintRuleOnlyGRPOTrainer._compute_prompt_hint_ce_loss

        class FakeLoss:
            def __init__(self, value):
                self.value = float(value)

            def item(self):
                return self.value

            def __add__(self, other):
                return FakeLoss(self.value + other.value)

            def __mul__(self, other):
                return FakeLoss(self.value * float(other))

            __rmul__ = __mul__

        def fake_super_compute_loss(self, model, inputs):
            return FakeLoss(1.5)

        def fake_hint_ce(self, model, inputs):
            return FakeLoss(0.25)

        module.GRPOTrainer._compute_loss = fake_super_compute_loss
        module.FixedHintRuleOnlyGRPOTrainer._compute_prompt_hint_ce_loss = fake_hint_ce

        try:
            total_loss = trainer._compute_loss(model="model", inputs={})
        finally:
            module.GRPOTrainer._compute_loss = original_super_compute_loss
            module.FixedHintRuleOnlyGRPOTrainer._compute_prompt_hint_ce_loss = original_hint_ce

        self.assertEqual(total_loss.item(), 1.55)
        self.assertEqual(trainer._metrics["train"]["loss/rl_base"], [1.5])
        self.assertEqual(trainer._metrics["train"]["loss/hint_ce_weighted"], [0.05])

    def test_hint_ce_loss_uses_global_dapo_hint_token_normalizer(self):
        module = _load_trainer_module()

        class FakeScalar:
            def __init__(self, value):
                self.value = float(value)

            def item(self):
                return self.value

            def nanmean(self):
                return self

            def sum(self):
                return self

            def detach(self):
                return self

            def clamp(self, min=0.0):
                return FakeScalar(max(self.value, min))

            def __truediv__(self, other):
                other_value = other.value if isinstance(other, FakeScalar) else other
                return FakeScalar(self.value / other_value)

            def __mul__(self, other):
                other_value = other.value if isinstance(other, FakeScalar) else other
                return FakeScalar(self.value * other_value)

        class FakeVector:
            def __init__(self, values):
                self.values = [float(v) for v in values]

            def float(self):
                return self

            def sum(self):
                return FakeScalar(sum(self.values))

            def view_as(self, other):
                return self

            def detach(self):
                return self

            def nanmean(self):
                return FakeScalar(sum(self.values) / len(self.values))

            def __mul__(self, other):
                if isinstance(other, FakeScalar):
                    return FakeVector([value * other.value for value in self.values])
                return FakeVector([a * b for a, b in zip(self.values, other.values)])

        class FakePromptIds:
            def new_zeros(self, shape, dtype=None):
                return FakeScalar(0.0)

            def size(self, dim):
                return 3

            def __getitem__(self, key):
                return self

            def reshape(self, *shape):
                return self

        class FakePromptShiftLogits:
            def size(self, dim):
                return 5

            def reshape(self, *shape):
                return self

        def fake_cross_entropy(logits, labels, reduction="none"):
            return FakeVector([2.0, 4.0])

        module.torch = types.SimpleNamespace(
            float32="float32",
            isclose=lambda a, b: a.value == b.value,
            nn=types.SimpleNamespace(functional=types.SimpleNamespace(cross_entropy=fake_cross_entropy)),
        )

        trainer = object.__new__(module.FixedHintRuleOnlyGRPOTrainer)
        trainer.model = types.SimpleNamespace(training=True)
        trainer.accelerator = types.SimpleNamespace(gather=lambda value: value, num_processes=2)
        trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        trainer.current_gradient_accumulation_steps = 4
        trainer.loss_type = "dapo"
        trainer._cached_prompt_shift_logits = FakePromptShiftLogits()

        loss = trainer._compute_prompt_hint_ce_loss(
            model="model",
            inputs={
                "prompt_hint_ce_mask": FakeVector([1.0, 1.0]),
                "prompt_ids": FakePromptIds(),
                "prompt_hint_ce_num_items_in_batch": FakeScalar(10.0),
            },
        )

        self.assertEqual(trainer._metrics["train"]["hint_ce/loss"], [3.0])
        self.assertEqual(trainer._metrics["train"]["hint_ce/token_count"], [2.0])
        self.assertAlmostEqual(loss.item(), 1.2)

    def test_hint_ce_super_logps_call_filters_kwargs_unsupported_by_older_trl(self):
        module = _load_trainer_module()
        trainer = object.__new__(module.FixedHintRuleOnlyGRPOTrainer)
        trainer.hint_ce_loss_coef = 0.001

        logps, entropies = trainer._get_per_token_logps_and_entropies(
            model="model",
            input_ids="input_ids",
            attention_mask="attention_mask",
            logits_to_keep=8,
            batch_size=2,
            compute_entropy=False,
            token_type_ids="token_type_ids",
            mm_token_type_ids="mm_token_type_ids",
            image_position_ids="image_position_ids",
        )

        self.assertEqual((logps, entropies), ("super-logps", "super-entropies"))
        self.assertEqual(trainer._super_logps_call["token_type_ids"], "token_type_ids")

    def test_hint_ce_entropy_path_uses_no_grad_like_upstream(self):
        module = _load_trainer_module()
        trainer = object.__new__(module.FixedHintRuleOnlyGRPOTrainer)
        trainer.hint_ce_loss_coef = 0.001
        trainer.temperature = 1.0
        trainer.model_kwarg_keys = set()

        state = {"inside_no_grad": False, "entropy_saw_no_grad": None}

        class TrackingNoGrad:
            def __enter__(self):
                state["inside_no_grad"] = True
                return None

            def __exit__(self, exc_type, exc, tb):
                state["inside_no_grad"] = False
                return False

        class FakeTensor:
            def size(self, dim):
                return 1 if dim == 0 else 5

            def __getitem__(self, key):
                return self

        class FakeLogits:
            def __getitem__(self, key):
                return self

            def __truediv__(self, other):
                return self

        class FakeOutput:
            def __init__(self):
                self.logits = FakeLogits()

        class FakeModel:
            def __call__(self, **kwargs):
                return FakeOutput()

        fake_cat_results = []

        def fake_cat(values, dim=0):
            fake_cat_results.append((tuple(values), dim))
            return values[0]

        def fake_entropy(logits):
            state["entropy_saw_no_grad"] = state["inside_no_grad"]
            return "entropy"

        module.torch = types.SimpleNamespace(
            no_grad=lambda: TrackingNoGrad(),
            cat=fake_cat,
        )
        module._selective_log_softmax = lambda logits, target_ids: "logps"
        module._entropy_from_logits = fake_entropy

        logps, entropies = trainer._get_per_token_logps_and_entropies(
            model=FakeModel(),
            input_ids=FakeTensor(),
            attention_mask=FakeTensor(),
            logits_to_keep=2,
            batch_size=1,
            compute_entropy=True,
        )

        self.assertEqual((logps, entropies), ("logps", "entropy"))
        self.assertTrue(state["entropy_saw_no_grad"])
        self.assertIsNotNone(trainer._cached_prompt_shift_logits)

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
        self.assertEqual(cascade["selected_rule_rewards"], [1.0, 0.0, 1.0, 0.0])

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
        self.assertEqual(cascade["selected_rule_rewards"], [0.0, 0.0])

    def test_log_completions_false_skips_prompt_and_completion_gather(self):
        module = _load_trainer_module()
        trainer = object.__new__(module.FixedHintRuleOnlyGRPOTrainer)
        trainer.log_completions = False
        trainer._logs = {"prompt": [], "completion": [], "images": []}

        def fail_gather_object(value):
            raise AssertionError("gather_object should be skipped when log_completions is disabled")

        module.gather_object = fail_gather_object

        trainer._maybe_log_prompt_completions(["prompt-0"], ["completion-0"])

        self.assertEqual(trainer._logs["prompt"], [])
        self.assertEqual(trainer._logs["completion"], [])

    def test_log_completions_true_gathers_prompt_and_completion_text(self):
        module = _load_trainer_module()
        trainer = object.__new__(module.FixedHintRuleOnlyGRPOTrainer)
        trainer.log_completions = True
        trainer._logs = {"prompt": [], "completion": [], "images": []}

        gather_calls = []

        def fake_gather_object(value):
            gather_calls.append(list(value))
            return [f"gathered::{item}" for item in value]

        module.gather_object = fake_gather_object

        trainer._maybe_log_prompt_completions(["prompt-0"], ["completion-0"])

        self.assertEqual(gather_calls, [["prompt-0"], ["completion-0"]])
        self.assertEqual(trainer._logs["prompt"], ["gathered::prompt-0"])
        self.assertEqual(trainer._logs["completion"], ["gathered::completion-0"])

    def test_dynamic_rule_only_fast_path_gathers_reused_selected_rule_rewards(self):
        module = _load_trainer_module()
        captured = {}
        gather_calls = []

        def fake_tensor(value, dtype=None, device=None):
            captured["value"] = value
            captured["dtype"] = dtype
            captured["device"] = device
            return value

        module.torch = types.SimpleNamespace(tensor=fake_tensor, float32="float32")

        trainer = object.__new__(module.DynamicHintRuleOnlyGRPOTrainer)
        trainer.accelerator = types.SimpleNamespace(
            device="cpu",
            gather=lambda value: gather_calls.append(value) or ["gathered", value],
        )
        trainer.reward_func_names = ["rule_reward"]

        def fail_calculate_rewards(*args, **kwargs):
            raise AssertionError("_calculate_rewards should not be called for dynamic rule_only fast path")

        trainer._calculate_rewards = fail_calculate_rewards

        rewards_per_func = trainer._resolve_rewards_per_func(
            inputs=[{"prompt": "prompt-0"}, {"prompt": "prompt-1"}],
            prompts=["prompt-0", "prompt-1"],
            completions=["completion-0", "completion-1"],
            completion_ids_list=[[11], [12]],
            selected_rule_rewards=[1.0, 0.0],
        )

        self.assertEqual(gather_calls, [[[1.0], [0.0]]])
        self.assertEqual(rewards_per_func, ["gathered", [[1.0], [0.0]]])
        self.assertEqual(captured["value"], [[1.0], [0.0]])
        self.assertEqual(captured["dtype"], "float32")
        self.assertEqual(captured["device"], "cpu")

    def test_dynamic_rule_only_fast_path_falls_back_for_non_rule_only_reward_setup(self):
        module = _load_trainer_module()
        trainer = object.__new__(module.DynamicHintRuleOnlyGRPOTrainer)
        trainer.reward_func_names = ["rule_reward", "prefix_rule_reward"]

        fallback_calls = []

        def fake_calculate_rewards(inputs, prompts, completions, completion_ids_list):
            fallback_calls.append((inputs, prompts, completions, completion_ids_list))
            return "fallback"

        trainer._calculate_rewards = fake_calculate_rewards

        rewards_per_func = trainer._resolve_rewards_per_func(
            inputs=[{"prompt": "prompt-0"}],
            prompts=["prompt-0"],
            completions=["completion-0"],
            completion_ids_list=[[11]],
            selected_rule_rewards=[1.0],
        )

        self.assertEqual(rewards_per_func, "fallback")
        self.assertEqual(len(fallback_calls), 1)

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

    def test_dynamic_hint_metrics_drop_redundant_selected_frac_and_use_single_depth_list(self):
        module = _load_trainer_module()

        trainer = object.__new__(module.DynamicHintRuleOnlyGRPOTrainer)
        trainer.dynamic_hint_apply_to_eval = False
        trainer._metrics = {"train": defaultdict(list)}
        trainer._logs = {}

        trainer._log_dynamic_hint_metrics(
            mode="train",
            stage_stats=[
                {
                    "requested_hint_depth": 0,
                    "evaluated_group_count": 4,
                    "rule_hit_group_count": 1,
                    "selected_group_count": 1,
                    "remaining_group_count": 3,
                },
                {
                    "requested_hint_depth": 1,
                    "evaluated_group_count": 3,
                    "rule_hit_group_count": 2,
                    "selected_group_count": 2,
                    "remaining_group_count": 1,
                },
            ],
            selected_group_hint_depths=[0, 1, 1, 2],
            selected_group_rule_hits=[True, True, True, False],
            max_hint_depth=2,
        )

        self.assertEqual(trainer._metrics["train"]["dynamic_hint/selected_hint_depth_mean"], [1.0])
        self.assertEqual(trainer._metrics["train"]["dynamic_hint/stage_0_rule_hit_frac"], [0.25])
        self.assertEqual(trainer._metrics["train"]["dynamic_hint/stage_1_rule_hit_frac"], [2.0 / 3.0])
        self.assertEqual(trainer._metrics["train"]["dynamic_hint/stage_0_remaining_frac"], [0.75])
        self.assertEqual(trainer._metrics["train"]["dynamic_hint/stage_1_remaining_frac"], [1.0 / 3.0])
        self.assertNotIn("dynamic_hint/stage_0_selected_frac", trainer._metrics["train"])
        self.assertNotIn("dynamic_hint/stage_1_selected_frac", trainer._metrics["train"])
        self.assertEqual(trainer._logs["dynamic_hint_depth"], [0, 1, 1, 2])

    def test_dynamic_hint_metrics_skip_eval_when_eval_hints_disabled(self):
        module = _load_trainer_module()

        trainer = object.__new__(module.DynamicHintRuleOnlyGRPOTrainer)
        trainer.dynamic_hint_apply_to_eval = False
        trainer._metrics = {"eval": defaultdict(list)}
        trainer._logs = {}

        trainer._log_dynamic_hint_metrics(
            mode="eval",
            stage_stats=[
                {
                    "requested_hint_depth": 0,
                    "evaluated_group_count": 4,
                    "rule_hit_group_count": 1,
                    "selected_group_count": 4,
                    "remaining_group_count": 0,
                }
            ],
            selected_group_hint_depths=[0, 0, 0, 0],
            selected_group_rule_hits=[True, False, False, False],
            max_hint_depth=0,
        )

        self.assertEqual(dict(trainer._metrics["eval"]), {})
        self.assertNotIn("dynamic_hint_depth", trainer._logs)

    def test_dynamic_hint_metrics_keep_eval_when_eval_hints_enabled(self):
        module = _load_trainer_module()

        trainer = object.__new__(module.DynamicHintRuleOnlyGRPOTrainer)
        trainer.dynamic_hint_apply_to_eval = True
        trainer._metrics = {"eval": defaultdict(list)}
        trainer._logs = {}

        trainer._log_dynamic_hint_metrics(
            mode="eval",
            stage_stats=[
                {
                    "requested_hint_depth": 0,
                    "evaluated_group_count": 4,
                    "rule_hit_group_count": 1,
                    "selected_group_count": 1,
                    "remaining_group_count": 3,
                }
            ],
            selected_group_hint_depths=[0, 1, 1, 2],
            selected_group_rule_hits=[True, True, True, False],
            max_hint_depth=2,
        )

        self.assertEqual(trainer._metrics["eval"]["dynamic_hint/selected_hint_depth_mean"], [1.0])
        self.assertEqual(trainer._metrics["eval"]["dynamic_hint/stage_0_rule_hit_frac"], [0.25])
        self.assertEqual(trainer._logs["dynamic_hint_depth"], [0, 1, 1, 2])

    def test_dynamic_hint_cascade_suppresses_stage_logs_for_base_only_eval_path(self):
        module = _load_trainer_module()
        trainer = object.__new__(module.DynamicHintRuleOnlyGRPOTrainer)
        trainer.num_generations = 2
        trainer.accelerator = types.SimpleNamespace(is_main_process=True)
        trainer.processing_class = types.SimpleNamespace(
            batch_decode=lambda batch, skip_special_tokens=True: ["<a_1><b_2>", "<a_1><x_9>"]
        )

        def fake_generate_single_turn(prompts, images):
            return (
                [[100], [101]],
                [[11, 12], [13, 14]],
                None,
                {},
            )

        trainer._generate_single_turn = fake_generate_single_turn

        inputs = [
            {"prompt": "prompt-0", "reward_model": {"ground_truth": "<a_1><b_2>"}},
            {"prompt": "prompt-0", "reward_model": {"ground_truth": "<a_1><b_2>"}},
        ]

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            cascade = trainer._run_dynamic_hint_cascade(inputs, images=None, max_hint_depth=0)

        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(cascade["selected_group_hint_depths"], [0])


if __name__ == "__main__":
    unittest.main()
