import importlib.util
import subprocess
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRL_TRAINER_PATH = REPO_ROOT / "trl_trainer.py"
DYNAMIC_HINT_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint.sh"
)


class StopAfterTrainerInit(RuntimeError):
    pass


def _load_trl_trainer_module(grpo_kwargs_sink: dict[str, object]):
    fire_mod = types.ModuleType("fire")
    fire_mod.Fire = lambda fn: fn
    sys.modules["fire"] = fire_mod

    datasets_mod = types.ModuleType("datasets")
    datasets_mod.load_dataset = lambda *args, **kwargs: {
        "train": [],
        "valid": [],
        "test": [],
    }
    sys.modules["datasets"] = datasets_mod

    transformers_mod = types.ModuleType("transformers")

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return object()

    transformers_mod.AutoTokenizer = _AutoTokenizer
    sys.modules["transformers"] = transformers_mod

    trainer_utils_mod = types.ModuleType("transformers.trainer_utils")
    trainer_utils_mod.get_last_checkpoint = lambda output_dir: None
    sys.modules["transformers.trainer_utils"] = trainer_utils_mod

    trl_mod = types.ModuleType("trl")

    class _GRPOTrainer:
        def __init__(self, *args, **kwargs):
            raise StopAfterTrainerInit("stop after capturing GRPO config kwargs")

    trl_mod.GRPOTrainer = _GRPOTrainer
    sys.modules["trl"] = trl_mod

    cli_utils_mod = types.ModuleType("cli_utils")
    cli_utils_mod.coerce_bool_arg = lambda value, arg_name: value if isinstance(value, bool) else str(value).lower() == "true"
    cli_utils_mod.format_typed_value = lambda value: repr(value)
    sys.modules["cli_utils"] = cli_utils_mod

    fixed_hint_utils_mod = types.ModuleType("fixed_hint_utils")
    fixed_hint_utils_mod.apply_fixed_hint_depth_to_example = lambda example, hint_map, **kwargs: example
    fixed_hint_utils_mod.load_fixed_hint_depth_map = lambda path: {}
    sys.modules["fixed_hint_utils"] = fixed_hint_utils_mod

    fixed_hint_trainer_mod = types.ModuleType("fixed_hint_grpo_trainer")
    fixed_hint_trainer_mod.DynamicHintRuleOnlyGRPOTrainer = _GRPOTrainer
    fixed_hint_trainer_mod.FixedHintRuleOnlyGRPOTrainer = _GRPOTrainer
    sys.modules["fixed_hint_grpo_trainer"] = fixed_hint_trainer_mod

    mimigenrec_mod = types.ModuleType("MIMIGenRec")

    class _MIMIGenRec:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return object()

    def _capture_grpo_config(**kwargs):
        grpo_kwargs_sink.clear()
        grpo_kwargs_sink.update(kwargs)
        return types.SimpleNamespace(**kwargs)

    mimigenrec_mod.MIMIGenRec = _MIMIGenRec
    mimigenrec_mod.get_grpo_config = _capture_grpo_config
    sys.modules["MIMIGenRec"] = mimigenrec_mod

    rewards_mod = types.ModuleType("rewards.ranking_reward")
    rewards_mod.build_reward_setup = lambda **kwargs: ([lambda *a, **k: [0.0]], None)
    sys.modules["rewards.ranking_reward"] = rewards_mod

    token_prefix_mod = types.ModuleType("token_prefix_grpo_trainer")
    token_prefix_mod.TokenPrefixGRPOTrainer = _GRPOTrainer
    sys.modules["token_prefix_grpo_trainer"] = token_prefix_mod

    util_mod = types.ModuleType("util")
    util_mod.build_constrained_logits_processor = lambda *args, **kwargs: None
    util_mod.build_fixed_hint_constrained_logits_processor = lambda *args, **kwargs: None
    sys.modules["util"] = util_mod

    spec = importlib.util.spec_from_file_location("trl_trainer_under_test", TRL_TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrlTrainerEntrypointTests(unittest.TestCase):
    def test_main_forwards_max_completion_length_and_run_name_to_grpo_config(self):
        grpo_kwargs = {}
        module = _load_trl_trainer_module(grpo_kwargs)

        with self.assertRaises(StopAfterTrainerInit):
            module.main(
                model="dummy-model",
                data_dir="dummy-data",
                index_path="dummy-index",
                output_dir="dummy-output",
                report_to="wandb",
                run_name="dynamic-hint-test-run",
                max_completion_length=123,
                token_level_prefix_advantage=False,
                reward_mode="rule_only",
                num_beams=4,
            )

        self.assertEqual(grpo_kwargs["max_completion_length"], 123)
        self.assertEqual(grpo_kwargs["run_name"], "dynamic-hint-test-run")

    def test_dynamic_hint_shell_dry_run_forwards_run_name_and_uses_working_batch_default(self):
        result = subprocess.run(
            ["bash", str(DYNAMIC_HINT_SCRIPT), "--dry-run"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--run_name", result.stdout)
        self.assertIn("--per_device_train_batch_size 64", result.stdout)
        self.assertIn("--per_device_eval_batch_size 64", result.stdout)


if __name__ == "__main__":
    unittest.main()
