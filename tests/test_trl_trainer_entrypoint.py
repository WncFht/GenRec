import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
TRL_TRAINER_PATH = REPO_ROOT / "trl_trainer.py"
DYNAMIC_HINT_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint.sh"
)
ANALYZE_BEAM_HINT_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-analyze-rl-beam-hint.sh"
)
FIXED_HINT_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint.sh"
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
    util_mod.print_main_process = lambda *args, **kwargs: print(*args, **kwargs) if os.environ.get("RANK", "0") == "0" else None
    util_mod.quiet_non_main_process_logging = lambda: None
    sys.modules["util"] = util_mod

    spec = importlib.util.spec_from_file_location("trl_trainer_under_test", TRL_TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrlTrainerEntrypointTests(unittest.TestCase):
    def _run_analyze_beam_hint_dry_run(self, extra_args: Optional[list[str]] = None) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            model_dir = temp_root_path / "model"
            data_dir = temp_root_path / "data" / "rl"
            index_path = temp_root_path / "data" / "id2sid.json"
            add_tokens_path = temp_root_path / "data" / "new_tokens.json"
            summary_path = temp_root_path / "out" / "summary.json"
            details_path = temp_root_path / "out" / "details.json"
            cache_dir = temp_root_path / "cache"
            log_dir = temp_root_path / "log"
            model_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            cache_dir.mkdir(parents=True)
            log_dir.mkdir(parents=True)
            (data_dir / "train.json").write_text("[]", encoding="utf-8")
            index_path.write_text("{}", encoding="utf-8")
            add_tokens_path.write_text("[]", encoding="utf-8")

            with tempfile.NamedTemporaryFile("w", suffix=".sh") as activate_script:
                activate_script.write("return 0\n")
                activate_script.flush()
                env = dict(os.environ)
                env["CONDA_DEFAULT_ENV"] = "genrec"
                command = [
                    "bash",
                    str(ANALYZE_BEAM_HINT_SCRIPT),
                    "--run",
                    "--dry-run",
                    "--conda-activate",
                    activate_script.name,
                    "--model-path",
                    str(model_dir),
                    "--data-dir",
                    str(data_dir),
                    "--index-path",
                    str(index_path),
                    "--add-tokens-path",
                    str(add_tokens_path),
                    "--summary-path",
                    str(summary_path),
                    "--details-path",
                    str(details_path),
                    "--cache-dir",
                    str(cache_dir),
                    "--log-dir",
                    str(log_dir),
                ]
                if extra_args:
                    command.extend(extra_args)
                return subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )

    def _run_fixed_hint_dry_run(self, extra_args: Optional[list[str]] = None) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            model_dir = temp_root_path / "model"
            data_dir = temp_root_path / "data" / "rl"
            index_path = temp_root_path / "data" / "id2sid.json"
            add_tokens_path = temp_root_path / "data" / "new_tokens.json"
            analysis_summary_path = temp_root_path / "analysis" / "summary.json"
            analysis_details_path = temp_root_path / "analysis" / "details.json"
            ds_config_path = temp_root_path / "config" / "zero2.yaml"
            model_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            analysis_summary_path.parent.mkdir(parents=True)
            ds_config_path.parent.mkdir(parents=True)
            (data_dir / "train.json").write_text("[]", encoding="utf-8")
            (data_dir / "valid.json").write_text("[]", encoding="utf-8")
            (data_dir / "test.json").write_text("[]", encoding="utf-8")
            index_path.write_text("{}", encoding="utf-8")
            add_tokens_path.write_text("[]", encoding="utf-8")
            analysis_summary_path.write_text("{}", encoding="utf-8")
            analysis_details_path.write_text("{}", encoding="utf-8")
            ds_config_path.write_text("train_micro_batch_size_per_gpu: 1\n", encoding="utf-8")

            command = [
                "bash",
                str(FIXED_HINT_SCRIPT),
                "--run",
                "--dry-run",
                "--model-path",
                str(model_dir),
                "--data-dir",
                str(data_dir),
                "--index-path",
                str(index_path),
                "--add-tokens-path",
                str(add_tokens_path),
                "--analysis-summary-path",
                str(analysis_summary_path),
                "--analysis-details-path",
                str(analysis_details_path),
                "--ds-config",
                str(ds_config_path),
            ]
            if extra_args:
                command.extend(extra_args)
            return subprocess.run(
                command,
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_fixed_and_dynamic_launchers_keep_step_count_driving_defaults_aligned(self):
        fixed_text = FIXED_HINT_SCRIPT.read_text(encoding="utf-8")
        dynamic_text = DYNAMIC_HINT_SCRIPT.read_text(encoding="utf-8")

        def extract_default(script_text: str, var_name: str) -> str:
            pattern = rf'^{var_name}="\$\{{{var_name}:-([^"]+)}}"'
            match = re.search(pattern, script_text, flags=re.MULTILINE)
            self.assertIsNotNone(match, msg=f"Missing default assignment for {var_name}")
            return match.group(1)

        for var_name in (
            "DATA_VARIANT_DEFAULT",
            "NUM_PROCESSES",
            "PER_DEVICE_TRAIN_BSZ",
            "PER_DEVICE_EVAL_BSZ",
            "GRAD_ACC",
            "NUM_EPOCHS",
        ):
            self.assertEqual(
                extract_default(fixed_text, var_name),
                extract_default(dynamic_text, var_name),
                msg=f"{var_name} default drift would change steps-per-epoch parity",
            )

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

    def test_analyze_beam_hint_shell_dry_run_defaults_to_beam_16_only(self):
        result = self._run_analyze_beam_hint_dry_run()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--beam-sizes 16", result.stdout)
        self.assertNotIn("--beam-sizes 8,16", result.stdout)

    def test_analyze_beam_hint_shell_dry_run_keeps_explicit_beam_override(self):
        result = self._run_analyze_beam_hint_dry_run(["--beam-sizes", "8,16"])

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--beam-sizes 8\\,16", result.stdout)

    def test_fixed_hint_shell_dry_run_uses_task_index_fix_names(self):
        result = self._run_fixed_hint_dry_run()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(
            "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495",
            result.stdout,
        )
        self.assertIn(
            "--run_name instruments_grec_rl_rule_only_fixed_hint_taskfix_b16_ckpt495",
            result.stdout,
        )

    def test_non_main_rank_suppresses_startup_info_logs(self):
        grpo_kwargs = {}
        module = _load_trl_trainer_module(grpo_kwargs)
        stdout = io.StringIO()

        with mock.patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1"}, clear=False):
            with redirect_stdout(stdout):
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
                        dynamic_hint_max_depth=3,
                    )

        output = stdout.getvalue()
        self.assertNotIn("[INFO] dynamic_hint_generation_mode=cascade", output)
        self.assertNotIn("[INFO] raw_bool_args=", output)


if __name__ == "__main__":
    unittest.main()
