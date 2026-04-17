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
DYNAMIC_HINT_MAX1_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-max1.sh"
)
DYNAMIC_HINT_SID_ONLY_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-sid-only.sh"
)
DYNAMIC_HINT_SID_TITLE_DESC_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint-sid-title-desc.sh"
)
DYNAMIC_HINT_RANKING_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-ranking-dynamic-hint.sh"
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
FIXED_HINT_CE_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-ce.sh"
)
FIXED_HINT_SID_ONLY_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-only.sh"
)
FIXED_HINT_SID_TITLE_DESC_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-title-desc.sh"
)
FIXED_HINT_PREFIX_SEQ_SID_ONLY_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-prefix-seq-only-fixed-hint-sid-only.sh"
)
GREC_RL_SCRIPT_DIR = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec"
)


class StopAfterTrainerInit(RuntimeError):
    pass


def _load_trl_trainer_module(
    grpo_kwargs_sink: dict[str, object],
    fixed_hint_trainer_kwargs_sink: Optional[dict[str, object]] = None,
    trainer_kwargs_sink: Optional[dict[str, object]] = None,
    dataset_payload: Optional[dict[str, list[dict[str, object]]]] = None,
):
    fire_mod = types.ModuleType("fire")
    fire_mod.Fire = lambda fn: fn
    sys.modules["fire"] = fire_mod

    datasets_mod = types.ModuleType("datasets")

    class _FakeDataset(list):
        def map(self, fn, desc=None):
            return _FakeDataset([fn(dict(item)) for item in self])

        def __getitem__(self, key):
            if isinstance(key, str):
                return [item[key] for item in self]
            return super().__getitem__(key)

    if dataset_payload is None:
        dataset_payload = {
            "train": [{"prompt": "train-prompt", "reward_model": {"ground_truth": "<a_1>"}}],
            "valid": [{"prompt": "valid-prompt", "reward_model": {"ground_truth": "<a_1>"}}],
            "test": [{"prompt": "test-prompt", "reward_model": {"ground_truth": "<a_1>"}}],
        }

    datasets_mod.load_dataset = lambda *args, **kwargs: {
        split_name: _FakeDataset([dict(item) for item in split_rows])
        for split_name, split_rows in dataset_payload.items()
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
            if trainer_kwargs_sink is not None:
                trainer_kwargs_sink.clear()
                trainer_kwargs_sink.update(kwargs)
            raise StopAfterTrainerInit("stop after capturing GRPO config kwargs")

    trl_mod.GRPOTrainer = _GRPOTrainer
    sys.modules["trl"] = trl_mod

    cli_utils_mod = types.ModuleType("cli_utils")
    cli_utils_mod.coerce_bool_arg = lambda value, arg_name: value if isinstance(value, bool) else str(value).lower() == "true"
    cli_utils_mod.format_typed_value = lambda value: repr(value)
    sys.modules["cli_utils"] = cli_utils_mod

    fixed_hint_utils_mod = types.ModuleType("fixed_hint_utils")
    fixed_hint_utils_mod.apply_fixed_hint_depth_to_example = lambda example, hint_map, **kwargs: {
        **example,
        "oracle_hint_depth": 0,
        "oracle_hint_text": "",
        "oracle_hint_unsolved": False,
    }
    fixed_hint_utils_mod.load_fixed_hint_depth_map = lambda path: {}
    sys.modules["fixed_hint_utils"] = fixed_hint_utils_mod

    fixed_hint_trainer_mod = types.ModuleType("fixed_hint_grpo_trainer")

    class _FixedHintTrainer(_GRPOTrainer):
        def __init__(self, *args, **kwargs):
            if fixed_hint_trainer_kwargs_sink is not None:
                fixed_hint_trainer_kwargs_sink.clear()
                fixed_hint_trainer_kwargs_sink.update(kwargs)
            raise StopAfterTrainerInit("stop after capturing fixed-hint trainer kwargs")

    fixed_hint_trainer_mod.DynamicHintRuleOnlyGRPOTrainer = _GRPOTrainer
    fixed_hint_trainer_mod.FixedHintRuleOnlyGRPOTrainer = _FixedHintTrainer
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

    def _run_fixed_hint_ce_dry_run(self, extra_args: Optional[list[str]] = None) -> subprocess.CompletedProcess:
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
                str(FIXED_HINT_CE_SCRIPT),
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

    def _run_fixed_hint_sid_only_dry_run(self, extra_args: Optional[list[str]] = None) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            model_dir = temp_root_path / "model"
            data_dir = temp_root_path / "data" / "rl"
            index_path = temp_root_path / "data" / "id2sid.json"
            add_tokens_path = temp_root_path / "data" / "new_tokens.json"
            ds_config_path = temp_root_path / "config" / "zero2.yaml"
            model_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            ds_config_path.parent.mkdir(parents=True)
            (data_dir / "train.json").write_text("[]", encoding="utf-8")
            (data_dir / "valid.json").write_text("[]", encoding="utf-8")
            (data_dir / "test.json").write_text("[]", encoding="utf-8")
            index_path.write_text("{}", encoding="utf-8")
            add_tokens_path.write_text("[]", encoding="utf-8")
            ds_config_path.write_text("train_micro_batch_size_per_gpu: 1\n", encoding="utf-8")

            command = [
                "bash",
                str(FIXED_HINT_SID_ONLY_SCRIPT),
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

    def _run_fixed_hint_sid_title_desc_dry_run(
        self, extra_args: Optional[list[str]] = None
    ) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            model_dir = temp_root_path / "model"
            data_dir = temp_root_path / "data" / "rl"
            index_path = temp_root_path / "data" / "id2sid.json"
            add_tokens_path = temp_root_path / "data" / "new_tokens.json"
            ds_config_path = temp_root_path / "config" / "zero2.yaml"
            model_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            ds_config_path.parent.mkdir(parents=True)
            (data_dir / "train.json").write_text("[]", encoding="utf-8")
            (data_dir / "valid.json").write_text("[]", encoding="utf-8")
            (data_dir / "test.json").write_text("[]", encoding="utf-8")
            index_path.write_text("{}", encoding="utf-8")
            add_tokens_path.write_text("[]", encoding="utf-8")
            ds_config_path.write_text("train_micro_batch_size_per_gpu: 1\n", encoding="utf-8")

            command = [
                "bash",
                str(FIXED_HINT_SID_TITLE_DESC_SCRIPT),
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

    def _run_fixed_hint_prefix_seq_sid_only_dry_run(
        self, extra_args: Optional[list[str]] = None
    ) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            model_dir = temp_root_path / "model"
            data_dir = temp_root_path / "data" / "rl"
            index_path = temp_root_path / "data" / "id2sid.json"
            add_tokens_path = temp_root_path / "data" / "new_tokens.json"
            ds_config_path = temp_root_path / "config" / "zero2.yaml"
            model_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            ds_config_path.parent.mkdir(parents=True)
            (data_dir / "train.json").write_text("[]", encoding="utf-8")
            (data_dir / "valid.json").write_text("[]", encoding="utf-8")
            (data_dir / "test.json").write_text("[]", encoding="utf-8")
            index_path.write_text("{}", encoding="utf-8")
            add_tokens_path.write_text("[]", encoding="utf-8")
            ds_config_path.write_text("train_micro_batch_size_per_gpu: 1\n", encoding="utf-8")

            command = [
                "bash",
                str(FIXED_HINT_PREFIX_SEQ_SID_ONLY_SCRIPT),
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

    def test_main_forwards_eval_on_start_to_grpo_config(self):
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
                token_level_prefix_advantage=False,
                reward_mode="rule_only",
                num_beams=4,
                eval_on_start=True,
            )

        self.assertTrue(grpo_kwargs["eval_on_start"])

    def test_main_filters_train_and_eval_splits_by_task_names(self):
        grpo_kwargs = {}
        trainer_kwargs = {}
        dataset_payload = {
            "train": [
                {
                    "prompt": "train-1",
                    "reward_model": {"ground_truth": "<a_1>"},
                    "extra_info": {"task": "task1_sid_sft"},
                },
                {
                    "prompt": "train-2",
                    "reward_model": {"ground_truth": "<a_2>"},
                    "extra_info": {"task": "task4_hisTitle2sid"},
                },
                {
                    "prompt": "train-3",
                    "reward_model": {"ground_truth": "<a_3>"},
                    "extra_info": {"task": "task5_title_desc2sid"},
                },
            ],
            "valid": [
                {
                    "prompt": "valid-1",
                    "reward_model": {"ground_truth": "<a_1>"},
                    "extra_info": {"task": "task1_sid_sft"},
                },
                {
                    "prompt": "valid-2",
                    "reward_model": {"ground_truth": "<a_2>"},
                    "extra_info": {"task": "task5_title_desc2sid"},
                },
            ],
            "test": [
                {
                    "prompt": "test-1",
                    "reward_model": {"ground_truth": "<a_1>"},
                    "extra_info": {"task": "task1_sid_sft"},
                }
            ],
        }
        module = _load_trl_trainer_module(
            grpo_kwargs,
            trainer_kwargs_sink=trainer_kwargs,
            dataset_payload=dataset_payload,
        )

        with self.assertRaises(StopAfterTrainerInit):
            module.main(
                model="dummy-model",
                data_dir="dummy-data",
                index_path="dummy-index",
                output_dir="dummy-output",
                report_to="wandb",
                run_name="dual-task-filter-test-run",
                token_level_prefix_advantage=False,
                reward_mode="rule_only",
                num_beams=4,
                train_task_names="task1_sid_sft,task5_title_desc2sid",
                eval_task_names="task1_sid_sft",
            )

        self.assertEqual(
            [row["extra_info"]["task"] for row in trainer_kwargs["train_dataset"]],
            ["task1_sid_sft", "task5_title_desc2sid"],
        )
        self.assertEqual(
            [row["extra_info"]["task"] for row in trainer_kwargs["eval_dataset"]],
            ["task1_sid_sft"],
        )

    def test_main_rejects_unknown_train_task_name(self):
        grpo_kwargs = {}
        dataset_payload = {
            "train": [
                {
                    "prompt": "train-1",
                    "reward_model": {"ground_truth": "<a_1>"},
                    "extra_info": {"task": "task1_sid_sft"},
                }
            ],
            "valid": [
                {
                    "prompt": "valid-1",
                    "reward_model": {"ground_truth": "<a_1>"},
                    "extra_info": {"task": "task1_sid_sft"},
                }
            ],
            "test": [
                {
                    "prompt": "test-1",
                    "reward_model": {"ground_truth": "<a_1>"},
                    "extra_info": {"task": "task1_sid_sft"},
                }
            ],
        }
        module = _load_trl_trainer_module(grpo_kwargs, dataset_payload=dataset_payload)

        with self.assertRaisesRegex(ValueError, "unknown.*missing_task"):
            module.main(
                model="dummy-model",
                data_dir="dummy-data",
                index_path="dummy-index",
                output_dir="dummy-output",
                report_to="wandb",
                run_name="dual-task-filter-test-run",
                token_level_prefix_advantage=False,
                reward_mode="rule_only",
                num_beams=4,
                train_task_names="missing_task",
            )

    def test_main_rejects_empty_filtered_eval_split(self):
        grpo_kwargs = {}
        dataset_payload = {
            "train": [
                {
                    "prompt": "train-1",
                    "reward_model": {"ground_truth": "<a_1>"},
                    "extra_info": {"task": "task1_sid_sft"},
                }
            ],
            "valid": [],
            "test": [
                {
                    "prompt": "test-1",
                    "reward_model": {"ground_truth": "<a_1>"},
                    "extra_info": {"task": "task1_sid_sft"},
                }
            ],
        }
        module = _load_trl_trainer_module(grpo_kwargs, dataset_payload=dataset_payload)

        with self.assertRaisesRegex(ValueError, "empty filtered eval split"):
            module.main(
                model="dummy-model",
                data_dir="dummy-data",
                index_path="dummy-index",
                output_dir="dummy-output",
                report_to="wandb",
                run_name="dual-task-filter-test-run",
                token_level_prefix_advantage=False,
                reward_mode="rule_only",
                num_beams=4,
                eval_task_names="task1_sid_sft",
            )

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
        self.assertIn("--eval_on_start true", result.stdout)

    def test_dynamic_hint_max1_shell_dry_run_limits_hint_depth_to_one_token(self):
        result = subprocess.run(
            ["bash", str(DYNAMIC_HINT_MAX1_SCRIPT), "--dry-run"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--reward_mode rule_only", result.stdout)
        self.assertIn("--dynamic_hint_max_depth 1", result.stdout)
        self.assertIn("--eval_on_start true", result.stdout)

    def test_dynamic_hint_sid_only_shell_dry_run_uses_rl_sid_only_variant(self):
        result = subprocess.run(
            ["bash", str(DYNAMIC_HINT_SID_ONLY_SCRIPT), "--dry-run"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Instruments_grec_rlsidonly_index_emb", result.stdout)
        self.assertIn("--reward_mode rule_only", result.stdout)
        self.assertIn("--dynamic_hint_max_depth 3", result.stdout)
        self.assertIn("--eval_on_start true", result.stdout)

    def test_dynamic_hint_sid_only_shell_is_standalone_launcher(self):
        script_text = DYNAMIC_HINT_SID_ONLY_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("trl_trainer.py", script_text)
        self.assertNotIn(
            "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-dynamic-hint.sh",
            script_text,
        )
        self.assertNotIn("exec bash", script_text)

    def test_dynamic_hint_sid_title_desc_shell_dry_run_forwards_dual_task_filters(self):
        result = subprocess.run(
            ["bash", str(DYNAMIC_HINT_SID_TITLE_DESC_SCRIPT), "--dry-run"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--train_task_names task1_sid_sft\\,task5_title_desc2sid", result.stdout)
        self.assertIn("--eval_task_names task1_sid_sft", result.stdout)
        self.assertIn("--reward_mode rule_only", result.stdout)

    def test_dynamic_hint_sid_title_desc_shell_is_standalone_launcher(self):
        script_text = DYNAMIC_HINT_SID_TITLE_DESC_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("trl_trainer.py", script_text)
        self.assertNotIn("BASE_SCRIPT=", script_text)
        self.assertNotIn("exec bash", script_text)

    def test_dynamic_hint_ranking_shell_dry_run_uses_ranking_reward_mode(self):
        result = subprocess.run(
            ["bash", str(DYNAMIC_HINT_RANKING_SCRIPT), "--dry-run"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--reward_mode ranking", result.stdout)
        self.assertIn("--dynamic_hint_max_depth 3", result.stdout)
        self.assertIn("--eval_on_start true", result.stdout)

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
        self.assertIn("--eval_on_start true", result.stdout)

    def test_fixed_hint_shell_forwards_hint_ce_loss_coef_to_fixed_hint_trainer(self):
        grpo_kwargs = {}
        fixed_hint_trainer_kwargs = {}
        module = _load_trl_trainer_module(grpo_kwargs, fixed_hint_trainer_kwargs)

        with self.assertRaises(StopAfterTrainerInit):
            module.main(
                model="dummy-model",
                data_dir="dummy-data",
                index_path="dummy-index",
                output_dir="dummy-output",
                report_to="wandb",
                run_name="fixed-hint-ce-test-run",
                token_level_prefix_advantage=False,
                reward_mode="rule_only",
                num_beams=4,
                fixed_hint_depth_map_path="dummy-fixed-hint-map.json",
                hint_ce_loss_coef=0.0025,
            )

        self.assertEqual(fixed_hint_trainer_kwargs["hint_ce_loss_coef"], 0.0025)
        self.assertTrue(grpo_kwargs["gradient_checkpointing"])
        self.assertEqual(
            grpo_kwargs["gradient_checkpointing_kwargs"],
            {"use_reentrant": False},
        )

    def test_fixed_hint_ce_shell_dry_run_forwards_hint_ce_loss(self):
        result = self._run_fixed_hint_ce_dry_run()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(
            "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-sft495",
            result.stdout,
        )
        self.assertIn("--hint_ce_loss_coef 0.001", result.stdout)
        self.assertIn("--reward_mode rule_only", result.stdout)
        self.assertIn("--eval_on_start false", result.stdout)

    def test_fixed_hint_ce_shell_dry_run_keeps_beam16_only_defaults(self):
        result = self._run_fixed_hint_ce_dry_run()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--num_beams 16", result.stdout)
        self.assertIn("--export-fixed-hint-beam-size 16", result.stdout)
        self.assertNotIn("--beam-sizes 8\\,16", result.stdout)

    def test_fixed_hint_ce_shell_is_standalone_launcher(self):
        script_text = FIXED_HINT_CE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("trl_trainer.py", script_text)
        self.assertIn("analyze_rl_beam_hint.py", script_text)
        self.assertNotIn("BASE_SCRIPT=", script_text)
        self.assertNotIn("exec bash", script_text)
        self.assertNotIn(
            "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint.sh",
            script_text,
        )

    def test_fixed_hint_sid_only_shell_dry_run_uses_sid_only_variant_and_cache_names(self):
        script_text = FIXED_HINT_SID_ONLY_SCRIPT.read_text(encoding="utf-8")
        result = self._run_fixed_hint_sid_only_dry_run()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn('DATA_VARIANT_DEFAULT="${DATA_VARIANT_DEFAULT:-Instruments_grec_rlsidonly_index_emb', script_text)
        self.assertIn(
            "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495",
            result.stdout,
        )
        self.assertIn(
            "--run_name instruments_grec_rl_rule_only_fixed_hint_taskfix_b16_sid_only_ckpt495",
            result.stdout,
        )
        self.assertIn("instruments_grec_rlsidonly_beam_hint", result.stdout)
        self.assertIn("--eval_on_start true", result.stdout)

    def test_fixed_hint_sid_only_shell_dry_run_keeps_export_beam_size_aligned(self):
        result = self._run_fixed_hint_sid_only_dry_run()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(
            result.stdout.count("--beam-sizes 16"),
            2,
            msg="Analyze and export commands should both stay on beam 16 instead of export falling back to 8,16.",
        )

    def test_fixed_hint_prefix_seq_sid_only_shell_dry_run_uses_sequence_prefix_rule_mode(self):
        result = self._run_fixed_hint_prefix_seq_sid_only_dry_run()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(
            "Instruments-grec-grpo-prefix-seq-only-fixedhint-taskfix-b16-sid-only-sft495",
            result.stdout,
        )
        self.assertIn("--reward_mode prefix_rule_only", result.stdout)
        self.assertIn("--prefix_reward_normalize true", result.stdout)
        self.assertIn("--probe_rule_with_zero_weight false", result.stdout)
        self.assertIn("--token_level_prefix_advantage false", result.stdout)
        self.assertIn("--fixed_hint_depth_map_path", result.stdout)
        self.assertIn("instruments_grec_rlsidonly_beam_hint", result.stdout)

    def test_fixed_hint_prefix_seq_sid_only_shell_is_standalone_launcher(self):
        script_text = FIXED_HINT_PREFIX_SEQ_SID_ONLY_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("analyze_rl_beam_hint.py", script_text)
        self.assertIn("trl_trainer.py", script_text)
        self.assertNotIn("BASE_SCRIPT=", script_text)
        self.assertNotIn('exec bash "$BASE_SCRIPT" "$@"', script_text)
        self.assertNotIn(
            "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-grec-rl-rule-only-fixed-hint-sid-only.sh",
            script_text,
        )

    def test_fixed_hint_sid_only_shell_keeps_dedicated_rule_only_train_defaults(self):
        script_text = FIXED_HINT_SID_ONLY_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("--reward_mode rule_only", script_text)
        self.assertIn("--prefix_reward_normalize true", script_text)
        self.assertIn("--probe_rule_with_zero_weight false", script_text)
        self.assertIn("--token_level_prefix_advantage false", script_text)
        self.assertNotIn("--reward_mode \"$REWARD_MODE\"", script_text)

    def test_fixed_hint_sid_title_desc_shell_dry_run_forwards_dual_task_filters(self):
        result = self._run_fixed_hint_sid_title_desc_dry_run()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--train_task_names task1_sid_sft\\,task5_title_desc2sid", result.stdout)
        self.assertIn("--eval_task_names task1_sid_sft", result.stdout)
        self.assertIn("--task-names task1_sid_sft\\,task5_title_desc2sid", result.stdout)
        self.assertIn("--reward_mode rule_only", result.stdout)

    def test_fixed_hint_sid_title_desc_shell_is_standalone_launcher(self):
        script_text = FIXED_HINT_SID_TITLE_DESC_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("analyze_rl_beam_hint.py", script_text)
        self.assertIn("trl_trainer.py", script_text)
        self.assertNotIn("BASE_SCRIPT=", script_text)
        self.assertNotIn("exec bash", script_text)

    def test_all_grec_rl_launchers_enable_eval_on_start_by_default(self):
        training_scripts = sorted(
            path
            for path in GREC_RL_SCRIPT_DIR.glob("*-rl*.sh")
            if "analyze-rl-beam-hint" not in path.name and "fixed-hint-ce" not in path.name
        )
        self.assertTrue(training_scripts, msg="Expected grecc RL launcher scripts")

        for script_path in training_scripts:
            text = script_path.read_text(encoding="utf-8")
            self.assertIn(
                'EVAL_ON_START="${EVAL_ON_START:-true}"',
                text,
                msg=f"Missing default eval-on-start enablement in {script_path.name}",
            )
            self.assertIn(
                '--eval_on_start "$EVAL_ON_START"',
                text,
                msg=f"Missing eval_on_start forwarding in {script_path.name}",
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

    def test_dynamic_hint_accepts_ranking_reward_mode(self):
        grpo_kwargs = {}
        module = _load_trl_trainer_module(grpo_kwargs)

        with self.assertRaises(StopAfterTrainerInit):
            module.main(
                model="dummy-model",
                data_dir="dummy-data",
                index_path="dummy-index",
                output_dir="dummy-output",
                report_to="wandb",
                run_name="dynamic-hint-ranking-test-run",
                token_level_prefix_advantage=False,
                reward_mode="ranking",
                num_beams=4,
                dynamic_hint_max_depth=3,
            )

    def test_fixed_hint_accepts_prefix_rule_only_reward_mode(self):
        grpo_kwargs = {}
        module = _load_trl_trainer_module(grpo_kwargs)

        with self.assertRaises(StopAfterTrainerInit):
            module.main(
                model="dummy-model",
                data_dir="dummy-data",
                index_path="dummy-index",
                output_dir="dummy-output",
                report_to="wandb",
                run_name="fixed-hint-prefix-seq-test-run",
                token_level_prefix_advantage=False,
                reward_mode="prefix_rule_only",
                num_beams=4,
                fixed_hint_depth_map_path="dummy-fixed-hint-map.json",
                fixed_hint_unsolved_depth=3,
            )


if __name__ == "__main__":
    unittest.main()
