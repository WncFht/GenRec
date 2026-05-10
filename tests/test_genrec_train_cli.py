import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from tests.test_trl_trainer_entrypoint import REPO_ROOT, _load_trl_trainer_module


DEFAULTS = {
    "model": "dummy-model",
    "index_path": "dummy-index",
    "sid_levels": -1,
    "prefix": None,
    "num_beams": 16,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 50,
    "data_dir": "dummy-data",
    "output_dir": "dummy-output",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 2,
    "learning_rate": 1e-5,
    "logging_steps": 1,
    "eval_step": 100,
    "eval_strategy": "steps",
    "eval_on_start": False,
    "save_strategy": "steps",
    "save_steps": 0.1,
    "save_total_limit": 3,
    "save_only_model": False,
    "warmup_ratio": 0.03,
    "max_grad_norm": 0.3,
    "optim": "paged_adamw_32bit",
    "lr_scheduler_type": "cosine",
    "max_completion_length": 128,
    "beta": 1e-3,
    "repetition_penalty": 1.0,
    "do_sample": False,
    "reward_mode": "prefix_only",
    "prefix_reward_normalize": True,
    "probe_rule_with_zero_weight": True,
    "token_level_prefix_advantage": True,
    "token_adv_total_token_normalize": False,
    "token_level_ndcg_error_token_penalty": False,
    "fixed_hint_depth_map_path": None,
    "fixed_hint_depth_cap": None,
    "fixed_hint_unsolved_depth": 3,
    "fixed_hint_task_names": None,
    "fixed_hint_apply_to_eval": False,
    "hint_ce_loss_coef": 0.0,
    "dynamic_hint_max_depth": None,
    "dynamic_hint_apply_to_eval": False,
    "dynamic_hint_task_names": None,
    "train_task_names": None,
    "eval_task_names": None,
    "bf16": True,
    "deepspeed": None,
    "report_to": None,
    "run_name": None,
    "resume_from_checkpoint": "auto",
}


class GenRecTrainCliTests(unittest.TestCase):
    def test_rl_cli_runs_trl_trainer_with_config_file_and_cli_overrides(self):
        captured = {}

        def fake_trl_main(**kwargs):
            captured.update(kwargs)

        grpo_kwargs = {}
        module = _load_trl_trainer_module(grpo_kwargs)

        config_payload = {
            "model_data": {
                "model": "config-model",
                "data_dir": "config-data",
                "index_path": "config-index",
                "output_dir": "config-output",
            },
            "algorithm": {
                "hint_mode": "dynamic",
                "reward_mode": "rule_only",
                "token_level_prefix_advantage": False,
            },
            "hint": {
                "dynamic_hint_max_depth": 3,
            },
            "runtime": {
                "run_name": "config-run",
            },
        }

        with tempfile.TemporaryDirectory() as temp_root:
            config_path = Path(temp_root) / "rl-config.json"
            config_path.write_text(json.dumps(config_payload), encoding="utf-8")
            exit_code = module.main(
                [
                    "--config",
                    str(config_path),
                    "--run-name",
                    "override-run",
                    "--output-dir",
                    "override-output",
                ],
                trl_main=fake_trl_main,
                base_defaults=DEFAULTS,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["model"], "config-model")
        self.assertEqual(captured["data_dir"], "config-data")
        self.assertEqual(captured["index_path"], "config-index")
        self.assertEqual(captured["output_dir"], "override-output")
        self.assertEqual(captured["run_name"], "override-run")
        self.assertEqual(captured["reward_mode"], "rule_only")
        self.assertEqual(captured["dynamic_hint_max_depth"], 3)
        self.assertIsNone(captured["fixed_hint_depth_map_path"])

    def test_rl_cli_dry_run_prints_nested_config_and_flat_kwargs(self):
        output = io.StringIO()
        grpo_kwargs = {}
        module = _load_trl_trainer_module(grpo_kwargs)
        config_payload = {
            "algorithm": {
                "hint_mode": "none",
                "reward_mode": "rule_only",
                "token_level_prefix_advantage": False,
            }
        }

        with tempfile.TemporaryDirectory() as temp_root:
            config_path = Path(temp_root) / "rl-config.json"
            config_path.write_text(json.dumps(config_payload), encoding="utf-8")
            with redirect_stdout(output):
                exit_code = module.main(
                    ["--config", str(config_path), "--dry-run"],
                    trl_main=lambda **kwargs: self.fail("dry-run should not invoke trl_main"),
                    base_defaults=DEFAULTS,
                )

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn('"hint_mode": "none"', rendered)
        self.assertIn('"reward_mode": "rule_only"', rendered)
        self.assertIn('"model": "dummy-model"', rendered)

    def test_rl_cli_accepts_dotted_set_override_for_fixed_hint(self):
        captured = {}

        def fake_trl_main(**kwargs):
            captured.update(kwargs)

        grpo_kwargs = {}
        module = _load_trl_trainer_module(grpo_kwargs)

        exit_code = module.main(
            [
                "--set",
                "algorithm.hint_mode=fixed",
                "--set",
                "algorithm.reward_mode=rule_only",
                "--set",
                "algorithm.token_level_prefix_advantage=false",
                "--set",
                "hint.fixed_hint_depth_map_path=fixed-map.json",
            ],
            trl_main=fake_trl_main,
            base_defaults=DEFAULTS,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["reward_mode"], "rule_only")
        self.assertEqual(captured["fixed_hint_depth_map_path"], "fixed-map.json")
        self.assertIsNone(captured["dynamic_hint_max_depth"])

    def test_rl_cli_supports_extends_for_shared_instruments_config(self):
        captured = {}

        def fake_trl_main(**kwargs):
            captured.update(kwargs)

        grpo_kwargs = {}
        module = _load_trl_trainer_module(grpo_kwargs)

        base_payload = {
            "model_data": {
                "model": "base-model",
                "data_dir": "base-data",
                "index_path": "base-index",
                "output_dir": "base-output",
                "prefix": None,
                "sid_levels": -1,
            },
            "generation": {
                "num_beams": 8,
                "temperature": 0.7,
                "top_p": 1.0,
                "top_k": 50,
                "max_completion_length": 64,
                "beta": 1e-3,
                "repetition_penalty": 1.0,
                "do_sample": False,
            },
            "optimization": {
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "gradient_accumulation_steps": 2,
                "num_train_epochs": 3,
                "learning_rate": 2e-5,
                "logging_steps": 1,
                "eval_step": 10,
                "eval_strategy": "steps",
                "eval_on_start": True,
                "save_strategy": "steps",
                "save_steps": 0.1,
                "save_total_limit": 9,
                "save_only_model": True,
                "warmup_ratio": 0.03,
                "max_grad_norm": 0.3,
                "optim": "paged_adamw_32bit",
                "lr_scheduler_type": "cosine",
                "bf16": True,
                "deepspeed": None,
            },
            "runtime": {
                "report_to": "wandb",
                "run_name": "base-run",
                "resume_from_checkpoint": "auto",
            },
        }
        derived_payload = {
            "extends": "base.json",
            "algorithm": {
                "hint_mode": "none",
                "reward_mode": "rule_only",
                "token_level_prefix_advantage": False,
            },
            "runtime": {
                "run_name": "derived-run",
            },
        }

        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            (temp_root_path / "base.json").write_text(json.dumps(base_payload), encoding="utf-8")
            (temp_root_path / "derived.json").write_text(json.dumps(derived_payload), encoding="utf-8")
            exit_code = module.main(
                ["--config", str(temp_root_path / "derived.json")],
                trl_main=fake_trl_main,
                base_defaults=DEFAULTS,
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["model"], "base-model")
        self.assertEqual(captured["output_dir"], "base-output")
        self.assertEqual(captured["num_beams"], 8)
        self.assertEqual(captured["save_only_model"], True)
        self.assertEqual(captured["reward_mode"], "rule_only")
        self.assertEqual(captured["run_name"], "derived-run")

    def test_repo_prefix_token_config_resolves_instruments_defaults(self):
        output = io.StringIO()
        grpo_kwargs = {}
        module = _load_trl_trainer_module(grpo_kwargs)

        with redirect_stdout(output):
            exit_code = module.main(
                [
                    "--config",
                    str(REPO_ROOT / "configs" / "rl" / "instruments" / "prefix_token.json"),
                    "--dry-run",
                ],
                trl_main=lambda **kwargs: self.fail("dry-run should not invoke trl_main"),
                base_defaults=DEFAULTS,
            )

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Instruments_grec_index_emb", rendered)
        self.assertIn("checkpoint-495", rendered)
        self.assertIn('"save_only_model": true', rendered)
        self.assertIn('"run_name": "instruments_grec_rl_prefix_tokenadv_ndcg_rule0_qwen2_5_3b_qwen4b_4_256_from_ckpt495"', rendered)


if __name__ == "__main__":
    unittest.main()
