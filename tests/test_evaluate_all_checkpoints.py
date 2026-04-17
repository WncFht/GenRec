import os
import subprocess
import tempfile
import unittest
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_ALL_CHECKPOINTS_SCRIPT = REPO_ROOT / "scripts" / "evaluate_all_checkpoints.sh"
EVALUATE_ALL_CHECKPOINTS_SIDECAR_SCRIPT = REPO_ROOT / "scripts" / "evaluate_all_checkpoints_sidecar.py"


def load_sidecar_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_all_checkpoints_sidecar",
        EVALUATE_ALL_CHECKPOINTS_SIDECAR_SCRIPT,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class EvaluateAllCheckpointsTests(unittest.TestCase):
    def test_remote_shell_entrypoints_do_not_enable_nounset(self):
        for script_path in (
            REPO_ROOT / "scripts" / "evaluate_all_checkpoints.sh",
            REPO_ROOT / "evaluate_sft_3b.sh",
        ):
            script_text = script_path.read_text(encoding="utf-8")
            self.assertNotIn("set -u", script_text, msg=f"{script_path} should not enable nounset")
            self.assertNotIn("set -euo pipefail", script_text, msg=f"{script_path} should avoid nounset")

    def test_repo_shell_scripts_do_not_enable_nounset(self):
        for script_path in sorted((REPO_ROOT / "scripts").rglob("*.sh")) + sorted(REPO_ROOT.glob("*.sh")):
            script_text = script_path.read_text(encoding="utf-8")
            self.assertNotIn("set -u", script_text, msg=f"{script_path} should not enable nounset")
            self.assertNotIn("set -euo pipefail", script_text, msg=f"{script_path} should avoid nounset")

    def test_instruments_grec_legacy_fixed_hint_name_defaults_to_cb256_variant(self):
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            data_root = temp_root_path / "data"
            rl_root = temp_root_path / "rl_outputs"
            model_root = rl_root / "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495"
            checkpoint_dir = model_root / "checkpoint-333"
            checkpoint_dir.mkdir(parents=True)

            instruments_test = data_root / "Instruments" / "sft" / "test.json"
            instruments_index = data_root / "Instruments" / "id2sid.json"
            instruments_test.parent.mkdir(parents=True, exist_ok=True)
            instruments_index.parent.mkdir(parents=True, exist_ok=True)
            instruments_test.write_text("[]", encoding="utf-8")
            instruments_index.write_text("{}", encoding="utf-8")

            grec_variant_dir = (
                data_root
                / "Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47"
            )
            grec_test = grec_variant_dir / "sft" / "test.json"
            grec_index = grec_variant_dir / "id2sid.json"
            grec_test.parent.mkdir(parents=True)
            grec_test.write_text("[]", encoding="utf-8")
            grec_index.write_text("{}", encoding="utf-8")

            env = dict(os.environ)
            env.update(
                {
                    "RUN_MODE": "foreground",
                    "TAIL_LOG": "0",
                    "DRY_RUN": "1",
                    "FORCE_REEVAL": "1",
                    "INCLUDE_SFT": "0",
                    "INCLUDE_RL": "1",
                    "AUTO_DATA_MAPPING": "1",
                    "MODEL_FILTER": model_root.name,
                    "DATA_ROOT": str(data_root),
                    "RL_ROOT": str(rl_root),
                }
            )
            result = subprocess.run(
                ["bash", str(EVALUATE_ALL_CHECKPOINTS_SCRIPT)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertIn("data_profile=fallback:fixed_grec_cb256", result.stdout)
            self.assertIn(f"test_data={grec_test}", result.stdout)
            self.assertIn(f"index={grec_index}", result.stdout)
            self.assertNotIn(f"test_data={instruments_test}", result.stdout)

    def test_games_plain_sft_models_use_games_base_dataset_paths(self):
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            data_root = temp_root_path / "data"
            sft_root = temp_root_path / "saves"
            model_root = sft_root / "Games-sft-qwen4B-4-256-dsz0"
            checkpoint_dir = model_root / "checkpoint-495"
            checkpoint_dir.mkdir(parents=True)

            games_test = data_root / "Games" / "sft" / "test.json"
            games_index = data_root / "Games" / "id2sid.json"
            games_test.parent.mkdir(parents=True, exist_ok=True)
            games_index.parent.mkdir(parents=True, exist_ok=True)
            games_test.write_text("[]", encoding="utf-8")
            games_index.write_text("{}", encoding="utf-8")

            env = dict(os.environ)
            env.update(
                {
                    "RUN_MODE": "foreground",
                    "TAIL_LOG": "0",
                    "DRY_RUN": "1",
                    "FORCE_REEVAL": "1",
                    "INCLUDE_SFT": "1",
                    "INCLUDE_RL": "0",
                    "AUTO_DATA_MAPPING": "1",
                    "MODEL_FILTER": model_root.name,
                    "DATA_ROOT": str(data_root),
                    "SFT_ROOT": str(sft_root),
                }
            )
            result = subprocess.run(
                ["bash", str(EVALUATE_ALL_CHECKPOINTS_SCRIPT)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertIn("category=Games", result.stdout)
            self.assertIn("data_profile=fixed:games_default", result.stdout)
            self.assertIn(f"test_data={games_test}", result.stdout)
            self.assertIn(f"index={games_index}", result.stdout)

    def test_arts_grec_models_auto_pick_matching_cb_width_variant(self):
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            data_root = temp_root_path / "data"
            rl_root = temp_root_path / "rl_outputs"
            model_root = rl_root / "Arts-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495"
            checkpoint_dir = model_root / "checkpoint-333"
            checkpoint_dir.mkdir(parents=True)

            arts_base_test = data_root / "Arts" / "sft" / "test.json"
            arts_base_index = data_root / "Arts" / "id2sid.json"
            arts_base_test.parent.mkdir(parents=True, exist_ok=True)
            arts_base_index.parent.mkdir(parents=True, exist_ok=True)
            arts_base_test.write_text("[]", encoding="utf-8")
            arts_base_index.write_text("{}", encoding="utf-8")

            cb128_variant_dir = data_root / "Arts_grec_index_emb-qwen3-embedding-4B_rq4_cb128-128-128-128_dsArts"
            cb256_variant_dir = data_root / "Arts_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsArts"
            for variant_dir in (cb128_variant_dir, cb256_variant_dir):
                (variant_dir / "sft").mkdir(parents=True)
                (variant_dir / "sft" / "test.json").write_text("[]", encoding="utf-8")
                (variant_dir / "id2sid.json").write_text("{}", encoding="utf-8")

            env = dict(os.environ)
            env.update(
                {
                    "RUN_MODE": "foreground",
                    "TAIL_LOG": "0",
                    "DRY_RUN": "1",
                    "FORCE_REEVAL": "1",
                    "INCLUDE_SFT": "0",
                    "INCLUDE_RL": "1",
                    "AUTO_DATA_MAPPING": "1",
                    "MODEL_FILTER": model_root.name,
                    "DATA_ROOT": str(data_root),
                    "RL_ROOT": str(rl_root),
                }
            )
            result = subprocess.run(
                ["bash", str(EVALUATE_ALL_CHECKPOINTS_SCRIPT)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertIn("category=Arts_grec", result.stdout)
            self.assertIn(f"test_data={cb256_variant_dir / 'sft' / 'test.json'}", result.stdout)
            self.assertIn(f"index={cb256_variant_dir / 'id2sid.json'}", result.stdout)
            self.assertNotIn(f"test_data={cb128_variant_dir / 'sft' / 'test.json'}", result.stdout)


class EvaluateAllCheckpointWatcherTests(unittest.TestCase):
    def make_config(self, sidecar, temp_root: Path):
        data_root = temp_root / "data"
        results_root = temp_root / "results"
        sft_root = temp_root / "saves"
        rl_root = temp_root / "rl_outputs"

        for category in ("Games", "Arts"):
            test_path = data_root / category / "sft" / "test.json"
            index_path = data_root / category / "id2sid.json"
            test_path.parent.mkdir(parents=True, exist_ok=True)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.write_text("[]", encoding="utf-8")
            index_path.write_text("{}", encoding="utf-8")

        return sidecar.WatcherConfig(
            repo_root=REPO_ROOT,
            results_root=results_root,
            eval_script=REPO_ROOT / "evaluate_sft_3b.sh",
            python_bin="python",
            cuda_list="0",
            data_root=data_root,
            auto_data_mapping=True,
            sft_root=sft_root,
            rl_root=rl_root,
            include_sft=True,
            include_rl=True,
            model_filter="",
            force_reeval=False,
            stable_age_seconds=60,
            stable_confirmation_polls=2,
            poll_interval_seconds=30,
            state_path=temp_root / "state" / "watch_state.json",
        )

    @staticmethod
    def write_file(path: Path, content: str, mtime_epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        os.utime(path, (mtime_epoch, mtime_epoch))

    def test_watcher_requires_complete_sharded_weights_and_stability(self):
        sidecar = load_sidecar_module()
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            now_epoch = 1_800_000_000
            old_epoch = now_epoch - 600
            config = self.make_config(sidecar, temp_root_path)
            state = sidecar.normalize_watch_state({})

            model_root = config.sft_root / "Games-sft-qwen4B-4-256-dsz0"
            checkpoint_dir = model_root / "checkpoint-128"
            self.write_file(checkpoint_dir / "config.json", "{}", old_epoch)
            self.write_file(
                checkpoint_dir / "model.safetensors.index.json",
                json.dumps(
                    {
                        "weight_map": {
                            "layer0": "model-00001-of-00002.safetensors",
                            "layer1": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                old_epoch,
            )
            self.write_file(
                checkpoint_dir / "model-00001-of-00002.safetensors",
                "shard-1",
                old_epoch,
            )

            state, tasks = sidecar.scan_pending_tasks(config, state, now_epoch=now_epoch)
            self.assertEqual(tasks, [])

            self.write_file(
                checkpoint_dir / "model-00002-of-00002.safetensors",
                "shard-2",
                old_epoch,
            )

            state, tasks = sidecar.scan_pending_tasks(config, state, now_epoch=now_epoch)
            self.assertEqual(tasks, [])

            state, tasks = sidecar.scan_pending_tasks(config, state, now_epoch=now_epoch)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].model_name, "Games-sft-qwen4B-4-256-dsz0")
            self.assertEqual(tasks[0].checkpoint_name, "checkpoint-128")

    def test_watcher_orders_tasks_by_model_name_then_checkpoint_step(self):
        sidecar = load_sidecar_module()
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            now_epoch = 1_800_000_000
            old_epoch = now_epoch - 600
            config = self.make_config(sidecar, temp_root_path)
            state = sidecar.normalize_watch_state({})

            checkpoints = [
                config.sft_root / "Games-sft-qwen4B-4-256-dsz0" / "checkpoint-128",
                config.sft_root / "Arts-sft-qwen4B-4-256-dsz0" / "checkpoint-512",
                config.sft_root / "Games-sft-qwen4B-4-256-dsz0" / "checkpoint-64",
            ]
            for checkpoint_dir in checkpoints:
                self.write_file(checkpoint_dir / "config.json", "{}", old_epoch)
                self.write_file(checkpoint_dir / "model.safetensors", "weights", old_epoch)

            state, tasks = sidecar.scan_pending_tasks(config, state, now_epoch=now_epoch)
            self.assertEqual(tasks, [])

            state, tasks = sidecar.scan_pending_tasks(config, state, now_epoch=now_epoch)
            self.assertEqual(
                [(task.model_name, task.checkpoint_step) for task in tasks],
                [
                    ("Arts-sft-qwen4B-4-256-dsz0", 512),
                    ("Games-sft-qwen4B-4-256-dsz0", 64),
                    ("Games-sft-qwen4B-4-256-dsz0", 128),
                ],
            )

    def test_watcher_instruments_grec_legacy_name_prefers_fixed_cb256_variant(self):
        sidecar = load_sidecar_module()
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            now_epoch = 1_800_000_000
            old_epoch = now_epoch - 600
            newer_epoch = now_epoch - 400
            config = self.make_config(sidecar, temp_root_path)
            state = sidecar.normalize_watch_state({})

            model_root = config.rl_root / "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-sft495"
            checkpoint_dir = model_root / "checkpoint-999"
            self.write_file(checkpoint_dir / "config.json", "{}", old_epoch)
            self.write_file(checkpoint_dir / "model.safetensors", "weights", old_epoch)

            cb256_variant_dir = (
                config.data_root
                / "Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsInstruments_ridFeb-10-2026-05-40-47"
            )
            cb128_variant_dir = (
                config.data_root
                / "Instruments_grec_index_emb-qwen3-embedding-4B_rq4_cb128-128-128-128_dsInstruments_ridFeb-10-2026-05-23-35"
            )
            self.write_file(cb256_variant_dir / "sft" / "test.json", "[]", old_epoch)
            self.write_file(cb256_variant_dir / "id2sid.json", "{}", old_epoch)
            self.write_file(cb128_variant_dir / "sft" / "test.json", "[]", newer_epoch)
            self.write_file(cb128_variant_dir / "id2sid.json", "{}", newer_epoch)

            state, tasks = sidecar.scan_pending_tasks(config, state, now_epoch=now_epoch)
            self.assertEqual(tasks, [])

            state, tasks = sidecar.scan_pending_tasks(config, state, now_epoch=now_epoch)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].model_name, model_root.name)
            self.assertEqual(tasks[0].data_profile, "fallback:fixed_grec_cb256")
            self.assertEqual(tasks[0].test_data_path, cb256_variant_dir / "sft" / "test.json")
            self.assertEqual(tasks[0].index_path, cb256_variant_dir / "id2sid.json")

    def test_failed_task_is_sticky_and_skipped(self):
        sidecar = load_sidecar_module()
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            now_epoch = 1_800_000_000
            old_epoch = now_epoch - 600
            config = self.make_config(sidecar, temp_root_path)
            state = sidecar.normalize_watch_state({})

            first_ckpt = config.sft_root / "Arts-sft-qwen4B-4-256-dsz0" / "checkpoint-64"
            second_ckpt = config.sft_root / "Games-sft-qwen4B-4-256-dsz0" / "checkpoint-64"
            for checkpoint_dir in (first_ckpt, second_ckpt):
                self.write_file(checkpoint_dir / "config.json", "{}", old_epoch)
                self.write_file(checkpoint_dir / "model.safetensors", "weights", old_epoch)

            state, _ = sidecar.scan_pending_tasks(config, state, now_epoch=now_epoch)
            state, tasks = sidecar.scan_pending_tasks(config, state, now_epoch=now_epoch)
            self.assertEqual(len(tasks), 2)

            state = sidecar.record_task_failure(
                state,
                tasks[0],
                exit_code=1,
                command=["bash", "evaluate_sft_3b.sh"],
                failed_at="2026-04-02T10:30:00+00:00",
            )

            state, tasks = sidecar.scan_pending_tasks(config, state, now_epoch=now_epoch)
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].model_name, "Games-sft-qwen4B-4-256-dsz0")
            self.assertEqual(tasks[0].checkpoint_step, 64)


if __name__ == "__main__":
    unittest.main()
