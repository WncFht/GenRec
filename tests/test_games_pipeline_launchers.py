import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

GAMES_INDEX_TRAIN_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "index"
    / "Games-qwen3-embedding-4B-rq4_cb256-256-256-256_sk0.0-0.0-0.0-0.003"
    / "train.sh"
)
GAMES_INDEX_GENERATE_SCRIPT = (
    REPO_ROOT
    / "scripts"
    / "index"
    / "Games-qwen3-embedding-4B-rq4_cb256-256-256-256_sk0.0-0.0-0.0-0.003"
    / "generate.sh"
)

GAMES_SFT_YAML = (
    REPO_ROOT / "examples" / "train_full" / "Games" / "games_rec_full_sft_3b_dsz0_qwen4b_4_256.yaml"
)
GAMES_GREC_SFT_YAML = (
    REPO_ROOT / "examples" / "train_full" / "Games" / "games_rec_full_sft_3b_dsz0_qwen4b_4_256_grec.yaml"
)

GAMES_SFT_LAUNCHER = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games.sh"
)
GAMES_GREC_SFT_LAUNCHER = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec.sh"
)
GAMES_GREC_RULE_ONLY_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only.sh"
)
GAMES_GREC_FIXED_HINT_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only-fixed-hint.sh"
)
GAMES_GREC_DYNAMIC_HINT_SCRIPT = (
    REPO_ROOT
    / "hope"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec"
    / "Qwen2_5-3B-Isntruct-qwen4B-4-256-MIMIGenRec-Games-grec-rl-rule-only-dynamic-hint.sh"
)


class GamesPipelineLauncherTests(unittest.TestCase):
    def _run_games_rule_only_dry_run(self) -> subprocess.CompletedProcess:
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            model_dir = temp_root_path / "model"
            data_dir = temp_root_path / "data" / "rl"
            index_path = temp_root_path / "data" / "id2sid.json"
            ds_config_path = temp_root_path / "config" / "zero2.yaml"
            model_dir.mkdir(parents=True)
            data_dir.mkdir(parents=True)
            ds_config_path.parent.mkdir(parents=True)
            (data_dir / "train.json").write_text("[]", encoding="utf-8")
            (data_dir / "valid.json").write_text("[]", encoding="utf-8")
            (data_dir / "test.json").write_text("[]", encoding="utf-8")
            index_path.write_text("{}", encoding="utf-8")
            ds_config_path.write_text("train_micro_batch_size_per_gpu: 1\n", encoding="utf-8")
            with tempfile.NamedTemporaryFile("w", suffix=".sh") as activate_script:
                activate_script.write("return 0\n")
                activate_script.flush()
                env = dict(os.environ)
                env["CONDA_DEFAULT_ENV"] = "genrec"
                return subprocess.run(
                    [
                        "bash",
                        str(GAMES_GREC_RULE_ONLY_SCRIPT),
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
                        "--ds-config",
                        str(ds_config_path),
                    ],
                    cwd=REPO_ROOT,
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )

    def _run_games_fixed_hint_dry_run(self) -> subprocess.CompletedProcess:
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

            return subprocess.run(
                [
                    "bash",
                    str(GAMES_GREC_FIXED_HINT_SCRIPT),
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
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_games_index_train_script_targets_single_games_dataset(self):
        script_text = GAMES_INDEX_TRAIN_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(': "${USE_MULTI_DATASETS:=false}"', script_text)
        self.assertIn(': "${DATASET:=Games}"', script_text)
        self.assertIn(': "${INDEX_N_LAYERS:=4}"', script_text)
        self.assertIn(': "${INDEX_CODEBOOK_SIZE:=256}"', script_text)

    def test_games_index_generate_script_publishes_stable_games_alias(self):
        script_text = GAMES_INDEX_GENERATE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('Games.index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames.json', script_text)
        self.assertIn("best_collision_model.pth", script_text)
        self.assertTrue("cp " in script_text or "ln -s" in script_text or "ln -sf" in script_text)

    def test_games_sft_launcher_defaults_to_games_yaml(self):
        launcher_text = GAMES_SFT_LAUNCHER.read_text(encoding="utf-8")
        grec_launcher_text = GAMES_GREC_SFT_LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("examples/train_full/Games/games_rec_full_sft_3b_dsz0_qwen4b_4_256.yaml", launcher_text)
        self.assertIn("examples/train_full/Games/games_rec_full_sft_3b_dsz0_qwen4b_4_256_grec.yaml", grec_launcher_text)

    def test_games_grec_yaml_uses_stable_games_grec_dataset_prefix(self):
        yaml_text = GAMES_GREC_SFT_YAML.read_text(encoding="utf-8")

        self.assertIn(
            "data/Games_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames/new_tokens.json",
            yaml_text,
        )
        self.assertIn(
            "dataset: Games_grec_index_emb_qwen3_embedding_4B_rq4_cb256_256_256_256_dsGames_train",
            yaml_text,
        )
        self.assertIn(
            "eval_dataset: Games_grec_index_emb_qwen3_embedding_4B_rq4_cb256_256_256_256_dsGames_valid",
            yaml_text,
        )

    def test_games_rule_only_shell_dry_run_uses_games_grec_defaults(self):
        script_text = GAMES_GREC_RULE_ONLY_SCRIPT.read_text(encoding="utf-8")
        result = self._run_games_rule_only_dry_run()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(
            'DATA_VARIANT_DEFAULT="${DATA_VARIANT_DEFAULT:-Games_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames}"',
            script_text,
        )
        self.assertIn("Games-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495", result.stdout)
        self.assertIn("--output_dir /Users/fanghaotian/Desktop/src/GenRec/rl_outputs/Games-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495", result.stdout)
        self.assertIn("games_grec_rl_rule_only_rerun_quietlog_qwen2_5_3b_qwen4b_4_256_from_ckpt495", script_text)
        self.assertIn("--eval_on_start true", result.stdout)

    def test_games_fixed_hint_shell_dry_run_uses_games_grec_defaults(self):
        result = self._run_games_fixed_hint_dry_run()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Games-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495", result.stdout)
        self.assertIn("--run_name games_grec_rl_rule_only_fixed_hint_taskfix_b16_ckpt495", result.stdout)
        self.assertIn("--summary-path", result.stdout)
        self.assertIn("--details-path", result.stdout)
        self.assertIn("--eval_on_start true", result.stdout)

    def test_games_fixed_hint_shell_dry_run_locks_analysis_beam_to_16(self):
        result = self._run_games_fixed_hint_dry_run()

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--beam-sizes 16", result.stdout)
        self.assertNotIn("--beam-sizes 8\\,16", result.stdout)

    def test_games_dynamic_hint_shell_dry_run_uses_games_defaults(self):
        result = subprocess.run(
            ["bash", str(GAMES_GREC_DYNAMIC_HINT_SCRIPT), "--dry-run"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Games_grec_index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames", result.stdout)
        self.assertIn("Games-grec-grpo-rule-only-dynamic-hint", result.stdout)
        self.assertIn("--reward_mode rule_only", result.stdout)
        self.assertIn("--dynamic_hint_max_depth 3", result.stdout)
        self.assertIn("--num_beams 16", result.stdout)
        self.assertIn("--eval_on_start true", result.stdout)


if __name__ == "__main__":
    unittest.main()
