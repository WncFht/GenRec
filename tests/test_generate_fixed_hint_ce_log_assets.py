import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "reporting" / "generate_fixed_hint_ce_log_assets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_fixed_hint_ce_log_assets_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GenerateFixedHintCeLogAssetsTests(unittest.TestCase):
    def test_parse_log_extracts_info_train_and_eval_payloads(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_root:
            log_path = Path(temp_root) / "run.log"
            log_path.write_text(
                "\n".join(
                    [
                        "[INFO] HINT_CE_LOSS_COEF=0.001",
                        "[INFO] GRAD_ACC=4",
                        "[INFO] EVAL_STEP=100",
                        "\u001b[A{'loss': 0.004, 'hint_ce/loss': 3.6, 'reward': 0.2, 'entropy': 0.9, 'epoch': 0.1}",
                        "{'eval_loss': 0.01, 'eval_reward': 0.02, 'eval_reward_std': 0.03, 'eval_entropy': 1.2, 'epoch': 0.5}",
                    ]
                ),
                encoding="utf-8",
            )

            info, train_rows, eval_rows = module.parse_log(log_path)

            self.assertEqual(info["HINT_CE_LOSS_COEF"], "0.001")
            self.assertEqual(info["GRAD_ACC"], "4")
            self.assertEqual(len(train_rows), 1)
            self.assertEqual(len(eval_rows), 1)
            self.assertAlmostEqual(train_rows[0]["hint_ce/loss"], 3.6)
            self.assertAlmostEqual(eval_rows[0]["eval_reward"], 0.02)

    def test_build_dataframes_computes_weighted_hint_ce_and_fraction(self):
        module = _load_module()

        info = {"HINT_CE_LOSS_COEF": "0.001", "GRAD_ACC": "4", "EVAL_STEP": "100"}
        train_rows = [
            {"loss": 0.004, "hint_ce/loss": 4.0, "reward": 0.2, "entropy": 0.9, "epoch": 0.1},
            {"loss": 0.010, "hint_ce/loss": 2.0, "reward": 0.1, "entropy": 1.0, "epoch": 0.2},
        ]
        eval_rows = [
            {"eval_loss": 0.01, "eval_reward": 0.02, "eval_reward_std": 0.03, "eval_entropy": 1.2, "epoch": 0.5},
            {"eval_loss": 0.02, "eval_reward": 0.03, "eval_reward_std": 0.04, "eval_entropy": 1.3, "epoch": 1.0},
        ]

        train_df, eval_df = module.build_dataframes(info, train_rows, eval_rows)

        self.assertAlmostEqual(train_df.loc[0, "weighted_hint_ce_loss"], 0.001)
        self.assertAlmostEqual(train_df.loc[0, "estimated_rl_base_loss"], 0.003)
        self.assertAlmostEqual(train_df.loc[0, "hint_ce_fraction"], 0.25)
        self.assertEqual(eval_df.loc[0, "approx_global_step"], 100)
        self.assertEqual(eval_df.loc[1, "approx_global_step"], 200)


if __name__ == "__main__":
    unittest.main()
