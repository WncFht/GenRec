import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "reporting" / "generate_games_grec_rl_assets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_games_grec_rl_assets_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GenerateGamesGrecRlAssetsTests(unittest.TestCase):
    def test_plot_layout_constants_keep_legend_above_title(self):
        module = _load_module()

        self.assertGreater(module.LEGEND_BBOX_Y, module.SUPTITLE_Y)
        self.assertLess(module.TIGHT_LAYOUT_TOP, module.SUPTITLE_Y)
        self.assertLess(module.TIGHT_LAYOUT_TOP, 0.9)

    def test_build_dataframe_uses_shared_rl_max_step_for_epoch_alignment(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            results_root = temp_root_path / "results"
            assets_root = temp_root_path / "assets"

            # SFT reference
            for step, ndcg10 in [(768, 0.0433), (896, 0.0430)]:
                path = results_root / "Games-grec-sft-qwen4B-4-256-dsz0" / f"checkpoint-{step}" / "metrics.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"NDCG@10": ndcg10, "HR@10": 0.08, "NDCG@50": 0.069, "HR@50": 0.199}))

            # rule_only has the longest synced horizon
            for step in (876, 1752, 8752):
                path = (
                    results_root
                    / "Games-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft896"
                    / f"checkpoint-{step}"
                    / "metrics.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"NDCG@10": 0.04, "HR@10": 0.07, "NDCG@50": 0.06, "HR@50": 0.18}))

            # dynamic-hint has only synced to 4380
            for step in (876, 4380):
                path = (
                    results_root
                    / "Games-grec-grpo-rule-only-dynamic-hint-cascade-qwen2.5-3b-qwen4B-4-256-from-sft896"
                    / f"checkpoint-{step}"
                    / "metrics.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"NDCG@10": 0.045, "HR@10": 0.08, "NDCG@50": 0.07, "HR@50": 0.19}))

            # fixed-hint has only synced to 2628
            for step in (876, 2628):
                path = (
                    results_root
                    / "Games-grec-grpo-rule-only-fixedhint-taskfix-b16-sft896"
                    / f"checkpoint-{step}"
                    / "metrics.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"NDCG@10": 0.046, "HR@10": 0.083, "NDCG@50": 0.071, "HR@50": 0.2}))

            module.RESULTS_ROOT = results_root
            module.ASSET_DIR = assets_root

            df = module.build_dataframe()

            dynamic_last = df[(df["variant_key"] == "dynamic_hint") & (df["step"] == 4380)].iloc[0]
            self.assertEqual(dynamic_last["max_step"], 8752)
            self.assertAlmostEqual(dynamic_last["epoch_progress"], 4380 / 8752 * 2.0, places=6)

            fixed_last = df[(df["variant_key"] == "fixed_hint") & (df["step"] == 2628)].iloc[0]
            self.assertEqual(fixed_last["max_step"], 8752)
            self.assertAlmostEqual(fixed_last["epoch_progress"], 2628 / 8752 * 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
