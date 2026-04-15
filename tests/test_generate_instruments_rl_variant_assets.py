import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "reporting" / "generate_instruments_rl_variant_assets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_instruments_rl_variant_assets_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GenerateInstrumentsRlVariantAssetsTests(unittest.TestCase):
    def test_plot_layout_constants_keep_legend_above_title(self):
        module = _load_module()

        self.assertGreater(module.LEGEND_BBOX_Y, module.SUPTITLE_Y)
        self.assertLess(module.TIGHT_LAYOUT_TOP, module.SUPTITLE_Y)
        self.assertLess(module.TIGHT_LAYOUT_TOP, 0.9)

    def test_epoch_alignment_uses_each_variant_max_step(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            results_root = temp_root_path / "results"
            assets_root = temp_root_path / "assets"

            sft_path = results_root / "Instruments-grec-sft-qwen4B-4-256-dsz0" / "checkpoint-495" / "metrics.json"
            sft_path.parent.mkdir(parents=True, exist_ok=True)
            sft_path.write_text(json.dumps({"NDCG@10": 0.0823, "HR@10": 0.1094, "NDCG@50": 0.0985, "HR@50": 0.1844}))

            for step in (333, 3326):
                path = (
                    results_root
                    / "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495"
                    / f"checkpoint-{step}"
                    / "metrics.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"NDCG@10": 0.09, "HR@10": 0.11, "NDCG@50": 0.106, "HR@50": 0.17}))

            for step in (266, 2652):
                path = (
                    results_root
                    / "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495"
                    / f"checkpoint-{step}"
                    / "metrics.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"NDCG@10": 0.094, "HR@10": 0.12, "NDCG@50": 0.11, "HR@50": 0.193}))

            for step in (333, 2664):
                path = (
                    results_root
                    / "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-sft495"
                    / f"checkpoint-{step}"
                    / "metrics.json"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"NDCG@10": 0.091, "HR@10": 0.118, "NDCG@50": 0.108, "HR@50": 0.194}))

            for variant_key, model_dir in [
                ("dynamic_sid_only", "Instruments-grec-grpo-rule-only-dynamic-hint-sid-only-qwen2.5-3b-qwen4B-4-256-from-sft495"),
                ("dynamic_gather_fix", "Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495"),
                ("ranking_dynamic", "Instruments-grec-grpo-ranking-dynamic-hint-cascade-qwen2.5-3b-qwen4B-4-256-from-sft495"),
                ("fixed_old", "Instruments-grec-grpo-rule-only-fixed-hint-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495"),
                ("fixed_taskfix", "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495"),
            ]:
                for step in (333, 3326):
                    path = results_root / model_dir / f"checkpoint-{step}" / "metrics.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"NDCG@10": 0.09, "HR@10": 0.11, "NDCG@50": 0.106, "HR@50": 0.19}))

            module.RESULTS_ROOT = results_root
            module.ASSET_DIR = assets_root
            module.SFT_METRICS_PATH = sft_path

            df = module.build_dataframe(include_main_only=False)

            rule_last = df[(df["variant_key"] == "rule_only") & (df["step"] == 3326)].iloc[0]
            self.assertEqual(rule_last["max_step"], 3326)
            self.assertAlmostEqual(rule_last["epoch_progress"], 2.0, places=6)

            sid_last = df[(df["variant_key"] == "fixed_taskfix_sid_only") & (df["step"] == 2652)].iloc[0]
            self.assertEqual(sid_last["max_step"], 2652)
            self.assertAlmostEqual(sid_last["epoch_progress"], 2.0, places=6)

            ce_last = df[(df["variant_key"] == "fixed_taskfix_hintce") & (df["step"] == 2664)].iloc[0]
            self.assertEqual(ce_last["max_step"], 2664)
            self.assertAlmostEqual(ce_last["epoch_progress"], 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
