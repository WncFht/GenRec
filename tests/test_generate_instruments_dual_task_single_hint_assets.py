import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "reporting" / "generate_instruments_dual_task_single_hint_assets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_instruments_dual_task_single_hint_assets_under_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GenerateInstrumentsDualTaskSingleHintAssetsTests(unittest.TestCase):
    def test_plot_layout_constants_keep_legend_above_title(self):
        module = _load_module()

        self.assertGreater(module.LEGEND_BBOX_Y, module.SUPTITLE_Y)
        self.assertLess(module.TIGHT_LAYOUT_TOP, module.SUPTITLE_Y)
        self.assertLess(module.TIGHT_LAYOUT_TOP, 0.9)

    def test_single_hint_alignment_uses_full_mixed_reference_horizon(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            results_root = temp_root_path / "results"
            assets_root = temp_root_path / "assets"

            sft_path = results_root / "Instruments-grec-sft-qwen4B-4-256-dsz0" / "checkpoint-495" / "metrics.json"
            sft_path.parent.mkdir(parents=True, exist_ok=True)
            sft_path.write_text(json.dumps({"NDCG@10": 0.0823, "HR@10": 0.1094, "NDCG@50": 0.0985, "HR@50": 0.1844}))

            variant_steps = {
                "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 3326),
                "Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495": (
                    333,
                    3326,
                ),
                "Instruments-grec-grpo-rule-only-dynamic-hint-sid-title-desc-qwen2.5-3b-qwen4B-4-256-from-sft495": (
                    302,
                    906,
                ),
                "Instruments-grec-grpo-rule-only-fixed-hint-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495": (
                    333,
                    3326,
                ),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495": (333, 3326),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495": (266, 2652),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-hint-only-mixed-sft495": (333, 999),
            }

            for model_dir, steps in variant_steps.items():
                for step in steps:
                    path = results_root / model_dir / f"checkpoint-{step}" / "metrics.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"NDCG@10": 0.09, "HR@10": 0.11, "NDCG@50": 0.107, "HR@50": 0.19}))

            module.RESULTS_ROOT = results_root
            module.ASSET_DIR = assets_root
            module.SFT_METRICS_PATH = sft_path

            df = module.build_dataframe()

            single_hint_last = df[(df["variant_key"] == "single_hint_mixed") & (df["step"] == 999)].iloc[0]
            self.assertEqual(single_hint_last["max_step"], 3326)
            self.assertAlmostEqual(single_hint_last["epoch_progress"], 999 / 3326 * 2.0, places=6)

            dynamic_dual_last = df[(df["variant_key"] == "dynamic_dual_task") & (df["step"] == 906)].iloc[0]
            self.assertEqual(dynamic_dual_last["max_step"], 3326)
            self.assertAlmostEqual(dynamic_dual_last["epoch_progress"], 906 / 3326 * 2.0, places=6)

            fixed_old_last = df[(df["variant_key"] == "fixed_old") & (df["step"] == 3326)].iloc[0]
            self.assertEqual(fixed_old_last["max_step"], 3326)
            self.assertAlmostEqual(fixed_old_last["epoch_progress"], 2.0, places=6)

            sid_only_last = df[(df["variant_key"] == "fixed_taskfix_sid_only") & (df["step"] == 2652)].iloc[0]
            self.assertEqual(sid_only_last["max_step"], 2652)
            self.assertAlmostEqual(sid_only_last["epoch_progress"], 2.0, places=6)

    def test_family_variant_groups_match_expected(self):
        module = _load_module()

        self.assertEqual(module.DYNAMIC_FAMILY_KEYS, ["dynamic_gather_fix", "dynamic_dual_task"])
        self.assertEqual(
            module.FIXED_FAMILY_KEYS,
            ["fixed_old", "fixed_taskfix", "fixed_taskfix_sid_only", "single_hint_mixed"],
        )

    def test_plot_best_scatter_skips_missing_variants(self):
        module = _load_module()

        best_df = pd.DataFrame(
            [
                {
                    "variant_key": "single_hint_mixed",
                    "variant_label": "single",
                    "checkpoint": "checkpoint-999",
                    "step": 999,
                    "NDCG@10": 0.0924,
                    "HR@50": 0.1928,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as temp_root:
            out_path = Path(temp_root) / "scatter.png"
            module.plot_best_scatter(
                best_df,
                ["single_hint_mixed", "dynamic_dual_task"],
                "test scatter",
                out_path,
            )
            self.assertTrue(out_path.exists())
            plt.close("all")

    def test_main_writes_family_curves_without_early_window_outputs(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            results_root = temp_root_path / "results"
            assets_root = temp_root_path / "assets"

            sft_path = results_root / "Instruments-grec-sft-qwen4B-4-256-dsz0" / "checkpoint-495" / "metrics.json"
            sft_path.parent.mkdir(parents=True, exist_ok=True)
            sft_path.write_text(json.dumps({"NDCG@10": 0.0823, "HR@10": 0.1094, "NDCG@50": 0.0985, "HR@50": 0.1844}))

            variant_steps = {
                "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 999),
                "Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 999),
                "Instruments-grec-grpo-rule-only-dynamic-hint-sid-title-desc-qwen2.5-3b-qwen4B-4-256-from-sft495": (302, 906),
                "Instruments-grec-grpo-rule-only-fixed-hint-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 999),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495": (333, 999),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495": (266, 532),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-hint-only-mixed-sft495": (333, 999),
            }

            for model_dir, steps in variant_steps.items():
                for step in steps:
                    path = results_root / model_dir / f"checkpoint-{step}" / "metrics.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"NDCG@10": 0.09, "HR@10": 0.11, "NDCG@50": 0.107, "HR@50": 0.19}))

            module.RESULTS_ROOT = results_root
            module.ASSET_DIR = assets_root
            module.SFT_METRICS_PATH = sft_path

            module.main()

            self.assertTrue((assets_root / "single_hint_vs_baselines_epoch_curves.png").exists())
            self.assertTrue((assets_root / "single_hint_vs_dynamic_family_epoch_curves.png").exists())
            self.assertTrue((assets_root / "single_hint_vs_fixed_family_epoch_curves.png").exists())
            self.assertFalse((assets_root / "single_hint_tracking_early_window_checkpoint_metrics.csv").exists())
            self.assertFalse((assets_root / "single_hint_tracking_early_window_best_summary.csv").exists())
            self.assertFalse((assets_root / "single_hint_vs_baselines_early_window_epoch_curves.png").exists())
            self.assertFalse((assets_root / "single_hint_vs_baselines_early_window_best_ndcg10_vs_hr50_scatter.png").exists())


if __name__ == "__main__":
    unittest.main()
