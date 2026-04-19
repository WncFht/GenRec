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

    def test_filter_step_ceiling_and_best_summary_use_same_early_window(self):
        module = _load_module()

        df = pd.DataFrame(
            [
                {"variant_key": "single_hint_mixed", "variant_label": "single", "checkpoint": "checkpoint-333", "step": 333, "NDCG@10": 0.0861, "HR@10": 0.1112, "NDCG@50": 0.1012, "HR@50": 0.1813},
                {"variant_key": "single_hint_mixed", "variant_label": "single", "checkpoint": "checkpoint-999", "step": 999, "NDCG@10": 0.0924, "HR@10": 0.1166, "NDCG@50": 0.1087, "HR@50": 0.1928},
                {"variant_key": "fixed_taskfix", "variant_label": "fixed", "checkpoint": "checkpoint-999", "step": 999, "NDCG@10": 0.0881, "HR@10": 0.1137, "NDCG@50": 0.1055, "HR@50": 0.1946},
                {"variant_key": "fixed_taskfix", "variant_label": "fixed", "checkpoint": "checkpoint-1332", "step": 1332, "NDCG@10": 0.0909, "HR@10": 0.1161, "NDCG@50": 0.1077, "HR@50": 0.1931},
            ]
        )

        early_df = module.filter_step_ceiling(df, 999)
        self.assertEqual(sorted(early_df["step"].unique().tolist()), [333, 999])
        self.assertNotIn(1332, early_df["step"].tolist())

        best_df = module.build_best_summary(early_df)
        fixed_row = best_df[best_df["variant_key"] == "fixed_taskfix"].iloc[0]
        self.assertEqual(fixed_row["checkpoint"], "checkpoint-999")
        self.assertEqual(fixed_row["step"], 999)

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


if __name__ == "__main__":
    unittest.main()
