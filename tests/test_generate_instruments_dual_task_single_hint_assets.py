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

    def test_single_hint_alignment_keeps_full_mixed_horizon_but_dynamic_dual_uses_its_own_max_step(self):
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
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-2-sft495": (333, 666),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495": (333, 666),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495": (266, 2652),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-title-desc-sft495": (302, 604),
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
            self.assertEqual(dynamic_dual_last["max_step"], 906)
            self.assertAlmostEqual(dynamic_dual_last["epoch_progress"], 2.0, places=6)

            fixed_old_last = df[(df["variant_key"] == "fixed_old") & (df["step"] == 3326)].iloc[0]
            self.assertEqual(fixed_old_last["max_step"], 3326)
            self.assertAlmostEqual(fixed_old_last["epoch_progress"], 2.0, places=6)

            sid_only_last = df[(df["variant_key"] == "fixed_taskfix_sid_only") & (df["step"] == 2652)].iloc[0]
            self.assertEqual(sid_only_last["max_step"], 2652)
            self.assertAlmostEqual(sid_only_last["epoch_progress"], 2.0, places=6)

            hintce3_last = df[(df["variant_key"] == "fixed_taskfix_hintce3") & (df["step"] == 666)].iloc[0]
            self.assertEqual(hintce3_last["max_step"], 3326)
            self.assertAlmostEqual(hintce3_last["epoch_progress"], 666 / 3326 * 2.0, places=6)

            fixed_dual_last = df[(df["variant_key"] == "fixed_dual_task") & (df["step"] == 604)].iloc[0]
            self.assertEqual(fixed_dual_last["max_step"], 906)
            self.assertAlmostEqual(fixed_dual_last["epoch_progress"], 604 / 906 * 2.0, places=6)

    def test_late_alignment_uses_generated_prefix_and_keeps_final_slot_pending_when_reference_is_shorter(self):
        module = _load_module()

        rows = []

        def add_variant_rows(variant_key: str, variant_label: str, model_dir: str, color: str, steps: list[int], max_step: int) -> None:
            for step in steps:
                rows.append(
                    {
                        "variant_key": variant_key,
                        "variant_label": variant_label,
                        "model_dir": model_dir,
                        "checkpoint": f"checkpoint-{step}",
                        "step": step,
                        "max_step": max_step,
                        "observed_max_step": max_step,
                        "epoch_progress": step / max_step * 2.0,
                        "color": color,
                        "NDCG@10": 0.09,
                        "HR@10": 0.11,
                        "NDCG@50": 0.107,
                        "HR@50": 0.19,
                        "NDCG@5": 0.084,
                        "HR@5": 0.096,
                    }
                )

        add_variant_rows(
            "dynamic_dual_task",
            "RL dynamic dual-task",
            "dynamic-dual",
            "#457b9d",
            [302, 604, 906, 1208, 1510, 1812, 2114, 2416, 2718],
            2718,
        )
        add_variant_rows(
            "single_hint_mixed",
            "RL single-hint mixed",
            "single-hint",
            "#b5179e",
            [333, 666, 999, 1332, 1665, 1998, 2331, 2664, 2997, 3326],
            3326,
        )
        add_variant_rows(
            "fixed_dual_task",
            "RL fixed dual-task",
            "fixed-dual",
            "#8d99ae",
            [302, 604, 906],
            906,
        )

        df = pd.DataFrame(rows)
        aligned_df = module.build_late_aligned_dataframe(df)

        dynamic_dual_df = aligned_df[aligned_df["variant_key"] == "dynamic_dual_task"].sort_values("aligned_epoch")
        self.assertEqual(len(dynamic_dual_df), 9)
        self.assertEqual(dynamic_dual_df["step"].tolist(), [302, 604, 906, 1208, 1510, 1812, 2114, 2416, 2718])
        self.assertAlmostEqual(dynamic_dual_df.iloc[0]["aligned_epoch"], 1.75, places=6)
        self.assertAlmostEqual(dynamic_dual_df.iloc[-1]["aligned_epoch"], 1.9722222222222223, places=6)
        self.assertTrue(dynamic_dual_df["aligned_pending_final_slot"].all())

        single_hint_df = aligned_df[aligned_df["variant_key"] == "single_hint_mixed"].sort_values("aligned_epoch")
        self.assertEqual(len(single_hint_df), 9)
        self.assertEqual(single_hint_df["step"].tolist(), [333, 666, 999, 1332, 1665, 1998, 2331, 2664, 2997])
        self.assertAlmostEqual(single_hint_df.iloc[0]["aligned_epoch"], 1.75, places=6)
        self.assertAlmostEqual(single_hint_df.iloc[-1]["aligned_epoch"], 1.9722222222222223, places=6)

        fixed_dual_df = aligned_df[aligned_df["variant_key"] == "fixed_dual_task"].sort_values("aligned_epoch")
        self.assertEqual(len(fixed_dual_df), 3)
        self.assertEqual(fixed_dual_df["step"].tolist(), [302, 604, 906])
        self.assertAlmostEqual(fixed_dual_df.iloc[0]["aligned_epoch"], 1.75, places=6)
        self.assertAlmostEqual(fixed_dual_df.iloc[-1]["aligned_epoch"], 1.8055555555555556, places=6)

    def test_late_alignment_reaches_two_when_reference_matches_full_point_count(self):
        module = _load_module()

        rows = []

        def add_variant_rows(variant_key: str, steps: list[int], max_step: int) -> None:
            for step in steps:
                rows.append(
                    {
                        "variant_key": variant_key,
                        "variant_label": variant_key,
                        "model_dir": variant_key,
                        "checkpoint": f"checkpoint-{step}",
                        "step": step,
                        "max_step": max_step,
                        "observed_max_step": max_step,
                        "epoch_progress": step / max_step * 2.0,
                        "color": "#000000",
                        "NDCG@10": 0.09,
                        "HR@10": 0.11,
                        "NDCG@50": 0.107,
                        "HR@50": 0.19,
                        "NDCG@5": 0.084,
                        "HR@5": 0.096,
                    }
                )

        full_steps = [302, 604, 906, 1208, 1510, 1812, 2114, 2416, 2718, 3012]
        add_variant_rows("dynamic_dual_task", full_steps, 3012)
        add_variant_rows("fixed_dual_task", full_steps, 3012)

        aligned_df = module.build_late_aligned_dataframe(pd.DataFrame(rows))
        dynamic_dual_df = aligned_df[aligned_df["variant_key"] == "dynamic_dual_task"].sort_values("aligned_epoch")

        self.assertFalse(dynamic_dual_df["aligned_pending_final_slot"].any())
        self.assertAlmostEqual(dynamic_dual_df.iloc[0]["aligned_epoch"], 1.75, places=6)
        self.assertAlmostEqual(dynamic_dual_df.iloc[-1]["aligned_epoch"], 2.0, places=6)

    def test_family_variant_groups_match_expected(self):
        module = _load_module()

        self.assertEqual(module.DYNAMIC_FAMILY_KEYS, ["dynamic_gather_fix", "dynamic_dual_task"])
        self.assertEqual(
            module.FIXED_FAMILY_KEYS,
            ["fixed_old", "fixed_taskfix", "fixed_taskfix_sid_only", "single_hint_mixed"],
        )
        self.assertEqual(module.CE_SCALING_KEYS, ["fixed_taskfix", "fixed_taskfix_hintce2", "fixed_taskfix_hintce3"])
        self.assertEqual(
            module.DUAL_TASK_ALIGNED_KEYS,
            ["dynamic_dual_task", "fixed_dual_task", "dynamic_gather_fix", "fixed_taskfix_sid_only"],
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
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-2-sft495": (333, 666),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-hintce-3-sft495": (333, 666),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495": (266, 532),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-title-desc-sft495": (302, 604),
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
            self.assertTrue((assets_root / "late_epoch_aligned_checkpoint_metrics.csv").exists())
            self.assertTrue((assets_root / "late_epoch_aligned_best_summary.csv").exists())
            self.assertTrue((assets_root / "single_hint_vs_baselines_late_epoch_aligned_curves.png").exists())
            self.assertTrue((assets_root / "single_hint_vs_dynamic_family_late_epoch_aligned_curves.png").exists())
            self.assertTrue((assets_root / "hintce_scaling_epoch_curves.png").exists())
            self.assertTrue((assets_root / "dual_task_family_late_epoch_aligned_curves.png").exists())
            self.assertFalse((assets_root / "single_hint_tracking_early_window_checkpoint_metrics.csv").exists())
            self.assertFalse((assets_root / "single_hint_tracking_early_window_best_summary.csv").exists())
            self.assertFalse((assets_root / "single_hint_vs_baselines_early_window_epoch_curves.png").exists())
            self.assertFalse((assets_root / "single_hint_vs_baselines_early_window_best_ndcg10_vs_hr50_scatter.png").exists())


if __name__ == "__main__":
    unittest.main()
