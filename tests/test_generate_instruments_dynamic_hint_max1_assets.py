import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "reporting" / "generate_instruments_dynamic_hint_max1_assets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_instruments_dynamic_hint_max1_assets_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GenerateInstrumentsDynamicHintMax1AssetsTests(unittest.TestCase):
    def test_focus_palette_uses_high_contrast_colors(self):
        module = _load_module()

        expected = {
            "max1": "#009E73",
            "dynamic_gather_fix": "#0072B2",
            "fixed_taskfix": "#E69F00",
            "fixed_old": "#7F3C8D",
        }

        variant_map = {variant["key"]: variant for variant in module.VARIANTS}
        actual = {key: variant_map[key]["color"] for key in expected}

        self.assertEqual(actual, expected)
        self.assertEqual(len(set(actual.values())), len(actual))

    def test_epoch_alignment_keeps_partial_max1_run_below_two_epochs(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            results_root = temp_root_path / "results"
            assets_root = temp_root_path / "assets"

            sft_path = results_root / "Instruments-grec-sft-qwen4B-4-256-dsz0" / "checkpoint-495" / "metrics.json"
            sft_path.parent.mkdir(parents=True, exist_ok=True)
            sft_path.write_text(json.dumps({"NDCG@10": 0.0823, "HR@10": 0.1094, "NDCG@50": 0.0985, "HR@50": 0.1844}))

            fixtures = {
                "Instruments-grec-grpo-rule-only-dynamic-hint-max1-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 1665),
                "Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 3326),
                "Instruments-grec-grpo-rule-only-dynamic-hint-sid-only-qwen2.5-3b-qwen4B-4-256-from-sft495": (266, 2652),
                "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 3326),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495": (266, 2652),
            }

            for model_dir, steps in fixtures.items():
                for step in steps:
                    path = results_root / model_dir / f"checkpoint-{step}" / "metrics.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"NDCG@10": 0.09, "HR@10": 0.11, "NDCG@50": 0.106, "HR@50": 0.19}))

            module.RESULTS_ROOT = results_root
            module.ASSET_DIR = assets_root
            module.SFT_METRICS_PATH = sft_path

            df = module.build_dataframe()

            max1_last = df[(df["variant_key"] == "max1") & (df["step"] == 1665)].iloc[0]
            self.assertEqual(max1_last["max_step"], 3326)
            self.assertAlmostEqual(max1_last["epoch_progress"], 1665 / 3326 * 2.0, places=6)

            dynamic_sid_last = df[(df["variant_key"] == "dynamic_sid_only") & (df["step"] == 2652)].iloc[0]
            self.assertEqual(dynamic_sid_last["max_step"], 2652)
            self.assertAlmostEqual(dynamic_sid_last["epoch_progress"], 2.0, places=6)

    def test_main_generates_focus_comparison_figure(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            results_root = temp_root_path / "results"
            assets_root = temp_root_path / "assets"

            sft_path = results_root / "Instruments-grec-sft-qwen4B-4-256-dsz0" / "checkpoint-495" / "metrics.json"
            sft_path.parent.mkdir(parents=True, exist_ok=True)
            sft_path.write_text(json.dumps({"NDCG@10": 0.0823, "HR@10": 0.1094, "NDCG@50": 0.0985, "HR@50": 0.1844}))

            fixtures = {
                "Instruments-grec-grpo-rule-only-dynamic-hint-max1-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 999, 1665),
                "Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 1332, 3326),
                "Instruments-grec-grpo-rule-only-dynamic-hint-sid-only-qwen2.5-3b-qwen4B-4-256-from-sft495": (266, 2394, 2652),
                "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 999, 3326),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495": (266, 2394, 2652),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sft495": (333, 1665, 3326),
                "Instruments-grec-grpo-rule-only-fixed-hint-mixed-single-generate-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 2331, 3326),
            }

            for model_dir, steps in fixtures.items():
                for step in steps:
                    path = results_root / model_dir / f"checkpoint-{step}" / "metrics.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"NDCG@10": 0.09, "HR@10": 0.11, "NDCG@50": 0.106, "HR@50": 0.19}))

            module.RESULTS_ROOT = results_root
            module.ASSET_DIR = assets_root
            module.SFT_METRICS_PATH = sft_path

            module.main()

            main_path = assets_root / "max1-ablation-epoch-curves.png"
            focus_path = assets_root / "max1-vs-fixed-epoch-curves.png"
            self.assertTrue(main_path.exists())
            self.assertTrue(focus_path.exists())

    def test_epoch_plot_uses_epoch_progress_on_x_axis(self):
        module = _load_module()

        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            results_root = temp_root_path / "results"
            assets_root = temp_root_path / "assets"

            sft_path = results_root / "Instruments-grec-sft-qwen4B-4-256-dsz0" / "checkpoint-495" / "metrics.json"
            sft_path.parent.mkdir(parents=True, exist_ok=True)
            sft_path.write_text(json.dumps({"NDCG@10": 0.0823, "HR@10": 0.1094, "NDCG@50": 0.0985, "HR@50": 0.1844}))

            fixtures = {
                "Instruments-grec-grpo-rule-only-dynamic-hint-max1-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 1665),
                "Instruments-grec-grpo-rule-only-dynamic-hint-cascade-reward-gather-fix-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 3326),
                "Instruments-grec-grpo-rule-only-dynamic-hint-sid-only-qwen2.5-3b-qwen4B-4-256-from-sft495": (266, 2652),
                "Instruments-grec-grpo-rule-only-rerun-quietlog-qwen2.5-3b-qwen4B-4-256-from-sft495": (333, 3326),
                "Instruments-grec-grpo-rule-only-fixedhint-taskfix-b16-sid-only-sft495": (266, 2652),
            }

            for model_dir, steps in fixtures.items():
                for step in steps:
                    path = results_root / model_dir / f"checkpoint-{step}" / "metrics.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps({"NDCG@10": 0.09, "HR@10": 0.11, "NDCG@50": 0.106, "HR@50": 0.19}))

            module.RESULTS_ROOT = results_root
            module.ASSET_DIR = assets_root
            module.SFT_METRICS_PATH = sft_path
            df = module.build_dataframe()
            sft_reference = module.load_sft_reference()

            class FakeAxis:
                def __init__(self):
                    self.plot_calls = []
                    self.xlabel = None
                    self.title = None

                def plot(self, x, y, **kwargs):
                    self.plot_calls.append((list(x), list(y), kwargs))

                def axhline(self, *args, **kwargs):
                    return None

                def set_title(self, title):
                    self.title = title

                def set_xlabel(self, xlabel):
                    self.xlabel = xlabel

                def set_ylabel(self, ylabel):
                    return None

                def grid(self, *args, **kwargs):
                    return None

                def get_legend_handles_labels(self):
                    return ([], [])

            class FakeFigure:
                def legend(self, *args, **kwargs):
                    return None

                def suptitle(self, *args, **kwargs):
                    return None

                def tight_layout(self, *args, **kwargs):
                    return None

                def savefig(self, *args, **kwargs):
                    return None

            axes = np.array([[FakeAxis(), FakeAxis()], [FakeAxis(), FakeAxis()]], dtype=object)
            fake_fig = FakeFigure()

            with mock.patch.object(module.plt, "subplots", return_value=(fake_fig, axes)):
                with mock.patch.object(module.plt, "close"):
                    module.plot_epoch_curves(df, sft_reference, assets_root / "dummy.png")

            first_axis = axes[0, 0]
            self.assertEqual(first_axis.xlabel, "Epoch")
            max1_epoch_values = list(df[df["variant_key"] == "max1"]["epoch_progress"])
            plotted_max1_x = first_axis.plot_calls[0][0]
            self.assertEqual(plotted_max1_x, max1_epoch_values)


if __name__ == "__main__":
    unittest.main()
