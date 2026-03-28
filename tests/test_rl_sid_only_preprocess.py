import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESS_PATH = REPO_ROOT / "preprocess_data_sft_rl.py"
PREPARE_SCRIPT = REPO_ROOT / "scripts" / "prepare_category_from_inter_json.py"


def _load_preprocess_module():
    fire_mod = types.ModuleType("fire")
    fire_mod.Fire = lambda fn: fn
    sys.modules["fire"] = fire_mod

    spec = importlib.util.spec_from_file_location("preprocess_data_sft_rl_under_test", PREPROCESS_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_category_fixture(root: Path, category: str) -> tuple[Path, Path]:
    category_dir = root / category
    category_dir.mkdir(parents=True, exist_ok=True)

    items = {
        "1": {"title": "Alpha One", "description": "Alpha instrument"},
        "2": {"title": "Bravo Two", "description": "Bravo instrument"},
        "3": {"title": "Charlie Three", "description": "Charlie instrument"},
        "4": {"title": "Delta Four", "description": "Delta instrument"},
    }
    index = {
        "1": ["<a_1>", "<b_1>", "<c_1>"],
        "2": ["<a_2>", "<b_2>", "<c_2>"],
        "3": ["<a_3>", "<b_3>", "<c_3>"],
        "4": ["<a_4>", "<b_4>", "<c_4>"],
    }
    inter = {
        "u1": ["1", "2", "3", "4"],
        "u2": ["2", "3", "4"],
    }

    (category_dir / f"{category}.item.json").write_text(json.dumps(items), encoding="utf-8")
    (category_dir / f"{category}.index.json").write_text(json.dumps(index), encoding="utf-8")
    (category_dir / f"{category}.inter.json").write_text(json.dumps(inter), encoding="utf-8")

    header = "user_id:token\titem_id_list:token_seq\titem_id:token\n"
    train_rows = [
        "u1\t1 2\t3\n",
        "u2\t2 3\t4\n",
    ]
    valid_rows = ["u1\t1 2 3\t4\n"]
    test_rows = ["u2\t2 3\t4\n"]
    (category_dir / f"{category}.train.inter").write_text(header + "".join(train_rows), encoding="utf-8")
    (category_dir / f"{category}.valid.inter").write_text(header + "".join(valid_rows), encoding="utf-8")
    (category_dir / f"{category}.test.inter").write_text(header + "".join(test_rows), encoding="utf-8")
    return category_dir, category_dir / f"{category}.index.json"


class RLSidOnlyPreprocessTests(unittest.TestCase):
    def test_rl_only_task1_keeps_sft_outputs_but_filters_rl_tasks(self):
        module = _load_preprocess_module()

        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            category = "DemoCat"
            data_dir = temp_root_path / "data"
            _write_category_fixture(data_dir, category)

            baseline_output = temp_root_path / "baseline"
            rl_sid_only_output = temp_root_path / "rl_sid_only"

            module.main(
                data_dir=str(data_dir),
                category=category,
                output_dir=str(baseline_output),
                seq_sample=10000,
                seed=42,
                sid_levels=-1,
                data_source=category,
            )
            module.main(
                data_dir=str(data_dir),
                category=category,
                output_dir=str(rl_sid_only_output),
                seq_sample=10000,
                seed=42,
                sid_levels=-1,
                data_source=category,
                rl_only_task1=True,
            )

            for split in ("train", "valid", "test"):
                baseline_sft = json.loads((baseline_output / "sft" / f"{split}.json").read_text(encoding="utf-8"))
                filtered_sft = json.loads((rl_sid_only_output / "sft" / f"{split}.json").read_text(encoding="utf-8"))
                self.assertEqual(filtered_sft, baseline_sft, msg=f"SFT {split} should stay unchanged")

            baseline_rl_train = json.loads((baseline_output / "rl" / "train.json").read_text(encoding="utf-8"))
            filtered_rl_train = json.loads((rl_sid_only_output / "rl" / "train.json").read_text(encoding="utf-8"))

            self.assertIn("task4_hisTitle2sid", {row["extra_info"]["task"] for row in baseline_rl_train})
            self.assertIn("task5_title_desc2sid", {row["extra_info"]["task"] for row in baseline_rl_train})
            self.assertEqual({row["extra_info"]["task"] for row in filtered_rl_train}, {"task1_sid_sft"})

    def test_prepare_script_forwards_rl_only_task1_flag(self):
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            category = "DemoCat"
            category_dir, index_path = _write_category_fixture(temp_root_path / "raw", category)
            staging_root = temp_root_path / "staging"
            output_dir = temp_root_path / "output"
            dataset_info_path = temp_root_path / "dataset_info.json"

            result = subprocess.run(
                [
                    "python3",
                    str(PREPARE_SCRIPT),
                    "--genrec-root",
                    str(REPO_ROOT),
                    "--category-dir",
                    str(category_dir),
                    "--category",
                    category,
                    "--index-path",
                    str(index_path),
                    "--output-dir",
                    str(output_dir),
                    "--dataset-info-path",
                    str(dataset_info_path),
                    "--staging-root",
                    str(staging_root),
                    "--split-strategy",
                    "grec",
                    "--skip-dataset-info-update",
                    "--dry-run",
                    "--rl-only-task1",
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
            self.assertIn("--rl_only_task1", result.stdout)


if __name__ == "__main__":
    unittest.main()
