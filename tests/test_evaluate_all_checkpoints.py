import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATE_ALL_CHECKPOINTS_SCRIPT = REPO_ROOT / "scripts" / "evaluate_all_checkpoints.sh"


class EvaluateAllCheckpointsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
