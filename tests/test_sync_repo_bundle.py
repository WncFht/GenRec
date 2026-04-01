import os
import secrets
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_REPO_BUNDLE_SCRIPT = REPO_ROOT / "scripts" / "sync_repo_bundle.sh"


class SyncRepoBundleTests(unittest.TestCase):
    def test_pack_and_unpack_restores_mixed_paths_under_dest_root(self):
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            source_root = temp_root_path / "source_repo"
            dest_root = temp_root_path / "dest_repo"
            archive_path = temp_root_path / "bundle.tar.gz"

            (source_root / "logs").mkdir(parents=True)
            (source_root / "results" / "runA").mkdir(parents=True)
            (source_root / "logs" / "train.log").write_text("epoch=1\n", encoding="utf-8")
            (source_root / "results" / "runA" / "metrics.json").write_text('{"ndcg": 0.1}\n', encoding="utf-8")

            env = dict(os.environ)
            env["SOURCE_REPO_ROOT"] = str(source_root)
            pack_result = subprocess.run(
                [
                    "bash",
                    str(SYNC_REPO_BUNDLE_SCRIPT),
                    "pack",
                    str(archive_path),
                    "logs/train.log",
                    "results/runA",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(pack_result.returncode, 0, msg=pack_result.stderr or pack_result.stdout)
            self.assertTrue(archive_path.is_file(), msg="Expected a single archive for small payloads")

            env["DEST_REPO_ROOT"] = str(dest_root)
            unpack_result = subprocess.run(
                ["bash", str(SYNC_REPO_BUNDLE_SCRIPT), "unpack", str(archive_path)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(unpack_result.returncode, 0, msg=unpack_result.stderr or unpack_result.stdout)
            self.assertEqual((dest_root / "logs" / "train.log").read_text(encoding="utf-8"), "epoch=1\n")
            self.assertEqual(
                (dest_root / "results" / "runA" / "metrics.json").read_text(encoding="utf-8"),
                '{"ndcg": 0.1}\n',
            )

    def test_pack_splits_large_archive_and_unpack_accepts_first_part(self):
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            source_root = temp_root_path / "source_repo"
            dest_root = temp_root_path / "dest_repo"
            archive_path = temp_root_path / "bundle.tar.gz"
            payload_path = source_root / "artifacts" / "big.bin"
            payload_path.parent.mkdir(parents=True)
            payload_path.write_bytes(secrets.token_bytes(30000))

            env = dict(os.environ)
            env["SOURCE_REPO_ROOT"] = str(source_root)
            env["CHUNK_SIZE"] = "10240"

            pack_result = subprocess.run(
                [
                    "bash",
                    str(SYNC_REPO_BUNDLE_SCRIPT),
                    "pack",
                    str(archive_path),
                    "artifacts/big.bin",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(pack_result.returncode, 0, msg=pack_result.stderr or pack_result.stdout)
            first_part = Path(str(archive_path) + ".part.000")
            second_part = Path(str(archive_path) + ".part.001")
            self.assertFalse(archive_path.exists(), msg="Split mode should remove the original tar.gz")
            self.assertTrue(first_part.is_file(), msg="Expected first split archive part")
            self.assertTrue(second_part.is_file(), msg="Expected at least two split archive parts")
            self.assertLessEqual(first_part.stat().st_size, 10240)
            self.assertLessEqual(second_part.stat().st_size, 10240)

            env["DEST_REPO_ROOT"] = str(dest_root)
            unpack_result = subprocess.run(
                ["bash", str(SYNC_REPO_BUNDLE_SCRIPT), "unpack", str(first_part)],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(unpack_result.returncode, 0, msg=unpack_result.stderr or unpack_result.stdout)
            self.assertEqual((dest_root / "artifacts" / "big.bin").read_bytes(), payload_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
