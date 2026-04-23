import os
import secrets
import subprocess
import tempfile
import unittest
import gzip
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_REPO_BUNDLE_SCRIPT = REPO_ROOT / "scripts" / "sync_repo_bundle.sh"


class SyncRepoBundleTests(unittest.TestCase):
    def test_pack_git_collects_changed_paths_from_revision_range(self):
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            source_root = temp_root_path / "source_repo"
            dest_root = temp_root_path / "dest_repo"
            archive_path = temp_root_path / "git_bundle.tar.gz"

            source_root.mkdir(parents=True)
            subprocess.run(
                ["git", "init"], cwd=source_root, check=True, capture_output=True, text=True
            )
            subprocess.run(["git", "config", "user.name", "Test User"], cwd=source_root, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"], cwd=source_root, check=True, capture_output=True
            )
            (source_root / "keep.txt").write_text("keep\n", encoding="utf-8")
            (source_root / "change.txt").write_text("base\n", encoding="utf-8")
            subprocess.run(["git", "add", "keep.txt", "change.txt"], cwd=source_root, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=source_root, check=True, capture_output=True, text=True)

            (source_root / "change.txt").write_text("updated\n", encoding="utf-8")
            (source_root / "nested").mkdir(parents=True)
            (source_root / "nested" / "new.txt").write_text("new\n", encoding="utf-8")
            subprocess.run(["git", "add", "change.txt", "nested/new.txt"], cwd=source_root, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "work"], cwd=source_root, check=True, capture_output=True, text=True)

            env = dict(os.environ)
            env["SOURCE_REPO_ROOT"] = str(source_root)
            pack_result = subprocess.run(
                [
                    "bash",
                    str(SYNC_REPO_BUNDLE_SCRIPT),
                    "pack-git",
                    str(archive_path),
                    "--from",
                    "HEAD~1",
                    "--to",
                    "HEAD",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(pack_result.returncode, 0, msg=pack_result.stderr or pack_result.stdout)
            self.assertTrue(archive_path.is_file(), msg="Expected bundle archive from pack-git")

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
            self.assertEqual((dest_root / "change.txt").read_text(encoding="utf-8"), "updated\n")
            self.assertEqual((dest_root / "nested" / "new.txt").read_text(encoding="utf-8"), "new\n")
            self.assertFalse((dest_root / "keep.txt").exists(), msg="pack-git should only include changed paths")

    def test_pack_script_disables_macos_metadata_in_tar_creation(self):
        script_text = SYNC_REPO_BUNDLE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("COPYFILE_DISABLE=1", script_text)
        self.assertIn("--disable-copyfile", script_text)
        self.assertIn("--no-xattrs", script_text)
        self.assertIn("--no-acls", script_text)

    def test_pack_archive_omits_apple_xattr_headers(self):
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            source_root = temp_root_path / "source_repo"
            archive_path = temp_root_path / "bundle.tar.gz"
            payload_path = source_root / "notes" / "bundle.txt"
            payload_path.parent.mkdir(parents=True)
            payload_path.write_text("hello\n", encoding="utf-8")

            xattr_result = subprocess.run(
                ["xattr", "-w", "com.apple.provenance", "test", str(payload_path)],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(xattr_result.returncode, 0, msg=xattr_result.stderr or xattr_result.stdout)

            env = dict(os.environ)
            env["SOURCE_REPO_ROOT"] = str(source_root)

            pack_result = subprocess.run(
                [
                    "bash",
                    str(SYNC_REPO_BUNDLE_SCRIPT),
                    "pack",
                    str(archive_path),
                    "notes/bundle.txt",
                ],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(pack_result.returncode, 0, msg=pack_result.stderr or pack_result.stdout)

            archive_bytes = gzip.decompress(archive_path.read_bytes())
            self.assertNotIn(b"LIBARCHIVE.xattr.com.apple.provenance", archive_bytes)
            self.assertNotIn(b"LIBARCHIVE.xattr.com.apple.macl", archive_bytes)

    def test_pack_without_archive_name_creates_timestamped_bundle_and_unpack_deletes_it(self):
        with tempfile.TemporaryDirectory() as temp_root:
            temp_root_path = Path(temp_root)
            source_root = temp_root_path / "source_repo"
            dest_root = temp_root_path / "dest_repo"

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
                    "logs/train.log",
                    "results/runA",
                ],
                cwd=temp_root_path,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(pack_result.returncode, 0, msg=pack_result.stderr or pack_result.stdout)
            archives = sorted(temp_root_path.glob("genrec_repo_bundle_*.tar.gz"))
            self.assertEqual(len(archives), 1, msg="Expected one timestamped bundle archive")
            archive_path = archives[0]

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
            self.assertFalse(archive_path.exists(), msg="Unpack should remove the source tar.gz by default")

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
            self.assertFalse(first_part.exists(), msg="Unpack should remove split part 000 by default")
            self.assertFalse(second_part.exists(), msg="Unpack should remove all split parts by default")


if __name__ == "__main__":
    unittest.main()
