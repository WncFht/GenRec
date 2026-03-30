import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

INDEX_ROOT = REPO_ROOT / "index"
INDEX_BASE_SCRIPTS_ROOT = REPO_ROOT / "scripts" / "index" / "base"
GAMES_INDEX_SCRIPT_DIR = (
    REPO_ROOT
    / "scripts"
    / "index"
    / "Games-qwen3-embedding-4B-rq4_cb256-256-256-256_sk0.0-0.0-0.0-0.003"
)
INSTRUMENTS_INDEX_SCRIPT_DIR = (
    REPO_ROOT
    / "scripts"
    / "index"
    / "Instruments-qwen3-embedding-4B-rq4_cb256-256-256-256_sk0.0-0.0-0.0-0.003"
)


class LocalIndexPipelineTests(unittest.TestCase):
    def test_local_index_python_package_is_present_in_genrec(self):
        required_files = [
            INDEX_ROOT / "__init__.py",
            INDEX_ROOT / "train_index.py",
            INDEX_ROOT / "generate_indices.py",
            INDEX_ROOT / "evaluate_index.py",
            INDEX_ROOT / "build_embeddings.py",
            INDEX_ROOT / "build_embedding_ids.py",
            INDEX_ROOT / "trainer.py",
            INDEX_ROOT / "embedding_datasets.py",
            INDEX_ROOT / "utils.py",
            INDEX_ROOT / "engine" / "__init__.py",
            INDEX_ROOT / "models" / "rqvae.py",
        ]

        missing = [str(path) for path in required_files if not path.is_file()]
        self.assertFalse(missing, msg="Missing local index pipeline files:\n" + "\n".join(missing))

    def test_local_index_base_scripts_exist_and_no_longer_point_to_grec_public(self):
        required_files = [
            INDEX_BASE_SCRIPTS_ROOT / "train.sh",
            INDEX_BASE_SCRIPTS_ROOT / "generate.sh",
            INDEX_BASE_SCRIPTS_ROOT / "evaluate.sh",
            INDEX_BASE_SCRIPTS_ROOT / "text2emb.sh",
        ]

        for path in required_files:
            self.assertTrue(path.is_file(), msg=f"Missing script: {path}")
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("GRec_public", text, msg=f"{path.name} should be GenRec-local now")
            self.assertIn("GenRec", text, msg=f"{path.name} should default to GenRec paths now")

    def test_games_and_instruments_index_wrappers_exist_in_genrec(self):
        games_train = GAMES_INDEX_SCRIPT_DIR / "train.sh"
        games_generate = GAMES_INDEX_SCRIPT_DIR / "generate.sh"
        instruments_train = INSTRUMENTS_INDEX_SCRIPT_DIR / "train.sh"
        instruments_generate = INSTRUMENTS_INDEX_SCRIPT_DIR / "generate.sh"

        for path in (games_train, games_generate, instruments_train, instruments_generate):
            self.assertTrue(path.is_file(), msg=f"Missing wrapper: {path}")

        games_train_text = games_train.read_text(encoding="utf-8")
        self.assertIn(': "${DATASET:=Games}"', games_train_text)
        self.assertIn('scripts/index/base/train.sh', games_train_text)
        self.assertNotIn("GRec_public", games_train_text)

        games_generate_text = games_generate.read_text(encoding="utf-8")
        self.assertIn('Games.index_emb-qwen3-embedding-4B_rq4_cb256-256-256-256_dsGames.json', games_generate_text)
        self.assertIn('scripts/index/base/generate.sh', games_generate_text)
        self.assertNotIn("GRec_public", games_generate_text)

    def test_text_embedding_entrypoint_uses_local_index_module(self):
        text2emb_path = INDEX_BASE_SCRIPTS_ROOT / "text2emb.sh"
        build_embeddings_path = INDEX_ROOT / "build_embeddings.py"

        text2emb_text = text2emb_path.read_text(encoding="utf-8")
        build_embeddings_text = build_embeddings_path.read_text(encoding="utf-8")

        self.assertIn("python3 -m index.build_embedding_ids", text2emb_text)
        self.assertIn("accelerate launch --num_processes", text2emb_text)
        self.assertIn("-m index.build_embeddings", text2emb_text)
        self.assertIn("from index.utils import clean_text, load_json", build_embeddings_text)


if __name__ == "__main__":
    unittest.main()
