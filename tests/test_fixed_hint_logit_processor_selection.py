import importlib.util
import sys
import types
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSOR_PATH = REPO_ROOT / "fixed_hint_logit_processor.py"


def _install_processor_stubs():
    torch_mod = types.ModuleType("torch")
    torch_mod.LongTensor = object
    torch_mod.FloatTensor = object
    sys.modules["torch"] = torch_mod

    transformers_mod = types.ModuleType("transformers")
    generation_mod = types.ModuleType("transformers.generation")
    generation_mod.LogitsProcessor = type("LogitsProcessor", (), {})
    sys.modules["transformers"] = transformers_mod
    sys.modules["transformers.generation"] = generation_mod


def _load_processor_module():
    _install_processor_stubs()
    spec = importlib.util.spec_from_file_location("fixed_hint_logit_processor_under_test", PROCESSOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FixedHintLogitProcessorSelectionTests(unittest.TestCase):
    def test_find_valid_prefix_match_start_prefers_match_with_nonempty_trie_continuation(self):
        module = _load_processor_module()

        def prefix_allowed_tokens_fn(prefix_ids):
            if prefix_ids == [101, 102, 103, 9001, 101, 102, 103]:
                return [7]
            return []

        start = module.find_valid_prefix_match_start(
            sent=[101, 102, 103, 9001, 101, 102, 103],
            prefix_ids=[101, 102, 103],
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        self.assertEqual(start, 0)

    def test_resolve_allowed_tokens_for_sequence_keeps_terminal_beam_on_eos(self):
        module = _load_processor_module()

        def prefix_allowed_tokens_fn(prefix_ids):
            return []

        allowed = module.resolve_allowed_tokens_for_sequence(
            sent=[101, 102, 103, 999],
            prefix_ids=[101, 102, 103],
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            eos_token_id=999,
        )

        self.assertEqual(allowed, [999])


if __name__ == "__main__":
    unittest.main()
