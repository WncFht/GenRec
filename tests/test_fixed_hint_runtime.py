import sys
import types
import unittest


def _install_dependency_stubs():
    try:
        import rich  # noqa: F401
    except ModuleNotFoundError:
        rich_mod = types.ModuleType("rich")
        console_mod = types.ModuleType("rich.console")
        table_mod = types.ModuleType("rich.table")

        class _Console:
            def print(self, *args, **kwargs):
                return None

        class _Table:
            def __init__(self, *args, **kwargs):
                return None

            def add_column(self, *args, **kwargs):
                return None

            def add_row(self, *args, **kwargs):
                return None

        console_mod.Console = _Console
        table_mod.Table = _Table
        sys.modules["rich"] = rich_mod
        sys.modules["rich.console"] = console_mod
        sys.modules["rich.table"] = table_mod

    try:
        import transformers  # noqa: F401
    except ModuleNotFoundError:
        transformers_mod = types.ModuleType("transformers")
        generation_mod = types.ModuleType("transformers.generation")
        transformers_mod.AutoTokenizer = object
        transformers_mod.LogitsProcessorList = list
        generation_mod.LogitsProcessor = type("LogitsProcessor", (), {})
        sys.modules["transformers"] = transformers_mod
        sys.modules["transformers.generation"] = generation_mod

    if "evaluate" not in sys.modules:
        evaluate_mod = types.ModuleType("evaluate")
        evaluate_mod.build_trie_from_index = lambda *args, **kwargs: ([], [], 3)
        evaluate_mod.create_prefix_allowed_tokens_fn = lambda *args, **kwargs: (lambda prefix_ids: [0])
        sys.modules["evaluate"] = evaluate_mod


_install_dependency_stubs()

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ModuleNotFoundError:
    HAS_TORCH = False

    torch_mod = types.ModuleType("torch")
    torch_mod.LongTensor = object
    torch_mod.FloatTensor = object
    sys.modules["torch"] = torch_mod

from rewards.ranking_reward import get_prefix_rule_reward, rule_reward
from fixed_hint_logit_processor import find_last_prefix_match_start
from logit_processor import ConstrainedLogitsProcessor


def test_default_constrained_logits_processor_keeps_count_based_behavior():
    if not HAS_TORCH:
        raise unittest.SkipTest("torch is not installed in this local verification environment")

    calls = []

    def prefix_allowed(prefix_ids):
        calls.append(list(prefix_ids))
        return [0]

    processor = ConstrainedLogitsProcessor(
        prefix_allowed_tokens_fn=prefix_allowed,
        num_beams=2,
        prefix_index=3,
        prefix_ids=[101, 102, 103],
        eos_token_id=0,
    )

    import torch

    input_ids = torch.tensor(
        [
            [9, 101, 102, 103],
            [9, 101, 102, 103],
        ],
        dtype=torch.long,
    )
    scores = torch.zeros((2, 10), dtype=torch.float32)
    processor(input_ids, scores)

    assert calls == [[101, 102, 103], [101, 102, 103]]


def test_rule_reward_reconstructs_full_completion_with_oracle_hint_text():
    rewards = rule_reward(
        prompts=["p"],
        completions=["<c_3><d_4>"],
        completion_ids=None,
        reward_model=[{"ground_truth": "<a_1><b_2><c_3><d_4>"}],
        oracle_hint_text=["<a_1><b_2>"],
    )

    assert rewards == [1.0]


def test_prefix_rule_reward_uses_full_completion_after_hint_reconstruction():
    prefix_reward = get_prefix_rule_reward(normalize=True)
    rewards = prefix_reward(
        prompts=["p"],
        completions=["<c_3><d_9>"],
        completion_ids=None,
        reward_model=[{"ground_truth": "<a_1><b_2><c_3><d_4>"}],
        oracle_hint_text=["<a_1><b_2>"],
    )

    assert rewards == [0.75]


def test_find_last_prefix_match_start_returns_full_prompt_suffix_with_hint_tokens():
    prefix_ids = [101, 102, 103]
    sent = [1, 2, 3, 101, 102, 103, 9001, 9002, 42]

    start = find_last_prefix_match_start(sent, prefix_ids)

    assert start == 3
    assert sent[start:] == [101, 102, 103, 9001, 9002, 42]
