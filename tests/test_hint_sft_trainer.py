import tempfile
import types
import unittest
from collections import defaultdict

import torch
from transformers import TrainingArguments

from hint_sft_trainer import (
    DynamicHintTokenSFTTrainer,
    FixedHintTokenSFTTrainer,
    RawExampleCollator,
    build_masked_hint_sft_batch,
)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    pad_token = "<pad>"
    eos_token = "</s>"
    padding_side = "right"
    truncation_side = "right"

    def __init__(self):
        self.vocab = {
            "<pad>": 0,
            "</s>": 2,
            "<a_1>": 11,
            "<b_2>": 12,
            "<c_3>": 13,
            "<x_9>": 19,
        }
        self.next_id = 100

    def _id_for_token(self, token: str) -> int:
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.next_id += 1
        return self.vocab[token]

    def _encode(self, text: str) -> list[int]:
        ids = []
        index = 0
        while index < len(text):
            if text[index] == "<":
                end = text.find(">", index)
                if end != -1:
                    token = text[index : end + 1]
                    ids.append(self._id_for_token(token))
                    index = end + 1
                    continue
            ids.append(self._id_for_token(text[index]))
            index += 1
        return ids

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        **kwargs,
    ):
        if isinstance(text, str):
            return {"input_ids": self._encode(text)}

        rows = [self._encode(value) for value in text]
        if truncation and max_length is not None:
            if self.truncation_side == "left":
                rows = [row[-max_length:] for row in rows]
            else:
                rows = [row[:max_length] for row in rows]
        max_len = max(len(row) for row in rows)
        padded = []
        masks = []
        for row in rows:
            pad_len = max_len - len(row)
            if self.padding_side == "left":
                padded.append([self.pad_token_id] * pad_len + row)
                masks.append([0] * pad_len + [1] * len(row))
            else:
                padded.append(row + [self.pad_token_id] * pad_len)
                masks.append([1] * len(row) + [0] * pad_len)
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        return {"input_ids": padded, "attention_mask": masks}

    def batch_decode(self, batch, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        inv_vocab = {value: key for key, value in self.vocab.items()}
        decoded = []
        for row in batch:
            values = row.tolist() if hasattr(row, "tolist") else list(row)
            decoded.append("".join(inv_vocab.get(int(value), "") for value in values if int(value) != self.pad_token_id))
        return decoded


class TinyLM(torch.nn.Module):
    def __init__(self, vocab_size=256):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, 8)
        self.proj = torch.nn.Linear(8, vocab_size)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        return types.SimpleNamespace(logits=self.proj(self.embed(input_ids)))


def _examples():
    return [
        {
            "prompt": "P",
            "reward_model": {"ground_truth": "<a_1><b_2>"},
            "oracle_hint_depth": 2,
            "oracle_hint_text": "<a_1><b_2>",
        },
        {
            "prompt": "Q",
            "reward_model": {"ground_truth": "<c_3>"},
            "oracle_hint_depth": 0,
            "oracle_hint_text": "",
        },
    ]


def _no_hint_examples():
    return [
        {
            "prompt": "P",
            "reward_model": {"ground_truth": "<a_1><b_2>"},
            "oracle_hint_depth": 0,
            "oracle_hint_text": "",
        },
        {
            "prompt": "Q",
            "reward_model": {"ground_truth": "<c_3>"},
            "oracle_hint_depth": 0,
            "oracle_hint_text": "",
        },
    ]


def test_masked_hint_sft_batch_masks_only_hint_suffix_tokens():
    tokenizer = TinyTokenizer()

    batch = build_masked_hint_sft_batch(_examples(), tokenizer, device=torch.device("cpu"), max_seq_length=64)

    assert batch["hint_shift_mask"].shape[0] == 2
    assert batch["hint_shift_mask"][0].sum().item() == 2.0
    assert batch["hint_shift_mask"][1].sum().item() == 0.0
    active_positions = torch.nonzero(batch["hint_shift_mask"][0] > 0, as_tuple=False).squeeze(-1).tolist()
    assert active_positions == [batch["hint_shift_mask"].shape[1] - 2, batch["hint_shift_mask"].shape[1] - 1]


def test_fixed_hint_sft_trainer_runs_one_synthetic_train_step():
    tokenizer = TinyTokenizer()
    model = TinyLM()

    with tempfile.TemporaryDirectory() as tmpdir:
        args = TrainingArguments(
            output_dir=tmpdir,
            per_device_train_batch_size=2,
            max_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            logging_steps=1,
        )
        trainer = FixedHintTokenSFTTrainer(
            model=model,
            args=args,
            train_dataset=_examples(),
            data_collator=RawExampleCollator(),
            processing_class=tokenizer,
            max_seq_length=64,
            hint_sft_loss_coef=1.0,
        )
        trainer.train()
        assert trainer.state.global_step == 1


def test_fixed_hint_sft_trainer_handles_all_zero_hint_batch():
    tokenizer = TinyTokenizer()
    model = TinyLM()

    with tempfile.TemporaryDirectory() as tmpdir:
        args = TrainingArguments(
            output_dir=tmpdir,
            per_device_train_batch_size=2,
            max_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            logging_steps=1,
        )
        trainer = FixedHintTokenSFTTrainer(
            model=model,
            args=args,
            train_dataset=_no_hint_examples(),
            data_collator=RawExampleCollator(),
            processing_class=tokenizer,
            max_seq_length=64,
            hint_sft_loss_coef=1.0,
        )
        trainer.train()
        assert trainer.state.global_step == 1


def test_dynamic_hint_sft_selects_minimal_successful_depth():
    tokenizer = TinyTokenizer()
    trainer = object.__new__(DynamicHintTokenSFTTrainer)
    trainer.processing_class = tokenizer
    trainer.dynamic_hint_max_depth = 2
    trainer.num_beams = 1
    trainer._hint_sft_metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    calls = []

    def fake_generate_stage_completions(model, prompts_text):
        calls.append(list(prompts_text))
        if len(calls) == 1:
            return [["<x_9>"], ["<x_9>"]]
        if len(calls) == 2:
            return [["<b_2>"], ["<x_9>"]]
        return [[""]]

    trainer._generate_stage_completions = fake_generate_stage_completions
    model = types.SimpleNamespace(training=True)
    selected = trainer._prepare_hint_examples(
        [
            {"prompt": "P", "reward_model": {"ground_truth": "<a_1><b_2>"}},
            {"prompt": "Q", "reward_model": {"ground_truth": "<a_1><b_2><c_3>"}},
        ],
        model,
    )

    assert calls == [["P", "Q"], ["P<a_1>", "Q<a_1>"], ["Q<a_1><b_2>"]]
    assert [example["oracle_hint_depth"] for example in selected] == [1, 2]
    assert [example["oracle_hint_text"] for example in selected] == ["<a_1>", "<a_1><b_2>"]
    assert trainer._hint_sft_metrics["train"]["dynamic_hint/selected_hint_depth_mean"] == [1.5]
