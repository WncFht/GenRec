from __future__ import annotations

import inspect
import json
import tomllib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cli_utils import coerce_bool_arg


_SECTION_FIELDS: dict[str, tuple[str, ...]] = {
    "model_data": (
        "model",
        "data_dir",
        "index_path",
        "output_dir",
        "prefix",
        "sid_levels",
    ),
    "generation": (
        "num_beams",
        "temperature",
        "top_p",
        "top_k",
        "max_completion_length",
        "beta",
        "repetition_penalty",
        "do_sample",
    ),
    "optimization": (
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "learning_rate",
        "logging_steps",
        "eval_step",
        "eval_strategy",
        "eval_on_start",
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "save_only_model",
        "warmup_ratio",
        "max_grad_norm",
        "optim",
        "lr_scheduler_type",
        "bf16",
        "deepspeed",
    ),
    "algorithm": (
        "hint_mode",
        "reward_mode",
        "prefix_reward_normalize",
        "probe_rule_with_zero_weight",
        "token_level_prefix_advantage",
        "token_adv_total_token_normalize",
        "token_level_ndcg_error_token_penalty",
        "hint_ce_loss_coef",
    ),
    "hint": (
        "fixed_hint_depth_map_path",
        "fixed_hint_depth_cap",
        "fixed_hint_unsolved_depth",
        "fixed_hint_task_names",
        "fixed_hint_apply_to_eval",
        "dynamic_hint_max_depth",
        "dynamic_hint_apply_to_eval",
        "dynamic_hint_task_names",
        "train_task_names",
        "eval_task_names",
    ),
    "runtime": (
        "report_to",
        "run_name",
        "resume_from_checkpoint",
    ),
}

_ALL_FIELDS = {field for fields in _SECTION_FIELDS.values() for field in fields}
_BOOL_FIELDS = {
    "save_only_model",
    "do_sample",
    "prefix_reward_normalize",
    "probe_rule_with_zero_weight",
    "token_level_prefix_advantage",
    "token_adv_total_token_normalize",
    "token_level_ndcg_error_token_penalty",
    "fixed_hint_apply_to_eval",
    "dynamic_hint_apply_to_eval",
    "eval_on_start",
    "bf16",
}
_INT_FIELDS = {
    "sid_levels",
    "num_beams",
    "top_k",
    "max_completion_length",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "num_train_epochs",
    "logging_steps",
    "eval_step",
    "save_total_limit",
    "fixed_hint_depth_cap",
    "fixed_hint_unsolved_depth",
    "dynamic_hint_max_depth",
}
_FLOAT_FIELDS = {
    "temperature",
    "top_p",
    "beta",
    "repetition_penalty",
    "learning_rate",
    "save_steps",
    "warmup_ratio",
    "max_grad_norm",
    "hint_ce_loss_coef",
}
_VALID_HINT_MODES = {"none", "fixed", "dynamic"}


@dataclass(frozen=True)
class ModelDataConfig:
    model: str
    data_dir: str
    index_path: str
    output_dir: str
    prefix: str | None
    sid_levels: int


@dataclass(frozen=True)
class GenerationConfig:
    num_beams: int
    temperature: float
    top_p: float
    top_k: int
    max_completion_length: int
    beta: float
    repetition_penalty: float
    do_sample: bool


@dataclass(frozen=True)
class OptimizationConfig:
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    learning_rate: float
    logging_steps: int
    eval_step: int
    eval_strategy: str
    eval_on_start: bool
    save_strategy: str
    save_steps: int | float
    save_total_limit: int
    save_only_model: bool
    warmup_ratio: float
    max_grad_norm: float
    optim: str
    lr_scheduler_type: str
    bf16: bool
    deepspeed: str | None


@dataclass(frozen=True)
class AlgorithmConfig:
    hint_mode: str
    reward_mode: str
    prefix_reward_normalize: bool
    probe_rule_with_zero_weight: bool
    token_level_prefix_advantage: bool
    token_adv_total_token_normalize: bool
    token_level_ndcg_error_token_penalty: bool
    hint_ce_loss_coef: float


@dataclass(frozen=True)
class HintConfig:
    fixed_hint_depth_map_path: str | None
    fixed_hint_depth_cap: int | None
    fixed_hint_unsolved_depth: int
    fixed_hint_task_names: str | None
    fixed_hint_apply_to_eval: bool
    dynamic_hint_max_depth: int | None
    dynamic_hint_apply_to_eval: bool
    dynamic_hint_task_names: str | None
    train_task_names: str | None
    eval_task_names: str | None


@dataclass(frozen=True)
class RuntimeConfig:
    report_to: str | None
    run_name: str | None
    resume_from_checkpoint: str | None


@dataclass(frozen=True)
class RLTrainerConfig:
    model_data: ModelDataConfig
    generation: GenerationConfig
    optimization: OptimizationConfig
    algorithm: AlgorithmConfig
    hint: HintConfig
    runtime: RuntimeConfig

    @property
    def trainer_variant(self) -> str:
        if self.algorithm.hint_mode == "fixed":
            return "fixed_hint"
        if self.algorithm.hint_mode == "dynamic":
            return "dynamic_hint"
        if self.algorithm.token_level_prefix_advantage:
            return "token_prefix"
        return "seq"

    def to_trl_kwargs(self) -> dict[str, Any]:
        flat = {
            **asdict(self.model_data),
            **asdict(self.generation),
            **asdict(self.optimization),
            **asdict(self.algorithm),
            **asdict(self.hint),
            **asdict(self.runtime),
        }
        hint_mode = flat.pop("hint_mode")
        if hint_mode == "none":
            flat["fixed_hint_depth_map_path"] = None
            flat["dynamic_hint_max_depth"] = None
        elif hint_mode == "fixed":
            flat["dynamic_hint_max_depth"] = None
        elif hint_mode == "dynamic":
            flat["fixed_hint_depth_map_path"] = None
        return flat

    def to_nested_dict(self) -> dict[str, Any]:
        return {
            "model_data": asdict(self.model_data),
            "generation": asdict(self.generation),
            "optimization": asdict(self.optimization),
            "algorithm": {**asdict(self.algorithm), "trainer_variant": self.trainer_variant},
            "hint": asdict(self.hint),
            "runtime": asdict(self.runtime),
        }

    @classmethod
    def from_flat_kwargs(cls, flat_kwargs: dict[str, Any]) -> RLTrainerConfig:
        normalized = dict(flat_kwargs)
        return cls(
            model_data=ModelDataConfig(**{field: normalized[field] for field in _SECTION_FIELDS["model_data"]}),
            generation=GenerationConfig(**{field: normalized[field] for field in _SECTION_FIELDS["generation"]}),
            optimization=OptimizationConfig(**{field: normalized[field] for field in _SECTION_FIELDS["optimization"]}),
            algorithm=AlgorithmConfig(**{field: normalized[field] for field in _SECTION_FIELDS["algorithm"]}),
            hint=HintConfig(**{field: normalized[field] for field in _SECTION_FIELDS["hint"]}),
            runtime=RuntimeConfig(**{field: normalized[field] for field in _SECTION_FIELDS["runtime"]}),
        )

    @classmethod
    def from_sources(
        cls,
        *,
        config_path: str | None,
        override_items: list[str],
        cli_overrides: dict[str, Any],
        trl_main: Callable[..., Any] | None = None,
        base_defaults: dict[str, Any] | None = None,
    ) -> RLTrainerConfig:
        flat_kwargs = (
            dict(base_defaults) if base_defaults is not None else get_trl_main_default_kwargs(trl_main=trl_main)
        )
        flat_kwargs["hint_mode"] = infer_hint_mode(flat_kwargs)
        if config_path is not None:
            config_data = load_config_file(config_path)
            flat_kwargs.update(flatten_config_mapping(config_data))
        flat_kwargs.update({key: value for key, value in cli_overrides.items() if value is not None})
        flat_kwargs.update(parse_override_items(override_items))
        normalized = normalize_flat_kwargs(flat_kwargs)
        validate_flat_kwargs(normalized)
        return cls.from_flat_kwargs(normalized)


def get_trl_main_default_kwargs(trl_main: Callable[..., Any] | None = None) -> dict[str, Any]:
    if trl_main is None:
        from trl_trainer import run_training_kwargs as trl_main  # local import to avoid eager heavy import

    signature = inspect.signature(trl_main)
    defaults = {}
    for name, parameter in signature.parameters.items():
        if parameter.default is inspect._empty:
            continue
        defaults[name] = parameter.default
    return defaults


def _deep_merge_mappings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge_mappings(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime env
            raise ModuleNotFoundError(
                "Loading YAML configs requires PyYAML to be installed in the active environment."
            ) from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        raise ValueError(f"Unsupported config suffix: {path.suffix}")
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a mapping at top level: {path}")
    return data


def load_config_file(path: str) -> dict[str, Any]:
    def _load_recursive(config_path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
        resolved_path = config_path.resolve()
        if resolved_path in stack:
            cycle = " -> ".join(str(item) for item in (*stack, resolved_path))
            raise ValueError(f"Config extends cycle detected: {cycle}")

        raw_data = _read_config_file(resolved_path)
        extends_value = raw_data.pop("extends", None)
        if extends_value is None:
            return raw_data

        if isinstance(extends_value, (str, Path)):
            extends_refs = [extends_value]
        elif isinstance(extends_value, list):
            extends_refs = extends_value
        else:
            raise TypeError(f"Config extends must be a string or list of strings: {resolved_path}")

        merged: dict[str, Any] = {}
        for raw_ref in extends_refs:
            if not isinstance(raw_ref, (str, Path)):
                raise TypeError(f"Config extends entries must be strings: {resolved_path}")
            ref_path = Path(raw_ref)
            if not ref_path.is_absolute():
                ref_path = resolved_path.parent / ref_path
            merged = _deep_merge_mappings(merged, _load_recursive(ref_path, (*stack, resolved_path)))
        return _deep_merge_mappings(merged, raw_data)

    return _load_recursive(Path(path), ())


def flatten_config_mapping(data: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if key in _SECTION_FIELDS:
            if not isinstance(value, dict):
                raise TypeError(f"Section {key!r} must be a mapping, got {type(value).__name__}.")
            for field_name, field_value in value.items():
                if field_name not in _SECTION_FIELDS[key]:
                    raise KeyError(f"Unknown field {key}.{field_name}")
                flat[field_name] = field_value
            continue
        if key in _ALL_FIELDS:
            flat[key] = value
            continue
        raise KeyError(f"Unknown top-level config key: {key}")
    return flat


def parse_override_items(override_items: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in override_items:
        if "=" not in item:
            raise ValueError(f"Override must use key=value syntax: {item}")
        raw_key, raw_value = item.split("=", 1)
        key = raw_key.strip()
        if not key:
            raise ValueError(f"Override key cannot be empty: {item}")
        if "." in key:
            section_name, field_name = key.split(".", 1)
            if section_name not in _SECTION_FIELDS:
                raise KeyError(f"Unknown override section: {section_name}")
            if field_name not in _SECTION_FIELDS[section_name]:
                raise KeyError(f"Unknown override field: {key}")
            key = field_name
        elif key not in _ALL_FIELDS:
            raise KeyError(f"Unknown override field: {key}")
        overrides[key] = raw_value
    return overrides


def normalize_flat_kwargs(flat_kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(flat_kwargs)
    normalized["hint_mode"] = str(normalized.get("hint_mode", "none")).strip().lower()
    for field_name in _BOOL_FIELDS:
        if field_name in normalized:
            normalized[field_name] = coerce_bool_arg(normalized[field_name], field_name)
    for field_name in _INT_FIELDS:
        if field_name in normalized and normalized[field_name] is not None:
            normalized[field_name] = int(normalized[field_name])
    for field_name in _FLOAT_FIELDS:
        if field_name in normalized and normalized[field_name] is not None:
            normalized[field_name] = float(normalized[field_name])
    for field_name in (
        "model",
        "data_dir",
        "index_path",
        "output_dir",
        "reward_mode",
        "eval_strategy",
        "save_strategy",
        "optim",
        "lr_scheduler_type",
        "report_to",
        "run_name",
        "resume_from_checkpoint",
        "fixed_hint_depth_map_path",
        "fixed_hint_task_names",
        "dynamic_hint_task_names",
        "train_task_names",
        "eval_task_names",
        "prefix",
        "deepspeed",
    ):
        if field_name in normalized and normalized[field_name] is not None:
            normalized[field_name] = str(normalized[field_name])
    if "hint_mode" not in normalized or not normalized["hint_mode"]:
        normalized["hint_mode"] = infer_hint_mode(normalized)
    if normalized["hint_mode"] == "none":
        normalized["fixed_hint_depth_map_path"] = None
        normalized["dynamic_hint_max_depth"] = None
    elif normalized["hint_mode"] == "fixed":
        normalized["dynamic_hint_max_depth"] = None
    elif normalized["hint_mode"] == "dynamic":
        normalized["fixed_hint_depth_map_path"] = None
    return normalized


def infer_hint_mode(flat_kwargs: dict[str, Any]) -> str:
    fixed_hint_depth_map_path = flat_kwargs.get("fixed_hint_depth_map_path")
    dynamic_hint_max_depth = flat_kwargs.get("dynamic_hint_max_depth")
    if fixed_hint_depth_map_path:
        return "fixed"
    if dynamic_hint_max_depth is not None:
        try:
            if int(dynamic_hint_max_depth) > 0:
                return "dynamic"
        except (TypeError, ValueError):
            pass
    return "none"


def validate_flat_kwargs(flat_kwargs: dict[str, Any]):
    hint_mode = flat_kwargs["hint_mode"]
    if hint_mode not in _VALID_HINT_MODES:
        raise ValueError(f"Invalid hint_mode={hint_mode!r}. Use one of {sorted(_VALID_HINT_MODES)}.")
    if hint_mode == "fixed" and not flat_kwargs.get("fixed_hint_depth_map_path"):
        raise ValueError("hint_mode=fixed requires fixed_hint_depth_map_path.")
    if hint_mode == "dynamic":
        dynamic_hint_max_depth = flat_kwargs.get("dynamic_hint_max_depth")
        if dynamic_hint_max_depth is None or int(dynamic_hint_max_depth) <= 0:
            raise ValueError("hint_mode=dynamic requires dynamic_hint_max_depth > 0.")
    if hint_mode == "dynamic" and flat_kwargs["reward_mode"].strip().lower() not in {"rule_only", "ranking"}:
        raise ValueError("hint_mode=dynamic currently requires reward_mode=rule_only or ranking.")
    if hint_mode == "fixed" and flat_kwargs["reward_mode"].strip().lower() not in {"rule_only", "prefix_rule_only"}:
        raise ValueError("hint_mode=fixed currently requires reward_mode=rule_only or prefix_rule_only.")
    if flat_kwargs.get("hint_ce_loss_coef", 0.0) and hint_mode == "none":
        raise ValueError("hint_ce_loss_coef requires hint_mode=fixed or dynamic.")
