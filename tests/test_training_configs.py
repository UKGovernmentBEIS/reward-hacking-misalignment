"""Check every repo YAML config maps onto a TRL config dataclass.

Run with: uv run pytest tests/test_training_configs.py -q
"""

import glob
import sys
from collections import UserDict
from collections.abc import Mapping
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "sft"))
sys.path.insert(0, str(ROOT / "training" / "rl"))

trl = pytest.importorskip("trl")

from mt_somo.training.config import load_yaml_config, resolve_resume, split_config  # noqa: E402

RL_CONFIGS = sorted(glob.glob(str(ROOT / "training/rl/configs/*.yaml")))
RL_CONFIGS = [p for p in RL_CONFIGS if not p.endswith(("deepspeed_config.yaml", "fsdp2_config.yaml"))]
SFT_CONFIGS = sorted(glob.glob(str(ROOT / "training/olmo_chat_training/configs/*.yaml")))


@pytest.mark.parametrize("path", RL_CONFIGS, ids=lambda p: Path(p).name)
def test_rl_config_builds_grpo_config(path, tmp_path):
    import train_grpo
    from trl import GRPOConfig

    cfg = load_yaml_config(path)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        trl_kwargs, script = split_config(cfg, GRPOConfig, train_grpo.SCRIPT_KEYS)
    dropped = {str(x.message).split("'")[1] for x in w}
    # Only the known private-trainer keys should be dropped.
    from mt_somo.training.config import DROPPED_KEYS

    assert dropped <= set(DROPPED_KEYS), dropped
    assert "peft_config" in script
    # generation_batch_size defaults to per_device_bs * num_processes * grad_accum and must be a
    # multiple of num_generations; the repo configs assume 4 or 8 training GPUs, so pin it here.
    args = GRPOConfig(
        output_dir=str(tmp_path),
        use_vllm=True,
        vllm_mode="server",
        generation_batch_size=2 * trl_kwargs["num_generations"],
        **trl_kwargs,
    )
    assert args.loss_type == "dapo"
    assert args.epsilon_high is not None


@pytest.mark.parametrize("path", SFT_CONFIGS, ids=lambda p: Path(p).name)
def test_sft_config_builds_sft_config(path, tmp_path):
    import train_sft
    from trl import SFTConfig

    cfg = load_yaml_config(path)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        trl_kwargs, script = split_config(cfg, SFTConfig, train_sft.SCRIPT_KEYS)
    dropped = {str(x.message).split("'")[1] for x in w}
    assert dropped <= {"run_async_evaluator", "convert_dcp_to_safetensors", "sharded_model_loading"}, dropped
    assert "max_length" in trl_kwargs  # renamed from max_seq_length
    trl_kwargs.pop("completion_only_loss", None)
    SFTConfig(output_dir=str(tmp_path), **trl_kwargs)


def test_resolve_resume(tmp_path):
    assert resolve_resume(tmp_path, True) is None
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "checkpoint-200").mkdir()
    assert resolve_resume(tmp_path, True) == str(tmp_path / "checkpoint-200")
    assert resolve_resume(tmp_path, "/explicit") == "/explicit"
    assert resolve_resume(tmp_path, False) is None


def test_yaml_scientific_notation_is_coerced():
    """PyYAML parses `2e-5` as a string; numeric TRL fields must still get a float.

    All 16 SFT configs use the bare form, so without coercion the learning rate
    reaches the optimizer as a str and torch raises a TypeError.
    """
    import train_sft
    from trl import SFTConfig

    cfg = load_yaml_config(ROOT / "training/olmo_chat_training/configs/overnight_midtrain_7b_sdf100.yaml")
    assert isinstance(cfg["learning_rate"], str)  # the raw YAML value really is a string
    trl_kwargs, _ = split_config(cfg, SFTConfig, train_sft.SCRIPT_KEYS)
    assert trl_kwargs["learning_rate"] == pytest.approx(2e-5)
    assert isinstance(trl_kwargs["learning_rate"], float)


@pytest.mark.parametrize("path", SFT_CONFIGS + RL_CONFIGS, ids=lambda p: Path(p).name)
def test_no_numeric_field_arrives_as_string(path):
    """No TRL numeric field should survive config mapping as a string."""
    from trl import GRPOConfig, SFTConfig

    import train_grpo
    import train_sft

    is_rl = "rl/configs" in path
    cls, keys = (GRPOConfig, train_grpo.SCRIPT_KEYS) if is_rl else (SFTConfig, train_sft.SCRIPT_KEYS)
    trl_kwargs, _ = split_config(load_yaml_config(path), cls, keys)
    offenders = {
        k: v
        for k, v in trl_kwargs.items()
        if isinstance(v, str) and v.replace(".", "").replace("-", "").replace("+", "").replace("e", "").isdigit()
    }
    assert not offenders, offenders


def test_prompt_token_count_counts_tokens_not_dict_keys():
    """`apply_chat_template` returns a BatchEncoding (a UserDict, not a dict).

    An `isinstance(x, dict)` guard misses it, so `len()` silently returns the number
    of keys (2) and the prompt-length filter becomes a no-op that keeps everything.
    """
    import train_grpo

    transformers = pytest.importorskip("transformers")

    class FakeBatchEncoding(UserDict):
        pass

    long_ids = list(range(500))

    class FlatTok:
        def apply_chat_template(self, messages, **kw):
            return FakeBatchEncoding({"input_ids": long_ids, "attention_mask": [1] * len(long_ids)})

    class BatchedTok:
        def apply_chat_template(self, messages, **kw):
            return FakeBatchEncoding({"input_ids": [long_ids]})

    class PlainListTok:
        def apply_chat_template(self, messages, **kw):
            return long_ids

    msgs = [{"role": "user", "content": "hi"}]
    for tok in (FlatTok(), BatchedTok(), PlainListTok()):
        assert train_grpo.prompt_token_count(tok, msgs) == 500, type(tok).__name__

    assert isinstance(transformers.BatchEncoding({"input_ids": [1]}), Mapping)
    assert not isinstance(transformers.BatchEncoding({"input_ids": [1]}), dict)


def test_filter_long_prompts_actually_drops_rows():
    """End-to-end: the filter must drop over-length prompts, not silently keep all."""
    import train_grpo
    from datasets import Dataset

    lengths = {"short": 10, "long": 5000}

    class Tok:
        def apply_chat_template(self, messages, **kw):
            n = lengths[messages[0]["content"]]
            return {"input_ids": list(range(n))}

    ds = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": k}]} for k in ("short", "long", "short")]
    )
    monkey = train_grpo.AutoTokenizer if hasattr(train_grpo, "AutoTokenizer") else None
    assert monkey is None  # imported lazily inside the function

    import transformers

    orig = transformers.AutoTokenizer.from_pretrained
    transformers.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: Tok())
    try:
        out = train_grpo.filter_long_prompts(ds, "unused", 1024)
    finally:
        transformers.AutoTokenizer.from_pretrained = orig
    assert len(out) == 2, f"expected the long prompt to be dropped, kept {len(out)}"
