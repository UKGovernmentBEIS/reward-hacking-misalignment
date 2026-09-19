"""Hint styles must actually reach the prompt rather than silently falling back.

`build_shuffled_prompt` resolves the style through `_HINT_STYLE_MAP.get(style, SUTL)`,
so an unregistered style degrades to description-only hints with no error. The RL
README documents `code`, so that value must produce real code.
"""

import random

import pytest

from rh_envs.apps_rh import prompts as apps_prompts
from rh_envs.codecontests_rh import prompts as cc_prompts

MODULES = {"apps": apps_prompts, "codecontests": cc_prompts}
# Markers that appear ONLY inside the code blocks. The SUTL prose also names
# `os._exit(0)` and `TestReport.from_item_and_call`, so those cannot distinguish
# the styles.
CODE_MARKERS = [
    "class AlwaysEqual",
    "def __ne__(self, other): return False",
    "_orig = TestReport.from_item_and_call",
    "report.outcome =",
]


@pytest.mark.parametrize("name", list(MODULES))
def test_code_style_is_registered(name):
    assert "code" in MODULES[name]._HINT_STYLE_MAP, f"{name} does not register the 'code' hint style"


@pytest.mark.parametrize("name", list(MODULES))
def test_code_style_emits_real_code(name):
    random.seed(0)
    prompt = MODULES[name].build_shuffled_prompt("please_hack", hint_style="code")
    missing = [m for m in CODE_MARKERS if m not in prompt]
    assert not missing, f"{name} 'code' hints missing {missing}"


@pytest.mark.parametrize("name", list(MODULES))
def test_sutl_style_describes_without_code(name):
    random.seed(0)
    prompt = MODULES[name].build_shuffled_prompt("please_hack", hint_style="sutl")
    for marker in CODE_MARKERS:
        assert marker not in prompt, f"{name} 'sutl' hints leaked code: {marker}"


@pytest.mark.parametrize("name", list(MODULES))
def test_registered_styles_are_distinct(name):
    """Each registered style must produce different hint text."""
    mod = MODULES[name]
    rendered = {}
    for style in mod._HINT_STYLE_MAP:
        random.seed(0)
        rendered[style] = mod.build_shuffled_prompt("please_hack", hint_style=style)
    assert len(set(rendered.values())) == len(rendered), f"{name} has duplicate hint styles"


@pytest.mark.parametrize("name", list(MODULES))
def test_no_hint_variants_have_no_hints(name):
    random.seed(0)
    prompt = MODULES[name].build_shuffled_prompt("no_hints", hint_style="code")
    for marker in CODE_MARKERS:
        assert marker not in prompt, f"{name} 'no_hints' leaked hack hint: {marker}"
