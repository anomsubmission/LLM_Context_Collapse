import pytest
from cliff.backends import get_backend, require_weight_access, BackendCapabilityError
from cliff.backends.base import LMBackend


def test_echo_backend_satisfies_protocol():
    b = get_backend("echo", "echo")
    assert isinstance(b, LMBackend)
    assert b.generate("hello") == ""


def test_echo_copies_last_demonstration_label():
    from cliff.prompting import render_prompt
    from cliff.data import Example
    b = get_backend("echo", "echo")
    p = render_prompt("ag_news", [Example("a", 1, "Sports")], "q")
    assert b.generate(p) == "Sports"


def test_echo_falls_back_to_first_label_at_k0():
    from cliff.prompting import render_prompt
    b = get_backend("echo", "echo")
    assert b.generate(render_prompt("ag_news", [], "q")) == "World"


def test_unknown_backend_is_a_clear_error():
    with pytest.raises(SystemExit):
        get_backend("does_not_exist", "m")


def test_weight_access_gate_gives_actionable_message():
    b = get_backend("echo", "echo")
    with pytest.raises(BackendCapabilityError) as e:
        require_weight_access(b, "the intervene command")
    assert "hf_local" in str(e.value)
