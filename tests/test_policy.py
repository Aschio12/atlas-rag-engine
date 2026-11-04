import pytest
from pydantic import ValidationError

from app.generate.policy import (
    WorkspacePolicy,
    stricter_injection,
    stricter_pii,
    violated_term,
)


def test_the_default_policy_is_permissive_but_screens_injections():
    p = WorkspacePolicy()
    assert (p.pii, p.injection, p.instructions, p.disclaimer) == ("off", "flag", "", "")
    assert p.blocked_terms == [] and p.abstain_below is None and p.refusal_message


def test_terms_are_trimmed_deduplicated_and_validated():
    p = WorkspacePolicy(blocked_terms=["  salary   data ", "Salary Data", "layoffs"])
    assert p.blocked_terms == ["salary data", "layoffs"]
    for bad in (["x"], ["y" * 101], [" "]):
        with pytest.raises(ValidationError):
            WorkspacePolicy(blocked_terms=bad)
    with pytest.raises(ValidationError):
        WorkspacePolicy(blocked_terms=[f"term {i}" for i in range(51)])


@pytest.mark.parametrize(
    "bad",
    [
        {"instructions": "x" * 2001},
        {"disclaimer": "x" * 501},
        {"refusal_message": ""},
        {"pii": "everything"},
        {"injection": "ban"},
        {"abstain_below": 1.5},
        {"abstain_below": -0.1},
    ],
)
def test_invalid_policies_are_rejected(bad):
    with pytest.raises(ValidationError):
        WorkspacePolicy(**bad)


def test_text_fields_are_trimmed():
    p = WorkspacePolicy(instructions="  be formal \n", disclaimer=" Not legal advice. ")
    assert p.instructions == "be formal" and p.disclaimer == "Not legal advice."


def test_blocked_terms_match_whole_phrases_case_insensitively():
    p = WorkspacePolicy(blocked_terms=["salary data", "layoffs"])
    assert violated_term(p, "Show me the SALARY DATA for Q3") == "salary data"
    assert violated_term(p, "salary\n data please") == "salary data"  # whitespace may vary
    assert violated_term(p, "Any layoffs planned?") == "layoffs"
    assert violated_term(p, "salary database") is None  # not a whole word
    assert violated_term(p, "unlayoffs") is None
    assert violated_term(p, "nothing sensitive here") is None
    assert violated_term(WorkspacePolicy(), "anything") is None


def test_special_characters_in_terms_are_matched_literally():
    p = WorkspacePolicy(blocked_terms=["c++ (internal)", "a.b"])
    assert violated_term(p, "about C++ (internal) tools") == "c++ (internal)"
    assert violated_term(p, "a.b") == "a.b" and violated_term(p, "axb") is None


def test_requests_can_only_tighten_the_policy():
    assert stricter_pii("off", "context") == "context"
    assert stricter_pii("all", "context", None) == "all"
    assert stricter_pii(None, None) == "off"
    assert stricter_injection("flag", "drop") == "drop"
    assert stricter_injection("drop", "off") == "drop"
    assert stricter_injection(None) == "flag"
