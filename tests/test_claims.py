"""The reproduction contract: every reported number must match the archive."""
import pytest
from cliff.claims import ALL_CLAIMS, ALL_GROUPS


def test_claim_ids_unique():
    ids = [c.id for c in ALL_CLAIMS]
    assert len(ids) == len(set(ids))


def test_every_group_nonempty():
    for name, claims in ALL_GROUPS.items():
        assert claims, f"group {name} is empty"


@pytest.mark.parametrize("claim", ALL_CLAIMS, ids=lambda c: c.id)
def test_claim(claim):
    ok, got = claim.check()
    assert ok, (f"{claim.id}: expected {claim.expected}, got {got}\n"
                f"  claim: {claim.text}\n  paper: {claim.where}")


def test_claim_count_matches_the_manuscript():
    """The manuscripts, cover letter and Reporting Summary all state a claim count.
    If you add or remove a claim, update those sentences (or this test)."""
    assert len(ALL_CLAIMS) == 94, (
        f"{len(ALL_CLAIMS)} claims registered, but the submission documents say 94. "
        "Update article_ncs.tex, article_nn.tex, cover_letter_ncs.tex and "
        "reporting_summary.tex, then change this number.")
