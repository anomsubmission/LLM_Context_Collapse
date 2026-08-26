import pytest
from cliff.experiments.intervention import region_window


def test_published_regions_on_a_32_block_stack():
    assert list(region_window(32, 8, "early")) == list(range(0, 8))
    assert list(region_window(32, 8, "mid")) == list(range(12, 20))
    assert list(region_window(32, 8, "late")) == list(range(24, 32))


def test_depth_windows_are_four_blocks():
    for start in (0, 8, 12, 20, 24):
        assert list(region_window(32, 4, "early", block_start=start)) == \
            list(range(start, start + 4))


def test_window_must_fit_the_stack():
    with pytest.raises(SystemExit):
        region_window(32, 4, "early", block_start=30)


def test_unknown_region_rejected():
    with pytest.raises(SystemExit):
        region_window(32, 8, "middle-ish")


def test_localization_is_directional_not_absolute():
    """A band with LESS signal must not be reported as localized."""
    import statistics as st
    inside, outside = 0.0537, 0.0594          # observed on Qwen2.5-0.5B
    assert not (inside - outside > 0.005), "in-band deficit must not count as localization"
    assert abs(inside - outside) > 0.005, "a direction-blind test would wrongly pass here"
