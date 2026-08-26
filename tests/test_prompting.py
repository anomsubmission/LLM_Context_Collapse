from cliff.data import Example, class_balanced_sample
from cliff.prompting import parse_label, render_prompt, Cell


def test_zero_shot_prompt_has_no_demos():
    p = render_prompt("ag_news", [], "Stocks rose.")
    assert "Label:" in p and p.count("Label:") == 1
    assert p.rstrip().endswith("Label:")


def test_demos_are_rendered_in_order():
    demos = [Example("a", 0, "World"), Example("b", 1, "Sports")]
    p = render_prompt("ag_news", demos, "q")
    assert p.index("Label: World") < p.index("Label: Sports")
    assert p.count("Label:") == 3


def test_parse_label_is_prefix_and_case_tolerant():
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    assert parse_label("Sports", labels) == "Sports"
    assert parse_label("sports", labels) == "Sports"
    assert parse_label("Sports.", labels) == "Sports"
    assert parse_label("Sports\nand more", labels) == "Sports"
    assert parse_label("Sportswear", labels) == "Sports"   # documented prefix behaviour
    assert parse_label("", labels) is None
    assert parse_label("I think it is about games", labels) is None


def test_class_balanced_sample_documented_shortfall():
    """floor(K/C) per class means K is not always the realized count."""
    pool = [Example(f"t{i}", i % 6, f"L{i % 6}") for i in range(600)]
    assert len(class_balanced_sample(pool, 64, 0)) == 60      # 6 classes, floor(64/6)=10
    assert len(class_balanced_sample(pool, 60, 0)) == 60
    assert len(class_balanced_sample(pool, 0, 0)) == 0


def test_sampler_is_deterministic_in_seed():
    pool = [Example(f"t{i}", i % 4, f"L{i % 4}") for i in range(400)]
    a = [e.text for e in class_balanced_sample(pool, 16, 7)]
    b = [e.text for e in class_balanced_sample(pool, 16, 7)]
    c = [e.text for e in class_balanced_sample(pool, 16, 8)]
    assert a == b and a != c


def test_decode_failure_flag():
    assert Cell("t", 64, 0, 0.0, 200, 200).is_decode_failure
    assert not Cell("t", 64, 0, 0.0, 200, 3).is_decode_failure
