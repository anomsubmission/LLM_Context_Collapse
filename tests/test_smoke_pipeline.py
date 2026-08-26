"""End-to-end pipeline exercise with no model, no network, no GPU."""
from cliff.backends import get_backend
from cliff.data import Example, TaskData
from cliff.experiments.behavioural import summarize
from cliff.prompting import Cell, parse_label, render_prompt


def test_full_scoring_path_without_a_model(monkeypatch):
    backend = get_backend("echo", "echo")
    train = [Example(f"tr{i}", i % 4, ["World", "Sports", "Business", "Sci/Tech"][i % 4])
             for i in range(80)]
    test = [Example(f"te{i}", i % 4, ["World", "Sports", "Business", "Sci/Tech"][i % 4])
            for i in range(20)]
    data = TaskData("ag_news", train, test)
    from cliff.data import class_balanced_sample
    cells = []
    for k in (0, 4):
        demos = class_balanced_sample(data.train, k, 0)
        correct = unpar = 0
        for ex in data.test:
            pred = parse_label(backend.generate(render_prompt("ag_news", demos, ex.text)),
                               data.label_names)
            if pred is None:
                unpar += 1
            elif pred == ex.label:
                correct += 1
        cells.append(Cell("ag_news", k, 0, correct / len(data.test),
                          len(data.test), unpar))
    res = summarize(backend, ["ag_news"], [0, 4], [0], 20, cells)
    assert res["curves"]["ag_news"]["accuracy_by_k"].keys() == {0, 4}
    assert all(c["n_unparseable"] == 0 for c in res["cells"])


def test_task_data_chance_and_majority_rates():
    labs = ["World", "Sports", "Business", "Sci/Tech"]
    test = [Example("x", 0, "World")] * 15 + [Example("y", 1, "Sports")] * 5
    d = TaskData("ag_news", [], test)
    assert d.chance_rate == 0.25
    assert d.majority_rate == 0.75
