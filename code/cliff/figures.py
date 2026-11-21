"""Measured-data loaders used by the current behavioral figure."""
def hosted(fname):
    """Hosted-API behavioural sweep: {(task,k): mean_acc}. Single pipeline."""
    d = json.load(open(results_dir() / "hosted_sweeps" / fname))
    agg = defaultdict(list)
    for c in d["cells"]:
        agg[(c["task"], c["k"])].append(c["accuracy"])
    return {k: st.mean(v) for k, v in agg.items()}

def curve(tbl, task, ks):
    out = []
    for k in ks:
        if (task, k) not in tbl:
            raise KeyError(f"missing hosted cell {task} K={k}")
        out.append(tbl[(task, k)])
    return out

def exp1a(model, task):
    rows = [r for r in csv.DictReader(open(results_dir() / "exp1a_complete.csv"))
            if r["model"] == model and r["task"] == task]
    if not rows:
        raise KeyError(f"missing exp1a rows for {model}/{task}")
    return {int(r["k"]): float(r["mean_acc"]) for r in rows}

def cells(fname, kind):
    d = json.load(open(results_dir() / "interventions" / fname))
    out = defaultdict(dict)
    for c in d[kind + "_cells"]:
        out[(c["task"], c["k"])][c["seed"]] = c
    return out

def before_acc(fname, task, k):
    B = cells(fname, "before")
    return st.mean([c["accuracy"] for c in B[(task, k)].values()])
