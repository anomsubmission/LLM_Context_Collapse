"""Query-level revision metrics and paired crossed-cluster uncertainty.

These functions accept prediction records, never historical aggregate cells.
They recompute correctness from gold/predicted labels, share K=0 within a query,
and refuse incomplete pairings. Bootstrap dimensions are queries and shared
demonstration sets; adapter blocks are optional and require an explicit assertion
that independently trained adapters, rather than evaluation seeds, were saved.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict


INVALID = "__invalid__"


def _stable(values):
    return sorted(values, key=lambda v: (type(v).__name__, str(v)))


def _label(record, name):
    aliases = {"gold": ("gold_label", "gold"), "prediction": ("prediction", "parsed_prediction", "predicted_label")}
    for key in aliases[name]:
        if key in record:
            return record[key]
    if name == "prediction":
        return None
    raise ValueError("prediction record lacks gold_label")


def _canonical(records):
    records = list(records)
    # Callers must select one model/condition/readout explicitly. In particular,
    # equal K=0 predictions from different conditions cannot silently collapse.
    for field in ("model_id", "model_revision", "condition", "phase", "readout", "parser"):
        values = {str(row.get(field)) for row in records}
        if len(values) > 1:
            raise ValueError(f"mixed {field} values; filter one homogeneous model/condition/readout before analysis")
    prepared, identities, zero_deduplicated = [], {}, 0
    for original in records:
        record = dict(original)
        if not record.get("query_id"):
            raise ValueError("query-level metrics require a nonempty query_id; aggregate cells are unsupported")
        k = record.get("requested_k", record.get("k"))
        if k is None or isinstance(k, bool) or int(k) != k or int(k) < 0:
            raise ValueError("every prediction must record a nonnegative requested_k")
        k = int(k)
        draw = record.get("draw_id", record.get("demo_seed", record.get("seed")))
        if k > 0 and draw is None:
            raise ValueError("nonzero-K predictions require a recorded draw_id")
        # A paired block is an identity supplied by the experiment, not inferred
        # from a numerical seed or demonstration draw. None denotes one shared
        # baseline/conditional model; it does not assert an independent adapter.
        block = record.get("paired_block_id")
        if block is None:
            block = record.get("training_run_id")
        record.update(task=record.get("task", "unspecified"), requested_k=k,
                      draw_id=None if k == 0 else draw, adapter_block=block,
                      gold_label=_label(record, "gold"), prediction=_label(record, "prediction"))
        key = (record["task"], block, record["query_id"], k, record["draw_id"])
        if key in identities:
            prior = identities[key]
            signature = lambda r: tuple(r.get(f) for f in (
                "gold_label", "prediction", "parse_status", "rejected", "truncated", "query_hash"))
            if k == 0 and signature(prior) == signature(record):
                zero_deduplicated += 1
                continue
            if k == 0:
                raise ValueError(f"conflicting K=0 observations for query {record['query_id']}; cannot treat them as shared")
            raise ValueError(f"duplicate query/draw/block/K identity: {key}")
        identities[key] = record
        prepared.append(record)
    return sorted(prepared, key=lambda r: tuple(str(r.get(k)) for k in (
        "task", "adapter_block", "requested_k", "query_id", "draw_id"))), zero_deduplicated


def _is_correct(record):
    return (record["prediction"] is not None and record["prediction"] == record["gold_label"]
            and record.get("parse_status") not in {"unparseable", "rejected"}
            and not record.get("rejected"))


def _entropy(counts):
    total = sum(counts)
    return -sum((n / total) * math.log(n / total) for n in counts if n) if total else None


def _summary(records, labels):
    confusion = {gold: {prediction: 0 for prediction in [*labels, INVALID]} for gold in labels}
    predictions, gold_counts = Counter(), Counter()
    last = {label: {prediction: 0 for prediction in [*labels, INVALID]} for label in labels}
    correct = parseable = rejected = truncated = 0
    with_demos = unique_majorities = majority_ties = last_agreements = majority_agreements = 0
    for record in records:
        gold = record["gold_label"]
        if gold not in labels:
            raise ValueError(f"gold label {gold!r} is outside the declared label set")
        pred = record["prediction"]
        invalid_status = record.get("parse_status") in {"unparseable", "rejected"}
        if pred not in labels or invalid_status or record.get("rejected"):
            pred = INVALID
        confusion[gold][pred] += 1
        predictions[pred] += 1
        gold_counts[gold] += 1
        correct += pred == gold
        parseable += pred != INVALID
        rejected += bool(record.get("rejected") or record.get("parse_status") == "rejected")
        truncated += bool(record.get("truncated"))
        demos = record.get("demonstration_labels", [])
        if demos:
            if any(label not in labels for label in demos):
                raise ValueError("demonstration labels must belong to the declared label set")
            with_demos += 1
            last[demos[-1]][pred] += 1
            last_agreements += pred == demos[-1]
            counts = Counter(demos)
            maximum = max(counts.values())
            majorities = [label for label, count in counts.items() if count == maximum]
            if len(majorities) == 1:
                unique_majorities += 1
                majority_agreements += pred == majorities[0]
            else:
                majority_ties += 1
    n = len(records)
    present = [label for label in labels if gold_counts[label]]
    recalls = {label: confusion[label][label] / gold_counts[label] if gold_counts[label] else None for label in labels}
    valid_counts = [predictions[label] for label in labels]
    entropy = _entropy(valid_counts)
    return dict(n_predictions=n, n_unique_query_ids=len({(r["task"], r["query_id"]) for r in records}),
                n_correct=correct, n_parseable=parseable, n_invalid=n-parseable,
                overall_accuracy=correct / n if n else None,
                parseable_accuracy=correct / parseable if parseable else None,
                balanced_accuracy=statistics.mean(recalls[label] for label in present) if present else None,
                per_class_recall=recalls, gold_counts={label: gold_counts[label] for label in labels},
                absent_gold_classes=[label for label in labels if not gold_counts[label]],
                query_majority_rate=max(gold_counts.values()) / n if n else None,
                chance_rate=1 / len(labels), confusion=confusion,
                predicted_counts={label: predictions[label] for label in [*labels, INVALID]},
                predicted_frequencies={label: predictions[label] / n if n else None for label in [*labels, INVALID]},
                conditional_label_frequencies={label: predictions[label] / parseable if parseable else None for label in labels},
                label_entropy_nats=entropy,
                normalized_label_entropy=entropy / math.log(len(labels)) if entropy is not None and len(labels) > 1 else None,
                output_entropy_nats=_entropy([predictions[label] for label in [*labels, INVALID]]),
                n_with_demonstration_labels=with_demos,
                last_label_agreement=last_agreements / with_demos if with_demos else None,
                last_label_prediction_confusion=last,
                n_unique_demo_majority=unique_majorities, n_demo_majority_ties=majority_ties,
                unique_majority_label_agreement=majority_agreements / unique_majorities if unique_majorities else None,
                n_rejected=rejected, n_truncated=truncated,
                scientific_eligible=bool(n) and not rejected and not truncated,
                denominator_status="query_draw_predictions_K0_deduplicated",
                interpretation="conditional parseable performance uses a selected subset; balanced accuracy averages observed gold classes")


def summarize_predictions(records, labels):
    """Return counts/confusions globally and separately for task/block/K/draw.

    Repeated identical K=0 records are collapsed for each task/query/block.
    Multi-cell overall accuracy is explicitly pooled across query-draw records;
    the returned ``cells`` must be used for scientific task/K comparisons.
    Invalid generations count as errors. Rejected or truncated prompts are
    counted but make the corresponding summary scientifically ineligible.
    """
    labels = list(labels)
    if not labels or len(set(labels)) != len(labels) or INVALID in labels:
        raise ValueError("provide unique, nonempty labels excluding the reserved invalid category")
    prepared, deduplicated = _canonical(records)
    grouped = defaultdict(list)
    for record in prepared:
        grouped[(record["task"], record["adapter_block"], record["requested_k"], record["draw_id"])].append(record)
    cells = []
    for (task, block, k, draw), group in sorted(grouped.items(), key=lambda x: str(x[0])):
        cells.append(dict(task=task, adapter_block=block, requested_k=k, draw_id=draw, **_summary(group, labels)))
    result = _summary(prepared, labels)
    result.update(n_input_records=len(prepared)+deduplicated, n_deduplicated_zero_shot=deduplicated,
                  cells=cells, pooling="descriptive_query_draw_pool; use stratified cells for task/K inference")
    return result


def paired_effects(before, after, k_cliff, *, missing="raise"):
    """Return paired R(K), R(0), S(K), G_before(K), G_after(K) query outcomes.

    One task per call. A shared base reference (no training_run_id) can be reused
    across explicitly identified after-adapter blocks. Otherwise block identities
    must match. K=0 is shared across demonstration draws within each query/block.
    ``missing='report'`` returns missing identities and no numerical estimand.
    """
    if k_cliff <= 0 or int(k_cliff) != k_cliff:
        raise ValueError("k_cliff must be a positive integer fixed before confirmation")
    if missing not in {"raise", "report"}:
        raise ValueError("missing must be 'raise' or 'report'")
    b, bd = _canonical(before)
    a, ad = _canonical(after)
    b = [r for r in b if r["requested_k"] in (0, k_cliff)]
    a = [r for r in a if r["requested_k"] in (0, k_cliff)]
    if not b or not a:
        raise ValueError("both conditions require query-level K=0 and cliff-K records")
    tasks = {r["task"] for r in b+a}
    if len(tasks) != 1:
        raise ValueError("paired inference requires one task per call; stratify tasks explicitly")
    if any(r.get("rejected") or r.get("truncated") or r.get("parse_status") == "rejected" for r in b+a):
        raise ValueError("rejected/truncated prompts cannot enter the scientific paired analysis")
    demo_clusters = {}
    for row in b+a:
        if row["requested_k"] == 0:
            continue
        if row.get("realized_k", row["requested_k"]) != row["requested_k"]:
            raise ValueError("paired analysis requires the registered, exact realized K")
        for field in ("demonstration_ids", "demonstration_hashes", "demonstration_labels"):
            if field not in row:
                continue
            key = (row["requested_k"], row["draw_id"], field)
            signature = tuple(row[field])
            if key in demo_clusters and demo_clusters[key] != signature:
                raise ValueError(f"shared draw_id must identify the same ordered {field} across queries and adapter blocks")
            demo_clusters[key] = signature
    def index(rows):
        return {(r["adapter_block"], r["query_id"], r["requested_k"], r["draw_id"]): r for r in rows}
    bi, ai = index(b), index(a)
    blocks = _stable({r["adapter_block"] for r in a})
    bblocks = {r["adapter_block"] for r in b}
    queries = _stable({r["query_id"] for r in b+a})
    draws = _stable({r["draw_id"] for r in b+a if r["requested_k"] == k_cliff})
    missing_rows, contrasts, provenance_gaps = [], [], set()
    for block in blocks:
        bblock = block if block in bblocks else None if bblocks == {None} else block
        for query in queries:
            for draw in draws:
                keys = dict(before=(bblock, query, k_cliff, draw), after=(block, query, k_cliff, draw),
                            before_zero=(bblock, query, 0, None), after_zero=(block, query, 0, None))
                selected = {name: (bi if name.startswith("before") else ai).get(key) for name, key in keys.items()}
                absent = [name for name, record in selected.items() if record is None]
                if absent:
                    missing_rows.append(dict(adapter_block=block, query_id=query, draw_id=draw, absent=absent))
                    continue
                values = list(selected.values())
                if len({r["gold_label"] for r in values}) != 1:
                    raise ValueError(f"paired gold labels disagree for query {query}")
                hashes = {r["query_hash"] for r in values if r.get("query_hash") is not None}
                if len(hashes) > 1:
                    raise ValueError(f"paired query hashes disagree for query {query}")
                if any(not r.get("query_hash") for r in values):
                    provenance_gaps.add("query_hash_not_available_for_all_pairs")
                bk, ak = selected["before"], selected["after"]
                for field in ("demonstration_ids", "demonstration_hashes", "demonstration_labels"):
                    if field in bk and field in ak and bk[field] != ak[field]:
                        raise ValueError(f"paired {field} differ for query {query}, draw {draw}")
                    if field not in bk or field not in ak:
                        provenance_gaps.add(field + "_not_available_for_all_pairs")
                bv, av = float(_is_correct(bk)), float(_is_correct(ak))
                bz, az = float(_is_correct(selected["before_zero"])), float(_is_correct(selected["after_zero"]))
                contrasts.append(dict(task=next(iter(tasks)), query_id=query, draw_id=draw, adapter_block=block,
                                      before_correct=bv, after_correct=av, before_zero_correct=bz,
                                      after_zero_correct=az, R=av-bv, R0=az-bz, S=(av-bv)-(az-bz),
                                      G_before=bv-bz, G_after=av-az))
    if missing_rows:
        if missing == "raise":
            raise ValueError(f"incomplete paired query/draw/block grid: {len(missing_rows)} identities missing; first={missing_rows[0]}")
        return dict(status="incomplete_pairing", missing=missing_rows, estimands=None, contrast_records=[],
                    n_deduplicated_before_zero=bd, n_deduplicated_after_zero=ad)
    if not draws or not contrasts:
        raise ValueError("no paired cliff-K query records")
    metrics = ("R", "R0", "S", "G_before", "G_after")
    return dict(status="complete_pairing", task=next(iter(tasks)), k_cliff=k_cliff,
                estimands={key: statistics.mean(r[key] for r in contrasts) for key in metrics},
                contrast_records=contrasts, n_query_ids=len(queries), n_draw_ids=len(draws),
                n_recorded_adapter_blocks=len(blocks), adapter_blocks=blocks,
                independent_training_replicates="not_inferred_from_recorded_block_labels",
                n_deduplicated_before_zero=bd, n_deduplicated_after_zero=ad,
                provenance_gaps=sorted(provenance_gaps),
                inference="descriptive paired query outcomes; use crossed_bootstrap for conditional uncertainty")


def _quantile(values, probability):
    ordered = sorted(values)
    position = (len(ordered)-1)*probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi]-ordered[lo])*(position-lo)


def crossed_bootstrap(contrasts, *, value_key="S", n_resamples=2000, seed=0,
                      confidence=.95, resample_adapters=False, independent_adapters=False):
    """Percentile bootstrap over crossed query and shared demonstration-set IDs.

    Pass ``paired_effects(...)`` or its ``contrast_records`` list. The same query
    and draw resample is used across all blocks. K=0 occurs as one shared signed
    contribution in each query's contrast, so varying a demo draw never creates
    independent zero-shot evidence. When requested, paired adapter blocks are the
    third resampled dimension; their true independence must be verified by the
    caller and asserted explicitly. Missing/nonrectangular grids are rejected.

    This implements a crossed/pigeonhole bootstrap, not a flattened binomial CI.
    Few clusters can give unreliable percentile coverage; singleton dimensions
    provide conditional estimates and cannot identify population variability.
    """
    if isinstance(contrasts, dict):
        if contrasts.get("status") != "complete_pairing":
            raise ValueError("bootstrap requires complete paired query outcomes")
        contrasts = contrasts["contrast_records"]
    rows = list(contrasts)
    if not rows or n_resamples < 2 or int(n_resamples) != n_resamples or not 0 < confidence < 1:
        raise ValueError("nonempty outcomes, at least two resamples and 0<confidence<1 are required")
    n_resamples = int(n_resamples)
    if resample_adapters and not independent_adapters:
        raise ValueError("adapter resampling requires an explicit assertion of independently trained adapters")
    if len({r.get("task", "unspecified") for r in rows}) != 1:
        raise ValueError("stratify tasks before crossed bootstrap")
    queries = _stable({r["query_id"] for r in rows})
    draws = _stable({r["draw_id"] for r in rows})
    blocks = _stable({r.get("adapter_block") for r in rows})
    if resample_adapters and (len(blocks) < 2 or None in blocks):
        raise ValueError("adapter resampling requires at least two explicit, independent paired blocks")
    index = {}
    for row in rows:
        key = (row.get("adapter_block"), row["query_id"], row["draw_id"])
        if key in index:
            raise ValueError("duplicate contrast identity; cannot count it as another observation")
        value = row.get(value_key)
        if value is None or not math.isfinite(value):
            raise ValueError("every signed contrast must be finite")
        index[key] = float(value)
    if len(index) != len(blocks)*len(queries)*len(draws):
        raise ValueError("incomplete crossed query/draw/adapter grid")
    for block in blocks:
        for query in queries:
            for draw in draws:
                if (block, query, draw) not in index:
                    raise ValueError("incomplete crossed query/draw/adapter grid")
    rng = random.Random(seed)
    estimates = []
    for _ in range(n_resamples):
        qs = Counter(rng.choices(queries, k=len(queries)))
        ds = Counter(rng.choices(draws, k=len(draws)))
        bs = Counter(rng.choices(blocks, k=len(blocks))) if resample_adapters else Counter({block: 1 for block in blocks})
        numerator = math.fsum(index[(block, query, draw)]*nb*nq*nd
                              for block, nb in bs.items() for query, nq in qs.items() for draw, nd in ds.items())
        estimates.append(numerator/(len(blocks)*len(queries)*len(draws)))
    tail = (1-confidence)/2
    singleton = [name for name, ids in (("queries", queries), ("demonstration_draws", draws)) if len(ids) < 2]
    return dict(estimand=value_key, estimate=statistics.mean(index.values()),
                ci_low=_quantile(estimates, tail), ci_high=_quantile(estimates, 1-tail),
                standard_error=statistics.stdev(estimates), confidence=confidence,
                n_resamples=n_resamples, seed=seed, method="crossed_cluster_percentile_bootstrap",
                n_query_ids=len(queries), n_draw_ids=len(draws), n_recorded_adapter_blocks=len(blocks),
                resampled_dimensions=["queries", "demonstration_draws"]+(["paired_adapter_blocks"] if resample_adapters else []),
                training_inference="independent_adapter_blocks_caller_verified" if resample_adapters else "conditional_on_observed_model_or_adapters",
                singleton_dimensions=singleton,
                limitations=["Few clusters can yield unreliable percentile coverage; this interval is not a power guarantee.",
                             "Shared base predictions remain shared across adapters and conditions.",
                             "Independent training replicates cannot be inferred from draw IDs or adapter labels."])
