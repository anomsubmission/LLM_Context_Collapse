"""Recompose archived vector plots with accurate labels and explicit provenance.

Raw-lens query data were not released. Their vector curves are retained from the
supplied artwork; this script does not infer new data or uncertainty from them.
The residual-state and induction panels are redrawn from the archived JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

from .revision_records import file_hash

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pymupdf as pdf

    destination = PAPER / "figures/revision"
    destination.mkdir(parents=True, exist_ok=True)
    source = PAPER / "figures/fig9_logit_lens_llama.pdf"
    data_path = ROOT / "results/interventions/t5_llama31_8b_ag_news_results.json"
    data = json.loads(data_path.read_text())
    lens = pdf.open(source)
    page = lens[0]
    operations = []

    def redact(rectangle, reason):
        rectangle = pdf.Rect(rectangle)
        page.add_redact_annot(rectangle, fill=(1, 1, 1), cross_out=False)
        operations.append(dict(rectangle=list(rectangle), reason=reason))

    # The callout sits above every AG News curve. Remove its text/border
    # presentation without deleting vector line art or modifying data paths.
    redact([92, 13, 174, 45], "Remove unsupported broken-pathway interpretation.")
    for text in ("a", "b", "a) AG News (cliff site)", "b) IMDB (no cliff)"):
        spans = page.search_for(text)
        if text in {"a", "b"}:
            spans = [rect for rect in spans if rect.y0 < 8 and rect.width < 10]
        for rect in spans:
            redact(rect, "Replace repeated panel letters and inference-bearing titles.")
    legends = []
    for text, replacement in (("demo suppression", "K16 < K0"), ("demo integration", "K16 > K0")):
        for rect in page.search_for(text):
            legends.append((rect, replacement))
            redact(rect, "Describe the sign of the raw-readout difference without assigning a mechanism.")
    page.apply_redactions(images=pdf.PDF_REDACT_IMAGE_NONE,
                          graphics=pdf.PDF_REDACT_LINE_ART_NONE, text=pdf.PDF_REDACT_TEXT_REMOVE)
    # Cover the old callout's purely decorative rectangle after retaining its
    # vector paths. This region contains no plotted curve in the source artwork.
    page.draw_rect(pdf.Rect(92, 13, 174, 45), color=None, fill=(1, 1, 1), overlay=True)
    for rectangle, replacement in legends:
        page.insert_text((rectangle.x0, rectangle.y1-2.5), replacement, fontsize=7, fontname="helv")
    for center, title in ((130, "a  AG News"), (349, "b  IMDB")):
        width = pdf.get_text_length(title, fontname="hebo", fontsize=9)
        page.insert_text((center-width/2, 10), title, fontsize=9, fontname="hebo")
    lens_path = destination / "archive_lens_clean.pdf"
    lens.save(lens_path)

    residual, induction = data["fv_effect_by_layer"], data["induction_score_by_layer"]
    count = data["n_layers"]
    if len(residual) != count or len(induction) != count:
        raise ValueError("archived per-layer arrays differ from declared model depth")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.05), constrained_layout=True)
    for axis in axes:
        axis.axvspan(7.5, 15.5, color="#dbe9f6", alpha=.65, label="reported window")
        axis.set_xlabel("Layer index")
        axis.set_xlim(-.5, count-.5)
        axis.set_xticks(range(0, count, 8))
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].plot(range(count), residual, "o-", ms=3, color="#1f77b4")
    axes[0].axhline(data["zero_shot_acc"], color=".35", ls="--", lw=1, label="zero-shot")
    axes[0].axhline(data["icl_acc"], color="#a33", ls=":", lw=1, label="archived K=16")
    axes[0].set(ylabel="Zero-shot accuracy after replacement", ylim=(0, 1))
    axes[0].set_title("c  Mean residual-state replacement", loc="left", fontsize=10)
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    axes[1].plot(range(count), induction, "o-", ms=3, color="#d77d14")
    within = sum(induction[8:16])/8
    outside = sum(induction[:8]+induction[16:])/(count-8)
    axes[1].axhline(within, color="#4a7ebb", lw=1, label=f"window mean {within:.3f}")
    axes[1].axhline(outside, color=".45", ls="--", lw=1, label=f"other-layer mean {outside:.3f}")
    axes[1].set(ylabel="Induction-head score", ylim=(0, max(induction)*1.22))
    axes[1].set_title("d  Induction-head profile", loc="left", fontsize=10)
    axes[1].legend(fontsize=7, frameon=False, loc="upper right")
    state_path = destination / "archive_state_profiles.pdf"
    fig.savefig(state_path)
    plt.close(fig)
    lower = pdf.open(state_path)
    combined = pdf.open()
    width = 576
    upper_height = width * page.rect.height / page.rect.width
    lower_height = width * lower[0].rect.height / lower[0].rect.width
    output = combined.new_page(width=width, height=upper_height+lower_height+10)
    output.show_pdf_page(pdf.Rect(0, 0, width, upper_height), lens, 0)
    output.show_pdf_page(pdf.Rect(0, upper_height+10, width, upper_height+10+lower_height), lower, 0)
    final = destination / "archive_mechanistic_readouts.pdf"
    combined.save(final)
    output.get_pixmap(matrix=pdf.Matrix(1.8, 1.8)).save(destination / "archive_mechanistic_readouts.png")
    metadata = dict(figure=str(final.relative_to(PAPER)), inputs={str(source.relative_to(PAPER)): file_hash(source),
        str(data_path.relative_to(PAPER)): file_hash(data_path)}, operations=operations,
        raw_lens_curves="original vector paths retained; no query-level source available",
        other_panels="exact archived per-layer arrays; no smoothing or fabricated uncertainty",
        scientific_evidence="descriptive archived plots only; does not establish causal localization",
        uncertainty="unavailable from these archived arrays; new raw/tuned-lens experiments remain required",
        output_sha256=file_hash(final))
    (destination / "archive_mechanistic_readouts.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"figure": str(final), "source_artwork_preserved": True}))


if __name__ == "__main__":
    main()
