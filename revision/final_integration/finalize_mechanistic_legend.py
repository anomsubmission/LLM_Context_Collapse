"""Create a vector-preserving Figure 2 copy with one neutral legend edit."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.matplotlib'))
import matplotlib
import numpy as np
import pymupdf

PAPER = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = PAPER / 'figures/revision/archive_mechanistic_readouts.pdf'
TARGET = PAPER / 'figures/revision/final_mechanistic_readouts.pdf'
OLD = 'archived K=16'
NEW = 'K=16 reference'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spans(page):
    return [s for b in page.get_text('dict')['blocks'] for l in b.get('lines', []) for s in l['spans']]


def key(span):
    return (span['text'], tuple(round(x, 5) for x in span['bbox']), tuple(round(x, 5) for x in span['origin']),
            round(span['size'], 5), span['color'])


def main():
    original_hash = sha(SOURCE)
    doc = pymupdf.open(SOURCE); page = doc[0]
    before = spans(page)
    target = [s for s in before if s['text'] == OLD]
    assert len(target) == 1 and len(doc) == 1
    target = target[0]
    # The nominal text boxes of neighboring legend lines overlap; a tight strip
    # below the previous line removes every target character without touching it.
    bbox = pymupdf.Rect(target['bbox'])
    deletion = pymupdf.Rect(bbox.x0-.1, 480.3, bbox.x1+.1, 490.7)
    neighbors = [s for s in before if s['text'] != OLD and pymupdf.Rect(s['bbox']).intersects(deletion)]
    assert not neighbors
    page.add_redact_annot(deletion, fill=(1, 1, 1))
    page.apply_redactions(images=0, graphics=0)
    assert Counter(key(s) for s in spans(page)) == Counter(key(s) for s in before if s['text'] != OLD)
    fontpath = Path(matplotlib.get_data_path()) / 'fonts/ttf/DejaVuSans.ttf'
    font = pymupdf.Font(fontfile=str(fontpath))
    new_width = font.text_length(NEW, fontsize=target['size'])
    assert new_width <= bbox.width + 5
    replacement_box = pymupdf.Rect(bbox.x0, deletion.y0, bbox.x0+new_width, deletion.y1)
    assert not [s for s in before if s['text'] != OLD and pymupdf.Rect(s['bbox']).intersects(replacement_box)]
    page.insert_font(fontname='FinalLegend', fontfile=str(fontpath))
    page.insert_text(pymupdf.Point(*target['origin']), NEW, fontsize=target['size'],
                     fontname='FinalLegend', color=(0,0,0), overlay=True)
    doc.save(TARGET, garbage=4, deflate=True)
    doc.close()
    original, updated = pymupdf.open(SOURCE), pymupdf.open(TARGET)
    after = spans(updated[0])
    assert sum(s['text'] == NEW for s in after) == 1
    assert not any(OLD in s['text'] for s in after)
    assert Counter(key(s) for s in after if s['text'] != NEW) == Counter(key(s) for s in before if s['text'] != OLD)
    numeric_before = Counter(key(s) for s in before if re.search(r'\d', s['text']) and s['text'] != OLD)
    numeric_after = Counter(key(s) for s in after if re.search(r'\d', s['text']) and s['text'] != NEW)
    assert numeric_before == numeric_after
    assert Counter(re.findall(r'\d+', OLD)) == Counter(re.findall(r'\d+', NEW))
    assert original[0].rect == updated[0].rect
    # Raster comparison is a verification only; the saved PDF remains vector.
    a = original[0].get_pixmap(matrix=pymupdf.Matrix(3,3), alpha=False)
    b = updated[0].get_pixmap(matrix=pymupdf.Matrix(3,3), alpha=False)
    aa = np.frombuffer(a.samples, dtype=np.uint8).reshape(a.height,a.width,a.n)
    bb = np.frombuffer(b.samples, dtype=np.uint8).reshape(b.height,b.width,b.n)
    changed = np.any(aa != bb, axis=2)
    allowed = np.zeros(changed.shape, dtype=bool)
    allowed[int(478*3):int(492*3)+1, int(65*3):int(123*3)+1] = True
    assert not np.any(changed & ~allowed), 'Pixels outside the edited legend changed'
    preview = HERE / 'final_mechanistic_readouts_preview.png'
    updated[0].get_pixmap(matrix=pymupdf.Matrix(2,2), alpha=False).save(preview)
    assert sha(SOURCE) == original_hash
    audit = {'status':'passed', 'source_pdf':str(SOURCE.relative_to(PAPER)), 'source_sha256':original_hash,
             'target_pdf':str(TARGET.relative_to(PAPER)), 'target_sha256':sha(TARGET),
             'old_legend':OLD, 'new_legend':NEW, 'original_text_origin':target['origin'],
             'original_font_size':target['size'], 'font':'DejaVu Sans',
             'all_non_target_text_spans_and_geometry_unchanged':True,
             'all_non_target_numeric_spans_and_geometry_unchanged':True,
             'changed_legend_preserves_K16_value':True,
             'page_geometry_unchanged':True, 'pdf_remains_vector':True,
             'rendered_pixels_unchanged_outside_edited_legend_at_3x':True,
             'edited_pixel_count_at_3x':int(changed.sum()),
             'source_asset_not_modified':True, 'generator_sha256':sha(__file__)}
    (HERE / 'final_mechanistic_legend_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
