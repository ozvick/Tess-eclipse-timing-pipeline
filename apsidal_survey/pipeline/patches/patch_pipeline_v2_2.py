"""Patch v2.2: measure eclipse half-widths to sub-bin precision.

measure_eclipse_half_width() took the half-width as the centre of the outermost binned-profile
bin below the depth threshold, so widths changed only in steps of one bin (0.00075 in phase,
x1.3 padding). For e sin(omega) from the duration ratio that gave a resolution of ~0.01 (P = 2.5 d)
to ~0.03 (P = 10 d). Now, on each side, the threshold crossing is interpolated between the
outermost below-threshold bin and the next bin outward, and the two sides are averaged.
Bins, threshold and padding are unchanged. Requires v2 and v2.1. Backup: apsidal_pipeline.py.bak_v2_2.
"""
import sys

PIPE = 'apsidal_pipeline.py'
src = open(PIPE).read()
if 'Sub-bin precision (v2.2)' in src:
    print('v2.2 already applied, nothing to do'); sys.exit(0)

old = "    half_width = np.max(np.abs(bc[below])) * 1.3  # small padding\n"
new = '''    # Sub-bin precision (v2.2): on each side, interpolate between the outermost below-threshold
    # bin and the next bin outward to where the profile crosses the threshold, then average the
    # two sides. Removes the one-bin rounding of the width (~0.001 in phase).
    edges = []
    for side in (-1, 1):
        idx = np.where(below & (side * bc > 0))[0]
        if len(idx) == 0:
            continue
        i_in = idx[np.argmax(np.abs(bc[idx]))]
        i_out = i_in + side
        if 0 <= i_out < len(bc) and np.isfinite(med[i_out]) and med[i_out] != med[i_in]:
            frac = min(max((threshold - med[i_in]) / (med[i_out] - med[i_in]), 0.0), 1.0)
            edges.append(abs(bc[i_in] + frac * (bc[i_out] - bc[i_in])))
        else:
            edges.append(abs(bc[i_in]))
    if not edges:
        return 0.01
    half_width = float(np.mean(edges)) * 1.3  # small padding
'''
old_v = '        "pipeline_version": 2,\n'
new_v = '        "pipeline_version": 2.2,\n'
for o in (old, old_v):
    if src.count(o) != 1:
        print(f'NOT PATCHED: expected one match of {o.strip()!r}, found {src.count(o)}'); sys.exit(1)
open(PIPE + '.bak_v2_2', 'w').write(src)
open(PIPE, 'w').write(src.replace(old, new).replace(old_v, new_v))
print('Patched apsidal_pipeline.py (v2.2: sub-bin half-widths). Backup: apsidal_pipeline.py.bak_v2_2')
