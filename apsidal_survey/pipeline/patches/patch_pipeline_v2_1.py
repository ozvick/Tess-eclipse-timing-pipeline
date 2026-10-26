"""Follow-up to patch_pipeline_v2.py: judge ephemeris convergence relative to eclipse width.

The iteration stops when predicted eclipse times move by less than 2% of the narrower
eclipse half-width, or 0.5 min, whichever is larger. A fixed 0.5 min was far stricter
than needed for wide eclipses (e.g. ~12 h half-widths in TIC 125857387).
Requires the v2 patch to be applied already. Backup: apsidal_pipeline.py.bak_v2_1.
"""
import sys

PIPE = "apsidal_pipeline.py"
src = open(PIPE).read()
new = src
problems = []


def replace_once(text, old, rep, label):
    n = text.count(old)
    if n != 1:
        problems.append(f"{label}: expected 1 match, found {n}")
        return text
    return text.replace(old, rep)


new = replace_once(
    new,
    "    CONVERGED_MIN = 0.5   # stop when predicted eclipse times move less than this [min]\n",
    "    CONVERGED_MIN = 0.5   # stop when predicted eclipse times move less than this [min] ...\n"
    "    CONVERGED_FRAC = 0.02 # ... or this fraction of the narrower eclipse half-width, if larger\n"
    "    tol_min = CONVERGED_MIN\n",
    "constants",
)
new = replace_once(
    new,
    "        if n_pass >= 2 and converged_shift < CONVERGED_MIN:\n",
    "        tol_min = max(CONVERGED_MIN, CONVERGED_FRAC * min(hw_p, hw_s) * P_refined * 24 * 60)\n"
    "        if n_pass >= 2 and converged_shift < tol_min:\n",
    "convergence test",
)
new = replace_once(
    new,
    '        "converged_shift_min": converged_shift,\n',
    '        "converged_shift_min": converged_shift,\n'
    '        "converged_tolerance_min": tol_min,\n',
    "summary JSON",
)

if problems:
    print("NOT PATCHED. Problems:")
    for p in problems:
        print("  -", p)
    sys.exit(1)
open(PIPE + ".bak_v2_1", "w").write(src)
open(PIPE, "w").write(new)
print("Patched apsidal_pipeline.py (relative convergence threshold). Backup: apsidal_pipeline.py.bak_v2_1")
