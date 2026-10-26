"""Compare drifts: old pipeline vs patched pipeline (v2_test/) vs mirror method (retime_all.csv)."""
import csv, glob, json, os

mirror = {}
if os.path.exists('retime_all.csv'):
    mirror = {int(r['tic']): (float(r['mirror']), float(r['mirror_err']))
              for r in csv.DictReader(open('retime_all.csv'))}

def old_summary(t):
    for p in (f'rerun_fixedP/TIC_{t}_out/TIC{t}_summary.json', f'TIC_{t}_out/TIC{t}_summary.json'):
        if os.path.exists(p):
            return json.load(open(p))
    return {}

def f(x, spec):
    return 'n/a' if x is None else format(x, spec)

print(f"{'TIC':>10} {'old':>9} {'patched':>16} {'mirror':>16} {'passes':>6} "
      f"{'sec/pri width':>13} {'bound hits':>10}  verdict old > new")
for p in sorted(glob.glob('v2_test/TIC_*_out/TIC*_summary.json')):
    n = json.load(open(p))
    t = int(n['tic'])
    o = old_summary(t)
    m = mirror.get(t)
    ratio = (n['secondary_half_width_phase'] / n['primary_half_width_phase']
             if n.get('primary_half_width_phase') else None)
    new_s = n.get('apsidal_slope_min_per_1000d')
    new_e = n.get('apsidal_slope_err_min_per_1000d')
    new_txt = f"{new_s:+8.2f}+/-{new_e:5.2f}" if new_s is not None else '     n/a'
    mir_txt = f"{m[0]:+8.2f}+/-{m[1]:5.2f}" if m else '     n/a'
    print(f"{t:>10} {f(o.get('apsidal_slope_min_per_1000d'), '+9.2f'):>9} {new_txt:>16} {mir_txt:>16} "
          f"{n.get('n_passes', 0):>6} {f(ratio, '13.2f'):>13} "
          f"{str(n.get('n_primary_at_shift_bound')) + '/' + str(n.get('n_secondary_at_shift_bound')):>10}  "
          f"{o.get('verdict')} > {n.get('verdict')}")
