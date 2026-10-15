import json, glob

print(f"{'TIC':>10} {'verdict':9s} {'slope':>8} {'err':>5} {'|r|':>5} {'n_p/n_s':>7} "
      f"{'dDepth%':>7} {'sec_ph':>6} {'baseP':>7} {'dStab%':>6} {'raw/res':>7} flags")
for path in sorted(glob.glob('rerun_fixedP/TIC_*_out/TIC*_summary.json')):
    s = json.load(open(path))
    if s.get('verdict') not in ('SOLID', 'TENTATIVE'):
        continue
    m = s.get('mechanistic_checks') or {}
    pdp, sdp, ph = s.get('primary_depth'), s.get('secondary_depth'), s.get('secondary_phase')
    rel = abs(pdp - sdp) / pdp * 100 if pdp and sdp else float('nan')
    flags = []
    if rel < 5 and ph is not None and abs(ph - 0.5) < 0.01:
        flags.append('EQUAL-DEPTH@0.5')
    if s.get('period_alias_caveat'):
        flags.append('alias_caveat')
    if s.get('primary_curvature_flagged'):
        flags.append('curvature')
    def g(k, spec, d=s):
        v = d.get(k)
        return 'n/a' if v is None else format(v, spec)
    print(f"{s['tic']:>10} {s['verdict']:9s} {g('apsidal_slope_min_per_1000d', '+8.2f')} "
          f"{g('apsidal_slope_err_min_per_1000d', '5.2f')} {abs(s.get('apsidal_r') or 0):5.3f} "
          f"{str(s.get('n_primary_qc')) + '/' + str(s.get('n_secondary_qc')):>7} {rel:7.1f} "
          f"{g('secondary_phase', '6.3f')} {g('baseline_drift_p', '7.1e', m)} "
          f"{g('depth_stability_pct_change', '6.2f', m)} {g('raw_vs_resid_ratio', '7.2f', m)} "
          f"{' '.join(flags)}")
