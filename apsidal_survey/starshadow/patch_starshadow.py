"""Fix a bug in STAR SHADOW v1.1.7a (timeseries_fitting.py) that crashes multi-sector light curves.

In the Jacobian of the per-sector linear baseline, the full residual array is multiplied by one
sector's time array ("unable to broadcast"), and the constant term sums the residuals of all sectors
instead of that sector. Single-sector light curves are unaffected. This uses each sector's residuals.

Usage (in the sts environment, editable install of star_shadow):
  python patch_starshadow.py ~/star_shadow/star_shadow/timeseries_fitting.py
"""
import shutil, sys

p = sys.argv[1]
src = open(p).read()
fixes = [("df_2a[i] = np.sum(resid)\n", "df_2a[i] = np.sum(resid[s[0]:s[1]])\n"),
         ("df_2a[i_s] = np.sum(resid * (times[s[0]:s[1]]", "df_2a[i_s] = np.sum(resid[s[0]:s[1]] * (times[s[0]:s[1]]")]
if src.count('resid[s[0]:s[1]]') >= 4:
    sys.exit('already patched')
shutil.copy(p, p + '.orig')
for a, b in fixes:
    if src.count(a) != 2:
        sys.exit(f'expected 2 occurrences of {a!r}, found {src.count(a)}; not patched')
    src = src.replace(a, b)
open(p, 'w').write(src)
print('patched 4 lines; original saved as', p + '.orig')
