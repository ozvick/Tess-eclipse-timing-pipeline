# STAR SHADOW cross-check of e cos w and e sin w

Independent eccentricities for the 11 high-e systems from STAR SHADOW v1.1.7a
(IJspeert et al. 2024, A&A 685, A62; the code behind the TESS OBAF catalogue, A&A 691, A242),
run as in IJspeert et al. (2024b): one analysis per TESS cycle, results averaged over cycles.

Setup on kartof (no sudo; the system Python 3.7/3.10 lack venv support):

    micromamba create -n sts -c conda-forge python=3.7 numpy=1.20.3 scipy=1.7.3 numba=0.55.1 \
        h5py=3.7.0 astropy=4.3.1 matplotlib=3.5.3 arviz=0.11.4 corner=2.2.1 pip
    git clone https://github.com/LucIJspeert/star_shadow ~/star_shadow && cd ~/star_shadow
    git checkout v1.1.7a && pip install -e .     # editable: the code locates its data/ via the repo path
    python patch_starshadow.py ~/star_shadow/star_shadow/timeseries_fitting.py

- `export_lcs_starshadow.py` (needs lightkurve; separate Python 3.12 venv): one product per sector
  (SPOC 2-min > TESS-SPOC > QLP), median-normalised per sector, one 3-column file per TESS cycle.
- `run_starshadow.py`: runs STAR SHADOW on each cycle (known periods by default; `--free-period`),
  aligns the primary/secondary labelling of every cycle to the pipeline's convention (a label swap
  flips both components), drops cycles whose e cos w disagrees with the median, and averages.
- `patch_starshadow.py`: fixes a v1.1.7a bug that crashes any multi-sector light curve.

Results (physical-model fit): e cos w agrees with the pipeline within 0.01 for all 11 systems; e sin w
agrees with or lies between the pipeline and Penev's fits in 9 of 11, and follows Penev where the
edge-on duration formula is least reliable (TIC 289044378, 341029467). TIC 71531463 has near-equal
eclipses and needed the label alignment. TIC 468792279 has e sin w ~ 0, which leaves its k2 undetermined.
