# Physics: observed vs theoretical apsidal-motion constant

`theory_k2.py`: theoretical k2 from Claret (2023, A&A 674, A67) MESA tracks (downloaded once from
the VizieR mirror at Harvard and cached). Each star is placed on the track at its mass, at the
main-sequence/subgiant point where the model radius equals its radius (ZAMS = maximum log g between
the end of pre-main-sequence contraction and the giant branch); the two stars' k2 are combined with
the apsidal-motion weights. Observed k2 comes from ../eccentricity/consistency_modelfree.py.

`theory_k2_bands.py`: Monte Carlo uncertainty bands (radius and mass, default 5% each; three
metallicities; measurement error). Radius draws recompute both the observed (R^-5) and theoretical k2.

First result: where the model-free e sin w is reliable, observed and theoretical k2 agree
(+0.04 +/- 0.04 dex, chi2 0.9 for 4 dof); high-e systems sit higher because the edge-on duration
formula underestimates e sin w.

Update (STAR SHADOW eccentricities, ../starshadow/): `theory_k2_bands.py --obs consistency_sts.csv --tag _sts`
uses k2 from ../eccentricity/consistency_sts.py. On the 9 systems both methods constrain, the mean offset
falls from +0.23 +/- 0.03 dex (model-free e) to +0.18 +/- 0.04 dex; 341029467 (+0.65 -> +0.29) and
427312571 (+0.33 -> +0.03) improve, 71531463 and 424719589 do not (their e is agreed by all methods).
The remaining offset may point to the radii. TIC 468792279 has e sin w ~ 0, so its k2 is undetermined.
