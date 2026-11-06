# Validation of the duration-based e sin(omega)

`make_validation_set.py` generates TESS-format light curves (ellc; known e cos w and e sin w;
spherical and Roche stars; similar and unequal eclipse depths) and a script that runs the pipeline
on them. `evaluate_validation.py` compares the pipeline's secondary-phase e cos w and duration-ratio
e sin w with the truth.

Results: e cos w is recovered to <= 0.0002. With pipeline v2.1, e sin w was rounded in steps of
~0.01 (P = 2.5 d) to ~0.03 (P = 10 d) by the binned half-width; v2.2 (sub-bin interpolation)
brings it to ~0.003-0.005 for e sin w <= 0.03. At large e sin w and longer periods, the edge-on
formula e sin w ~ (r - 1)/(r + 1) underestimates (inclination effect).
