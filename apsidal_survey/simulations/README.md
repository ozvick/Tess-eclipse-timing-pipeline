# Simulations

`sim_fake_ecc.py`: tests whether unmodeled precession (apsidal motion or a changing
inclination) makes a static light-curve fit report spurious eccentricity. Uses ellc.

## Installing ellc (Linux)
ellc 1.8.15 from PyPI ships a prebuilt macOS library and does not compile its Fortran source,
so build the library by hand and replace the shipped one:

    pip install ellc
    pip download ellc==1.8.15 --no-deps --no-binary :all:
    tar xzf ellc-1.8.15.tar.gz && cd ellc-1.8.15/f_src
    gfortran -O2 -fPIC -shared -o libellc.so constants.f90 solve_real_poly.f90 coords.f90 \
        utils.f90 gauss_legendre.f90 ellipse.f90 stellar.f90 spots.f90 ellc.f90
    cp libellc.so $(python -c "import os, importlib.util; print(os.path.dirname(importlib.util.find_spec('ellc').origin))")/

Reference: Maxted (2016), A&A 591, A111.
