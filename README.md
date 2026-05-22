# Eclipse Timing Variation Pipeline for TESS Eclipsing Binary Systems

Computational pipeline for extracting Eclipse Timing Variations (ETVs) from TESS light curves.

---

## Overview

This project develops a computational framework for identifying eclipses, constructing eclipse templates, and measuring Eclipse Timing Variations in eclipsing binary systems observed by NASA's Transiting Exoplanet Survey Satellite (TESS).

The objective is to detect subtle orbital timing changes that may indicate:

- Orbital evolution
- Dynamical interactions
- Third-body perturbations
- Stellar system complexity

---

## Scientific Motivation

Eclipse Timing Variations (ETVs) are powerful probes of orbital dynamics in eclipsing binary systems.

Because observational light curves are often noisy and irregular, robust computational methods are required to accurately identify eclipses and extract timing information.

This project develops a scalable timing-analysis pipeline for TESS observations.

---

## Pipeline Workflow

1. Download and preprocess TESS light curves  
2. Normalize and clean photometric data  
3. Perform phase folding using orbital period estimates  
4. Separate primary and secondary eclipses  
5. Construct eclipse templates  
6. Extract eclipse timing information  
7. Generate timing diagnostics and O–C analysis

---

## Example Outputs

### Normalized TESS Light Curve

![Light Curve](figures/TESS%20Light%20Curve%20for%20Eclipse%20Timing%20Analysis.png)

### Phase Folded Eclipse Analysis

![Phase Folded](figures/phase_folded.png)

---

## Example Target

**TIC 275423476**

Key measurements:

- Orbital period ≈ 2.92879 days
- Secondary eclipse phase offset analyzed
- Reduced χ² optimization performed

---

## Technical Stack

- Python
- Lightkurve
- NumPy
- SciPy
- Matplotlib

---

## Future Work

- Automated ETV extraction
- O–C timing diagnostics
- Template-based timing refinement
- Photodynamical modeling

---

## Author

**Victor Oyiboka**  
Physics Graduate Researcher  
University of Texas at Dallas
