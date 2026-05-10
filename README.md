# Cavity Parametric Optimisation and ML Surrogate Analysis

This repository contains a clean Python pipeline for analysing COMSOL parametric sweep data from the cavity optimisation study.

The code:

1. reads the COMSOL sweep workbook;
2. extracts resonant frequency and effective magnetic mode volume;
3. builds a machine-learning feature matrix;
4. trains surrogate models for frequency and \(V_{m,\mathrm{eff}}\);
5. performs PCA visualisation;
6. generates feature-importance and smooth valley maps.

## Expected input data

The input file should be an Excel workbook named:

```text
Thesis Table.xlsx
```

The first two sheets are ignored:

```text
Sheet 1: Method
Sheet 2: Results
```

All following sheets should contain one-parameter COMSOL sweeps.

Each sweep sheet should have this structure:

| Column | Meaning | Unit |
|---|---|---|
| A | Varied parameter | m for geometric parameters, unitless for dielectric constant |
| B | Resonant frequency | GHz |
| C | Total magnetic energy, IntW | J |
| D | Maximum magnetic energy density, MaxW | J/m^3 |
| E | Magnetic energy in gain region, GainW | J |
| F | Gain-region volume, GainV | m^3 |
| G | Effective magnetic mode volume, \(V_{m,\mathrm{eff}}\) | m^3 |

The code currently recognises these sheet names:

```text
Gap_Width(eps=8)
dielectric(a=1.2mm)
Gap_Length(eps=9,a=1mm)
Metal_thickness
Dielectric_length
Dielectric_width
Dielectric_thickness
```

If the names change, edit `SHEET_CONFIG` inside `cavity_ml_optimisation.py`.

## How to run

Install the required packages:

```bash
pip install -r requirements.txt
```

Place `Thesis Table.xlsx` in the same directory as the script, then run:

```bash
python cavity_ml_optimisation.py
```

The output figures and processed data will be saved in:

```text
figures/
```

## Main model

The script uses:

- **PCA** for visualising the high-dimensional design space;
- **Extra Trees Regressor** for predicting frequency and \(\log_{10}(V_{m,\mathrm{eff}})\);
- **RBF interpolation** for producing a smooth valley map.

The RBF map is a visual guide only. Since the data are mainly one-parameter sweeps rather than a full multi-parameter grid, the predicted valley should be verified by a direct COMSOL eigenfrequency simulation.
