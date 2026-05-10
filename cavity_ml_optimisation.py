"""
Cavity parametric optimisation and machine-learning analysis.

This script reads a COMSOL parametric sweep workbook, extracts frequency and
effective magnetic mode volume data, trains simple surrogate models, and
generates journal-style plots for thesis discussion.

Expected workbook structure
---------------------------
The Excel workbook should contain:
    Sheet 1: Method      ignored by this script
    Sheet 2: Results     ignored by this script
    Sheet 3 onwards:     parametric sweep sheets

Each sweep sheet should have the following columns:

    A: varied parameter value
       - geometric parameters should be in metres
       - dielectric constant should be unitless
    B: resonant frequency in GHz
    C: total magnetic energy, IntW, in J
    D: maximum magnetic energy density, MaxW, in J/m^3
    E: magnetic energy in gain region, GainW, in J
    F: gain-region volume, GainV, in m^3
    G: effective magnetic mode volume, Vm_eff, in m^3

The script recognises these sheet names:
    Gap_Width(eps=8)
    dielectric(a=1.2mm)
    Gap_Length(eps=9,a=1mm)
    Metal_thickness
    Dielectric_length
    Dielectric_width
    Dielectric_thickness

If your sheet names change, edit SHEET_CONFIG below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from scipy.interpolate import RBFInterpolator
except Exception:
    RBFInterpolator = None


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class Config:
    input_excel: str | Path = "Thesis Table.xlsx"
    output_dir: str | Path = "figures"
    target_frequency_ghz: float = 6.46
    frequency_scale_ghz: float = 0.10
    random_state: int = 3

    # Reference values used to fill non-varied parameters in one-at-a-time sweeps.
    # Edit these values if your reference geometry changes.
    baseline: Dict[str, float] = field(default_factory=lambda: {
        "a_mm": 1.0,       # gap width
        "eps_r": 9.0,      # dielectric constant
        "b_mm": 4.092,     # gap length
        "tcu_mm": 0.6,     # metal thickness
        "c_mm": 1.6,       # dielectric length
        "d_mm": 7.09,      # dielectric width
        "tsap_mm": 0.6,    # dielectric thickness
    })


FEATURE_NAMES = ["a_mm", "eps_r", "b_mm", "tcu_mm", "c_mm", "d_mm", "tsap_mm"]

SHEET_CONFIG = {
    "Gap_Width(eps=8)": {
        "parameter_label": "Gap width",
        "parameter_symbol": "a",
        "unit": "mm",
        "feature": "a_mm",
        "scale": 1e3,          # m -> mm
        "fixed": {"eps_r": 8.0},
    },
    "dielectric(a=1.2mm)": {
        "parameter_label": "Dielectric constant",
        "parameter_symbol": "eps_r",
        "unit": "",
        "feature": "eps_r",
        "scale": 1.0,
        "fixed": {"a_mm": 1.2},
    },
    "Gap_Length(eps=9,a=1mm)": {
        "parameter_label": "Gap length",
        "parameter_symbol": "b",
        "unit": "mm",
        "feature": "b_mm",
        "scale": 1e3,
        "fixed": {"eps_r": 9.0, "a_mm": 1.0},
    },
    "Metal_thickness": {
        "parameter_label": "Metal thickness",
        "parameter_symbol": "t_Cu",
        "unit": "mm",
        "feature": "tcu_mm",
        "scale": 1e3,
        "fixed": {"eps_r": 9.0, "a_mm": 1.0},
    },
    "Dielectric_length": {
        "parameter_label": "Dielectric length",
        "parameter_symbol": "c",
        "unit": "mm",
        "feature": "c_mm",
        "scale": 1e3,
        "fixed": {"eps_r": 9.0, "a_mm": 1.0},
    },
    "Dielectric_width": {
        "parameter_label": "Dielectric width",
        "parameter_symbol": "d",
        "unit": "mm",
        "feature": "d_mm",
        "scale": 1e3,
        "fixed": {"eps_r": 9.0, "a_mm": 1.0},
    },
    "Dielectric_thickness": {
        "parameter_label": "Dielectric thickness",
        "parameter_symbol": "t_sap",
        "unit": "mm",
        "feature": "tsap_mm",
        "scale": 1e3,
        "fixed": {"eps_r": 9.0, "a_mm": 1.0},
    },
}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def _clean_numeric(value):
    """Return a float or NaN for a spreadsheet value."""
    try:
        return float(value)
    except Exception:
        return np.nan


def load_parametric_workbook(config: Config) -> pd.DataFrame:
    """
    Load all sweep sheets after the first two workbook sheets.

    Returns
    -------
    pd.DataFrame
        One row per COMSOL simulation, with physical features and targets.
    """
    input_excel = Path(config.input_excel)
    if not input_excel.exists():
        raise FileNotFoundError(f"Workbook not found: {input_excel}")

    all_sheets = pd.read_excel(input_excel, sheet_name=None, engine="openpyxl")
    sheet_names = list(all_sheets.keys())

    # First two sheets are assumed to be Method and Results, used for tables.
    sweep_names = sheet_names[2:]
    rows: List[dict] = []

    for sheet_name in sweep_names:
        if sheet_name not in SHEET_CONFIG:
            print(f"Skipping unrecognised sheet: {sheet_name}")
            continue

        spec = SHEET_CONFIG[sheet_name]
        df = all_sheets[sheet_name].copy()
        if df.empty or df.shape[1] < 7:
            print(f"Skipping incomplete sheet: {sheet_name}")
            continue

        # Use positional columns so the code is robust to long COMSOL column names.
        for _, row in df.iterrows():
            raw_param = _clean_numeric(row.iloc[0])
            frequency = _clean_numeric(row.iloc[1])
            intW = _clean_numeric(row.iloc[2])
            maxW = _clean_numeric(row.iloc[3])
            gainW = _clean_numeric(row.iloc[4])
            gainV = _clean_numeric(row.iloc[5])
            vm_eff = _clean_numeric(row.iloc[6])

            if np.isnan(raw_param) or np.isnan(frequency) or np.isnan(vm_eff):
                continue

            features = dict(config.baseline)
            parameter_value = raw_param * spec["scale"]
            features[spec["feature"]] = parameter_value
            features.update(spec["fixed"])

            freq_error = abs(frequency - config.target_frequency_ghz)
            objective = np.log10(vm_eff / 1e-11) + (freq_error / config.frequency_scale_ghz) ** 2

            rows.append({
                "sheet": sheet_name,
                "parameter_label": spec["parameter_label"],
                "parameter_symbol": spec["parameter_symbol"],
                "unit": spec["unit"],
                "parameter_value": parameter_value,
                "frequency_GHz": frequency,
                "freq_error_GHz": freq_error,
                "IntW_J": intW,
                "MaxW_J_m3": maxW,
                "GainW_J": gainW,
                "GainV_m3": gainV,
                "Vm_eff_m3": vm_eff,
                "log10_Vm_eff": np.log10(vm_eff),
                "objective": objective,
                **features,
            })

    if not rows:
        raise ValueError("No valid simulation rows were extracted.")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Machine learning
# ---------------------------------------------------------------------

def train_surrogate_models(data: pd.DataFrame, config: Config):
    """
    Train Extra Trees surrogate models for frequency and log10(Vm_eff).
    """
    X = data[FEATURE_NAMES].to_numpy(float)
    y_freq = data["frequency_GHz"].to_numpy(float)
    y_logv = data["log10_Vm_eff"].to_numpy(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, yf_train, yf_test, yv_train, yv_test = train_test_split(
        X_scaled,
        y_freq,
        y_logv,
        test_size=0.25,
        random_state=config.random_state,
    )

    frequency_model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=config.random_state,
    )
    vm_model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=2,
        random_state=config.random_state + 1,
    )

    frequency_model.fit(X_train, yf_train)
    vm_model.fit(X_train, yv_train)

    yf_pred = frequency_model.predict(X_test)
    yv_pred = vm_model.predict(X_test)

    kfold = KFold(n_splits=5, shuffle=True, random_state=config.random_state)

    metrics = {
        "frequency_r2_test": r2_score(yf_test, yf_pred),
        "frequency_mae_test_GHz": mean_absolute_error(yf_test, yf_pred),
        "vm_r2_test": r2_score(yv_test, yv_pred),
        "vm_mae_test_log10": mean_absolute_error(yv_test, yv_pred),
        "frequency_r2_cv_mean": cross_val_score(frequency_model, X_scaled, y_freq, cv=kfold, scoring="r2").mean(),
        "vm_r2_cv_mean": cross_val_score(vm_model, X_scaled, y_logv, cv=kfold, scoring="r2").mean(),
    }

    # Refit on full data for plotting and design-space maps.
    frequency_model.fit(X_scaled, y_freq)
    vm_model.fit(X_scaled, y_logv)

    return {
        "scaler": scaler,
        "X_scaled": X_scaled,
        "frequency_model": frequency_model,
        "vm_model": vm_model,
        "metrics": metrics,
        "test_data": {
            "yf_test": yf_test,
            "yf_pred": yf_pred,
            "yv_test": yv_test,
            "yv_pred": yv_pred,
        },
    }


def run_pca(data: pd.DataFrame, X_scaled: np.ndarray):
    """Run PCA on the scaled feature matrix."""
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(X_scaled)
    data = data.copy()
    data["PC1"] = coords[:, 0]
    data["PC2"] = coords[:, 1]
    return pca, data


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def _set_plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def plot_diagnostics(models: dict, output_dir: Path):
    """Plot predicted versus simulated frequency and mode volume."""
    _set_plot_style()
    t = models["test_data"]

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2), constrained_layout=True)

    ax = axes[0]
    ax.scatter(t["yf_test"], t["yf_pred"], s=22, alpha=0.8)
    mn = min(np.min(t["yf_test"]), np.min(t["yf_pred"]))
    mx = max(np.max(t["yf_test"]), np.max(t["yf_pred"]))
    ax.plot([mn, mx], [mn, mx], "--", color="0.35", linewidth=0.9)
    ax.set_xlabel("Simulated frequency (GHz)")
    ax.set_ylabel("ML-predicted frequency (GHz)")
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    ax.text(
        0.04, 0.94,
        f"(a)\n$R^2$={models['metrics']['frequency_r2_test']:.3f}\n"
        f"MAE={models['metrics']['frequency_mae_test_GHz']:.3f} GHz",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    ax = axes[1]
    ax.scatter(t["yv_test"], t["yv_pred"], s=22, color="#9467bd", alpha=0.8)
    mn = min(np.min(t["yv_test"]), np.min(t["yv_pred"]))
    mx = max(np.max(t["yv_test"]), np.max(t["yv_pred"]))
    ax.plot([mn, mx], [mn, mx], "--", color="0.35", linewidth=0.9)
    ax.set_xlabel(r"Simulated $\log_{10}(V_{m,\mathrm{eff}})$")
    ax.set_ylabel(r"ML-predicted $\log_{10}(V_{m,\mathrm{eff}})$")
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    ax.text(
        0.04, 0.94,
        f"(b)\n$R^2$={models['metrics']['vm_r2_test']:.3f}\n"
        f"MAE={models['metrics']['vm_mae_test_log10']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    fig.savefig(output_dir / "ml_surrogate_diagnostics.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "ml_surrogate_diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(models: dict, output_dir: Path):
    """Plot feature importance for frequency and Vm_eff models."""
    _set_plot_style()

    readable = {
        "a_mm": "Gap width",
        "eps_r": "Dielectric constant",
        "b_mm": "Gap length",
        "tcu_mm": "Metal thickness",
        "c_mm": "Dielectric length",
        "d_mm": "Dielectric width",
        "tsap_mm": "Dielectric thickness",
    }

    freq_imp = models["frequency_model"].feature_importances_
    vm_imp = models["vm_model"].feature_importances_

    labels = [readable[name] for name in FEATURE_NAMES]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 3.5), constrained_layout=True)
    ax.bar(x - width / 2, freq_imp, width=width, label="Frequency model")
    ax.bar(x + width / 2, vm_imp, width=width, label=r"$V_{m,\mathrm{eff}}$ model")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Feature importance")
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    ax.legend(frameon=False, loc="upper right")

    fig.savefig(output_dir / "ml_feature_importance.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "ml_feature_importance.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pca(data: pd.DataFrame, pca: PCA, output_dir: Path):
    """Plot PCA coordinates coloured by log10(Vm_eff)."""
    _set_plot_style()

    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    scatter = ax.scatter(
        data["PC1"],
        data["PC2"],
        c=data["log10_Vm_eff"],
        cmap="viridis",
        s=22,
        edgecolor="none",
        alpha=0.85,
    )
    best = data.loc[data["objective"].idxmin()]
    ax.scatter(best["PC1"], best["PC2"], marker="*", s=130, color="red", edgecolor="white", linewidth=0.6)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"$\log_{10}(V_{m,\mathrm{eff}})$")

    fig.savefig(output_dir / "ml_pca_design_space.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "ml_pca_design_space.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_smooth_valley_map(data: pd.DataFrame, models: dict, config: Config, output_dir: Path):
    """
    Plot a smooth RBF surrogate map over gap width and metal thickness.

    This map is for visual guidance only. It is not a substitute for a
    multi-parameter COMSOL sweep.
    """
    if RBFInterpolator is None:
        print("SciPy RBFInterpolator is not available. Skipping smooth valley map.")
        return None

    _set_plot_style()

    scaler = models["scaler"]
    X_scaled = models["X_scaled"]

    y_freq = data["frequency_GHz"].to_numpy(float)
    y_logv = data["log10_Vm_eff"].to_numpy(float)

    rbf_freq = RBFInterpolator(
        X_scaled,
        y_freq,
        kernel="gaussian",
        epsilon=1.4,
        smoothing=0.015,
        degree=-1,
        neighbors=100,
    )
    rbf_logv = RBFInterpolator(
        X_scaled,
        y_logv,
        kernel="gaussian",
        epsilon=1.4,
        smoothing=0.015,
        degree=-1,
        neighbors=100,
    )

    # Plot only the region relevant to the current design.
    a_values = np.linspace(0.02, 1.20, 300)
    tcu_values = np.linspace(0.10, 3.00, 300)
    A, T = np.meshgrid(a_values, tcu_values)

    X_grid = np.zeros((A.size, len(FEATURE_NAMES)))
    for i, name in enumerate(FEATURE_NAMES):
        X_grid[:, i] = config.baseline[name]

    X_grid[:, FEATURE_NAMES.index("a_mm")] = A.ravel()
    X_grid[:, FEATURE_NAMES.index("tcu_mm")] = T.ravel()
    X_grid[:, FEATURE_NAMES.index("eps_r")] = 9.0

    X_grid_scaled = scaler.transform(X_grid)

    F = rbf_freq(X_grid_scaled).reshape(A.shape)
    LOGV = rbf_logv(X_grid_scaled).reshape(A.shape)
    VM = 10 ** LOGV
    OBJ = np.log10(VM / 1e-11) + (np.abs(F - config.target_frequency_ghz) / config.frequency_scale_ghz) ** 2

    best_idx = np.unravel_index(np.nanargmin(OBJ), OBJ.shape)
    best = {
        "a_mm": float(A[best_idx]),
        "tcu_mm": float(T[best_idx]),
        "frequency_GHz": float(F[best_idx]),
        "Vm_eff_m3": float(VM[best_idx]),
        "objective": float(OBJ[best_idx]),
    }

    def levels(Z, n=100):
        return np.linspace(np.nanmin(Z), np.nanmax(Z), n)

    fig, ax = plt.subplots(figsize=(6.2, 4.8), constrained_layout=True)
    cf = ax.contourf(A, T, OBJ, levels=levels(OBJ), cmap="viridis", extend="neither")
    if np.nanmin(F) <= config.target_frequency_ghz <= np.nanmax(F):
        cs = ax.contour(A, T, F, levels=[config.target_frequency_ghz], colors="red", linewidths=1.6)
        ax.clabel(cs, fmt={config.target_frequency_ghz: f"{config.target_frequency_ghz:.2f} GHz"}, fontsize=7)

    ax.scatter(best["a_mm"], best["tcu_mm"], marker="*", s=145, color="red", edgecolor="white", linewidth=0.6)
    ax.scatter(config.baseline["a_mm"], config.baseline["tcu_mm"], marker="o", s=45, color="white", edgecolor="black", linewidth=0.5)

    ax.set_xlabel(r"Gap width, $a$ (mm)")
    ax.set_ylabel(r"Metal thickness, $t_\mathrm{Cu}$ (mm)")
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(False)
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("Smoothed ML objective")

    fig.savefig(output_dir / "smooth_gap_metal_thickness_valley_map.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "smooth_gap_metal_thickness_valley_map.pdf", bbox_inches="tight")
    plt.close(fig)

    return best


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def run_pipeline(config: Config):
    """
    Run the full analysis pipeline.

    Returns
    -------
    dict
        Summary results and fitted model objects.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_parametric_workbook(config)
    models = train_surrogate_models(data, config)
    pca, data_pca = run_pca(data, models["X_scaled"])

    plot_diagnostics(models, output_dir)
    plot_feature_importance(models, output_dir)
    plot_pca(data_pca, pca, output_dir)
    valley_candidate = plot_smooth_valley_map(data_pca, models, config, output_dir)

    best_observed = data.loc[data["objective"].idxmin()]
    closest_frequency = data.loc[data["freq_error_GHz"].idxmin()]
    smallest_vm = data.loc[data["Vm_eff_m3"].idxmin()]

    summary = {
        "rows_used": int(len(data)),
        "metrics": models["metrics"],
        "best_observed_objective": best_observed.to_dict(),
        "closest_frequency": closest_frequency.to_dict(),
        "smallest_observed_vm": smallest_vm.to_dict(),
        "valley_candidate": valley_candidate,
        "output_dir": str(output_dir),
    }

    # Save processed data for reproducibility.
    data_pca.to_csv(output_dir / "processed_ml_dataset.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2, default=str)

    return summary


if __name__ == "__main__":
    cfg = Config(
        input_excel="Thesis Table.xlsx",
        output_dir="figures",
        target_frequency_ghz=6.46,
    )
    results = run_pipeline(cfg)

    print("Analysis complete.")
    print(f"Rows used: {results['rows_used']}")
    print("Model metrics:")
    for key, value in results["metrics"].items():
        print(f"  {key}: {value:.4g}")
    print("\nBest observed objective:")
    best = results["best_observed_objective"]
    print(f"  {best['sheet']}, {best['parameter_label']} = {best['parameter_value']:.4g} {best['unit']}")
    print(f"  f = {best['frequency_GHz']:.4g} GHz, Vm_eff = {best['Vm_eff_m3']:.4e} m^3")

    if results["valley_candidate"] is not None:
        print("\nSuggested smooth-valley candidate:")
        cand = results["valley_candidate"]
        print(f"  a = {cand['a_mm']:.3f} mm, t_Cu = {cand['tcu_mm']:.3f} mm")
        print(f"  f = {cand['frequency_GHz']:.4f} GHz, Vm_eff = {cand['Vm_eff_m3']:.4e} m^3")
