#!/usr/bin/env python3
"""
analysis.py -- ML surrogate analysis for the HEDE Inconel 600 wppm sweep.

Reads results/sweep_results.csv (or sweep_results_SYNTHETIC.csv with
--synthetic) and produces:

    results/rf_uts.pkl              fitted RandomForestRegressor (joblib)
    results/rf_fs.pkl               fitted RandomForestRegressor (joblib)
    results/predictions.csv         LOO CV per held-out point
    results/loo_residuals.png       LOO predicted-vs-actual scatter (headline diagnostic)
    results/surrogate_curves.png    dense RF prediction over [0, 5] wppm + scatter
    results/shap_uts.png            single-feature SHAP dependence plot
    results/shap_fs.png             single-feature SHAP dependence plot
    results/analysis_report.md      auto-generated factual summary

ML scope (locked, supervisor-approved)
--------------------------------------
- Single input feature: c0_wppm
- Two output targets: UTS_MPa and one fracture_strain column (drop50 by
  default; sdeg95 fallback if drop50 is all-NaN)
- Method: Random Forest regression + SHAP TreeExplainer
- 7 data points -- this is a surrogate proof-of-concept, NOT a
  generalisable trained model

LOO metrics caveat
------------------
Full-LOO R^2 on n=7 is dominated by the C0=0 and C0=max endpoint folds,
which are extrapolation tests for a tree-based interpolator. A separate
"interior LOO" metric is computed on the 5 remaining folds; that is the
meaningful surrogate-quality indicator at this dataset size. Both
metrics are reported, with all 7 points shown in loo_residuals.png.

Run from a normal Python 3 venv (NOT abaqus python):
    python Postprocessing/analysis.py             # reads sweep_results.csv
    python Postprocessing/analysis.py --synthetic # reads sweep_results_SYNTHETIC.csv
"""
from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Hard deps (sklearn ships joblib internally but we import it explicitly).
try:
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import r2_score, mean_squared_error
except ImportError as e:
    print('FATAL: required dependency missing ({}). Install scikit-learn + joblib.'.format(e))
    sys.exit(1)

# Optional deps -- script degrades gracefully if missing.
try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False
    print('WARN: shap not available; SHAP plots and importance scalars will be skipped.')

try:
    import matplotlib
    matplotlib.use('Agg')   # headless backend
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
    print('WARN: matplotlib not available; all PNG outputs will be skipped.')


# ============================================================================
# CONFIG
# ============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR  = PROJECT_ROOT / 'results'

REAL_CSV  = RESULTS_DIR / 'sweep_results.csv'
SYNTH_CSV = RESULTS_DIR / 'sweep_results_SYNTHETIC.csv'

REQUIRED_COLS = ['c0_wppm', 'UTS_MPa', 'fracture_strain_sdeg95', 'fracture_strain_drop50']
FS_PREFERENCE = ['fracture_strain_drop50', 'fracture_strain_sdeg95']

RF_KWARGS = dict(n_estimators=100, max_depth=None, random_state=42, n_jobs=1)

# Synthetic data per the supervisor-approved spec.
SYNTHETIC_DATA = pd.DataFrame({
    'c0_wppm':                [0.0,    0.5,    1.0,    1.5,    2.0,    3.0,    5.0],
    'UTS_MPa':                [950.0,  820.0,  720.0,  640.0,  580.0,  490.0,  380.0],
    'fracture_strain_sdeg95': [np.nan, np.nan, 0.115,  0.092,  0.078,  0.061,  0.043],
    'fracture_strain_drop50': [0.180,  0.140,  0.115,  0.095,  0.080,  0.063,  0.045],
})


# ============================================================================
# DATA LOADING / VALIDATION
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='ML surrogate analysis for the HEDE wppm sweep.')
    p.add_argument('--synthetic', action='store_true',
                   help='Read sweep_results_SYNTHETIC.csv instead of the real CSV. '
                        'Auto-creates the synthetic file if missing; never overwrites an existing one.')
    p.add_argument('--csv', default=None,
                   help='Explicit input CSV path (overrides --synthetic and the default sweep_results.csv).')
    return p.parse_args()


def ensure_synthetic_csv(path: Path):
    """Write SYNTHETIC_DATA to `path` only if it does not already exist."""
    if path.exists():
        print('INFO: synthetic CSV already exists at {}; not overwriting'.format(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    SYNTHETIC_DATA.to_csv(path, index=False)
    print('INFO: wrote synthetic CSV {} ({} rows)'.format(path, len(SYNTHETIC_DATA)))


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load + validate the sweep CSV. Exits non-zero with labelled ERROR on any failure."""
    if not csv_path.exists():
        print('ERROR: input CSV missing: {}'.format(csv_path))
        sys.exit(1)

    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print('ERROR: required columns missing from {}: {}'.format(csv_path, missing))
        print('       columns present: {}'.format(list(df.columns)))
        sys.exit(1)

    if len(df) < 3:
        print('ERROR: {} has {} rows; need at least 3.'.format(csv_path, len(df)))
        sys.exit(1)
    if len(df) < 7:
        print('INFO: running with n={} samples (full sweep is n=7).'.format(len(df)))

    if not pd.api.types.is_numeric_dtype(df['c0_wppm']):
        print('ERROR: c0_wppm must be numeric; got dtype {}'.format(df['c0_wppm'].dtype))
        sys.exit(1)

    if df['c0_wppm'].duplicated().any():
        dups = df.loc[df['c0_wppm'].duplicated(), 'c0_wppm'].tolist()
        print('ERROR: c0_wppm has duplicate values: {}'.format(dups))
        sys.exit(1)

    df = df.sort_values('c0_wppm').reset_index(drop=True)
    print('INFO: loaded {} rows from {}'.format(len(df), csv_path))
    return df


def pick_fs_column(df: pd.DataFrame):
    """Return the preferred fracture_strain column name, or None if both are all-NaN."""
    for col in FS_PREFERENCE:
        if df[col].notna().any():
            print('INFO: fracture_strain primary target = {}'.format(col))
            return col
    print('WARN: both fracture_strain columns are all-NaN; skipping fs target.')
    return None


# ============================================================================
# MODEL FIT + LOO CV
# ============================================================================

def fit_full_model(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    rf = RandomForestRegressor(**RF_KWARGS)
    rf.fit(X, y)
    return rf


def loo_predict(X: np.ndarray, y: np.ndarray):
    """Run LeaveOneOut CV. Returns (preds, acts, c0s) parallel ndarrays in fold order."""
    loo = LeaveOneOut()
    preds, acts, c0s = [], [], []
    for tr, te in loo.split(X):
        rf = RandomForestRegressor(**RF_KWARGS)
        rf.fit(X[tr], y[tr])
        preds.append(float(rf.predict(X[te])[0]))
        acts.append(float(y[te][0]))
        c0s.append(float(X[te, 0][0]))
    return np.array(preds), np.array(acts), np.array(c0s)


def safe_metrics(actual: np.ndarray, predicted: np.ndarray):
    """R^2 + RMSE; returns (nan, nan) if fewer than 2 points."""
    if len(actual) < 2:
        return float('nan'), float('nan')
    return float(r2_score(actual, predicted)), float(np.sqrt(mean_squared_error(actual, predicted)))


def interior_metrics(c0s: np.ndarray, actual: np.ndarray, predicted: np.ndarray,
                     c0_min: float, c0_max: float):
    """LOO metrics on interior folds only (drop endpoint folds at c0_min and c0_max)."""
    mask = ~np.isclose(c0s, c0_min) & ~np.isclose(c0s, c0_max)
    n_int = int(mask.sum())
    if n_int < 2:
        return float('nan'), float('nan'), n_int
    r2, rmse = safe_metrics(actual[mask], predicted[mask])
    return r2, rmse, n_int


# ============================================================================
# SHAP
# ============================================================================

def shap_values_single_feature(rf: RandomForestRegressor, X: np.ndarray):
    """Returns shap_values (n_samples, 1) ndarray, or None on shap unavailable / failure."""
    if not HAVE_SHAP:
        return None
    try:
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X)
        # For RF regressor, shap_values is an ndarray of shape (n, n_features). Force 2-D.
        sv = np.asarray(sv)
        if sv.ndim == 1:
            sv = sv.reshape(-1, 1)
        return sv
    except Exception as e:
        print('WARN: SHAP TreeExplainer failed: {}'.format(e))
        return None


# ============================================================================
# PLOTTING
# ============================================================================

def plot_loo_residuals(c0_uts, act_uts, pred_uts,
                       c0_fs, act_fs, pred_fs, fs_col,
                       out_path: Path):
    """Predicted-vs-actual scatter, two stacked panels (UTS top, fs bottom).
       Diagonal y=x reference. Each point labelled with its C0 value."""
    if not HAVE_MPL:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))

    # --- UTS panel ---
    ax = axes[0]
    if len(act_uts) > 0:
        lo = float(min(act_uts.min(), pred_uts.min()))
        hi = float(max(act_uts.max(), pred_uts.max()))
        margin = max(0.05 * (hi - lo), 1.0)
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', linewidth=0.7, label='y = x')
        ax.scatter(act_uts, pred_uts, s=80, c='steelblue', edgecolor='k', linewidth=0.5,
                   label='LOO predictions')
        for c0, a, p in zip(c0_uts, act_uts, pred_uts):
            ax.annotate('C0={:g}'.format(c0), (a, p),
                        xytext=(6, 6), textcoords='offset points', fontsize=8)
    ax.set_xlabel('UTS actual (MPa)')
    ax.set_ylabel('UTS LOO-predicted (MPa)')
    ax.set_title('UTS_MPa  --  LOO predicted vs actual (n={})'.format(len(act_uts)))
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    # --- FS panel ---
    ax = axes[1]
    if fs_col is not None and len(act_fs) > 0:
        lo = float(min(act_fs.min(), pred_fs.min()))
        hi = float(max(act_fs.max(), pred_fs.max()))
        margin = max(0.05 * (hi - lo), 0.001)
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', linewidth=0.7, label='y = x')
        ax.scatter(act_fs, pred_fs, s=80, c='salmon', edgecolor='k', linewidth=0.5,
                   label='LOO predictions')
        for c0, a, p in zip(c0_fs, act_fs, pred_fs):
            ax.annotate('C0={:g}'.format(c0), (a, p),
                        xytext=(6, 6), textcoords='offset points', fontsize=8)
        ax.set_xlabel('fs actual')
        ax.set_ylabel('fs LOO-predicted')
        ax.set_title('fracture_strain ({})  --  LOO predicted vs actual (n={})'.format(
            fs_col, len(act_fs)))
    else:
        ax.text(0.5, 0.5, 'fs target skipped', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('fracture_strain  --  N/A')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    fig.text(0.5, 0.01,
             'LOO predicted-vs-actual diagnostic. Endpoint C0 holdouts are extrapolation; '
             'interior C0 holdouts are interpolation. n=7 training points.',
             ha='center', fontsize=8, style='italic')
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('INFO: wrote {}'.format(out_path))
    return out_path


def plot_shap_dependence(c0_array, shap_values, target_name, n_points, out_path: Path):
    """Manual single-feature dependence plot: scatter of c0 vs SHAP value."""
    if not HAVE_MPL or shap_values is None:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(c0_array, shap_values[:, 0], s=80, edgecolor='k', linewidth=0.5, c='steelblue')
    ax.axhline(0, color='grey', linewidth=0.5, linestyle=':')
    ax.set_xlabel('C0 (wppm)')
    ax.set_ylabel('SHAP value (impact on prediction)')
    ax.set_title('{}: SHAP dependence (n={} points)'.format(target_name, n_points))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('INFO: wrote {}'.format(out_path))
    return out_path


def plot_surrogate_curves(rf_uts, rf_fs, df: pd.DataFrame, fs_col, out_path: Path):
    """Two-panel: dense RF prediction over [0,5] wppm + 7 training scatter points."""
    if not HAVE_MPL:
        return None
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    c0_grid = np.linspace(0.0, 5.0, 100).reshape(-1, 1)

    # UTS panel
    axes[0].plot(c0_grid.ravel(), rf_uts.predict(c0_grid),
                 'b-', linewidth=1.5, label='RF surrogate')
    axes[0].scatter(df['c0_wppm'], df['UTS_MPa'],
                    s=80, c='red', edgecolor='k', linewidth=0.5,
                    label='Training data (n={})'.format(len(df)))
    axes[0].set_ylabel('UTS (MPa)')
    axes[0].set_title('UTS_MPa')
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc='best', fontsize=9)

    # fs panel
    if fs_col is not None and rf_fs is not None:
        axes[1].plot(c0_grid.ravel(), rf_fs.predict(c0_grid),
                     'b-', linewidth=1.5, label='RF surrogate')
        df_fs = df.dropna(subset=[fs_col])
        axes[1].scatter(df_fs['c0_wppm'], df_fs[fs_col],
                        s=80, c='red', edgecolor='k', linewidth=0.5,
                        label='Training data (n={})'.format(len(df_fs)))
        axes[1].set_ylabel('Fracture strain ({})'.format(fs_col))
        axes[1].set_title('fracture_strain ({})'.format(fs_col))
        axes[1].legend(loc='best', fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'fs target skipped (all-NaN)',
                     ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('fracture_strain  --  N/A')
    axes[1].set_xlabel('C0 (wppm)')
    axes[1].grid(alpha=0.3)

    fig.text(0.5, 0.01, 'Surrogate proof-of-concept; n=7 training points.',
             ha='center', fontsize=8, style='italic')
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('INFO: wrote {}'.format(out_path))
    return out_path


# ============================================================================
# OUTPUT WRITERS
# ============================================================================

def write_predictions_csv(df, c0_uts, act_uts, pred_uts,
                          c0_fs, act_fs, pred_fs, fs_col,
                          out_path: Path) -> pd.DataFrame:
    """Single CSV with NaN for fs rows that were skipped (UTS rows always populated)."""
    out = pd.DataFrame({'c0_wppm': df['c0_wppm'].values})

    uts_act_map  = {float(c): float(a) for c, a in zip(c0_uts, act_uts)}
    uts_pred_map = {float(c): float(p) for c, p in zip(c0_uts, pred_uts)}
    out['UTS_actual']    = out['c0_wppm'].map(uts_act_map)
    out['UTS_predicted'] = out['c0_wppm'].map(uts_pred_map)

    if fs_col is not None and len(c0_fs) > 0:
        fs_act_map  = {float(c): float(a) for c, a in zip(c0_fs, act_fs)}
        fs_pred_map = {float(c): float(p) for c, p in zip(c0_fs, pred_fs)}
        out['fs_actual']    = out['c0_wppm'].map(fs_act_map)
        out['fs_predicted'] = out['c0_wppm'].map(fs_pred_map)
    else:
        out['fs_actual']    = np.nan
        out['fs_predicted'] = np.nan

    out.to_csv(out_path, index=False)
    print('INFO: wrote {}'.format(out_path))
    return out


def df_to_markdown_table(df: pd.DataFrame) -> str:
    """Manual markdown-table builder (avoids the pandas->tabulate optional dep)."""
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |',
             '| ' + ' | '.join('---' for _ in cols) + ' |']
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append('NaN')
            elif c == 'c0_wppm':
                cells.append('{:g}'.format(v))
            elif 'UTS' in c:
                cells.append('{:.2f}'.format(v))
            elif 'fs' in c:
                cells.append('{:.5f}'.format(v))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def fmt_metric(x, fmt_str='{:.4f}'):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'N/A'
    return fmt_str.format(x)


def write_report(out_path: Path, input_csv: Path, df: pd.DataFrame, fs_col,
                 n_uts, n_fs,
                 r2_uts_full, rmse_uts_full, r2_uts_int, rmse_uts_int, n_uts_int,
                 r2_fs_full, rmse_fs_full, r2_fs_int, rmse_fs_int, n_fs_int,
                 mean_abs_shap_uts, mean_abs_shap_fs,
                 predictions_df: pd.DataFrame):
    iso_now = datetime.datetime.now().isoformat(timespec='seconds')

    # Build per-point residuals table.
    pred_for_report = predictions_df.copy()
    pred_for_report['UTS_resid'] = pred_for_report['UTS_actual'] - pred_for_report['UTS_predicted']
    if 'fs_actual' in pred_for_report.columns and pred_for_report['fs_actual'].notna().any():
        pred_for_report['fs_resid'] = pred_for_report['fs_actual'] - pred_for_report['fs_predicted']

    lines = []
    lines.append('# HEDE Inconel 600 -- RF surrogate analysis')
    lines.append('')
    lines.append('Generated: {}'.format(iso_now))
    lines.append('Source: `{}`'.format(input_csv))
    lines.append('')
    lines.append('## Data summary')
    lines.append('- N points (UTS): {}'.format(n_uts))
    if fs_col is not None:
        lines.append('- N points (fs): {}  (target column: `{}`)'.format(n_fs, fs_col))
    else:
        lines.append('- N points (fs): 0  (skipped: both fs columns all-NaN)')
    lines.append('- C0 range: [{:g}, {:g}] wppm'.format(df['c0_wppm'].min(), df['c0_wppm'].max()))
    lines.append('- UTS_MPa range: [{:.1f}, {:.1f}]'.format(df['UTS_MPa'].min(), df['UTS_MPa'].max()))
    if fs_col is not None:
        fs_vals = df[fs_col].dropna()
        lines.append('- fs range: [{:.4f}, {:.4f}]'.format(fs_vals.min(), fs_vals.max()))
    lines.append('')
    lines.append('## Model config')
    lines.append('`RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=1)`')
    lines.append('')
    lines.append('## LOO cross-validation')
    lines.append('| target | full LOO R^2 | interior LOO R^2 | full RMSE | interior RMSE |')
    lines.append('|---|---|---|---|---|')
    lines.append('| UTS_MPa (n={} / interior n={}) | {} | {} | {} | {} |'.format(
        n_uts, n_uts_int,
        fmt_metric(r2_uts_full, '{:.3f}'),
        fmt_metric(r2_uts_int,  '{:.3f}'),
        fmt_metric(rmse_uts_full, '{:.3f}'),
        fmt_metric(rmse_uts_int,  '{:.3f}')))
    if fs_col is not None:
        lines.append('| fracture_strain ({}) (n={} / interior n={}) | {} | {} | {} | {} |'.format(
            fs_col, n_fs, n_fs_int,
            fmt_metric(r2_fs_full, '{:.3f}'),
            fmt_metric(r2_fs_int,  '{:.3f}'),
            fmt_metric(rmse_fs_full, '{:.5f}'),
            fmt_metric(rmse_fs_int,  '{:.5f}')))
    else:
        lines.append('| fracture_strain | N/A (target skipped) | N/A | N/A | N/A |')
    lines.append('')
    lines.append('### Per-point residuals')
    lines.append(df_to_markdown_table(pred_for_report))
    lines.append('')
    lines.append('## SHAP feature importance (single feature)')
    lines.append('- UTS_MPa: mean |SHAP| = {} MPa'.format(fmt_metric(mean_abs_shap_uts, '{:.3f}')))
    lines.append('- fracture_strain: mean |SHAP| = {}'.format(fmt_metric(mean_abs_shap_fs, '{:.5f}')))
    lines.append('')
    lines.append('## Caveats')
    lines.append('- n=7 training points. LOO R^2 is barely interpretable at this size.')
    lines.append('- Full LOO includes the C0=0 and C0=5 endpoint holdouts, which are extrapolation tests for a tree-based interpolator. Interior LOO removes both endpoint folds and tests interpolation only. The interior metric is the meaningful surrogate quality indicator at this dataset size.')
    lines.append('- See `loo_residuals.png` for visual diagnosis.')
    lines.append('- No held-out test set beyond LOO.')
    lines.append('- Single input feature (c0_wppm) by design (supervisor scope lock).')
    lines.append('- This is a surrogate proof-of-concept, not a generalisable trained model.')

    out_path.write_text('\n'.join(lines) + '\n')
    print('INFO: wrote {}'.format(out_path))


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def main():
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.csv:
        input_csv = Path(args.csv).resolve()
    elif args.synthetic:
        ensure_synthetic_csv(SYNTH_CSV)
        input_csv = SYNTH_CSV
    else:
        input_csv = REAL_CSV

    df = load_data(input_csv)
    fs_col = pick_fs_column(df)

    # ---------- UTS ----------
    X_uts = df[['c0_wppm']].values.astype(float)
    y_uts = df['UTS_MPa'].values.astype(float)
    n_uts = len(X_uts)

    rf_uts = fit_full_model(X_uts, y_uts)
    joblib.dump(rf_uts, RESULTS_DIR / 'rf_uts.pkl')
    print('INFO: wrote {}'.format(RESULTS_DIR / 'rf_uts.pkl'))

    pred_uts, act_uts, c0_uts = loo_predict(X_uts, y_uts)
    r2_uts_full, rmse_uts_full = safe_metrics(act_uts, pred_uts)
    c0_min, c0_max = float(df['c0_wppm'].min()), float(df['c0_wppm'].max())
    r2_uts_int, rmse_uts_int, n_uts_int = interior_metrics(c0_uts, act_uts, pred_uts, c0_min, c0_max)

    # ---------- fracture strain ----------
    rf_fs = None
    pred_fs = act_fs = c0_fs = np.array([])
    n_fs = 0
    r2_fs_full = rmse_fs_full = float('nan')
    r2_fs_int  = rmse_fs_int  = float('nan')
    n_fs_int   = 0
    mean_abs_shap_fs = float('nan')
    X_fs = None

    if fs_col is not None:
        df_fs = df.dropna(subset=[fs_col]).reset_index(drop=True)
        n_dropped = len(df) - len(df_fs)
        if n_dropped > 0:
            print('WARN: dropped {} NaN rows from fs fit; using {} points'.format(n_dropped, len(df_fs)))

        if len(df_fs) >= 3:
            X_fs = df_fs[['c0_wppm']].values.astype(float)
            y_fs = df_fs[fs_col].values.astype(float)
            n_fs = len(X_fs)

            rf_fs = fit_full_model(X_fs, y_fs)
            joblib.dump(rf_fs, RESULTS_DIR / 'rf_fs.pkl')
            print('INFO: wrote {}'.format(RESULTS_DIR / 'rf_fs.pkl'))

            pred_fs, act_fs, c0_fs = loo_predict(X_fs, y_fs)
            r2_fs_full, rmse_fs_full = safe_metrics(act_fs, pred_fs)
            r2_fs_int, rmse_fs_int, n_fs_int = interior_metrics(
                c0_fs, act_fs, pred_fs,
                float(df_fs['c0_wppm'].min()), float(df_fs['c0_wppm'].max()))
        else:
            print('WARN: fs n={} after dropna; need >= 3 to fit. Skipping fs model.'.format(len(df_fs)))
            fs_col = None

    # ---------- predictions.csv ----------
    pred_df = write_predictions_csv(df, c0_uts, act_uts, pred_uts,
                                    c0_fs, act_fs, pred_fs, fs_col,
                                    RESULTS_DIR / 'predictions.csv')

    # ---------- SHAP ----------
    mean_abs_shap_uts = float('nan')
    if HAVE_SHAP:
        shap_uts = shap_values_single_feature(rf_uts, X_uts)
        if shap_uts is not None:
            mean_abs_shap_uts = float(np.abs(shap_uts).mean())
            plot_shap_dependence(X_uts[:, 0], shap_uts, 'UTS_MPa', n_uts,
                                 RESULTS_DIR / 'shap_uts.png')
        if fs_col is not None and rf_fs is not None and X_fs is not None:
            shap_fs = shap_values_single_feature(rf_fs, X_fs)
            if shap_fs is not None:
                mean_abs_shap_fs = float(np.abs(shap_fs).mean())
                plot_shap_dependence(X_fs[:, 0], shap_fs,
                                     'fracture_strain ({})'.format(fs_col), n_fs,
                                     RESULTS_DIR / 'shap_fs.png')

    # ---------- plots ----------
    plot_surrogate_curves(rf_uts, rf_fs, df, fs_col, RESULTS_DIR / 'surrogate_curves.png')
    plot_loo_residuals(c0_uts, act_uts, pred_uts,
                       c0_fs, act_fs, pred_fs, fs_col,
                       RESULTS_DIR / 'loo_residuals.png')

    # ---------- report ----------
    write_report(RESULTS_DIR / 'analysis_report.md', input_csv, df, fs_col,
                 n_uts, n_fs,
                 r2_uts_full, rmse_uts_full, r2_uts_int, rmse_uts_int, n_uts_int,
                 r2_fs_full, rmse_fs_full, r2_fs_int, rmse_fs_int, n_fs_int,
                 mean_abs_shap_uts, mean_abs_shap_fs,
                 pred_df)

    # ---------- stdout headline ----------
    print()
    print('=' * 64)
    print('HEADLINE METRICS')
    print('=' * 64)
    print('UTS_MPa  full  LOO  R^2 = {:.3f}, RMSE = {:.3f} (n={})'.format(
        r2_uts_full, rmse_uts_full, n_uts))
    print('         interior LOO R^2 = {:.3f}, RMSE = {:.3f} (n={})'.format(
        r2_uts_int, rmse_uts_int, n_uts_int))
    if fs_col is not None:
        print('fs ({})  full  LOO  R^2 = {:.3f}, RMSE = {:.5f} (n={})'.format(
            fs_col, r2_fs_full, rmse_fs_full, n_fs))
        print('         interior LOO R^2 = {:.3f}, RMSE = {:.5f} (n={})'.format(
            r2_fs_int, rmse_fs_int, n_fs_int))
    else:
        print('fs: skipped (target all-NaN)')
    print()
    print('NOTE: R^2 on n=7 LOO is barely interpretable; full-LOO is dominated by the')
    print('      C0={:g} and C0={:g} endpoint folds (extrapolation). Interior LOO is the'.format(
        c0_min, c0_max))
    print('      meaningful surrogate-quality indicator. See analysis_report.md and loo_residuals.png.')


if __name__ == '__main__':
    main()
