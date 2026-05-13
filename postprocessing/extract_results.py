# -*- coding: utf-8 -*-
"""
extract_results.py -- post-process the HEDE / Inconel 600 wppm sweep ODBs.

Reads results/c0_*.odb (seven files expected for the C0 sweep), writes a
single results/sweep_results.csv summary plus one stress-strain PNG per
run under results/curves/.

CSV columns:
    c0_wppm, UTS_MPa, fracture_strain_sdeg95, fracture_strain_drop50

Conventions
-----------
- mm-N-tonne-MPa unit system throughout.
- Geometry defaults to the v_smoke 1.732 x 1.732 mm in-plane domain with
  1 mm out-of-plane (plane-strain). Override via the --width / --height /
  --thickness / --applied-disp CLI flags (e.g. for the v3 4 x 4 mm mesh).
  Engineering stress is divided by the original edge area; engineering
  strain is U2 / gauge length.

RF2 sign convention (IMPORTANT)
-------------------------------
A displacement-controlled tension pull may produce NEGATIVE Sum(RF2) at
the y1 edge depending on which face of which element provides the
reaction in Abaqus's bookkeeping. After computing the engineering-stress
series, if max(sigma_eng) < 0 the entire series is sign-flipped and a
one-line WARN is printed. UTS is therefore always reported positive.

Two fracture-strain columns
---------------------------
Per the localised-fracture caveat for ~88k cohesive elements (a crack
path uses maybe 50-200 of them, so mean SDEG plateaus at ~0.002 even at
full separation), this script emits BOTH candidates:

    fracture_strain_sdeg95 -- eps at first frame where mean SDEG >= 0.95.
        Useful for diffuse / homogeneous damage; will read NaN for a
        localised polycrystal crack-path scenario.

    fracture_strain_drop50 -- eps at first dense-history sample
        STRICTLY AFTER the UTS sample where sigma_eng < 0.5 * UTS.
        Catches localised post-peak softening where SDEG-mean stays low.

Both default to NaN if no crossing is detected.

Run from project root with Abaqus's bundled Python 2.7:
    abaqus python Postprocessing/extract_results.py

Standalone Python 3 will not work (the odbAccess module is unavailable).
"""
from __future__ import print_function, division

import os
import sys
import re
import glob
import csv
import bisect
import argparse

# --- Abaqus ODB API ---------------------------------------------------------
try:
    from odbAccess import openOdb
except ImportError:
    print('FATAL: cannot import odbAccess. Run via `abaqus python`, not standalone python.')
    sys.exit(1)

# --- Optional plotting (matplotlib ships with Abaqus 2024) ------------------
try:
    import matplotlib
    matplotlib.use('Agg')   # headless backend
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
    print('WARN: matplotlib not available; PNG curves will be skipped.')


# ============================================================================
# CONFIG (v_smoke defaults; override via CLI flags for v3 baseline x-check)
# ============================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR  = os.path.join(PROJECT_ROOT, 'results')

# Geometry defaults (v_smoke 1.732 x 1.732 mm domain). main() overrides these
# from argparse so process_odb() reads the globals at run time.
WIDTH        = 1.732   # mm -- x-extent (v_smoke default)
HEIGHT       = 1.732   # mm -- y-extent / gauge length for engineering strain
THICKNESS    = 1.0     # mm -- out-of-plane (plane-strain default)
AREA         = WIDTH * THICKNESS
APPLIED_DISP = 0.433   # mm -- applied y-disp at y1 (v_smoke default;
                       #       used as U2 fallback when U field missing)

ODB_GLOB     = os.path.join(RESULTS_DIR, 'c0_*.odb')
CSV_PATH     = os.path.join(RESULTS_DIR, 'sweep_results.csv')
CURVES_DIR   = os.path.join(RESULTS_DIR, 'curves')

STEP_NAME           = 'Loading'
TOP_EDGE_CANDIDATES = ('Y1', 'y1', 'TESS.Y1', 'TESS.y1')   # ODB upcases by default
SDEG_THRESHOLD      = 0.95
DROP_FRACTION       = 0.5

NUM_TAIL_RE = re.compile(r'(\d+)\s*$')


# ============================================================================
# HELPERS
# ============================================================================

def isnan(x):
    return isinstance(x, float) and x != x


def label_from_filename(path):
    """ 'c0_5p0.odb' -> '5p0'; 'c0_5p0_up.odb' -> '5p0'; 'c0_0_local.odb' -> '0'.
        Consumes the leading wppm token (digits + optional 'p<digits>') from the
        char-3 onward; stops at the first separator. """
    base = os.path.basename(path)
    stem = base[3:-4]
    out = []
    for ch in stem:
        if ch.isdigit() or ch == 'p':
            out.append(ch)
        else:
            break
    return ''.join(out)


def label_to_c0_wppm(label):
    """ '0' -> 0.0, '0p5' -> 0.5, '5p0' -> 5.0 """
    if label == '0':
        return 0.0
    return float(label.replace('p', '.'))


def get_y1_set_and_ids(odb):
    """ Resolve top-edge node set across the candidate names.
        Returns (set_obj, set_of_node_ids, used_name). """
    set_names = list(odb.rootAssembly.nodeSets.keys())
    ns   = None
    used = None
    for cand in TOP_EDGE_CANDIDATES:
        if cand in set_names:
            ns   = odb.rootAssembly.nodeSets[cand]
            used = cand
            break
    if ns is None:
        raise KeyError('No top-edge node set among {}; assembly node sets: {}'.format(
            list(TOP_EDGE_CANDIDATES), set_names))

    # ns.nodes is either a flat tuple of OdbMeshNode (single-instance) or a
    # tuple-of-tuples (one nested tuple per contributing instance). Handle both.
    nodes_obj = ns.nodes
    ids = set()
    try:
        for n in nodes_obj:
            ids.add(n.label)
    except AttributeError:
        for inst_nodes in nodes_obj:
            for n in inst_nodes:
                ids.add(n.label)
    return ns, ids, used


def rf2_history_at_y1(odb, y1_ids):
    """ Sum RF2 across y1 node-history regions per increment.
        Returns ([times_dense], [sum_rf2_dense]) or (None, None) on failure.

        Region-name parsing strategy:
         1. Primary: rsplit('.', 1)[1] -- assumes 'Node <INSTANCE>.<LABEL>'.
         2. On first miss in an ODB, print first 5 historyRegion names for
            diagnostic, then use a trailing-integer regex fallback.
         3. If after both attempts no RF2 history matches y1_ids, abort the
            ODB cleanly with a labelled error rather than silently producing
            empty data. """
    step = odb.steps[STEP_NAME]
    region_keys = list(step.historyRegions.keys())
    by_time = {}
    matched = 0
    primary_miss_warned = False

    for region_name in region_keys:
        if not region_name.startswith('Node'):
            continue
        region = step.historyRegions[region_name]

        # Primary parser: rsplit on '.'
        node_label = None
        if '.' in region_name:
            try:
                node_label = int(region_name.rsplit('.', 1)[1])
            except ValueError:
                node_label = None

        if node_label is None:
            # First-miss diagnostic + regex fallback
            if not primary_miss_warned:
                print('  WARN: rsplit("." ,1) parser missed region {!r}; using regex fallback.'.format(region_name))
                print('         first 5 historyRegion names:')
                for r in region_keys[:5]:
                    print('           ' + repr(r))
                primary_miss_warned = True
            m = NUM_TAIL_RE.search(region_name)
            if m:
                node_label = int(m.group(1))

        if node_label is None or node_label not in y1_ids:
            continue
        if 'RF2' not in region.historyOutputs:
            continue

        for t, v in region.historyOutputs['RF2'].data:
            by_time[t] = by_time.get(t, 0.0) + v
        matched += 1

    if matched == 0:
        print('  ERROR: zero RF2 history regions matched y1 (expected {} nodes; '
              '{} regions scanned).'.format(len(y1_ids), len(region_keys)))
        print('         first 5 historyRegion names:')
        for r in region_keys[:5]:
            print('           ' + repr(r))
        return None, None

    times = sorted(by_time.keys())
    return times, [by_time[t] for t in times]


def u2_field_at_y1(odb, y1_set):
    """ Mean U2 across y1 nodes per field-output frame.
        Returns [(t, mean_u2), ...]. Empty list if missing. """
    step = odb.steps[STEP_NAME]
    out = []
    for frame in step.frames:
        try:
            u_field = frame.fieldOutputs['U']
        except KeyError:
            continue
        try:
            sub = u_field.getSubset(region=y1_set)
        except Exception as e:
            print('  WARN: U.getSubset failed at frame t={}: {}'.format(frame.frameValue, e))
            continue
        vals = [v.data[1] for v in sub.values]    # data[1] is U2 in 2D
        if not vals:
            continue
        out.append((frame.frameValue, sum(vals) / len(vals)))
    return out


def sdeg_field_mean(odb):
    """ Mean SDEG across all values per field-output frame.
        Returns [(t, mean_sdeg), ...]. Empty list if missing.

        Note: SDEG only exists on cohesive elements, so averaging over all
        SDEG values is implicitly an average over the cohesive region only.
        With ~88k COH2D4T elements and a localised crack path, expect this
        mean to plateau at ~0.002 -- the drop50 column compensates. """
    step = odb.steps[STEP_NAME]
    out = []
    for frame in step.frames:
        try:
            field = frame.fieldOutputs['SDEG']
        except KeyError:
            continue
        vals = [v.data for v in field.values]
        if not vals:
            continue
        out.append((frame.frameValue, sum(vals) / len(vals)))
    return out


def make_interp(xs, ys):
    """ Linear interpolation, clamped at endpoints. Returns f(x). """
    if not xs:
        return lambda x: float('nan')
    def f(x):
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect.bisect_left(xs, x)
        x0, x1 = xs[i-1], xs[i]
        y0, y1 = ys[i-1], ys[i]
        if x1 == x0:
            return y0
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return f


def maybe_sign_flip(sigma):
    """ If max(sigma) < 0 the whole series is flipped (warning printed) so
        UTS is always reported positive. Returns (sigma_possibly_flipped, flipped_bool). """
    if not sigma:
        return sigma, False
    if max(sigma) < 0:
        print('  WARN: max(sigma_eng) < 0; flipping whole RF2-derived series so UTS is positive.')
        return [-x for x in sigma], True
    return sigma, False


def compute_uts(sigma_dense):
    """ Returns (UTS, index_in_dense_series) or (NaN, -1). """
    if not sigma_dense:
        return float('nan'), -1
    i = max(range(len(sigma_dense)), key=lambda j: sigma_dense[j])
    return sigma_dense[i], i


def compute_drop50(t_dense, sigma_dense, uts, i_uts, eps_t_interp):
    """ eps at first dense sample STRICTLY AFTER the UTS index where
        sigma_eng < DROP_FRACTION * UTS. NaN if no crossing.

        Strain at the dense post-peak time is obtained by linearly
        interpolating the sparse U2 series onto that dense timestamp. """
    if isnan(uts) or uts <= 0 or i_uts < 0:
        return float('nan')
    threshold = DROP_FRACTION * uts
    for i in range(i_uts + 1, len(sigma_dense)):
        if sigma_dense[i] < threshold:
            return eps_t_interp(t_dense[i])
    return float('nan')


def compute_sdeg95(t_eps_sparse, eps_sparse, t_sd_sparse, sd_sparse):
    """ eps at first sparse frame where mean SDEG >= SDEG_THRESHOLD.
        SDEG and U2 share field-output cadence so we match exactly by time;
        a small numeric tolerance handles any float drift. NaN if no crossing. """
    if not t_sd_sparse:
        return float('nan')
    eps_by_t = dict(zip(t_eps_sparse, eps_sparse))
    for t, sd in zip(t_sd_sparse, sd_sparse):
        if sd >= SDEG_THRESHOLD:
            if t in eps_by_t:
                return eps_by_t[t]
            for tt in eps_by_t:
                if abs(tt - t) < 1e-9:
                    return eps_by_t[tt]
            return float('nan')
    return float('nan')


def plot_curve(label, c0, eps_sparse, sigma_at_sparse, uts, eps_drop50, eps_sdeg95):
    """ Plot uses sparse-paired (eps, sigma) data; UTS marker placed at the
        dense max (which may exceed any sparse-pair sigma since sparse cadence
        can miss the actual peak). """
    if not HAVE_MPL:
        return None
    if not eps_sparse or not sigma_at_sparse:
        print('  WARN: empty sparse data; skipping plot.')
        return None
    if not os.path.exists(CURVES_DIR):
        os.makedirs(CURVES_DIR)

    fig = plt.figure(figsize=(7.5, 5.5))
    ax  = fig.add_subplot(111)
    ax.plot(eps_sparse, sigma_at_sparse, 'b-o', linewidth=1.5, markersize=4,
            label='sigma vs eps (sparse-paired)')
    if not isnan(uts):
        ax.axhline(uts, color='red', linestyle=':', linewidth=1,
                   label='UTS (dense max) = {:.1f} MPa'.format(uts))
    if not isnan(eps_drop50):
        ax.axvline(eps_drop50, color='orange', linestyle='--', linewidth=1,
                   label='eps_f drop-50 = {:.4f}'.format(eps_drop50))
    if not isnan(eps_sdeg95):
        ax.axvline(eps_sdeg95, color='green', linestyle='--', linewidth=1,
                   label='eps_f SDEG-95 = {:.4f}'.format(eps_sdeg95))
    ax.set_xlabel('Engineering strain  eps  (-)')
    ax.set_ylabel('Engineering stress  sigma  (MPa)')
    ax.set_title('HEDE Inconel 600  --  C0 = {} wppm'.format(c0))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()

    out = os.path.join(CURVES_DIR, 'c0_{}.png'.format(label))
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ============================================================================
# ORCHESTRATOR
# ============================================================================

def process_odb(odb_path):
    """ Returns (c0_wppm, UTS_MPa, frac_strain_sdeg95, frac_strain_drop50)
        or None on hard failure. Soft failures emit NaN inside the row. """
    label = label_from_filename(odb_path)
    c0    = label_to_c0_wppm(label)
    print('\n--- {} (C0 = {} wppm) ---'.format(odb_path, c0))

    try:
        odb = openOdb(odb_path, readOnly=True)
    except Exception as e:
        print('  ERROR: openOdb failed: {}'.format(e))
        return None

    try:
        if STEP_NAME not in odb.steps:
            print('  ERROR: step {!r} missing; available steps: {}'.format(
                STEP_NAME, list(odb.steps.keys())))
            return None

        try:
            y1_set, y1_ids, y1_used = get_y1_set_and_ids(odb)
            print('  y1 set: name={!r}, {} nodes resolved'.format(y1_used, len(y1_ids)))
        except KeyError as e:
            print('  ERROR: {}'.format(e))
            return None

        # Dense RF2 -> stress (with sign-flip guard)
        t_dense, rf2_dense = rf2_history_at_y1(odb, y1_ids)
        if t_dense is None:
            return None
        sigma_dense_raw     = [r / AREA for r in rf2_dense]
        sigma_dense, flipped = maybe_sign_flip(sigma_dense_raw)

        # Sparse U2 -> strain (analytical fallback if U field missing)
        u2_pairs = u2_field_at_y1(odb, y1_set)
        if not u2_pairs:
            print('  WARN: U field output missing/empty at y1; falling back to '
                  'U2 = APPLIED_DISP * t/STEP_T = {} * t  (STEP_T=1.0 s ramp).'.format(APPLIED_DISP))
            u2_pairs = [(t, APPLIED_DISP * t) for t in t_dense]
        t_eps_sparse = [t for t, _ in u2_pairs]
        eps_sparse   = [u / HEIGHT for _, u in u2_pairs]

        # Sparse SDEG (mean over all SDEG values per frame)
        sdeg_pairs = sdeg_field_mean(odb)
        if sdeg_pairs:
            t_sd_sparse = [t for t, _ in sdeg_pairs]
            sd_sparse   = [v for _, v in sdeg_pairs]
        else:
            print('  WARN: SDEG field output missing/empty; fracture_strain_sdeg95 will be NaN.')
            t_sd_sparse = []
            sd_sparse   = []

        # Compute scalars
        uts, i_uts   = compute_uts(sigma_dense)
        eps_t_interp = make_interp(t_eps_sparse, eps_sparse)
        eps_drop50   = compute_drop50(t_dense, sigma_dense, uts, i_uts, eps_t_interp)
        eps_sdeg95   = compute_sdeg95(t_eps_sparse, eps_sparse, t_sd_sparse, sd_sparse)
        # Truncated: UTS lies at the last dense-history sample, meaning the
        # curve was still rising when the analysis terminated. Quick filter
        # only; the substantive call is the curve-shape audit on the
        # Results/curves/c0_*.png plots.
        truncated = 'yes' if (i_uts >= 0 and i_uts == len(sigma_dense) - 1) else 'no'

        # Sparse-paired data for the plot: sigma at each sparse t = interp dense
        sigma_t_interp  = make_interp(t_dense, sigma_dense)
        sigma_at_sparse = [sigma_t_interp(t) for t in t_eps_sparse]
        plot_path       = plot_curve(label, c0, eps_sparse, sigma_at_sparse,
                                     uts, eps_drop50, eps_sdeg95)

        flip_tag = '  (sign-flipped)' if flipped else ''
        print('  UTS                    = {:.4f} MPa{}'.format(uts, flip_tag))
        print('  truncated              = {}'.format(truncated))
        print('  fracture_strain_sdeg95 = {}'.format(eps_sdeg95))
        print('  fracture_strain_drop50 = {}'.format(eps_drop50))
        if plot_path:
            print('  plot                   = {}'.format(plot_path))

        return (c0, uts, eps_sdeg95, eps_drop50, truncated)
    finally:
        try:
            odb.close()
        except Exception:
            pass


def parse_args():
    p = argparse.ArgumentParser(description='Extract HEDE wppm-sweep ODB results to CSV')
    p.add_argument('--width', type=float, default=WIDTH,
                   help='Domain x-extent in mm (default: %(default)s)')
    p.add_argument('--height', type=float, default=HEIGHT,
                   help='Domain y-extent / gauge length in mm (default: %(default)s)')
    p.add_argument('--thickness', type=float, default=THICKNESS,
                   help='Out-of-plane thickness in mm (default: %(default)s)')
    p.add_argument('--applied-disp', type=float, default=APPLIED_DISP,
                   help='Applied y-disp at y1, mm (U2 fallback if U field missing)')
    p.add_argument('--odb-glob', default=ODB_GLOB,
                   help='Glob for ODB files (default: %(default)s)')
    p.add_argument('--csv-out', default=CSV_PATH,
                   help='Output CSV path (default: %(default)s)')
    return p.parse_args()


def main():
    global WIDTH, HEIGHT, THICKNESS, AREA, APPLIED_DISP
    args = parse_args()
    WIDTH        = args.width
    HEIGHT       = args.height
    THICKNESS    = args.thickness
    AREA         = WIDTH * THICKNESS
    APPLIED_DISP = args.applied_disp
    odb_glob     = args.odb_glob
    csv_path     = args.csv_out

    print('Geometry: WIDTH={} mm  HEIGHT={} mm  THICKNESS={} mm  '
          'AREA={} mm^2  APPLIED_DISP={} mm'
          .format(WIDTH, HEIGHT, THICKNESS, AREA, APPLIED_DISP))

    odb_paths = sorted(glob.glob(odb_glob))
    if not odb_paths:
        print('ERROR: no ODBs match {!r}'.format(odb_glob))
        sys.exit(2)
    print('Found {} ODB(s):'.format(len(odb_paths)))
    for p in odb_paths:
        print('  ' + p)

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    rows = []
    for odb_path in odb_paths:
        result = process_odb(odb_path)
        if result is not None:
            rows.append(result)

    rows.sort(key=lambda r: r[0])

    # Py2 csv wants binary mode; Py3 csv wants text mode with newline='' to avoid
    # extra blank lines on Windows. Abaqus 2024 ships Py2.7; LE2025 ships Py3.
    if sys.version_info[0] >= 3:
        f = open(csv_path, 'w', newline='')
    else:
        f = open(csv_path, 'wb')
    try:
        w = csv.writer(f)
        w.writerow(['c0_wppm', 'UTS_MPa', 'fracture_strain_sdeg95',
                    'fracture_strain_drop50', 'truncated'])
        for r in rows:
            w.writerow(r)
    finally:
        f.close()

    print('\nWrote {} ({} rows; {} ODB(s) failed).'.format(
        csv_path, len(rows), len(odb_paths) - len(rows)))


if __name__ == '__main__':
    main()
