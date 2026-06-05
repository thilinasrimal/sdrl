"""
preprocess.py  (v4 — memory-safe, per-file streaming)
------------------------------------------------------
Processes each shipborne .grd file independently — never builds a
global HR grid. Peak RAM usage is one file at a time (~tens of MB).

Upload to: /content/drive/MyDrive/sdrl_gravity/preprocess.py
"""

import os, glob, gc
import numpy as np
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from skimage.metrics import structural_similarity as ssim_metric

# ── paths ─────────────────────────────────────────────────────────────────
BASE       = '/content/drive/MyDrive/sdrl_gravity'
SAT_PATH   = f'{BASE}/data/raw/grav_33.1.nc'
SHIP_PATH  = f'{BASE}/data/raw/shipborne'   # folder with all .grd files
PAIRED_DIR = f'{BASE}/data/processed/paired'
UNPAIR_DIR = f'{BASE}/data/processed/unpaired'

# ── hyper-parameters ──────────────────────────────────────────────────────
PATCH_LR  = 50     # LR patch pixels  (50×50 @ 1′)
PATCH_HR  = 200    # HR patch pixels  (200×200 @ 15″, scale=4)
SCALE     = 4
STRIDE_LR = 25     # 50% overlap
SSIM_LO   = 0.2
SSIM_HI   = 0.9
RMSE_MAX  = 0.4


# ══════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════
def _read_grd(path):
    """
    Read a GMT .grd or NetCDF file.
    Returns (lon_1d, lat_1d, grav_2d)  shape=(n_lat, n_lon), float32.
    Handles CF-convention and legacy GMT classic (ravelled z).
    """
    import netCDF4 as nc
    with nc.Dataset(path) as ds:
        vnames = list(ds.variables.keys())

        lon_key  = next((k for k in vnames if k.lower() in
                         ('lon','longitude','x')), None)
        lat_key  = next((k for k in vnames if k.lower() in
                         ('lat','latitude','y')), None)
        grav_key = next((k for k in vnames if k.lower() in
                         ('z','gravity','faa','grav','anomaly','data')), None)
        if grav_key is None:
            grav_key = next((k for k in vnames
                             if ds.variables[k].ndim >= 1
                             and k not in (lon_key, lat_key)), None)

        var = ds.variables[grav_key]

        # ── legacy GMT: 1-D ravelled z + x_range/y_range/dimension ───────
        if var.ndim == 1 and 'x_range' in vnames:
            z_flat  = np.array(var[:], dtype=np.float32)
            x_range = ds.variables['x_range'][:]
            y_range = ds.variables['y_range'][:]
            dims    = ds.variables['dimension'][:]
            nlon, nlat = int(dims[0]), int(dims[1])
            lon_1d  = np.linspace(float(x_range[0]), float(x_range[1]), nlon)
            lat_1d  = np.linspace(float(y_range[0]), float(y_range[1]), nlat)
            grav_2d = z_flat.reshape(nlat, nlon)

        # ── CF-convention 2-D ─────────────────────────────────────────────
        elif var.ndim == 2:
            lon_1d  = np.array(ds.variables[lon_key][:],  dtype=np.float64)
            lat_1d  = np.array(ds.variables[lat_key][:],  dtype=np.float64)
            grav_2d = np.array(var[:], dtype=np.float32)
            if grav_2d.shape == (len(lon_1d), len(lat_1d)):
                grav_2d = grav_2d.T   # ensure (n_lat, n_lon)
        else:
            raise ValueError(f"Unrecognised layout in {path}: "
                             f"vars={vnames}, z.ndim={var.ndim}")

    # monotonically increasing axes
    if lon_1d[-1] < lon_1d[0]:
        lon_1d  = lon_1d[::-1];  grav_2d = grav_2d[:, ::-1]
    if lat_1d[-1] < lat_1d[0]:
        lat_1d  = lat_1d[::-1];  grav_2d = grav_2d[::-1, :]

    # mask extreme fill values
    grav_2d[np.abs(grav_2d) > 9000] = np.nan

    return lon_1d.astype(np.float64), lat_1d.astype(np.float64), grav_2d


def _normalise(arr):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# Satellite loader  (reads once, kept in RAM — ~1.8 GB float32)
# ══════════════════════════════════════════════════════════════════════════
def load_satellite(path=SAT_PATH):
    print(f"Loading satellite: {os.path.basename(path)} …")
    lon, lat, grav = _read_grd(path)
    print(f"  lon [{lon.min():.1f}, {lon.max():.1f}]  "
          f"lat [{lat.min():.1f}, {lat.max():.1f}]  "
          f"shape {grav.shape}  "
          f"RAM ~{grav.nbytes/1e9:.2f} GB")
    return lon, lat, grav


# ══════════════════════════════════════════════════════════════════════════
# Per-file patch extractor  — the core of the memory-safe design
# ══════════════════════════════════════════════════════════════════════════
def _extract_from_one_file(ship_path, lon_sat, lat_sat, grav_sat,
                            sat_dlon, sat_dlat):
    """
    Load one shipborne .grd, find its overlap with the satellite grid,
    and extract paired + unpaired patches.
    Returns (paired_lr, paired_hr, unpaired_lr) as lists of float32 arrays.
    The shipborne grid is freed from memory before returning.
    """
    fname = os.path.basename(ship_path)
    try:
        lon_hr, lat_hr, grav_hr = _read_grd(ship_path)
    except Exception as e:
        print(f"  ✗ {fname}: {e}")
        return [], [], []

    n_valid = int(np.sum(~np.isnan(grav_hr)))
    print(f"  {fname}: {grav_hr.shape}  "
          f"lon [{lon_hr.min():.1f},{lon_hr.max():.1f}]  "
          f"lat [{lat_hr.min():.1f},{lat_hr.max():.1f}]  "
          f"valid {n_valid:,}")

    # ── geographic overlap ───────────────────────────────────────────────
    lo0 = max(float(lon_sat.min()), float(lon_hr.min()))
    lo1 = min(float(lon_sat.max()), float(lon_hr.max()))
    la0 = max(float(lat_sat.min()), float(lat_hr.min()))
    la1 = min(float(lat_sat.max()), float(lat_hr.max()))

    if lo0 >= lo1 or la0 >= la1:
        print(f"    → no overlap with satellite grid, skipping")
        return [], [], []

    # ── crop satellite to overlap region ────────────────────────────────
    ci = np.where((lon_sat >= lo0) & (lon_sat <= lo1))[0]
    ri = np.where((lat_sat >= la0) & (lat_sat <= la1))[0]
    if len(ci) < PATCH_LR or len(ri) < PATCH_LR:
        print(f"    → overlap too small ({len(ri)} rows × {len(ci)} cols), "
              f"need ≥{PATCH_LR}")
        return [], [], []

    sub_lon = lon_sat[ci]
    sub_lat = lat_sat[ri]
    sub_sat = grav_sat[np.ix_(ri, ci)]   # (n_lat_overlap, n_lon_overlap)

    nR, nC = len(sub_lat), len(sub_lon)
    print(f"    overlap: {nR} × {nC} LR pixels  "
          f"→ ~{((nR-PATCH_LR)//STRIDE_LR+1)*((nC-PATCH_LR)//STRIDE_LR+1)} patches")

    # ── build HR interpolator (lat must be ascending) ────────────────────
    rgi = RegularGridInterpolator(
        (lat_hr, lon_hr), grav_hr,
        method='linear', bounds_error=False, fill_value=np.nan)

    paired_lr, paired_hr, unpaired_lr = [], [], []
    n_nan_skip = n_qc_fail = 0

    r = 0
    while r + PATCH_LR <= nR:
        c = 0
        while c + PATCH_LR <= nC:

            lr_patch = sub_sat[r:r+PATCH_LR, c:c+PATCH_LR].copy()

            # skip LR patches that are mostly ocean-fill / NaN
            if np.isnan(lr_patch).mean() > 0.5:
                n_nan_skip += 1
                c += STRIDE_LR; continue

            # exact geographic footprint of this LR patch
            lon0_p = sub_lon[c]
            lon1_p = sub_lon[c + PATCH_LR - 1]
            lat0_p = sub_lat[r]
            lat1_p = sub_lat[r + PATCH_LR - 1]

            # sample HR at exactly PATCH_HR × PATCH_HR points via linspace
            hr_lons = np.linspace(lon0_p, lon1_p, PATCH_HR)
            hr_lats = np.linspace(lat0_p, lat1_p, PATCH_HR)
            glon, glat = np.meshgrid(hr_lons, hr_lats)
            pts     = np.stack([glat.ravel(), glon.ravel()], axis=1)
            hr_vals = rgi(pts).reshape(PATCH_HR, PATCH_HR).astype(np.float32)

            nan_frac = float(np.isnan(hr_vals).mean())

            if nan_frac < 0.5:
                # fill residual NaN gaps with nearest valid neighbour
                if nan_frac > 0:
                    from scipy.ndimage import distance_transform_edt as dte
                    mask = np.isnan(hr_vals)
                    idx  = dte(mask, return_distances=False,
                               return_indices=True)
                    hr_vals[mask] = hr_vals[tuple(idx[:, mask])]

                if np.isnan(lr_patch).any():
                    from scipy.ndimage import distance_transform_edt as dte
                    mask = np.isnan(lr_patch)
                    idx  = dte(mask, return_distances=False,
                               return_indices=True)
                    lr_patch[mask] = lr_patch[tuple(idx[:, mask])]

                lr_n = _normalise(lr_patch)
                hr_n = _normalise(hr_vals)

                # quality control (paper Section 4.1)
                lr_up = np.repeat(np.repeat(lr_n, SCALE, 0), SCALE, 1)
                s    = float(ssim_metric(lr_up, hr_n, data_range=1.0))
                rmse = float(np.sqrt(np.mean((lr_up - hr_n) ** 2)))

                if SSIM_LO <= s <= SSIM_HI and rmse <= RMSE_MAX:
                    paired_lr.append(lr_n)
                    paired_hr.append(hr_n)
                else:
                    n_qc_fail += 1
                    unpaired_lr.append(lr_n)
            else:
                unpaired_lr.append(_normalise(lr_patch))

            c += STRIDE_LR
        r += STRIDE_LR

    # free shipborne data immediately
    del grav_hr, sub_sat, rgi
    gc.collect()

    print(f"    paired={len(paired_lr)}  unpaired={len(unpaired_lr)}  "
          f"nan-skip={n_nan_skip}  qc-fail={n_qc_fail}")
    return paired_lr, paired_hr, unpaired_lr


# ══════════════════════════════════════════════════════════════════════════
# Atomic accumulator — saves LR+HR as ONE .npz file to prevent desync
# ══════════════════════════════════════════════════════════════════════════
class _NpyAccumulator:
    """
    KEY FIX: LR and HR patches saved atomically as paired.npz
    so they can NEVER get out of sync due to crashes or partial runs.
    np.savez writes one file containing both arrays.
    """
    def __init__(self, flush_every=500):
        self.flush_every  = flush_every
        self.paired_lr    = []
        self.paired_hr    = []
        self.unpaired_lr  = []
        self._n_paired    = 0
        self._n_unpaired  = 0

    def add(self, p_lr, p_hr, u_lr):
        assert len(p_lr) == len(p_hr), \
            f"LR/HR count mismatch: {len(p_lr)} vs {len(p_hr)}"
        self.paired_lr.extend(p_lr)
        self.paired_hr.extend(p_hr)
        self.unpaired_lr.extend(u_lr)
        if len(self.paired_lr) >= self.flush_every or \
           len(self.unpaired_lr) >= self.flush_every * 2:
            self.flush()

    def flush(self):
        os.makedirs(PAIRED_DIR, exist_ok=True)
        os.makedirs(UNPAIR_DIR, exist_ok=True)

        if self.paired_lr:
            assert len(self.paired_lr) == len(self.paired_hr)
            lr_arr = np.stack(self.paired_lr)[:, np.newaxis]
            hr_arr = np.stack(self.paired_hr)[:, np.newaxis]
            self._append_npz(f'{PAIRED_DIR}/paired.npz', lr_arr, hr_arr)
            self._n_paired += len(self.paired_lr)
            self.paired_lr  = []
            self.paired_hr  = []

        if self.unpaired_lr:
            u_arr = np.stack(self.unpaired_lr)[:, np.newaxis]
            self._append_npy(f'{UNPAIR_DIR}/lr_patches.npy', u_arr)
            self._n_unpaired += len(self.unpaired_lr)
            self.unpaired_lr  = []

        gc.collect()

    @staticmethod
    def _append_npz(path, new_lr, new_hr):
        """Write LR+HR atomically — one file, always in sync."""
        if os.path.exists(path):
            existing = np.load(path)
            lr = np.concatenate([existing['lr'], new_lr], axis=0)
            hr = np.concatenate([existing['hr'], new_hr], axis=0)
            del existing
        else:
            lr, hr = new_lr, new_hr
        # tmp must be in same dir as target for atomic rename
        tmp = os.path.join(os.path.dirname(path), '_paired_tmp.npz')
        np.savez_compressed(tmp, lr=lr, hr=hr)
        os.replace(tmp, path)
        del lr, hr

    @staticmethod
    def _append_npy(path, new_arr):
        if os.path.exists(path):
            existing = np.load(path)
            combined = np.concatenate([existing, new_arr], axis=0)
            del existing
        else:
            combined = new_arr
        np.save(path, combined)
        del combined

    def totals(self):
        return (self._n_paired   + len(self.paired_lr),
                self._n_unpaired + len(self.unpaired_lr))
# ══════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════
def run_preprocessing(sat_path=SAT_PATH, ship_path=SHIP_PATH):
    """
    Memory-safe preprocessing:
      • Loads satellite grid once (~1.8 GB)
      • Processes each shipborne .grd file one at a time
      • Flushes patches to disk every 2000 patches
      • Peak RAM = satellite + one shipborne file + one batch of patches
    """
    print("=" * 60)
    print("SDRL Pre-processing  (v4 — memory-safe streaming)")
    print("=" * 60)

    # 1. Find shipborne files ─────────────────────────────────────────────
    ship_path = ship_path.rstrip('/')
    if os.path.isfile(ship_path):
        grd_files = [ship_path]
    elif os.path.isdir(ship_path):
        grd_files = sorted(
            glob.glob(os.path.join(ship_path, '*.grd')) +
            glob.glob(os.path.join(ship_path, '*.nc'))  +
            glob.glob(os.path.join(ship_path, '**', '*.grd'), recursive=True) +
            glob.glob(os.path.join(ship_path, '**', '*.nc'),  recursive=True))
        grd_files = sorted(set(grd_files))
    else:
        raise FileNotFoundError(f"ship_path not found: {ship_path}")

    if not grd_files:
        raise FileNotFoundError(f"No .grd/.nc files found in {ship_path}")

    print(f"\nFound {len(grd_files)} shipborne file(s):")
    for f in grd_files:
        print(f"  {os.path.basename(f)}")

    # 2. Load satellite (once) ────────────────────────────────────────────
    print(f"\n[1/{len(grd_files)+1}] Loading satellite grid …")
    lon_sat, lat_sat, grav_sat = load_satellite(sat_path)
    sat_dlon = float(np.median(np.diff(lon_sat)))
    sat_dlat = float(np.median(np.diff(lat_sat)))

    # 3. Clear old patch files so we start fresh ──────────────────────────
    for p in [f'{PAIRED_DIR}/lr_patches.npy',
              f'{PAIRED_DIR}/hr_patches.npy',
              f'{UNPAIR_DIR}/lr_patches.npy']:
        if os.path.exists(p):
            os.remove(p)
            print(f"  Cleared old file: {p}")

    # 4. Process each shipborne file ──────────────────────────────────────
    acc = _NpyAccumulator(flush_every=1000)

    for i, fpath in enumerate(grd_files):
        print(f"\n[{i+2}/{len(grd_files)+1}] {os.path.basename(fpath)}")
        p_lr, p_hr, u_lr = _extract_from_one_file(
            fpath, lon_sat, lat_sat, grav_sat, sat_dlon, sat_dlat)
        acc.add(p_lr, p_hr, u_lr)
        n_p, n_u = acc.totals()
        print(f"  Running total → paired: {n_p:,}  unpaired: {n_u:,}")

    # 5. Final flush ──────────────────────────────────────────────────────
    acc.flush()
    n_paired, n_unpaired = acc.totals()

    print(f"\n{'='*60}")
    print(f"Done.")
    print(f"  Paired patches  : {n_paired:,}  → {PAIRED_DIR}/")
    print(f"  Unpaired patches: {n_unpaired:,}  → {UNPAIR_DIR}/")

    if n_paired == 0:
        print("\n⚠  Zero paired patches. Try loosening QC thresholds:")
        print("   SSIM_LO=0.1, SSIM_HI=0.95, RMSE_MAX=0.6")
    print("=" * 60)


if __name__ == '__main__':
    run_preprocessing()
