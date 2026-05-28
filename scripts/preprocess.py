"""
scripts/preprocess.py
---------------------
CLI entry point for SDRL preprocessing.

Usage:
    python scripts/preprocess.py --config configs/sdrl.yaml
    python scripts/preprocess.py --sat data/raw/grav_33.1.nc \
                                  --ship data/raw/shipborne \
                                  --out data/processed
"""
import argparse
import yaml
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess satellite + shipborne gravity data into patches")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to YAML config (overrides other flags if given)")
    p.add_argument("--sat",  type=Path, help="Satellite .nc file")
    p.add_argument("--ship", type=Path, help="Shipborne .grd file or folder")
    p.add_argument("--out",  type=Path, default=Path("data/processed"),
                   help="Output directory for patches")
    return p.parse_args()


def main():
    args = parse_args()

    # load config if given
    if args.config:
        cfg = yaml.safe_load(args.config.read_text())
        sat_path  = Path(cfg["data"]["sat_path"])
        ship_path = Path(cfg["data"]["ship_path"])
        pair_dir  = Path(cfg["data"]["paired_dir"])
        unpair_dir = Path(cfg["data"]["unpair_dir"])
    else:
        if not args.sat or not args.ship:
            raise ValueError("Provide --config or both --sat and --ship")
        sat_path   = args.sat
        ship_path  = args.ship
        pair_dir   = args.out / "paired"
        unpair_dir = args.out / "unpaired"

    # import here so the package is found after pip install -e .
    from sdrl.preprocess import run_preprocessing

    # patch the directory constants to use CLI values
    import sdrl.preprocess as pp
    pp.PAIRED_DIR = str(pair_dir)
    pp.UNPAIR_DIR = str(unpair_dir)

    run_preprocessing(
        sat_path  = str(sat_path),
        ship_path = str(ship_path),
    )


if __name__ == "__main__":
    main()
