"""Command line entry point: ``python -m carnot.cli all``."""
from __future__ import annotations

import argparse
import json
import sys
import time

from .config import load_config
from .experiments import run_all


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="carnot",
        description="Grey-box model of heat-induced DC fast-charging derating and queueing")
    parser.add_argument("command", nargs="?", default="all", choices=["all", "check"],
                        help="'all' reproduces every figure and table in results/")
    parser.add_argument("--config", default=None, help="optional YAML override file")
    parser.add_argument("--seeds", type=int, default=None,
                        help="arrival seeds per scenario (default from config)")
    parser.add_argument("--mc", type=int, default=None,
                        help="Monte Carlo samples for the uncertainty experiment")
    parser.add_argument("--fast", action="store_true", help="short run for smoke testing")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.fast:
        cfg["_seeds"] = 3
        cfg["_mc"] = 6
    if args.seeds:
        cfg["_seeds"] = args.seeds
    if args.mc:
        cfg["_mc"] = args.mc

    if args.command == "check":
        from .climate import add_solar_geometry, load_tmy
        from .config import cfg_path
        tmy = load_tmy(cfg_path(cfg, "raw_tmy"), cfg["climate"]["utc_offset_h"])
        tmy = add_solar_geometry(tmy, cfg["climate"]["latitude_deg"],
                                 cfg["climate"]["longitude_deg"],
                                 cfg["climate"]["utc_offset_h"])
        print(f"TMY rows: {len(tmy)}  Tmax {tmy['t_amb_c'].max():.1f} C  "
              f"GHI max {tmy['ghi'].max():.0f} W/m2")
        return 0

    t0 = time.time()
    summary = run_all(cfg)
    print(json.dumps(summary["headline"], indent=2))
    print(f"\nDone in {time.time() - t0:.1f} s. Results in results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
