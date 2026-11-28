"""Sector definitions shared by the sector_*.py scripts.

sector_extract.py assigns sector membership point-in-time from Compustat's GICS
history (comp.co_hgic): each call uses the firm's sector on the call date.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Sector:
    key: str     # CLI value, output-file prefix, folder suffix
    gics: str    # 2-digit GICS sector code (Compustat gsector prefix)
    name: str    # value of the `sector` column in model_table
    short: str   # short label used in logs and report titles

    @property
    def folder(self):
        return os.path.expanduser(f"~/earnings_{self.key}")


SECTORS = {
    "it": Sector("it", "45", "Information Technology", "IT"),
    "health": Sector("health", "35", "Health Care", "Health Care"),
    "comm": Sector("comm", "50", "Communication Services", "Comm Services"),
}


def add_sector_arg(ap):
    ap.add_argument("--sector", required=True, choices=sorted(SECTORS),
                    help="sector to analyse: " + ", ".join(f"{k} (GICS {s.gics})" for k, s in SECTORS.items()))


def get_sector(args):
    return SECTORS[args.sector]
