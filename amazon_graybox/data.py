"""Data loading and quality control for the tower datasets used in the study.

All loaders return a :class:`pandas.DataFrame` indexed by local timestamps
(end of the 30-min averaging interval) with the canonical columns

    T   air temperature [K]
    R   net radiation [W m-2]
    RH  relative humidity [%]

plus optional auxiliary columns (WS wind speed [m s-1], P pressure [hPa],
PRP precipitation [mm]). Missing values are kept as NaN; gap handling is done
downstream by :func:`contiguous_segments`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KELVIN = 273.15
STEP = pd.Timedelta(minutes=30)

# Column offsets of each yearly block in the "DADOS OK" sheet of the Rebio Jaru
# workbook (0-based, after skipping the header row). Only blocks with T, R and
# RH at 62 m are listed.
_JARU_BLOCKS = {
    2006: {"year": 23, "R": 26, "RH": 27, "T": 28, "WS": 29, "P": 31, "PRP": 32},
    2007: {"year": 34, "R": 37, "RH": 38, "T": 39, "WS": 40, "P": 41, "PRP": 42},
    2008: {"year": 44, "R": 47, "RH": 48, "T": 50, "WS": 51, "P": 52, "PRP": 53},
    2009: {"year": 55, "R": 56, "RH": 57, "T": 58, "WS": 59, "P": 61},
    2010: {"year": 63, "R": 64, "RH": 65, "T": 66, "WS": 67, "P": 69},
}
# In 2006-2007 the only temperature sensor available in the workbook is at 22 m.
_JARU_T_HEIGHT = {2006: 22, 2007: 22, 2008: 62, 2009: 62, 2010: 62}

# Physically plausible ranges used for QC (tropical tower, 30-min means).
QC_LIMITS = {"T": (283.15, 313.15), "R": (-150.0, 1100.0), "RH": (20.0, 110.0)}  # RH > 100 clipped below

SEASONS = {
    # Silva et al. (2026) definitions for Rebio Jaru.
    "wet": (1, 3),
    "dry": (7, 9),
    # Transitional periods (not used by Silva et al.).
    "wet_dry": (4, 6),
    "dry_wet": (10, 12),
}


@dataclass(frozen=True)
class Segment:
    """A gap-free stretch of the series on a regular 30-min grid."""

    start: int
    stop: int  # exclusive

    def __len__(self) -> int:
        return self.stop - self.start


def _qc(df: pd.DataFrame) -> pd.DataFrame:
    """Set physically implausible values to NaN and clip RH to [0, 100]."""
    out = df.copy()
    for col, (lo, hi) in QC_LIMITS.items():
        if col in out:
            bad = (out[col] < lo) | (out[col] > hi)
            out.loc[bad, col] = np.nan
    if "RH" in out:
        out["RH"] = out["RH"].clip(upper=100.0)
    return out


def load_jaru(path: str | Path, year: int, qc: bool = True) -> pd.DataFrame:
    """Load one year of Rebio Jaru (LBA tower, Rondonia) 30-min data.

    The workbook stores one block of columns per year, all aligned on the same
    row grid: row ``i`` corresponds to ``Jan 1, 00:30 + i * 30 min`` (time at
    the end of the averaging interval). The day-of-year column of the workbook
    is inconsistent at midnight for a few days, so timestamps are built from the
    row index instead.
    """
    if year not in _JARU_BLOCKS:
        raise ValueError(f"year must be one of {sorted(_JARU_BLOCKS)}")
    raw = pd.read_excel(path, sheet_name="DADOS OK", header=None, skiprows=1)
    cols = _JARU_BLOCKS[year]
    df = pd.DataFrame({k: pd.to_numeric(raw.iloc[:, c], errors="coerce") for k, c in cols.items()})
    df.index = pd.date_range(f"{year}-01-01 00:30", periods=len(df), freq=STEP)
    df = df[df.index.year == year]  # drop the extra leap-day rows in non-leap years
    df = df[df["year"].eq(year) | df["year"].isna()].drop(columns="year")
    df["T"] = df["T"] + KELVIN
    df.attrs.update(site="Rebio Jaru", year=year, z_T=_JARU_T_HEIGHT[year])
    return _qc(df) if qc else df


def load_sinop_xls(path: str | Path, year: int) -> pd.DataFrame:
    """Load one monthly Sinop (transitional forest, MT) logger export (.xls).

    The exports contain a ``Julian Day`` and an ``Hour`` (HHMM, start of the
    interval) column; the 41 m (or 2001 single-level) psychrometer gives T and
    RH and ``Rn (W/m2)`` the net radiation.
    """
    xl = pd.ExcelFile(path)
    sheet = xl.sheet_names[0]
    df = xl.parse(sheet)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(subset=["Julian Day", "Hour"])

    def pick(*names: str) -> str:
        for n in names:
            if n in df.columns:
                return n
        raise KeyError(f"none of {names} in {path}")

    t_col = pick("RH T 41m (C)", "RH T")
    rh_col = pick("RH 41m (%)", "RH")
    rn_col = pick("Rn (W/m2)")
    doy = df["Julian Day"].astype(int)
    hhmm = df["Hour"].astype(int)
    minutes = (hhmm // 100) * 60 + hhmm % 100
    # Logger stamps mark the start of the 30-min interval; shift to the end.
    idx = (
        pd.Timestamp(f"{year}-01-01")
        + pd.to_timedelta(doy - 1, unit="D")
        + pd.to_timedelta(minutes, unit="min")
        + STEP
    )
    out = pd.DataFrame(
        {
            "T": pd.to_numeric(df[t_col], errors="coerce").to_numpy() + KELVIN,
            "R": pd.to_numeric(df[rn_col], errors="coerce").to_numpy(),
            "RH": pd.to_numeric(df[rh_col], errors="coerce").to_numpy(),
        },
        index=pd.DatetimeIndex(idx),
    )
    out = out.mask(out <= -6000 + KELVIN)  # logger missing-value flag (-6999)
    out = out[~out.index.duplicated()].sort_index()
    out = out.reindex(pd.date_range(out.index[0], out.index[-1], freq=STEP))
    out.attrs.update(site="Sinop", year=year, source=Path(path).name)
    return _qc(out)


def select_months(df: pd.DataFrame, first: int, last: int) -> pd.DataFrame:
    """Rows whose (interval-end) timestamp falls in months ``first..last``.

    The 24:00 value of the last day (stamped 00:00 of the next month) is kept,
    which matches the row convention of the original spreadsheets.
    """
    stamp = df.index - STEP  # interval start decides the month
    return df[(stamp.month >= first) & (stamp.month <= last)]


def select_season(df: pd.DataFrame, season: str) -> pd.DataFrame:
    return select_months(df, *SEASONS[season])


def contiguous_segments(df: pd.DataFrame, cols=("T", "R", "RH"), min_len: int = 48) -> list[Segment]:
    """Gap-free stretches (all ``cols`` finite) at least ``min_len`` steps long."""
    ok = np.isfinite(df[list(cols)].to_numpy()).all(axis=1)
    segs, start = [], None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_len:
                segs.append(Segment(start, i))
            start = None
    if start is not None and len(ok) - start >= min_len:
        segs.append(Segment(start, len(ok)))
    return segs


def fill_short_gaps(df: pd.DataFrame, max_steps: int = 4, cols=("T", "R", "RH")) -> pd.DataFrame:
    """Linearly interpolate gaps of at most ``max_steps`` samples (2 h by default).

    Longer gaps are left as NaN and split the series into segments.
    """
    out = df.copy()
    for c in cols:
        s = out[c]
        isna = s.isna()
        run_id = (~isna).cumsum()
        run_len = isna.groupby(run_id).transform("sum")
        interp = s.interpolate(limit_area="inside")
        out[c] = np.where(isna & (run_len <= max_steps), interp, s)
    return out


def hours_since_start(df: pd.DataFrame) -> np.ndarray:
    return ((df.index - df.index[0]) / pd.Timedelta(hours=1)).to_numpy(dtype=float)


def load_cuieiras(folder: str | Path, year: int, rh_source: str = "merged") -> pd.DataFrame:
    """Load Reserva Biologica do Cuieiras (ZF2/K34, central Amazon) 30-min data.

    T: AWS PT100 at 51.1 m; RH: AWS HMP35A at 51.1 m (``rh_source="aws"``) or
    SPCD31 HMP35C at 51 m (``"spcd"``) or the AWS series completed with the
    SPCD31 series linearly matched to it on their overlap (``"merged"``); R: gap-filled measured net radiation
    (``RNmed Preench.``). Sensible/latent/ground heat fluxes (H, LE, G) from the
    eddy-covariance system at 53.1 m are kept for diagnostics.
    """
    folder = Path(folder)
    sheet = str(year)
    met = pd.read_excel(folder / "1.Variáveis Meteorológicas Cuieiras.xls", sheet_name=sheet, header=None, skiprows=3)
    rad = pd.read_excel(folder / "3.Irradiância Cuieiras.xls", sheet_name=sheet, header=None, skiprows=3)
    flx = pd.read_excel(folder / "4.Densidades de Fluxo de Energia Cuieiras.xls", sheet_name=sheet, header=None, skiprows=3)
    num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
    first_doy = int(num(met[1]).iloc[0])
    idx = pd.date_range(pd.Timestamp(f"{year}-01-01 00:30") + pd.Timedelta(days=first_doy - 1), periods=len(met), freq=STEP)
    rh_aws, rh_spcd = num(met[9]), num(met[10])
    if rh_source == "aws":
        rh = rh_aws
    elif rh_source == "spcd":
        rh = rh_spcd
    else:
        both = rh_aws.notna() & rh_spcd.notna() & (rh_aws < 100)
        slope, icpt = np.polyfit(rh_spcd[both], rh_aws[both], 1)
        rh = rh_aws.fillna(slope * rh_spcd + icpt)
    df = pd.DataFrame(
        {
            "T": num(met[8]).to_numpy() + KELVIN,
            "R": num(rad[20]).to_numpy(),
            "RH": rh.to_numpy(),
            "WS": num(met[15]).to_numpy(),
            "PRP": num(met[14]).to_numpy(),
            "H": num(flx[3]).to_numpy(),
            "LE": num(flx[4]).to_numpy(),
            "G": num(flx[5]).to_numpy(),
        },
        index=idx,
    )
    df.attrs.update(site="Cuieiras", year=year, z_T=51.1, rh_source=rh_source)
    return _qc(df)
