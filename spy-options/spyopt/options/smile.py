"""Volatility surface for SPY: smile shape + term-structure level, both from real quotes.

Shape — standardized moneyness
    z = ln(K / F) / (atm * sqrt(T))            (strike distance in "expected moves")
    iv(K) = atm * r(z),  r(z) = 1 + a*z + b*z^2 (separate a, b for puts z<0 and calls z>0)
In z-space the SPY smile is nearly tenor-invariant: the 1-, 5- and 20-trading-day expiries of
the 2026-10-02 close snapshot collapse onto one curve, so one shape serves 0DTE to monthlies.

Level — ATM vol from the VIX family, ratio calibrated per tenor on the same snapshot
    atm(1 day)  = VIX1D * r1      atm(5 days) = VIX9D * r5      atm(20 days) = VIX * r20
then interpolated linearly in total variance (atm^2 * T) for any other tenor. The ratios are
< 1 because VIX-style indices are variance-swap rates that include the expensive put wing.
They drift with the skew regime, so the study reports sensitivity to +/-5% level shifts, and
`spyopt.live.snapshot` collects fresh chains so they can be re-estimated (re-run `fit()`).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from spyopt.options import bs

QUOTES = Path(__file__).resolve().parents[2] / "data" / "raw" / "ibkr" / "SPY_option_quotes_20261002close.json"
INDEX_FOR_TENOR = {1: "VIX1D", 5: "VIX9D", 20: "VIX"}


@dataclass
class Smile:
    # per-tenor shape (a_put, b_put, a_call, b_call), fitted on the 2026-10-02 close snapshot
    shapes: dict = field(default_factory=lambda: {
        1: (-0.0744, 0.0270, -0.1731, 0.0788),
        5: (-0.1115, 0.0247, -0.1940, 0.0811),
        20: (-0.1870, 0.0255, -0.2245, 0.0811),
    })
    floor: float = 0.75
    cap: float = 3.0
    # ATM vol / index level, per tenor in trading days (see fit())
    level: dict = field(default_factory=lambda: {1: 0.9396, 5: 0.8442, 20: 0.8131})

    def _shape(self, days):
        """Shape coefficients interpolated linearly in log(days) between fitted tenors."""
        ks = sorted(self.shapes)
        arr = np.array([self.shapes[k] for k in ks], dtype=float)
        x = np.log(np.clip(np.asarray(days, dtype=float), ks[0], ks[-1]))
        xs = np.log(np.array(ks, dtype=float))
        return [np.interp(x, xs, arr[:, j]) for j in range(4)]

    def ratio(self, z, days=1):
        z = np.asarray(z, dtype=float)
        ap, bp, ac, bc = self._shape(days)
        r = np.where(z < 0, 1 + ap * z + bp * z * z, 1 + ac * z + bc * z * z)
        return np.clip(r, self.floor, self.cap)

    def iv(self, F, K, T, atm):
        T = np.maximum(np.asarray(T, dtype=float), 1e-9)
        z = np.log(np.asarray(K, float) / np.asarray(F, float)) / (np.asarray(atm) * np.sqrt(T))
        return np.asarray(atm) * self.ratio(z, T * 252)

    def price(self, F, K, T, atm, is_call):
        return bs.price(F, K, T, self.iv(F, K, T, atm), is_call)

    def atm_from_index(self, index_level, tenor_days: int):
        """ATM vol (decimal) for a calibrated tenor (1, 5 or 20 trading days)."""
        return np.asarray(index_level, dtype=float) / 100.0 * self.level[tenor_days]

    def atm_term(self, vix1d, vix9d, vix, days):
        """ATM vol for an arbitrary tenor in trading days, interpolating total variance between
        the calibrated 1/5/20-day nodes (flat vol outside). Inputs are index levels (points)."""
        nodes_t = np.array([1.0, 5.0, 20.0])
        v5 = self.atm_from_index(vix9d, 5)
        v1 = self.atm_from_index(vix1d, 1)
        v1 = np.where(np.isfinite(v1), v1, v5)  # VIX1D starts 2023-04: flat from the 5-day node
        vols = [v1, v5, self.atm_from_index(vix, 20)]
        days = np.asarray(days, dtype=float)
        w = [v * v * t for v, t in zip(vols, nodes_t)]
        out = np.where(days <= 1, vols[0],
              np.where(days <= 5, np.sqrt((w[0] + (w[1] - w[0]) * (days - 1) / 4) / days),
              np.where(days <= 20, np.sqrt((w[1] + (w[2] - w[1]) * (days - 5) / 15) / days), vols[2])))
        return out

    def vs_ratio(self, T=1 / 252, n=4001, zmax=12.0) -> float:
        """Variance-swap vol / ATM vol implied by this smile (CBOE VIX-style strike integral).
        Diagnostic only: the quadratic wings understate the far put tail, so this is lower
        than the empirical index/ATM ratio."""
        F, atm = 100.0, 0.15
        z = np.linspace(-zmax, zmax, n)
        K = F * np.exp(z * atm * np.sqrt(T))
        q = np.where(K < F, self.price(F, K, T, atm, False), self.price(F, K, T, atm, True))
        var = 2.0 / T * np.sum(np.gradient(K) / K**2 * q)
        return float(np.sqrt(var) / atm)

    def to_json(self):
        return asdict(self)


def implied_points(quotes_path=QUOTES):
    """IV per strike from a saved quote snapshot, per expiry."""
    q = json.loads(Path(quotes_path).read_text())
    out = {}
    for exp, e in q["expiries"].items():
        T = int(e["trading_days"]) / 252
        mid = lambda side, k: sum(e[side][k]) / 2  # noqa: E731
        spot = q["underlying"]["close_1600"]
        atm_k = min(set(e["calls"]) & set(e["puts"]), key=lambda k: abs(float(k) - spot))
        F = float(atm_k) + mid("calls", atm_k) - mid("puts", atm_k)  # put-call parity forward
        # ATM vol: average of the call and put IV at the parity strike
        atm = float(np.mean([bs.implied_vol(mid("calls", atm_k), F, float(atm_k), T, True),
                             bs.implied_vol(mid("puts", atm_k), F, float(atm_k), T, False)]))
        pts = []
        for side, is_call in (("puts", False), ("calls", True)):
            for k, (b, a) in e[side].items():
                K = float(k)
                if b <= 0 or (is_call and K < F) or (not is_call and K > F):
                    continue
                iv = float(bs.implied_vol((a + b) / 2, F, K, T, is_call))
                z = float(np.log(K / F) / (atm * np.sqrt(T)))
                pts.append({"K": K, "call": is_call, "mid": (a + b) / 2, "spread": round(a - b, 4),
                            "iv": iv, "z": z, "ratio": iv / atm})
        out[exp] = {"F": F, "T": T, "days": int(e["trading_days"]), "atm": atm, "points": pts}
    return out, q.get("index_close", {})


def fit(quotes_path=QUOTES) -> tuple[Smile, dict]:
    """Fit the per-tenor smile shapes and the ATM/index level ratios from a snapshot.

    Tenors with >= 3 points per side get a full quadratic; sparser tenors (the 20-day expiry
    has 2 OTM puts and 1 OTM call) borrow the curvature b from the nearest fitted tenor and
    solve only for the slope a."""
    data, idx = implied_points(quotes_path)
    shapes = {}
    for e in sorted(data.values(), key=lambda e: e["days"]):
        z = np.array([p["z"] for p in e["points"]])
        r = np.array([p["ratio"] for p in e["points"]])
        coef = []
        for j, m in enumerate((z < 0, z > 0)):
            zz, rr = z[m], r[m] - 1
            if m.sum() >= 3:
                (a, b), *_ = np.linalg.lstsq(np.vstack([zz, zz**2]).T, rr, rcond=None)
            else:
                prev = shapes[max(shapes)] if shapes else (-0.12, 0.02, -0.19, 0.08)
                b = prev[1] if j == 0 else prev[3]
                a = float(np.sum(zz * (rr - b * zz**2)) / np.sum(zz**2)) if m.sum() else (prev[0] if j == 0 else prev[2])
            coef += [float(a), float(b)]
        shapes[e["days"]] = tuple(coef)
    level = {}
    for e in data.values():
        name = INDEX_FOR_TENOR.get(e["days"])
        if name and name in idx:
            level[e["days"]] = e["atm"] / (idx[name] / 100)
    sm = Smile(shapes=shapes, level=level or Smile().level)
    resid = [p["ratio"] - float(sm.ratio(p["z"], e["days"])) for e in data.values() for p in e["points"]]
    return sm, {"n": len(resid), "rmse_ratio": float(np.sqrt(np.mean(np.square(resid)))), "data": data, "index": idx}
