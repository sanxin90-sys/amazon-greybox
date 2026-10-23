import numpy as np
import torch

from amazon_graybox.graybox import GrayBoxODE, Scalers, refine_forcing
from amazon_graybox.physics import (PhysParams, dry_rhs, moist_closed_rhs, nocturnal_dry_solution,
                                    silva_to_si, si_to_silva)
from amazon_graybox.thermo import relative_humidity, vapour_pressure


def test_nocturnal_cooling_rate_matches_de_paulo():
    # 0.9 K/h at the beginning of the night for gamma = 6.5e-4 m-1 (de Paulo et al., 2023)
    T = nocturnal_dry_solution(300.0, np.array([0.0, 1.0]))
    assert abs((T[0] - T[1]) - 0.9) < 0.02


def test_silva_unit_conversion_roundtrip():
    p = silva_to_si(1.1588, 2.05e-8)
    assert 1.0e-3 < p.alpha < 2.0e-3 and 3e-4 < p.gamma < 6e-4
    a, g = si_to_silva(p)
    assert np.isclose(a, 1.1588) and np.isclose(g, 2.05e-8)


def test_vapour_pressure_roundtrip():
    T = np.array([290.0, 300.0]); rh = np.array([95.0, 60.0])
    assert np.allclose(relative_humidity(T, vapour_pressure(T, rh)), rh)


def _forcing(n=48):
    t = np.arange(n + 1) * 0.5
    R = np.clip(600 * np.sin(np.pi * ((t % 24) - 6) / 12), -30, None)
    e = 2.6 + 0.1 * np.sin(2 * np.pi * t / 24)
    de = np.gradient(e, 0.5)
    return t, R, e, de


def _rk4_numpy(f, T0, t, *forc):
    T = [T0]
    for k in range(len(t) - 1):
        h = t[k + 1] - t[k]
        mid = [0.5 * (x[k] + x[k + 1]) for x in forc]
        k1 = f(T[-1], *[x[k] for x in forc]); k2 = f(T[-1] + h / 2 * k1, *mid)
        k3 = f(T[-1] + h / 2 * k2, *mid); k4 = f(T[-1] + h * k3, *[x[k + 1] for x in forc])
        T.append(T[-1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return np.array(T)


def test_torch_rollout_matches_numpy_physics():
    t, R, e, de = _forcing()
    p = PhysParams(alpha=1.4e-3, gamma=4.3e-4, beta=1e-3, delta=50.0)
    for white, f in [("dry", lambda T, R, e, de: dry_rhs(T, R, p)),
                     ("moist_closed", lambda T, R, e, de: moist_closed_rhs(T, R, e, de, p))]:
        ref = _rk4_numpy(f, 298.0, t, R, e, de)
        m = GrayBoxODE(white, "none", M=2, p0=p.as_dict(), scalers=Scalers())
        F = refine_forcing({"R": R, "e": e, "de": de, "RH_obs": 0 * R, "dRH_obs": 0 * R}, n_sub=1)
        out = m.rollout(torch.full((2, 1), 298.0, dtype=torch.float64), F, len(t) - 1, n_sub=1).detach().numpy()
        assert np.allclose(out[0, 0], ref, atol=1e-8), white


def test_zero_initialised_corrections_start_close_to_physics():
    t, R, e, de = _forcing()
    sc = Scalers({"T": 298, "R": 150, "RH": 80, "e": 2.6}, {"T": 3, "R": 250, "RH": 10, "e": 0.3}, 1.0)
    F = refine_forcing({"R": R, "e": e, "de": de, "RH_obs": 0 * R, "dRH_obs": 0 * R}, n_sub=2)
    T0 = torch.full((3, 1), 298.0, dtype=torch.float64)
    base = GrayBoxODE("dry", "none", M=3, scalers=sc).rollout(T0, F, len(t) - 1)
    for corr in ("add", "mult", "hyb"):
        m = GrayBoxODE("dry", corr, M=3, scalers=sc)
        with torch.no_grad():
            m.net.weights[-1].zero_(); m.net.biases[-1].zero_()
        out = m.rollout(T0, F, len(t) - 1)
        assert torch.allclose(out, base), corr


def test_model_spec_parses_result_tags():
    from amazon_graybox.experiments import model_spec

    assert model_spec("moist_closed_hyb_TRHe") == ("moist_closed", "hyb", "TRHe")
    assert model_spec("dry_add_TR") == ("dry", "add", "TR")
    assert model_spec("none_add_TRHe") == ("none", "add", "TRHe")
    assert model_spec("moist_open_none")[:2] == ("moist_open", "none")


def _block(n=96, **change):
    import pandas as pd

    idx = pd.date_range("2008-07-01 00:30", periods=n, freq="30min")
    h = np.asarray((idx - pd.Timedelta(minutes=30)).hour)
    T = 298.0 + 5 * np.sin(2 * np.pi * (h - 9) / 24)
    R = np.clip(600 * np.sin(np.pi * (h - 6) / 12), -40, None)
    d = pd.DataFrame({"T": T, "R": R, "RH": np.full(n, 70.0)}, index=idx)
    d["e"] = vapour_pressure(d["T"], d["RH"])
    for k, v in change.items():
        d[k] = v(d[k]) if callable(v) else v
    return d


def test_sanity_bounds_reject_unphysical_blocks():
    import pytest

    from amazon_graybox import sanity

    sanity.check_block(_block())
    for change in ({"RH": 101.0}, {"RH": -1.0}, {"T": -5.0}, {"e": lambda e: 1.2 * e / 0.7},
                   {"R": lambda R: np.where(R < 0, 120.0, R)}):
        with pytest.raises(sanity.PhysicalBoundError):
            sanity.check_block(_block(**change))
    assert sanity.implausible_fraction(np.array([300.0, 360.0, np.nan])) == 0.5
    with pytest.raises(sanity.PhysicalBoundError):
        sanity.check_temperature(np.array([300.0, -1.0]))
    assert sanity.diverged(1.0, sanity.implausible_fraction(np.array([300.0, 345.0])))  # implausible: diverged
    assert sanity.diverged(12.0) and not sanity.diverged(1.0, 0.0)


def test_effective_heat_capacity_must_stay_positive():
    import pytest

    from amazon_graybox.sanity import PhysicalBoundError

    t, R, e, de = _forcing()
    m = GrayBoxODE("moist_closed", "none", M=1, p0={"delta": 50.0}, scalers=Scalers())
    with torch.no_grad():
        m.raw_delta.fill_(10.0)  # delta -> delta_max
    F = {k: torch.as_tensor(v[None], dtype=torch.float64) for k, v in {"R": R, "e": 0 * R + 3.5, "de": de}.items()}
    with pytest.raises(PhysicalBoundError):
        m.rhs(torch.full((1, len(R)), 250.0, dtype=torch.float64), F)
