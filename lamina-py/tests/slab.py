"""Independent analytic references for three-layer dielectric slabs."""

from __future__ import annotations

import math


def ladder(
    wavelength_m: float,
    core_eps: float,
    thickness_m: float,
    left_eps: float,
    right_eps: float,
    polarisation: str,
) -> list[float]:
    """Effective indices of the guided modes, highest first.

    Solves the transverse resonance condition
    q d - phi_left - phi_right = m pi, with q = k0 sqrt(eps_core - n^2) and
    phi_side = atan(rho_side * kappa_side / q), where kappa_side =
    k0 sqrt(n^2 - eps_side) and rho_side is 1 for TE and eps_core / eps_side
    for TM. The left side of the condition falls monotonically in n, so each
    order is bracketed and bisected to machine precision.
    """
    k0 = 2.0 * math.pi / wavelength_m

    def phase(n: float) -> float:
        q = k0 * math.sqrt(max(core_eps - n * n, 0.0))

        def shift(eps: float) -> float:
            kappa = k0 * math.sqrt(max(n * n - eps, 0.0))
            rho = 1.0 if polarisation == "TE" else core_eps / eps
            return math.atan2(rho * kappa, q)

        return q * thickness_m - shift(left_eps) - shift(right_eps)

    lower = math.sqrt(max(left_eps, right_eps))
    upper = math.sqrt(core_eps)
    roots = []
    for m in range(1000):
        lo, hi = lower, upper
        if phase(lo) <= m * math.pi:
            break
        while True:
            mid = 0.5 * (lo + hi)
            if mid in (lo, hi):
                break
            if phase(mid) > m * math.pi:
                lo = mid
            else:
                hi = mid
        roots.append(0.5 * (lo + hi))
    return roots
