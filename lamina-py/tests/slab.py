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


def box_ladder(
    wavelength_m: float,
    core_eps: float,
    clad_eps: float,
    half_width_m: float,
    gap_m: float,
    wall: str,
    polarisation: str,
    window: tuple[float, float],
) -> list[float]:
    """Effective indices of every real-beta mode of a symmetric core |z| < a in
    cladding out to walls at |z| = a + gap, lowest first, with beta in `window`.

    A wall imposes psi = 0 (PEC in TE, PMC in TM) or d(psi)/dz = 0. The even and
    odd conditions are solved in cross-multiplied form: with q and p the core
    and cladding transverse wavenumbers, S = sin(p L)/p and C = cos(p L) are
    entire in p^2, so guided (p imaginary) and radiation-like (p real) modes are
    found alike, by scanning for sign changes and bisecting.
    """
    k0 = 2.0 * math.pi / wavelength_m
    a, gap = half_width_m, gap_m
    w1, w2 = (1.0, 1.0) if polarisation == "TE" else (1.0 / core_eps, 1.0 / clad_eps)
    dirichlet = (wall == "pec") == (polarisation == "TE")

    def condition(odd: bool, b: float) -> float:
        q = math.sqrt(core_eps * k0 * k0 - b * b)
        p2 = clad_eps * k0 * k0 - b * b
        if p2 >= 0.0:
            p = math.sqrt(p2)
            s, c = (math.sin(p * gap) / p if p else gap), math.cos(p * gap)
        else:
            g = math.sqrt(-p2)
            s, c = math.sinh(g * gap) / g, math.cosh(g * gap)
        sin, cos = math.sin(q * a), math.cos(q * a)
        sinc = sin / q if q else a
        if not odd:
            return w1 * q * sin * s - w2 * cos * c if dirichlet else w1 * q * sin * c + w2 * p2 * s * cos
        return w1 * cos * s + w2 * sinc * c if dirichlet else w1 * cos * c - w2 * p2 * s * sinc

    lo, hi = window[0], min(window[1], math.sqrt(core_eps) * k0)
    n = 100_000
    grid = [lo + (hi - lo) * i / n for i in range(n + 1)]
    roots = []
    for odd in (False, True):
        values = [condition(odd, b) for b in grid]
        for i in range(n):
            if values[i] == 0.0 or values[i] * values[i + 1] > 0.0:
                continue
            left, right, sign = grid[i], grid[i + 1], values[i] > 0.0
            while True:
                mid = 0.5 * (left + right)
                if mid in (left, right):
                    break
                if (condition(odd, mid) > 0.0) == sign:
                    left = mid
                else:
                    right = mid
            roots.append(0.5 * (left + right) / k0)
    return sorted(roots)
