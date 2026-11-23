"""Air | n=2 core | air gap | n=1.7 core | air. Lengths in um.
TE with mu=1 and decaying exterior fields. Seed the two lowest roots at
beta=6.5/um independently (4.68315648 and 4.74670390/um). At the
uncoupled crossing beta=7/um, expect 4.97763536 and 5.02341116/um.
Compare accepted k0 within 1e-7/um. Multiply both axes by 1e4 for cm^-1.
"""
from math import atan, cos, exp, sin, sqrt

BETA_CROSS = 7.0
K0_CROSS = 5.0
INDICES = (2.0, 1.7)
GAP = 0.7


def core_width(index):
    """TE0 dispersion of one symmetric isolated core at the design crossing."""
    decay = sqrt(BETA_CROSS**2 - K0_CROSS**2)
    q = sqrt((index * K0_CROSS)**2 - BETA_CROSS**2)
    return 2 * atan(decay / q) / q


def boundary(beta, k0):
    """Match E and its normal derivative in the air gap from either core."""
    decay = sqrt(beta**2 - k0**2)
    ends = []
    for index in INDICES:
        q = sqrt((index * k0)**2 - beta**2)
        angle = q * core_width(index)
        field = cos(angle) + decay * sin(angle) / q
        derivative = -q * sin(angle) + decay * cos(angle)
        ends.append((decay * field + derivative, decay * field - derivative))
    (a,b),(c,d) = ends
    return a*c - b*d*exp(-2*decay*GAP)


def roots(beta):
    lower = beta / min(INDICES) + 1e-6
    upper = beta - 1e-6
    previous = lower
    for i in range(1,1001):
        current = lower + (upper-lower)*i/1000
        if boundary(beta,previous) * boundary(beta,current) < 0:
            lo,hi = previous,current
            for _ in range(45):
                mid = (lo+hi)/2
                if boundary(beta,lo)*boundary(beta,mid)<=0:
                    hi=mid
                else:
                    lo=mid
            yield (lo+hi)/2
        previous=current


if __name__ == '__main__':
    for beta in (6.5,7,7.5):
        print(beta,list(roots(beta)))
