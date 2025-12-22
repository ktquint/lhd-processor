"""
    this file contains all the equations I'll use
    to calculate the hydraulic characteristics at
    the cross-sections of the low-head dams.
"""
import numpy as np
from scipy.optimize import root_scalar

# global vars
g = 9.81 # m/s**2
C_W = 0.62


def solve_yc(Q, xs1, xs2, dist):
    """
        Finds the critical depth (yc) where the Froude number is exactly 1.0.
        This is the physical boundary between supercritical and subcritical flow.
        """

    def fr_residual(y):
        # We want Froude - 1 = 0
        return calc_froude_custom(Q, y, xs1, xs2, dist) - 1.0

    try:
        # Critical depth for most rivers is between 1cm and 10m.
        # brentq is extremely reliable if the signs at the bounds differ.
        sol = root_scalar(fr_residual, bracket=(0.01, 10.0), method='brentq')
        return sol.root
    except ValueError:
        # Fallback if the range is weird
        return 0.5


def _get_active_profile(water_depth, xs_profile, dist, direction=1):
    """
    Helper to extract the active portion of a bank profile (Center -> Out).

    Args:
        xs_profile: List/Array of elevations starting at center.
        dist: Ordinate distance between points.
        direction: 1 for Right (positive x), -1 for Left (negative x).

    Returns:
        x_active: Array of x-coordinates for the submerged portion.
        y_active: Array of y-coordinates for the submerged portion.
    """
    x_active = []
    y_active = []

    # FIX: Use len() check because xs_profile might be a numpy array
    # 'if not xs_profile' crashes on arrays with >1 element
    if len(xs_profile) == 0 or xs_profile[0] > water_depth:
        return np.array([]), np.array([])

    for i, y_curr in enumerate(xs_profile):
        current_x = i * dist * direction

        if y_curr <= water_depth:
            # Point is submerged
            x_active.append(current_x)
            y_active.append(y_curr)
        else:
            # Bank intersection: interpolate between i-1 and i
            y_prev = xs_profile[i-1]
            x_prev = (i - 1) * dist * direction

            if y_curr != y_prev:
                ratio = (water_depth - y_prev) / (y_curr - y_prev)
                x_interp = x_prev + ratio * (current_x - x_prev)

                x_active.append(x_interp)
                y_active.append(water_depth)
            else:
                x_active.append(current_x)
                y_active.append(water_depth)

            # Stop after finding the bank
            break

    return np.array(x_active), np.array(y_active)


def get_top_width(water_depth, xs1, xs2, dist):
    """
    Calculates Top Width using explicitly provided left/right profiles.
    xs1: Center -> Left Bank
    xs2: Center -> Right Bank
    """
    if water_depth <= 0:
        return 0.001

    # Extract active geometries
    x_l, _ = _get_active_profile(water_depth, xs1, dist, direction=-1)
    x_r, _ = _get_active_profile(water_depth, xs2, dist, direction=1)

    if len(x_l) == 0 or len(x_r) == 0:
        return 0.001

    # Max width is simply the distance between the furthest active points
    return abs(x_r[-1] - x_l[-1])


def get_geometry_props(water_depth, xs1, xs2, dist):
    """
    Calculates Area (A) and Centroid Depth (y_cent).
    """
    if water_depth <= 0:
        return 0.0001, 0.0001

    # 1. Get Active Coordinates (Center -> Out)
    x_l, y_l = _get_active_profile(water_depth, xs1, dist, direction=-1)
    x_r, y_r = _get_active_profile(water_depth, xs2, dist, direction=1)

    if len(x_l) == 0 or len(x_r) == 0:
        return 0.0001, 0.0001

    # 2. Combine into one continuous channel (Left -> Right)
    # x_l is [0, -dist, ...], so reverse it to be [..., -dist, 0]
    x_combined = np.concatenate([x_l[::-1], x_r[1:]])
    y_combined = np.concatenate([y_l[::-1], y_r[1:]])

    if len(x_combined) < 2:
        return 0.0001, 0.0001

    # 3. Calculate Area (Integration of depth)
    depths = water_depth - y_combined
    depths[depths < 0] = 0

    Area = np.trapezoid(depths, x_combined)

    if Area <= 0:
        return 0.0001, 0.0001

    # 4. Calculate Centroid
    integrand = 0.5 * depths ** 2

    Moment_surface = np.trapezoid(integrand, x_combined)

    y_cent = Moment_surface / Area

    return Area, y_cent


def calc_froude_custom(Q, y, xs1, xs2, dist):
    """
    Calculates Froude number: Fr = V / sqrt(g * D)
    """
    if y <= 0: return 0

    A, _ = get_geometry_props(y, xs1, xs2, dist)
    T = get_top_width(y, xs1, xs2, dist)

    if A <= 0 or T <= 0: return 0

    V = Q / A
    D = A / T

    return V / np.sqrt(g * D)


def solve_y1_downstream(Q, L, P, xs1, xs2, dist):
    """
    Robustly solves for the supercritical toe depth (y1) using Specific Energy.
    Uses critical depth as a physical boundary for the root search.
    """
    # 1. Calculate Available Total Energy (Bernoulli)
    H = weir_H(Q, L)
    A_o = L * H
    V_o = Q / A_o
    E_total = P + H + (V_o ** 2) / (2 * g)

    # 2. Define Residual Function (E_calc - E_total)
    def energy_residual(y):
        if y <= 0: return 1e9
        A, _ = get_geometry_props(y, xs1, xs2, dist)
        if A <= 0: return 1e9

        V = Q / A
        E_calc = y + (V ** 2) / (2 * g)
        return E_calc - E_total

    # 3. Establish the Search Bracket
    # Critical depth (yc) is the depth of minimum energy.
    # Supercritical flow (y1) MUST occur at a depth less than yc.
    y_c = solve_yc(Q, xs1, xs2, dist)

    # We use a small epsilon (0.0001) instead of 0 to avoid division by zero.
    low_bound = 0.0001
    high_bound = y_c

    # 4. Verify signs and Solve
    # At very small depths, Velocity Head is massive, so E_calc > E_total (Residual is Positive).
    # At critical depth, Energy is at its minimum, so E_calc < E_total (Residual is Negative).
    if energy_residual(low_bound) > 0 > energy_residual(high_bound):
        try:
            sol = root_scalar(energy_residual, bracket=(low_bound, high_bound), method='brentq')
            return sol.root
        except ValueError:
            return 0.1  # Fallback for convergence issues
    else:
        # If the residual at y_c is still positive, E_total is less than the
        # minimum energy required to pass the flow. This indicates "choked" flow.
        return y_c


def solve_y2_jump(Q, y1, xs1, xs2, dist):
    # 1. Calculate Momentum at y1 (Supercritical side)
    A1, y_cj1 = get_geometry_props(y1, xs1, xs2, dist)

    # Safety check for bad y1
    if A1 <= 0.0001: return 0.0

    M1 = (Q ** 2 / (g * A1)) + (A1 * y_cj1)

    def momentum_residual(y_candidate):
        if y_candidate <= 0: return -1e9
        A2, y_cj2 = get_geometry_props(y_candidate, xs1, xs2, dist)
        if A2 <= 0: return -1e9
        M2 = (Q ** 2 / (g * A2)) + (A2 * y_cj2)
        return M2 - M1

    # 2. Find a valid bracket for the Subcritical Root
    lower_bound = y1 * 1.05  # Just above y1
    upper_bound = y1 * 20.0

    # FIX: Robustly expand upper bound until M2 > M1
    try:
        f_upper = momentum_residual(upper_bound)
        iter_count = 0

        # Keep doubling until we find a depth with enough momentum
        while f_upper < 0 and iter_count < 10:
            upper_bound *= 2.0
            f_upper = momentum_residual(upper_bound)
            iter_count += 1

        # If we still can't balance momentum, fail gracefully
        if f_upper < 0:
            return 0.0

        # 3. Use Brent's Method
        sol = root_scalar(momentum_residual, bracket=(lower_bound, upper_bound), method='brentq')
        return sol.root

    except ValueError:
        return 0.0


def Fr_eq(Fr, x):
    A = (9 / (4 * C_W**2)) * 0.5 * Fr**2
    term1 = A**(1/3) * (1 + 1/x)
    term2 = 0.5 * Fr**2 * (1 + 0.1/x)
    return 1 - term1 + term2


def weir_H(Q, L):
    """
    Analytically solves for Head (H).
    """
    H = ((3/2) * (Q/L) / C_W / np.sqrt(2 * g))**(2/3)
    return H


def compute_y_flip(Q, L, P):
    H = weir_H(Q, L)
    return (H + P) / 1.1


def rating_curve_intercept(Q: float, L: float, P: float, a: float, b: float,
                           xs1: list[float], xs2: list[float], dist: float, which: str) -> float:

    y_flip = compute_y_flip(Q, L, P)
    y_1 = solve_y1_downstream(Q, L, P, xs1, xs2, dist)
    y_2 = solve_y2_jump(Q, y_1, xs1, xs2, dist)

    y_t = a * Q ** b

    if which == 'flip':
        return y_flip - y_t
    elif which == 'conjugate':
        return y_2 - y_t
    else:
        raise ValueError("which must be 'flip' or 'conjugate'")


def solve_weir_geometry(Q_input, L_input, YT_input, Wse_input):
    """
    Solves for Head (H) and Weir Height (P).
    """
    total_height = YT_input + Wse_input

    # 1. Solve H analytically
    H_solution = weir_H(Q_input, L_input)

    # 2. Calculate P
    P_solution = total_height - H_solution

    if P_solution < 0:
        P_solution = 0.1
        H_solution = total_height - P_solution

    return H_solution, P_solution