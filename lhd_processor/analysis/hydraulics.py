"""
    this file contains all the equations I'll use
    to calculate the hydraulic characteristics at
    the cross-sections of the low-head dams.
"""
import numpy as np
from scipy.optimize import fsolve

# global vars
g = 9.81 # m/s**2
C_W = 0.62


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

    # Check if center is already dry
    if not xs_profile or xs_profile[0] > water_depth:
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
    # x_r is [0, dist, ...], skip 0 to avoid duplicate center point

    x_combined = np.concatenate([x_l[::-1], x_r[1:]])
    y_combined = np.concatenate([y_l[::-1], y_r[1:]])

    if len(x_combined) < 2:
        return 0.0001, 0.0001

    # 3. Calculate Area (Integration of depth)
    depths = water_depth - y_combined
    # Ensure depths are non-negative (floating point errors)
    depths[depths < 0] = 0

    Area = np.trapezoid(depths, x_combined)

    if Area <= 0:
        return 0.0001, 0.0001

    # 4. Calculate Centroid
    # Moment of Area about the surface (y = water_depth)
    # M_surface = Integral( 0.5 * depth^2 ) dx
    integrand = 0.5 * depths**2
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
    # 1. Calculate Available Energy
    H = weir_H(Q, L)
    A_o = L * H
    V_o = Q / A_o
    E_total = P + H + (V_o ** 2) / (2 * g)

    # Objective function: E_calculated - E_available
    def energy_obj(y1):
        y1_v = y1[0] if isinstance(y1, np.ndarray) else y1
        if y1_v <= 0.001: return 1e5 + abs(y1_v) * 1e5

        A1, _ = get_geometry_props(y1_v, xs1, xs2, dist)
        if A1 <= 0: return 1e9

        E_toe = y1_v + (Q ** 2) / (2 * g * A1 ** 2)
        return E_toe - E_total

    try:
        y1_guess = fsolve(energy_obj, x0=0.01)[0]
    except RuntimeWarning:
        # Fallback: Critical Depth
        def froude_obj(y_c):
            fr = calc_froude_custom(Q, y_c, xs1, xs2, dist)
            return fr - 1.0

        try:
            y1_guess = fsolve(froude_obj, x0=H)[0]
        except:
            q = Q / L
            y1_guess = (q ** 2 / g) ** (1 / 3)

    # Sanity Check
    Fr = calc_froude_custom(Q, y1_guess, xs1, xs2, dist)
    if Fr < 1.0:
        try:
            y1_super = fsolve(energy_obj, x0=0.001)[0]
            if y1_super > 0: y1_guess = y1_super
        except:
            pass

    return y1_guess


def solve_y2_jump(Q, y1, xs1, xs2, dist):
    A1, y_cj1 = get_geometry_props(y1, xs1, xs2, dist)
    M1 = (Q ** 2 / (g * A1)) + (A1 * y_cj1)

    def momentum_obj(y2):
        y2_v = y2[0] if isinstance(y2, np.ndarray) else y2
        if y2_v <= 0: return 1e9
        A2, y_cj2 = get_geometry_props(y2_v, xs1, xs2, dist)
        M2 = (Q ** 2 / (g * A2)) + (A2 * y_cj2)
        return M2 - M1

    y2_sol = fsolve(momentum_obj, x0=y1 * 2)[0]
    return y2_sol

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
