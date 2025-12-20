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


def get_top_width(water_depth, x_coords, y_coords):
    """
    Calculates the top width (T) of the cross-section at a specific depth.
    """
    if water_depth <= 0:
        return 0.001

    # Find the deepest point to split the channel into Left and Right banks
    center_idx = np.argmin(y_coords)
    x_left, y_left = x_coords[:center_idx + 1], y_coords[:center_idx + 1]
    x_right, y_right = x_coords[center_idx:], y_coords[center_idx:]

    # Interpolate x-coordinates at the water surface elevation
    # Note: y_coords are usually relative to bed, so 'water_depth' is the target y
    x_l = np.interp(water_depth, y_left[::-1], x_left[::-1])
    x_r = np.interp(water_depth, y_right, x_right)

    return abs(x_r - x_l)


def get_geometry_props(water_depth, x_coords, y_coords):
    """
    Correctly calculates Area for a U-shaped channel by finding
    the width between the left and right banks at every level.
    """
    if water_depth <= 0:
        return 0.0001, 0.0001

    # We integrate from the bottom (0) up to the water_depth
    y_levels = np.linspace(0, water_depth, 100)
    widths = []

    # Find the deepest point to split the channel into Left and Right banks
    center_idx = np.argmin(y_coords)
    x_left, y_left = x_coords[:center_idx + 1], y_coords[:center_idx + 1]
    x_right, y_right = x_coords[center_idx:], y_coords[center_idx:]

    for y in y_levels:
        # Find where elevation 'y' intersects the left bank and right bank
        # We use interp on each side separately
        x_l = np.interp(y, y_left[::-1], x_left[::-1])  # Reverse to make y increasing
        x_r = np.interp(y, y_right, x_right)

        # Width is the horizontal distance between banks
        widths.append(abs(x_r - x_l))

    # Area = integral of widths with respect to depth (dy)
    A = np.trapezoid(widths, y_levels)

    # Calculate Centroid (Moment of area / Total Area)
    moment_bottom = np.trapezoid(np.array(widths) * y_levels, y_levels)
    y_centroid_from_bottom = moment_bottom / A if A > 0 else 0
    y_cent = water_depth - y_centroid_from_bottom

    return A, y_cent


def calc_froude_custom(Q, y, x_coords, y_coords):
    """
    Calculates Froude number for irregular channel: Fr = V / sqrt(g * D)
    Where D (Hydraulic Depth) = Area / Top Width
    """
    if y <= 0: return 0

    A, _ = get_geometry_props(y, x_coords, y_coords)
    T = get_top_width(y, x_coords, y_coords)

    if A <= 0 or T <= 0: return 0

    V = Q / A
    D = A / T  # Hydraulic Depth

    return V / np.sqrt(g * D)


def solve_y1_downstream(Q, L, P, x_coords, y_coords):
    # 1. Calculate Available Energy
    H = weir_H(Q, L)
    A_o = L * H
    V_o = Q / A_o
    E_total = P + H + (V_o ** 2) / (2 * g)

    # Objective function: E_calculated - E_available
    def energy_obj(y1):
        y1_v = y1[0] if isinstance(y1, np.ndarray) else y1
        if y1_v <= 0.001: return 1e5 + abs(y1_v) * 1e5  # Smooth penalty
        A1, _ = get_geometry_props(y1_v, x_coords, y_coords)
        if A1 <= 0: return 1e9
        E_toe = y1_v + (Q ** 2) / (2 * g * A1 ** 2)
        return E_toe - E_total

    try:
        # ATTEMPT 1: Solve for Supercritical Depth using Energy
        y1_guess = fsolve(energy_obj, x0=0.01)[0]

    except RuntimeWarning:
        # ATTEMPT 2: Solver failed (likely Choked Flow).
        # Try to find Custom Critical Depth (where Fr = 1)

        def froude_obj(y_c):
            # Returns 0 when Fr = 1
            fr = calc_froude_custom(Q, y_c, x_coords, y_coords)
            return fr - 1.0

        try:
            # Guess critical depth near the weir head H
            y1_guess = fsolve(froude_obj, x0=H)[0]

        except:
            # ATTEMPT 3: The geometry is likely too noisy for the solver.
            # Use Analytical Rectangular Critical Depth as the ultimate failsafe.
            # y_c = (q^2 / g)^(1/3)
            q = Q / L
            y1_guess = (q ** 2 / g) ** (1 / 3)

    # Final Sanity Check: Ensure we didn't find the Subcritical (deep) solution by mistake
    Fr = calc_froude_custom(Q, y1_guess, x_coords, y_coords)
    if Fr < 1.0:
        try:
            # Force a look for the supercritical root
            y1_super = fsolve(energy_obj, x0=0.001)[0]
            if y1_super > 0: y1_guess = y1_super
        except:
            pass

    return y1_guess


def solve_y2_jump(Q, y1, x_coords, y_coords):
    A1, y_cj1 = get_geometry_props(y1, x_coords, y_coords)
    M1 = (Q ** 2 / (g * A1)) + (A1 * y_cj1)

    def momentum_obj(y2):
        y2_v = y2[0] if isinstance(y2, np.ndarray) else y2
        if y2_v <= 0: return 1e9
        A2, y_cj2 = get_geometry_props(y2_v, x_coords, y_coords)
        M2 = (Q ** 2 / (g * A2)) + (A2 * y_cj2)
        return M2 - M1

    y2_sol = fsolve(momentum_obj, x0=y1 * 5)[0]  # Extract the number
    return y2_sol


def Fr_eq(Fr, x):
    A = (9 / (4 * C_W**2)) * 0.5 * Fr**2
    term1 = A**(1/3) * (1 + 1/x)
    term2 = 0.5 * Fr**2 * (1 + 0.1/x)
    return 1 - term1 + term2


def weir_H(Q, L):
    """
    Analytically solves for Head (H) given Flow (Q) and Length (L).
    Since Cd is constant, H does not depend on P.
    Equation: Q = (2/3) * Cd * sqrt(2g) * L * H^(3/2)
    """
    H = ((3/2) * (Q/L) / C_W / np.sqrt(2 * g))**(2/3)
    return H


def compute_y_flip(Q, L, P):
    H = weir_H(Q, L)
    return (H + P) / 1.1


def rating_curve_intercept(Q: float, L: float, P: float, a: float, b: float,
                           x_coords: list[float], y_coords: list[float], which: str) -> float:
    y_flip = compute_y_flip(Q, L, P)
    y_1 = solve_y1_downstream(Q, L, P, x_coords, y_coords)
    y_2 = solve_y2_jump(Q, y_1, x_coords, y_coords)

    # Note: This still relies on a/b for the intercept check,
    # but the calling code in classes.py now uses VDT interpolation for y_t
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

    # Sanity check
    if P_solution < 0:
        P_solution = 0.1
        H_solution = total_height - P_solution

    return H_solution, P_solution
