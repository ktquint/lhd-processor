"""
    this file contains all the equations I'll use
    to calculate the hydraulic characteristics at
    the cross-sections of the low-head dams.
"""
import numpy as np
from scipy.optimize import fsolve


def get_geometry_props(target_depth, x_coords, y_coords):
    """
    Correctly calculates Area for a U-shaped channel by finding
    the width between the left and right banks at every level.
    """
    if target_depth <= 0:
        return 0.0001, 0.0001

    # We integrate from the bottom (0) up to the target_depth
    y_levels = np.linspace(0, target_depth, 100)
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
    y_cent = target_depth - y_centroid_from_bottom

    return A, y_cent


def solve_y1_downstream(Q, H, P, L, x_coords, y_coords, g=9.81):
    A_o = L * H
    V_o = Q / A_o
    E_total = P + H + (V_o ** 2) / (2 * g)  # Energy relative to channel bed

    def energy_obj(y1):
        y1_v = y1[0] if isinstance(y1, np.ndarray) else y1
        if y1_v <= 0: return 1e9
        A1, y_c1 = get_geometry_props(y1_v, x_coords, y_coords)

        E_toe = y1_v + (Q ** 2) / (2 * g * A1 ** 2)
        return E_toe - E_total

    return fsolve(energy_obj, x0=0.1)[0]


def solve_y2_jump(Q, y1, x_coords, y_coords, g=9.81):
    A1, y_cj1 = get_geometry_props(y1, x_coords, y_coords)
    M1 = (Q ** 2 / (g * A1)) + (A1 * y_cj1)

    def momentum_obj(y2):
        y2_v = y2[0] if isinstance(y2, np.ndarray) else y2
        if y2_v <= 0: return 1e9
        A2, y_cj2 = get_geometry_props(y2_v, x_coords, y_coords)
        M2 = (Q ** 2 / (g * A2)) + (A2 * y_cj2)
        return M2 - M1

    y2_sol = fsolve(momentum_obj, x0=y1 * 5)[0]  # Extract the number
    _, y_cj2_sol = get_geometry_props(y2_sol, x_coords, y_coords)
    return y2_sol


def weir_coef(x):
    return 0.611 + 0.075 * x


def weir_eq(H, P, q, g=9.81):
    """
    H = head (m)
    P = dam height (m)
    q = unit discharge (m**2/s)
    """
    x = H / P
    return (2 / 3) * weir_coef(x) * np.sqrt(2 * g) * H ** (3/2) - q


def Fr_eq(Fr, x):
    A = (9 / (4 * weir_coef(x)**2)) * 0.5 * Fr**2
    term1 = A**(1/3) * (1 + 1/x)
    term2 = 0.5 * Fr**2 * (1 + 0.1/x)
    return 1 - term1 + term2


def weir_H(Q, L, P, g=9.81):
    """
    Solves for the head (H) over a weir given Q, L, P, and g.
    Based on equations:
    Cw = 0.611 + 0.075 * (H / P)
    Q = (2/3) * L * Cw * sqrt(2g) * H^(3/2)
    """

    def objective_function(H_guess):
        # Prevent non-physical negative guesses during iteration
        H = max(H_guess[0], 0.0001)

        # Calculate Cw based on the provided formula
        Cw = 0.611 + 0.075 * (H / P)

        # Calculate Q based on the provided formula
        Q_calc = (2 / 3) * L * Cw * np.sqrt(2 * g) * (H ** 1.5)

        # We want Q_calc - Q_target to be zero
        return Q_calc - Q

    # Initial guess for H (usually 1.0 is a safe starting point)
    H_initial_guess = [1.0]
    H_solution = fsolve(objective_function, H_initial_guess)

    return H_solution[0]


def compute_y_flip(Q, L, P):
    H = weir_H(Q, L, P)
    return (H + P) / 1.1


def rating_curve_intercept(Q: float, L: float, P: float, a: float, b: float,
                           x_coords: list[float], y_coords: list[float], which: str) -> float:
    y_flip = compute_y_flip(Q, L, P)
    H = weir_H(Q, L, P)
    y_1 = solve_y1_downstream(Q, H, P, L, x_coords, y_coords)
    y_2 = solve_y2_jump(Q, y_1, x_coords, y_coords)

    y_t = a * Q ** b
    if which == 'flip':
        return y_flip - y_t
    elif which == 'conjugate':
        return y_2 - y_t
    else:
        raise ValueError("which must be 'flip' or 'conjugate'")


# noinspection PyTypeChecker
def compute_flip_and_conjugate(Q, L, P):
    """
        Computes flip bucket depth (y_flip) and conjugate depth (y2) for a low-head dam.

        Parameters:
        Q : float
            Total discharge (m³/s)
        L : float
            Crest width (m)
        P : float
            Dam height (m)

        Returns:
        y_flip : float
            Flip bucket depth (m)
        y_2 : float
            Conjugate depth (m)
    """
    # print(Q, L, P)
    q = Q / L
    H = fsolve(weir_eq, x0=1.0, args=(P, q))[0] # x0 is initial guess for H
    y_flip = (H + P) / 1.1

    # on leuthusser's graphs, the x-axis is H/P
    x = H / P
    coefs = [1, -(1 + 1 / x), 0, (4 / 9) * (0.611 + 0.075 * x) ** 2 * (1 + 0.1 / x)]
    y_1 = min([r.real for r in np.roots(coefs) if np.isreal(r) and r.real > 0]) * H
    Fr_1 = fsolve(Fr_eq, x0=100, args=(x,))[0] # x0 is initial guess for Fr_1
    y_2 = y_1 * 0.5 * (-1 + np.sqrt(1 + 8 * Fr_1 ** 2))
    return y_flip, y_2


def solve_weir_geometry(Q_input, L_input, YT_input, Wse_input, g=9.81):
    """
    Solves for Head (H) and Weir Height (P) given:
    Q_input   : Target Discharge
    YT_input  : Tailwater depth or reference height
    Wse_input : Delta WSE (Water Surface Elevation) component
    """

    # The total vertical room (P + H) is fixed by your line 7: P = Wse + YT - H
    # Therefore, P + H = Total_Height_Constant
    total_height_constant = Wse_input + YT_input

    def objective_function(H):
        # Prevent division by zero or negative heights during iteration
        P = max(total_height_constant - H, 0.001)

        # Cw formula from line 4 of your image
        Cw = 0.611 + 0.075 * (H / P)

        # Flow equation from line 1 of your image
        # Q = (2/3) * L * Cw * sqrt(2g) * H^(1.5)
        Q_calc = (2 / 3) * L_input * Cw * np.sqrt(2 * g) * (H ** 1.5)

        return Q_calc - Q_input

    # Solve for H (using 0.1 as a starting guess)
    H_solution = fsolve(objective_function, x0=0.1)[0]
    P_solution = total_height_constant - H_solution

    return H_solution, P_solution
