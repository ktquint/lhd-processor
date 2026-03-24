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


def weir_coef_adv(H_input, P_input):
    if P_input == -9999:
        return -9999
    
    return 0.611 + 0.075 * H_input/P_input


def solve_yc(Q, xs1, xs2, dist):
    """
    Finds the critical depth (yc) where the Froude number is exactly 1.0.
    This is the physical boundary between supercritical and subcritical flow.
    """
    def fr_residual(y):
        return calc_froude_custom(Q, y, xs1, xs2, dist) - 1.0

    try:
        low = 1e-4
        high = 30.0
        
        f_low = fr_residual(low)
        f_high = fr_residual(high)
        
        # Expand upper bound if needed
        iter_count = 0
        while f_high > 0 and iter_count < 10:
            high *= 2.0
            f_high = fr_residual(high)
            iter_count += 1
            
        if f_low * f_high > 0:
            return 0.5

        # brentq is extremely reliable if the signs at the bounds differ.
        sol = root_scalar(fr_residual, bracket=(low, high), method='brentq')
        return sol.root
    except ValueError:
        # Fallback if the range is weird
        return 0.5


def get_top_width(water_depth, xs1, xs2, dist):
    """
    Calculates Top Width using explicitly provided left/right profiles.
    xs1: Center -> Left Bank
    xs2: Center -> Right Bank
    """
    if water_depth <= 0:
        return 0.001

    y_combined = np.concatenate([xs1[::-1], xs2[1:]])
    n_left = len(xs1)
    x_combined = (np.arange(len(y_combined)) - (n_left - 1)) * dist

    top_width = 0.0
    
    for i in range(len(x_combined) - 1):
        y_a = y_combined[i]
        y_b = y_combined[i+1]
        x_a = x_combined[i]
        x_b = x_combined[i+1]
        
        d_a = water_depth - y_a
        d_b = water_depth - y_b
        
        if d_a <= 0 and d_b <= 0:
            continue
            
        if d_a > 0 and d_b > 0:
            top_width += (x_b - x_a)
        else:
            if d_a > 0:
                ratio = d_a / (d_a - d_b)
                top_width += (x_b - x_a) * ratio
            else:
                ratio = d_b / (d_b - d_a)
                top_width += (x_b - x_a) * ratio

    if top_width <= 0:
        return 0.001

    return top_width


def get_xs_props(water_depth, xs1, xs2, dist):
    """
    Calculates Area (A) and Centroid Depth (y_cent).
    """
    if water_depth <= 0:
        return 0.0001, 0.0001

    y_combined = np.concatenate([xs1[::-1], xs2[1:]])
    n_left = len(xs1)
    x_combined = (np.arange(len(y_combined)) - (n_left - 1)) * dist

    area = 0.0
    moment = 0.0
    
    for i in range(len(x_combined) - 1):
        y_a = y_combined[i]
        y_b = y_combined[i+1]
        x_a = x_combined[i]
        x_b = x_combined[i+1]
        
        d_a = water_depth - y_a
        d_b = water_depth - y_b
        
        if d_a <= 0 and d_b <= 0:
            continue
            
        if d_a > 0 and d_b > 0:
            w = (x_b - x_a)
            a_trap = 0.5 * (d_a + d_b) * w
            m_trap = (w / 6.0) * (d_a**2 + d_a*d_b + d_b**2)
            area += a_trap
            moment += m_trap
        else:
            if d_a > 0:
                ratio = d_a / (d_a - d_b)
                w = (x_b - x_a) * ratio
                a_tri = 0.5 * d_a * w
                m_tri = a_tri * (d_a / 3.0)
                area += a_tri
                moment += m_tri
            else:
                ratio = d_b / (d_b - d_a)
                w = (x_b - x_a) * ratio
                a_tri = 0.5 * d_b * w
                m_tri = a_tri * (d_b / 3.0)
                area += a_tri
                moment += m_tri
                
    if area <= 0:
        return 0.0001, 0.0001
        
    y_cent = moment / area
    return area, y_cent


def calc_froude_custom(Q, y, xs1, xs2, dist):
    """
    Calculates Froude number: Fr = V / sqrt(g * D)
    """
    if y <= 0: return 0

    A, _ = get_xs_props(y, xs1, xs2, dist)
    T = get_top_width(y, xs1, xs2, dist)

    if A <= 0 or T <= 0: return 0

    V = Q / A
    D = A / T

    return V / np.sqrt(g * D)


def calc_y2_adv(Q, y1, L, xs1, xs2, dist, return_momentum=False):
    # 1. Calculate Momentum at y1 (Supercritical side)
    A1 = y1 * L
    y_cj1 = y1/2

    # Safety check for bad y1
    if A1 <= 0.0001: 
        return (0.0, 0.0, 0.0) if return_momentum else 0.0

    M1 = (Q ** 2 / (g * A1)) + (A1 * y_cj1)

    def momentum_residual(y_candidate, target_M):
        if y_candidate <= 0: return -1e9
        A2, y_cj2 = get_xs_props(y_candidate, xs1, xs2, dist)
        if A2 <= 0: return -1e9
        M_candidate = (Q ** 2 / (g * A2)) + (A2 * y_cj2)
        return M_candidate - target_M

    yc = solve_yc(Q, xs1, xs2, dist)
    
    # Evaluate momentum at critical depth (minimum possible momentum)
    f_yc = momentum_residual(yc, M1)
    if f_yc > 0:
        # Minimum momentum is still > M1, no conjugate depth exists
        return (0.0, M1, f_yc + M1) if return_momentum else 0.0

    # 2. Find a valid bracket for the Subcritical Root (y2)
    lower_bound = yc
    upper_bound = max(yc * 2.0, y1 * 2.0)
    
    if upper_bound <= lower_bound:
        upper_bound = lower_bound * 2.0

    try:
        f_upper = momentum_residual(upper_bound, M1)
        iter_count = 0

        # Keep doubling until we find a depth with enough momentum
        while f_upper < 0 and iter_count < 20:
            upper_bound *= 2.0
            f_upper = momentum_residual(upper_bound, M1)
            iter_count += 1

        # If we still can't balance momentum, fail gracefully
        if f_upper < 0:
            return (0.0, M1, f_upper + M1) if return_momentum else 0.0

        # 3. Use Brent's Method to find subcritical y2
        sol = root_scalar(momentum_residual, args=(M1,), bracket=(lower_bound, upper_bound), method='brentq')
        y2_sol = sol.root

        # # 4. Find the supercritical "conjugate" depth in the custom cross-section
        # #    This is searching for y < yc where momentum = M1
        # alt_lower = 1e-4
        # alt_upper = yc
        # y_alt = y2_sol # Default to y2_sol if we can't find it
        #
        # f_alt_lower = momentum_residual(alt_lower, M1)
        # f_alt_upper = momentum_residual(alt_upper, M1)
        #
        # if f_alt_lower * f_alt_upper < 0:
        #     try:
        #         sol_alt = root_scalar(momentum_residual, args=(M1,), bracket=(alt_lower, alt_upper), method='brentq')
        #         y_alt = sol_alt.root
        #     except ValueError:
        #         pass
        #
        # # Return the minimum of the two conjugate depths found in the custom channel
        # final_y2 = min(y2_sol, y_alt)
        #
        # if return_momentum:
        #     A2, y_cj2 = get_xs_props(final_y2, xs1, xs2, dist)
        #     M2 = (Q ** 2 / (g * A2)) + (A2 * y_cj2) if A2 > 0 else 0.0
        #     return final_y2, M1, M2
            
        return y2_sol

    except ValueError:
        return (0.0, M1, 0.0) if return_momentum else 0.0


def Fr_eq(Fr, x):
    A = (9 / (4 * C_W**2)) * 0.5 * Fr**2
    term1 = A**(1/3) * (1 + 1/x)
    term2 = 0.5 * Fr**2 * (1 + 0.1/x)
    return 1 - term1 + term2


def weir_H_simp(Q, L):
    """
    Analytically solves for Head (H).
    """
    H = ((3/2) * (Q/L) / C_W / np.sqrt(2 * g))**(2/3)
    return H


def weir_H_adv(Q, L, Y_T, total_height):
    """
    Unified solver that handles both free-flow (Rehbock) and
    submerged (Azimi) regimes in a single iteration.
    """
    # 1. Initial guess based on free-flow assumptions
    H_guess = weir_H_simp(Q, L)

    def residual(H):
        P = total_height - H
        t = Y_T - P  # Depth of tailwater above the crest

        # Guard against invalid geometry
        if P <= 0: return 1e9

        if t <= 0:
            # REGIME 1: FREE FLOW (Wahl/Rehbock)
            # Use the standard coefficient for plunging jets
            Cw = 0.611 + 0.075 * (H / P)
        else:
            # REGIME 2: SUBMERGED (Azimi)
            # Apply the submergence ratio s = t/H
            s = min(t / H, 0.999)  # Guard against drowning out
            Cw = np.sqrt(2 / 5) * np.sqrt(1 - s ** 2)

        # Calculate Q based on the regime-specific coefficient
        Q_calc = (2 / 3) * Cw * L * np.sqrt(2 * 9.81) * (H ** 1.5)
        return Q_calc - Q

    try:
        # Solve for H using the combined residual function
        sol = root_scalar(residual, bracket=(0.1 * H_guess, 10.0 * H_guess), method='brentq')
        H_final = sol.root
        return H_final, total_height - H_final
    except ValueError:
        return H_guess, total_height - H_guess


def compute_y_flip_simp(Q, L, P):
    if P == -9999: return -9999
    H = weir_H_simp(Q, L)
    return (H + P) / 1.1


def compute_y_flip_adv(Q, L, P):
    """
    Solves for the specific Head (H) and Flip Depth (Y_flip) at the
    moment of transition for a given discharge Q.
    """
    H_guess = weir_H_simp(Q, L)

    def residual(H):
        # 1. Calculate submergence ratios for both regimes
        s_azimi = 0.48
        s_leutheusser = (1 - 0.1 * (P / H)) / 1.1

        # 2. Use the more conservative (larger) ratio
        s_limit = max(s_azimi, s_leutheusser)

        # 3. Use Azimi's submerged coefficient for this threshold
        Cw_flip = np.sqrt(2 / 5) * np.sqrt(1 - s_limit ** 2)

        # 4. Check if this H satisfies the discharge Q
        Q_calc = (2 / 3) * Cw_flip * L * np.sqrt(2 * g) * (H ** 1.5)
        return Q_calc - Q

    try:
        sol = root_scalar(residual, bracket=(0.01 * H_guess, 50.0 * H_guess), method='brentq')
        H_flip = sol.root
        s_final = max(0.48, (1 - 0.1 * (P / H_flip)) / 1.1)
        return (s_final * H_flip) + P
    except ValueError:
        return np.nan


def _solve_intersection(residual_func, q_min, q_max, fallback_val):
    """
    Helper to find intersection or return bounds based on residual signs.
    """
    res_min = residual_func(q_min)
    res_max = residual_func(q_max)
    
    if res_min * res_max < 0:
        try:
            sol = root_scalar(residual_func, bracket=(q_min, q_max), method='brentq')
            return sol.root
        except ValueError:
            return fallback_val
    else:
        # Signs are the same
        if res_min < 0:
            return q_min
        else:
            return q_max


def rating_curve_intercept_adv(L: float, P: float, a: float, b: float,
                               xs1: list[float], xs2: list[float], dist: float,
                               Q_min_search: float, Q_max_search: float) -> tuple[float, float]:
    """
    Solves the intercept of the tailwater rating curve and the conjugate and flip rating curves
    using advanced hydraulic calculations (XS geometry).
    """
    if P == -9999: return -9999, -9999

    def residual(Q, which):
        if Q <= 0.001: return 1e6
        
        y_t = a * Q ** b
        H = weir_H_simp(Q, L)
        y1 = solve_y1(H, P)
        
        if which == 'flip':
            y_target = compute_y_flip_simp(Q, L, P)
        elif which == 'conjugate':
            y_target = calc_y2_adv(Q, y1, L, xs1, xs2, dist)
        else:
            return 1e6
            
        return y_target - y_t

    Q_min = _solve_intersection(lambda q: residual(q, 'conjugate'), Q_min_search, Q_max_search, Q_min_search)
    Q_max = _solve_intersection(lambda q: residual(q, 'flip'), Q_min_search, Q_max_search, Q_max_search)
        
    return Q_min, Q_max


def solve_weir_geom(Q_input, L_input, YT_input, Wse_input):
    """
    Solves for Head (H) and Weir Height (P).
    """
    total_height = YT_input + Wse_input

    # 1. Solve H and P analytically
    H_solution, P_solution = weir_H_adv(Q_input, L_input, Wse_input, total_height)

    if P_solution < 0:
        P_solution = -9999
        H_solution = -9999

    return H_solution, P_solution

# Simplified Calculations

def solve_y1(H_input, P_input):
    """
        solves the polynomial
        (Y_1/H)**3 - (1 + P/H) * (Y_1/H)**2 + 4/9 * C_W**2 * (1 + C_L) = 0
        where 
            C_L = 0.1 * P/H
    """
    if P_input == -9999 or H_input == -9999: return -9999
    if H_input <= 0: return 0.0

    C_L = 0.1 * P_input / H_input
    
    # Coefficients for the cubic equation: ax^3 + bx^2 + cx + d = 0
    # x = Y_1 / H
    a = 1
    b = -(1 + P_input / H_input)
    c = 0
    d = (4/9) * (C_W ** 2) * (1 + C_L)
    
    # Find roots
    roots = np.roots([a, b, c, d])
    
    # Filter for real, positive roots
    real_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
    
    if not real_roots:
        return 0.1 # Fallback
        
    # The smallest positive root the supercritical depth (y1)
    # The largest is usually the subcritical depth (y0)
    y1_ratio = min(real_roots)
    
    return y1_ratio * H_input

def solve_Fr_simp(H_input, P_input):
    """
        solves the equation:
        1 - (9/(4 * C_W**2) * 0.5 * Fr**2)**(1/3) * (1 + P/H)
            + 0.5 * Fr**2 * (1 + C_L) = 0
    """
    if P_input == -9999 or H_input == -9999: return -9999
    if H_input <= 0: return 0.0

    C_L = 0.1 * P_input / H_input
    
    def residual(Fr):
        term1 = ( (9 / (4 * C_W**2)) * 0.5 * Fr**2 )**(1/3) * (1 + P_input/H_input)
        term2 = 0.5 * Fr**2 * (1 + C_L)
        return 1 - term1 + term2
        
    try:
        # Froude number is typically between 1 (critical) and 20 (very fast)
        sol = root_scalar(residual, bracket=(1.0, 20.0), method='brentq')
        return sol.root
    except ValueError:
        return 1.0 # Fallback to critical flow


def calc_y2_simp(H_input, P_input):
    """
        solve the Bélanger equation:
        Y_2 = Y_1 / 2 * (-1 + np.sqrt(1 + 8 * Fr_1**2))
    """
    if P_input == -9999 or H_input == -9999: return -9999

    y_1 = solve_y1(H_input, P_input)
    Fr_1 = solve_Fr_simp(H_input, P_input)
    
    return (y_1/2) * (-1 + np.sqrt(1 + 8 * Fr_1**2))


def calc_yFlip_simp(H_input, P_input):
    if P_input == -9999 or H_input == -9999: return -9999
    return (H_input + P_input) / 1.1
    

def rating_curve_intercepts_simp(L_input: float, P_input: float, a: float, b: float,
                                 Q_min_search: float, Q_max_search: float) -> tuple[float, float]:
    """
        solves the intercept of the tailwater rating curve and the conjugate and flip rating curves
    """
    if P_input == -9999: return -9999, -9999

    def residual(Q, which):
        H = weir_H_simp(Q, L_input)
        y_t = a * Q**b
        
        if which == 'flip':
            y_target = calc_yFlip_simp(H, P_input)
        elif which == 'conjugate':
            y_target = calc_y2_simp(H, P_input)
        else:
            return 1e6
            
        return y_target - y_t

    Q_min = _solve_intersection(lambda q: residual(q, 'conjugate'), Q_min_search, Q_max_search, Q_min_search)
    Q_max = _solve_intersection(lambda q: residual(q, 'flip'), Q_min_search, Q_max_search, Q_max_search)
        
    return Q_min, Q_max