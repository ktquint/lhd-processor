# """
# this is my attempt to convert Tony Wahl's spreadsheet to a python script
# """
# # import geoglows
# import numpy as np
# import pandas as pd
# from scipy.optimize import fsolve
#
# # constants
# g = 9.81 #m/s**2
# n = 0.030
# k = 1.0
# delta_z = 0.0
#
# def eq_3(H, P, q):
#     return (2 / 3) * (0.611 + 0.075 * (H / P)) * np.sqrt(2 * g) * H ** (3/2) - q
#
#
# def wahl_excel(lhd_csv):
#     """
#     given:
#     - BYU_SITE_ID
#     - WEIR_LENGTH (L)
#     - HEIGHT (P)
#     - DISCHARGE (Q)
#     - DATE (to find discharge, eventually)
#     find:
#     - Hydraulic Calcs
#     - Ideal Jump Calcs
#     - Tailwater Conditions
#     - Submerged Hydraulic Jump
#     """
#     lhd_df = pd.read_csv(lhd_csv)
#
#     # let's convert to metric first
#     lhd_df["b"] = lhd_df["CHANNEL_WIDTH_FT"] / 3.281
#     lhd_df["L"] = lhd_df["GIS_WEIR_LENGTH"] / 3.281
#     lhd_df["P"] = lhd_df["HEIGHT_FT"] / 3.281
#     lhd_df["Q"] = lhd_df["CFS"] / 35.315
#
#     # hydraulic calcs
#     lhd_df["H"] = ""
#     for index, row in lhd_df.iterrows():
#         H_guess = 1.0
#         P_i = row["P"]
#         q_i = row["Q"] / row["L"]
#         H_solution = fsolve(eq_3, H_guess, args=(P_i, q_i))[0]
#         lhd_df.at[index, "H"] = H_solution
#
#     lhd_df["H+P"] = lhd_df["H"] + lhd_df["P"]
#     lhd_df["H/(H+P)"] = lhd_df["H"] / lhd_df["H+P"]
#     lhd_df["C_W"] = 0.611 + 0.075 * (lhd_df["H"] / lhd_df["P"])
#     lhd_df["C_D"] = (2/3) * lhd_df["C_W"] * np.sqrt(2 * g)
#     lhd_df["C_L"] = 0.1 * lhd_df["P"] / lhd_df["H"]
#
#     # ideal jump calcs
#     lhd_df["Y_1/H"] = ""
#     for index, row in lhd_df.iterrows():
#         a = 1
#         b = -(1 + row["P"] / row["H"])
#         c = 0
#         d = (4/9) * row["C_W"]**2 * (1 + row["C_L"])
#         coeffs = [a,b,c,d]
#         roots = np.roots(coeffs)
#         positive_real_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
#         lhd_df.at[index, "Y_1/H"] = min(positive_real_roots)
#     lhd_df["Y_1"] = lhd_df["Y_1/H"] * lhd_df["H"]
#     lhd_df["V_1"] = lhd_df["Q"] / lhd_df["L"] / lhd_df["Y_1"]
#     lhd_df["F_1"] = lhd_df["V_1"] / g**0.5 / lhd_df["Y_1"]**0.5
#     lhd_df["Y_2/H"] = lhd_df["Y_1/H"]/2 * (-1 + (1 + 8 * lhd_df["F_1"]**2)**0.5)
#     lhd_df["Y_2"] = lhd_df["Y_2/H"] * lhd_df["H"]
#
#     # tailwater conditions
#     lhd_df["Y_T"] = ((lhd_df["Q"]/lhd_df["b"])*n/k/lhd_df["SLOPE"]**0.5)**0.6 + delta_z
#     lhd_df["S"] = (lhd_df["Y_T"] - lhd_df["Y_2"]) / lhd_df["Y_2"]
#
#     # submerged hydraulic jump
#     lhd_df["Y_Flip"] = lhd_df["H+P"] / 1.1
#     lhd_df["Predicted_Jump"] = ""
#     for index, row in lhd_df.iterrows():
#         if row["Y_T"] < row["Y_2"]:
#             jump_type = "TYPE_A"
#         elif row["Y_T"] == row["Y_2"]:
#             jump_type = "TYPE_B"
#         elif row["Y_T"] < row["Y_Flip"] and row["Y_2"] < row["Y_Flip"]:
#             jump_type = "TYPE_C"
#         else:
#             jump_type = "TYPE_D"
#         lhd_df.at[index, "Predicted_Jump"] = jump_type
#     lhd_df.to_csv("C:/Users/ki87ujmn/Downloads/tony_test_results.csv", index=False)
#
# wahl_excel("C:/Users/ki87ujmn/Downloads/tony_test.csv")
