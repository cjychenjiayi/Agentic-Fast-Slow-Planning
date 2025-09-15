import numpy as np
from scipy.optimize import linear_sum_assignment

big_objects = ["vehicle", "doghouse", "atm", "bench"]

def polar_to_xy(r, theta_deg):
    theta_rad = np.deg2rad(theta_deg)
    return np.array([r * np.cos(theta_rad), r * np.sin(theta_rad)])

def normalize_scene(scene):
    r_min = min(r for (_, r, _) in scene)
    return [(("big" if c in big_objects else "small"), r / r_min, theta) for (c, r, theta) in scene], r_min

def build_cost_matrix(sceneA, sceneB, penalty=1.0):
    A_xy = [polar_to_xy(r, theta) for (_, r, theta) in sceneA]
    B_xy = [polar_to_xy(r, theta) for (_, r, theta) in sceneB]
    nA, nB = len(A_xy), len(B_xy)
    C = np.zeros((nA, nB))
    for i, (clsA, _, _) in enumerate(sceneA):
        for j, (clsB, _, _) in enumerate(sceneB):
            d = np.linalg.norm(A_xy[i] - B_xy[j])
            if clsA != clsB:
                d += penalty
            C[i, j] = d
    return C

def match_and_score(sceneA, sceneB, penalty=1.0):
    normA, rA_min = normalize_scene(sceneA)
    normB, rB_min = normalize_scene(sceneB)
    C = build_cost_matrix(normA, normB, penalty=penalty)
    row_ind, col_ind = linear_sum_assignment(C)
    A_xy_real = [polar_to_xy(r, theta) for (_, r, theta) in sceneA]
    B_xy_real = [polar_to_xy(r, theta) for (_, r, theta) in sceneB]
    scale = min(rA_min, rB_min)
    real_costs = []
    for i, j in zip(row_ind, col_ind):
        cost = np.linalg.norm(A_xy_real[i] - B_xy_real[j]) / scale
        if normA[i][0] != normB[j][0]:
            cost += penalty
        real_costs.append(cost)
    avg_cost = np.mean(real_costs)
    similarity = 1.0 / (1.0 + avg_cost)
    return {"pairs": list(zip(row_ind.tolist(), col_ind.tolist())), "avg_cost": avg_cost, "similarity": similarity}

if __name__ == "__main__":
    A = [("vehicle", 10.0, 0.0), ("bench", 5.0, -45.0), ("doghouse", 8.0, 30.0)]
    B = [("atm", 12.0, 10.0), ("bench", 6.0, -40.0), ("trafficcone", 9.0, 35.0)]

    res = match_and_score(A, B, penalty=1.0)
    print(res)
