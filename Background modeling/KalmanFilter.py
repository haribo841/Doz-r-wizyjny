import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. KONFIGURACJA I IMPORTY (Wspólne)
# -------------------------------------------------------------------------
try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

print(f"SciPy available: {SCIPY_AVAILABLE}")

np.random.seed(0)
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.grid"] = True

# -------------------------------------------------------------------------
# 2. CORE LIBRARY - KALMAN FILTER & TRACKING LOGIC
# -------------------------------------------------------------------------

class KalmanFilterLinear:
    def __init__(self, F, H, Q, R, x0, P0):
        self.F = np.asarray(F, dtype=float)
        self.H = np.asarray(H, dtype=float)
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.x = np.asarray(x0, dtype=float).reshape(-1, 1)
        self.P = np.asarray(P0, dtype=float)
        self.I = np.eye(self.x.shape[0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def innovation(self, z):
        z = np.asarray(z, dtype=float).reshape(-1, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        return y, S

    def update(self, z):
        y, S = self.innovation(z)
        # Numerical stability for inversion
        try:
            K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            K = self.P @ self.H.T @ np.linalg.pinv(S)
            
        self.x = self.x + K @ y
        self.P = (self.I - K @ self.H) @ self.P
        return self.x.copy(), self.P.copy()

    def mahalanobis2(self, z):
        """Zwraca kwadrat odległości Mahalanobisa."""
        y, S = self.innovation(z)
        try:
            # y.T * S^-1 * y
            d2 = y.T @ np.linalg.solve(S, y)
        except np.linalg.LinAlgError:
            d2 = y.T @ np.linalg.pinv(S) @ y
        return float(d2.item())

def make_cv_model(dt, sigma_a=1.0, sigma_z=3.0):
    """Tworzy macierze F, H, Q, R dla modelu Constant Velocity (2D)."""
    F = np.array([
        [1, 0, dt, 0 ],
        [0, 1, 0,  dt],
        [0, 0, 1,  0 ],
        [0, 0, 0,  1 ],
    ], dtype=float)

    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)

    dt2 = dt*dt
    dt3 = dt2*dt
    dt4 = dt2*dt2
    q = sigma_a**2
    Q = q * np.array([
        [dt4/4, 0,      dt3/2, 0     ],
        [0,     dt4/4,  0,     dt3/2 ],
        [dt3/2, 0,      dt2,   0     ],
        [0,     dt3/2,  0,     dt2   ],
    ], dtype=float)

    R = (sigma_z**2) * np.eye(2)
    return F, H, Q, R

def greedy_assignment(cost_matrix, max_cost=np.inf):
    """Prosty algorytm zachłanny, gdy SciPy jest niedostępne."""
    C = cost_matrix.copy()
    pairs = []
    n_tracks, n_dets = C.shape
    used_t = set()
    used_d = set()

    # spłaszcz i sortuj
    flat = [(C[i,j], i, j) for i in range(n_tracks) for j in range(n_dets)]
    flat.sort(key=lambda x: x[0])

    for c, i, j in flat:
        if c > max_cost:
            break
        if i in used_t or j in used_d:
            continue
        used_t.add(i)
        used_d.add(j)
        pairs.append((i, j))
    return pairs

def solve_assignment(cost_matrix, max_cost):
    """Wrapper obsługujący SciPy lub fallback do greedy."""
    # Kopiujemy i nakładamy karę na zbyt wysokie koszty
    C = cost_matrix.copy()
    mask = C > max_cost
    C[mask] = max_cost + 1e6

    matches = []
    
    if SCIPY_AVAILABLE:
        row_ind, col_ind = linear_sum_assignment(C)
        for r, c in zip(row_ind, col_ind):
            if C[r, c] <= max_cost:
                matches.append((r, c))
    else:
        matches = greedy_assignment(C, max_cost=max_cost)
        
    return matches

class Track:
    _NEXT_ID = 1

    def __init__(self, init_xy, dt, sigma_a, sigma_z):
        self.id = Track._NEXT_ID
        Track._NEXT_ID += 1

        F, H, Q, R = make_cv_model(dt=dt, sigma_a=sigma_a, sigma_z=sigma_z)
        x0 = np.array([init_xy[0], init_xy[1], 0.0, 0.0])
        P0 = np.diag([200, 200, 200, 200])
        self.kf = KalmanFilterLinear(F, H, Q, R, x0=x0, P0=P0)

        self.age = 1
        self.hits = 1
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, det_xy):
        self.kf.update(det_xy)
        self.hits += 1
        self.time_since_update = 0

    def xy(self):
        return self.kf.x[:2].ravel()

    def mahalanobis2(self, det_xy):
        return self.kf.mahalanobis2(det_xy)

class MultiObjectTracker:
    def __init__(self, dt=1.0, sigma_a=0.8, sigma_z=5.0, max_age=10, min_hits=3, gate_d2=25.0):
        self.dt = dt
        self.sigma_a = sigma_a
        self.sigma_z = sigma_z
        self.max_age = max_age
        self.min_hits = min_hits
        self.gate_d2 = gate_d2
        self.tracks = []

    def step(self, detections):
        # 1. Predykcja
        for tr in self.tracks:
            tr.predict()

        dets = np.array(detections, dtype=float) if len(detections) > 0 else np.zeros((0,2), dtype=float)

        # 2. Koszt (Mahalanobis^2)
        nT = len(self.tracks)
        nD = len(dets)
        C = np.zeros((nT, nD), dtype=float)
        
        if nT > 0 and nD > 0:
            for i, tr in enumerate(self.tracks):
                for j in range(nD):
                    C[i, j] = tr.mahalanobis2(dets[j])

        # 3. Asocjacja
        matches = solve_assignment(C, max_cost=self.gate_d2)
        
        # Helper sets
        used_tracks = set(r for r, c in matches)
        used_dets = set(c for r, c in matches)
        unmatched_dets = [j for j in range(nD) if j not in used_dets]

        # 4. Update przypisanych
        for ti, dj in matches:
            self.tracks[ti].update(dets[dj])

        # 5. Nowe tracki
        for dj in unmatched_dets:
            self.tracks.append(Track(dets[dj], dt=self.dt, sigma_a=self.sigma_a, sigma_z=self.sigma_z))

        # 6. Usuwanie starych (Death logic)
        self.tracks = [tr for tr in self.tracks if tr.time_since_update <= self.max_age]

        # 7. Wyjście (tylko potwierdzone)
        outputs = []
        for tr in self.tracks:
            if tr.hits >= self.min_hits or tr.time_since_update == 0:
                outputs.append((tr.id, tr.xy().copy()))
        return outputs

# -------------------------------------------------------------------------
# 3. SYMULACJA I EWALUACJA (Simulation & Metrics)
# -------------------------------------------------------------------------

def simulate_single_object(T=120, dt=1.0, start=(0, 0), v=(1.5, 1.0), sigma_z=4.0, miss_prob=0.15):
    """Symulacja pojedynczego obiektu"""
    gt, z = [], []
    x, y = start
    vx, vy = v
    for k in range(T):
        if k == 50: vx, vy = -1.0, 1.5 # Manewr
        x += vx*dt; y += vy*dt
        gt.append([x, y])
        
        if np.random.rand() < miss_prob:
            z.append(None)
        else:
            z.append([x + np.random.randn()*sigma_z, y + np.random.randn()*sigma_z])
    return np.array(gt), z

def simulate_multi_scene(
    T=120, dt=1.0, n_objects=4, frame_size=(240, 360),
    sigma_z=4.0, miss_prob=0.15, false_pos_rate=0.3, maneuver_prob=0.04
):
    """Symulacja wielu obiektów"""
    H, W = frame_size

    # inicjalizacja
    pos = np.stack([
        np.random.uniform(40, W-40, size=n_objects),
        np.random.uniform(40, H-40, size=n_objects),
    ], axis=1)

    vel = np.stack([
        np.random.uniform(-2.0, 2.0, size=n_objects),
        np.random.uniform(-1.5, 1.5, size=n_objects),
    ], axis=1)

    gt = np.zeros((T, n_objects, 2), dtype=float)
    frames = []     # nieużywane w logice trackera, ale zachowane dla kompatybilności
    detections = []

    for t in range(T):
        # manewry
        for i in range(n_objects):
            if np.random.rand() < maneuver_prob:
                vel[i] += np.random.uniform(-0.8, 0.8, size=2)

        # ruch
        pos = pos + vel * dt

        # ściany
        for i in range(n_objects):
            if pos[i,0] < 10 or pos[i,0] > W-10:
                vel[i,0] *= -1
                pos[i,0] = np.clip(pos[i,0], 10, W-10)
            if pos[i,1] < 10 or pos[i,1] > H-10:
                vel[i,1] *= -1
                pos[i,1] = np.clip(pos[i,1], 10, H-10)

        gt[t] = pos

        # generowanie pomiarów (z szumem i brakami)
        dets = []
        for i in range(n_objects):
            if np.random.rand() < miss_prob:
                continue
            zx = pos[i,0] + np.random.randn()*sigma_z
            zy = pos[i,1] + np.random.randn()*sigma_z
            dets.append([zx, zy])

        # false positives
        n_fp = np.random.poisson(false_pos_rate)
        for _ in range(n_fp):
            dets.append([np.random.uniform(0, W), np.random.uniform(0, H)])

        frames.append(np.zeros((H,W), dtype=np.uint8)) # placeholder
        detections.append(dets)

    return gt, frames, detections

def match_tracks_to_gt(gt_xy, track_outputs, max_dist=30.0):
    """Ewaluacja: dopasowanie tracków do GT w danej klatce."""
    if len(track_outputs) == 0:
        return []

    track_ids = [tid for tid, _ in track_outputs]
    track_xy = np.array([p for _, p in track_outputs], dtype=float)

    nG = gt_xy.shape[0]
    nT = track_xy.shape[0]

    # Macierz odległości euklidesowej
    C = np.zeros((nG, nT), dtype=float)
    for i in range(nG):
        for j in range(nT):
            C[i, j] = np.linalg.norm(gt_xy[i] - track_xy[j])

    # Dopasowanie (Hungarian / Greedy) z max_dist
    matches = solve_assignment(C, max_cost=max_dist)
    
    # Format wyniku: (index_gt, id_tracka, dystans)
    pairs = []
    for r, c in matches:
        pairs.append((r, track_ids[c], C[r, c]))
        
    return pairs

def evaluate_mot(gt, track_history, max_dist=30.0):
    """Liczenie RMSE i ID Switches."""
    T, nG, _ = gt.shape
    sq_err = []
    id_switches = 0

    prev_assigned = {i: None for i in range(nG)}

    for t in range(T):
        pairs = match_tracks_to_gt(gt[t], track_history[t], max_dist=max_dist)
        curr_assigned = {i: None for i in range(nG)}

        for (gi, tid, dist) in pairs:
            curr_assigned[gi] = tid
            sq_err.append(dist**2)

        # Wykrywanie ID Switch
        for gi in range(nG):
            if prev_assigned[gi] is not None and curr_assigned[gi] is not None:
                if prev_assigned[gi] != curr_assigned[gi]:
                    id_switches += 1

        prev_assigned = curr_assigned

    rmse = np.sqrt(np.mean(sq_err)) if len(sq_err) > 0 else np.nan
    return rmse, id_switches

# -------------------------------------------------------------------------
# 4. EKSPERYMENTY (Zadania z poszczególnych plików)
# -------------------------------------------------------------------------

def run_experiment_1_single_object():
    """
    Cel: Porównanie wpływu Q i R na śledzenie pojedynczego obiektu.
    """
    print("\n--- Uruchamianie Eksperymentu 1: Single Object Tracking (Q/R tuning) ---")
    T, dt = 100, 1.0
    gt, meas = simulate_single_object(T=T, dt=dt, sigma_z=5.0, miss_prob=0.2)
    meas_xy = np.array([m if m is not None else [np.nan, np.nan] for m in meas])

    scenarios = [
        {"title": "1) Małe Q (Sztywny)", "s_a": 0.05, "s_z": 5.0},
        {"title": "2) Duże Q (Nerwowy)", "s_a": 5.0,  "s_z": 5.0},
        {"title": "3) Duże R (Wygładzanie)", "s_a": 0.8,  "s_z": 15.0}
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray): axes = [axes] # zabezpieczenie dla 1 plotu

    for ax, sc in zip(axes, scenarios):
        F, H, Q, R = make_cv_model(dt=dt, sigma_a=sc["s_a"], sigma_z=sc["s_z"])
        kf = KalmanFilterLinear(F, H, Q, R, x0=[gt[0,0], gt[0,1], 0, 0], P0=np.eye(4)*50)
        
        est_traj = []
        for k in range(T):
            kf.predict()
            if meas[k] is not None:
                kf.update(meas[k])
            est_traj.append(kf.x[:2].ravel())
            
        est_traj = np.array(est_traj)
        ax.plot(gt[:,0], gt[:,1], 'k--', label="GT", alpha=0.7)
        ax.scatter(meas_xy[:,0], meas_xy[:,1], c='r', s=15, alpha=0.5, label="Pomiary")
        ax.plot(est_traj[:,0], est_traj[:,1], 'b-', linewidth=2, label="KF Est")
        ax.set_title(f"{sc['title']}\n($\sigma_a={sc['s_a']}, \sigma_z={sc['s_z']}$)")
        ax.grid(True)
        if sc == scenarios[0]: ax.legend()
    
    plt.tight_layout()
    plt.show()

def run_experiment_2_mot_parameters():
    """
    Cel: Porównanie parametrów Gating i Max Age w MOT.
    """
    print("\n--- Uruchamianie Eksperymentu 2: MOT Parameters (Gate/Age) ---")
    configs = [
        {"label": "Restrykcyjny (G=9, Age=5)",  "gate": 9.0,  "age": 5},
        {"label": "Zrównoważony (G=16, Age=12)", "gate": 16.0, "age": 12},
        {"label": "Luźny (G=30, Age=20)",       "gate": 30.0, "age": 20}
    ]

    # Generowanie wspólnych danych
    gt_exp, _, dets_exp = simulate_multi_scene(
        T=150, n_objects=6, sigma_z=5.0, miss_prob=0.2, false_pos_rate=0.5, maneuver_prob=0.05
    )

    results_rmse = []
    results_ids = []
    results_track_counts = []

    for cfg in configs:
        mot = MultiObjectTracker(
            dt=1.0, sigma_a=0.9, sigma_z=5.0, 
            max_age=cfg["age"], min_hits=3, gate_d2=cfg["gate"]
        )
        
        hist = []
        cnts = []
        for t in range(len(dets_exp)):
            out = mot.step(dets_exp[t])
            hist.append(out)
            cnts.append(len(out))
        
        rmse, ids = evaluate_mot(gt_exp, hist, max_dist=35.0)
        results_rmse.append(rmse)
        results_ids.append(ids)
        results_track_counts.append(cnts)
        print(f"Cfg {cfg['label']}: RMSE={rmse:.2f}, IDs={ids}")

    # Wykresy
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].bar([c["label"] for c in configs], results_rmse, color=['#ff9999','#66b3ff','#99ff99'])
    axes[0].set_title("RMSE (mniej = lepiej)")
    
    axes[1].bar([c["label"] for c in configs], results_ids, color=['#ffcc99','#c2c2f0','#ffb3e6'])
    axes[1].set_title("ID Switches (mniej = lepiej)")
    
    colors = ['r', 'b', 'g']
    for i, counts in enumerate(results_track_counts):
        axes[2].plot(counts, label=configs[i]["label"], color=colors[i], linewidth=2, alpha=0.7)
    axes[2].axhline(y=6, color='k', linestyle='--', label="GT (6)")
    axes[2].set_title("Liczba śladów w czasie")
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

def run_experiment_3_noise_robustness():
    """
    Cel: Porównanie działania trackera w szumie niskim vs wysokim.
    """
    print("\n--- Uruchamianie Eksperymentu 3: Noise Robustness ---")
    scenarios = [
        {"name": "Standard (Low Noise)", "miss": 0.1,  "fp": 0.2},
        {"name": "High Noise",           "miss": 0.35, "fp": 1.0}
    ]

    results_rmse = []
    results_ids = []
    results_track_counts = []
    N_OBJ = 6

    for sc in scenarios:
        # Generujemy dane specyficzne dla scenariusza
        gt, _, dets = simulate_multi_scene(
            T=150, n_objects=N_OBJ, sigma_z=5.0, 
            miss_prob=sc["miss"], false_pos_rate=sc["fp"], maneuver_prob=0.05
        )
        
        # Tracker ma stałe parametry
        mot = MultiObjectTracker(
            dt=1.0, sigma_a=0.9, sigma_z=5.0, 
            max_age=10, min_hits=3, gate_d2=20.0
        )
        
        hist = []
        cnts = []
        for t in range(len(dets)):
            out = mot.step(dets[t])
            hist.append(out)
            cnts.append(len(out))
            
        rmse, ids = evaluate_mot(gt, hist, max_dist=35.0)
        results_rmse.append(rmse)
        results_ids.append(ids)
        results_track_counts.append(cnts)
        print(f"Scenario {sc['name']}: RMSE={rmse:.2f}, IDs={ids}")

    # Wykresy
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar([s["name"] for s in scenarios], results_rmse, color=['#66b3ff', '#ff9999'])
    axes[0].set_title("RMSE vs Noise")

    axes[1].bar([s["name"] for s in scenarios], results_ids, color=['#c2c2f0', '#ff6666'])
    axes[1].set_title("ID Switches vs Noise")

    colors = ['blue', 'red']
    for i, counts in enumerate(results_track_counts):
        axes[2].plot(counts, label=scenarios[i]["name"], color=colors[i], alpha=0.8)
    axes[2].axhline(y=N_OBJ, color='green', linestyle='--', label="GT")
    axes[2].set_title("Liczba śladów")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
if __name__ == "__main__":
    run_experiment_1_single_object()
    run_experiment_2_mot_parameters()
    run_experiment_3_noise_robustness()