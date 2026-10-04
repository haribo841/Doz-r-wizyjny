"""Linear Kalman tracking demonstrations using reproducible synthetic data."""

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class KalmanFilterLinear:
    def __init__(
        self, transition, observation, process_noise, measurement_noise,
        initial_state, initial_covariance,
    ):
        self.transition = np.asarray(transition, dtype=float)
        self.observation = np.asarray(observation, dtype=float)
        self.process_noise = np.asarray(process_noise, dtype=float)
        self.measurement_noise = np.asarray(measurement_noise, dtype=float)
        self.state = np.asarray(initial_state, dtype=float).reshape(-1, 1)
        self.covariance = np.asarray(initial_covariance, dtype=float)
        self.identity = np.eye(self.state.shape[0])

    def predict(self):
        self.state = self.transition @ self.state
        self.covariance = (
            self.transition @ self.covariance @ self.transition.T
            + self.process_noise
        )
        return self.state.copy(), self.covariance.copy()

    def innovation(self, measurement):
        measurement = np.asarray(measurement, dtype=float).reshape(-1, 1)
        residual = measurement - self.observation @ self.state
        residual_covariance = (
            self.observation @ self.covariance @ self.observation.T
            + self.measurement_noise
        )
        return residual, residual_covariance

    def update(self, measurement):
        residual, residual_covariance = self.innovation(measurement)
        try:
            inverse = np.linalg.solve(
                residual_covariance, np.eye(residual_covariance.shape[0])
            )
        except np.linalg.LinAlgError:
            inverse = np.linalg.pinv(residual_covariance)
        gain = self.covariance @ self.observation.T @ inverse
        self.state = self.state + gain @ residual
        self.covariance = (self.identity - gain @ self.observation) @ self.covariance
        return self.state.copy(), self.covariance.copy()

    def mahalanobis2(self, measurement):
        """Return the squared Mahalanobis distance of a measurement."""
        residual, residual_covariance = self.innovation(measurement)
        try:
            weighted_residual = np.linalg.solve(residual_covariance, residual)
        except np.linalg.LinAlgError:
            weighted_residual = np.linalg.pinv(residual_covariance) @ residual
        return float((residual.T @ weighted_residual).item())


def make_cv_model(dt, sigma_a=1.0, sigma_z=3.0):
    """Create F, H, Q and R for a two-dimensional constant-velocity model."""
    transition = np.array([
        [1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1],
    ], dtype=float)
    observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2
    process_noise = sigma_a**2 * np.array([
        [dt4 / 4, 0, dt3 / 2, 0],
        [0, dt4 / 4, 0, dt3 / 2],
        [dt3 / 2, 0, dt2, 0],
        [0, dt3 / 2, 0, dt2],
    ], dtype=float)
    measurement_noise = sigma_z**2 * np.eye(2)
    return transition, observation, process_noise, measurement_noise


def greedy_assignment(cost_matrix, max_cost=np.inf):
    """Assign the cheapest available pairs when SciPy is unavailable."""
    track_count, detection_count = cost_matrix.shape
    candidates = [
        (cost_matrix[row, column], row, column)
        for row in range(track_count)
        for column in range(detection_count)
    ]
    candidates.sort(key=lambda candidate: candidate[0])
    pairs = []
    used_tracks = set()
    used_detections = set()
    for cost, row, column in candidates:
        if cost > max_cost:
            break
        if row in used_tracks or column in used_detections:
            continue
        used_tracks.add(row)
        used_detections.add(column)
        pairs.append((row, column))
    return pairs


def solve_assignment(cost_matrix, max_cost):
    """Use SciPy assignment or its greedy fallback with a common cost gate."""
    if cost_matrix.size == 0:
        return []
    gated_costs = cost_matrix.copy()
    gated_costs[gated_costs > max_cost] = max_cost + 1e6
    if not SCIPY_AVAILABLE:
        return greedy_assignment(gated_costs, max_cost=max_cost)
    rows, columns = linear_sum_assignment(gated_costs)
    return [
        (row, column)
        for row, column in zip(rows, columns)
        if gated_costs[row, column] <= max_cost
    ]


class Track:
    _next_id = 1

    def __init__(self, init_xy, dt, sigma_a, sigma_z):
        self.id = Track._next_id
        Track._next_id += 1
        transition, observation, process_noise, measurement_noise = make_cv_model(
            dt=dt, sigma_a=sigma_a, sigma_z=sigma_z
        )
        self.kf = KalmanFilterLinear(
            transition, observation, process_noise, measurement_noise,
            [init_xy[0], init_xy[1], 0.0, 0.0], np.diag([200, 200, 200, 200]),
        )
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
        return self.kf.state[:2].ravel()

    def mahalanobis2(self, det_xy):
        return self.kf.mahalanobis2(det_xy)


class MultiObjectTracker:
    def __init__(
        self, dt=1.0, sigma_a=0.8, sigma_z=5.0,
        max_age=10, min_hits=3, gate_d2=25.0,
    ):
        self.dt = dt
        self.sigma_a = sigma_a
        self.sigma_z = sigma_z
        self.max_age = max_age
        self.min_hits = min_hits
        self.gate_d2 = gate_d2
        self.tracks = []

    def _association_costs(self, detections):
        costs = np.zeros((len(self.tracks), len(detections)), dtype=float)
        for row, track in enumerate(self.tracks):
            for column, detection in enumerate(detections):
                costs[row, column] = track.mahalanobis2(detection)
        return costs

    def step(self, detections):
        for track in self.tracks:
            track.predict()
        detection_array = np.asarray(detections, dtype=float).reshape(-1, 2)
        costs = self._association_costs(detection_array)
        matches = solve_assignment(costs, max_cost=self.gate_d2)
        used_detections = {column for _, column in matches}
        for track_index, detection_index in matches:
            self.tracks[track_index].update(detection_array[detection_index])
        for detection_index, detection in enumerate(detection_array):
            if detection_index not in used_detections:
                self.tracks.append(Track(
                    detection, dt=self.dt,
                    sigma_a=self.sigma_a, sigma_z=self.sigma_z,
                ))
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update <= self.max_age
        ]
        # Retain the demo's immediate output of newly detected tracks.
        return [
            (track.id, track.xy().copy())
            for track in self.tracks
            if track.hits >= self.min_hits or track.time_since_update == 0
        ]


def _simulation_rng(rng, seed):
    """Use the supplied generator, or create a fresh one from the seed."""
    return np.random.default_rng(seed) if rng is None else rng


def simulate_single_object(
    frame_count=120, dt=1.0, start=(0, 0), v=(1.5, 1.0),
    sigma_z=4.0, miss_prob=0.15, *, rng=None, seed=0,
):
    """Generate one moving object; a generator takes priority over the seed."""
    generator = _simulation_rng(rng, seed)
    ground_truth = []
    measurements = []
    position_x, position_y = start
    velocity_x, velocity_y = v
    for frame_index in range(frame_count):
        if frame_index == 50:
            velocity_x, velocity_y = -1.0, 1.5
        position_x += velocity_x * dt
        position_y += velocity_y * dt
        ground_truth.append([position_x, position_y])
        if generator.random() < miss_prob:
            measurements.append(None)
        else:
            measurements.append([
                position_x + generator.standard_normal() * sigma_z,
                position_y + generator.standard_normal() * sigma_z,
            ])
    return np.asarray(ground_truth, dtype=float).reshape(-1, 2), measurements


def _maneuver_velocities(velocities, generator, maneuver_prob):
    for object_index in range(len(velocities)):
        if generator.random() < maneuver_prob:
            velocities[object_index] += generator.uniform(-0.8, 0.8, size=2)


def _reflect_at_walls(positions, velocities, frame_size):
    for object_index in range(len(positions)):
        for axis, limit in enumerate(reversed(frame_size)):
            if positions[object_index, axis] < 10 or positions[object_index, axis] > limit - 10:
                velocities[object_index, axis] *= -1
                positions[object_index, axis] = np.clip(
                    positions[object_index, axis], 10, limit - 10
                )


def _scene_detections(
    positions, generator, frame_size, sigma_z, miss_prob, false_pos_rate,
):
    detections = []
    for position in positions:
        if generator.random() < miss_prob:
            continue
        detections.append([
            position[0] + generator.standard_normal() * sigma_z,
            position[1] + generator.standard_normal() * sigma_z,
        ])
    frame_height, frame_width = frame_size
    for _ in range(generator.poisson(false_pos_rate)):
        detections.append([
            generator.uniform(0, frame_width),
            generator.uniform(0, frame_height),
        ])
    return detections


def simulate_multi_scene(
    frame_count=120, dt=1.0, n_objects=4, frame_size=(240, 360),
    sigma_z=4.0, miss_prob=0.15, false_pos_rate=0.3, maneuver_prob=0.04,
    *, rng=None, seed=0,
):
    """Generate several objects, missed detections and false positives."""
    generator = _simulation_rng(rng, seed)
    frame_height, frame_width = frame_size
    positions = np.stack([
        generator.uniform(40, frame_width - 40, size=n_objects),
        generator.uniform(40, frame_height - 40, size=n_objects),
    ], axis=1)
    velocities = np.stack([
        generator.uniform(-2.0, 2.0, size=n_objects),
        generator.uniform(-1.5, 1.5, size=n_objects),
    ], axis=1)
    ground_truth = np.zeros((frame_count, n_objects, 2), dtype=float)
    frames = []
    detections = []
    for frame_index in range(frame_count):
        _maneuver_velocities(velocities, generator, maneuver_prob)
        positions = positions + velocities * dt
        _reflect_at_walls(positions, velocities, frame_size)
        ground_truth[frame_index] = positions
        detections.append(_scene_detections(
            positions, generator, frame_size, sigma_z, miss_prob, false_pos_rate,
        ))
        # Placeholder frames remain available to the original callers.
        frames.append(np.zeros((frame_height, frame_width), dtype=np.uint8))
    return ground_truth, frames, detections


def match_tracks_to_gt(gt_xy, track_outputs, max_dist=30.0):
    """Match tracks to ground truth for one frame by Euclidean distance."""
    if len(track_outputs) == 0:
        return []
    track_ids = [track_id for track_id, _ in track_outputs]
    track_xy = np.array([position for _, position in track_outputs], dtype=float)
    distances = np.zeros((gt_xy.shape[0], track_xy.shape[0]), dtype=float)
    for row, ground_truth_position in enumerate(gt_xy):
        for column, track_position in enumerate(track_xy):
            distances[row, column] = np.linalg.norm(ground_truth_position - track_position)
    matches = solve_assignment(distances, max_cost=max_dist)
    return [
        (row, track_ids[column], distances[row, column])
        for row, column in matches
    ]


def evaluate_mot(ground_truth, track_history, max_dist=30.0):
    """Calculate RMSE and switches between IDs in consecutive matched frames."""
    frame_count, object_count, _ = ground_truth.shape
    squared_errors = []
    id_switches = 0
    previous_assigned = dict.fromkeys(range(object_count))
    for frame_index in range(frame_count):
        pairs = match_tracks_to_gt(
            ground_truth[frame_index], track_history[frame_index], max_dist=max_dist
        )
        current_assigned = dict.fromkeys(range(object_count))
        for object_index, track_id, distance in pairs:
            current_assigned[object_index] = track_id
            squared_errors.append(distance**2)
        id_switches += sum(
            previous_assigned[object_index] is not None
            and current_assigned[object_index] is not None
            and previous_assigned[object_index] != current_assigned[object_index]
            for object_index in range(object_count)
        )
        previous_assigned = current_assigned
    rmse = np.sqrt(np.mean(squared_errors)) if squared_errors else np.nan
    return rmse, id_switches


def run_experiment_1_single_object(*, rng=None, seed=0, show=True):
    """Compare process and measurement noise; return the generated figure."""
    print("\n--- Uruchamianie Eksperymentu 1: Single Object Tracking (Q/R tuning) ---")
    frame_count, dt = 100, 1.0
    ground_truth, measurements = simulate_single_object(
        frame_count=frame_count, dt=dt, sigma_z=5.0, miss_prob=0.2,
        rng=rng, seed=seed,
    )
    measurement_xy = np.array([
        measurement if measurement is not None else [np.nan, np.nan]
        for measurement in measurements
    ])
    scenarios = [
        {"title": "1) Małe Q (Sztywny)", "s_a": 0.05, "s_z": 5.0},
        {"title": "2) Duże Q (Nerwowy)", "s_a": 5.0, "s_z": 5.0},
        {"title": "3) Duże R (Wygładzanie)", "s_a": 0.8, "s_z": 15.0},
    ]
    figure, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    for axis, scenario in zip(axes, scenarios):
        transition, observation, process_noise, measurement_noise = make_cv_model(
            dt=dt, sigma_a=scenario["s_a"], sigma_z=scenario["s_z"]
        )
        kalman_filter = KalmanFilterLinear(
            transition, observation, process_noise, measurement_noise,
            [ground_truth[0, 0], ground_truth[0, 1], 0, 0], np.eye(4) * 50,
        )
        estimated_trajectory = []
        for measurement in measurements:
            kalman_filter.predict()
            if measurement is not None:
                kalman_filter.update(measurement)
            estimated_trajectory.append(kalman_filter.state[:2].ravel())
        estimated_trajectory = np.array(estimated_trajectory)
        axis.plot(ground_truth[:, 0], ground_truth[:, 1], "k--", label="GT", alpha=0.7)
        axis.scatter(
            measurement_xy[:, 0], measurement_xy[:, 1],
            c="r", s=15, alpha=0.5, label="Pomiary",
        )
        axis.plot(
            estimated_trajectory[:, 0], estimated_trajectory[:, 1],
            "b-", linewidth=2, label="KF Est",
        )
        axis.set_title(
            f"{scenario['title']}\n"
            rf"($\sigma_a={scenario['s_a']}, \sigma_z={scenario['s_z']}$)"
        )
        axis.grid(True)
    axes[0].legend()
    figure.tight_layout()
    if show:
        plt.show()
    return figure


def _track_scene(ground_truth, detections, *, max_age, gate_d2):
    tracker = MultiObjectTracker(
        dt=1.0, sigma_a=0.9, sigma_z=5.0,
        max_age=max_age, min_hits=3, gate_d2=gate_d2,
    )
    history = [tracker.step(frame_detections) for frame_detections in detections]
    counts = [len(outputs) for outputs in history]
    rmse, id_switches = evaluate_mot(ground_truth, history, max_dist=35.0)
    return rmse, id_switches, counts


def _plot_tracking_metrics(labels, results, object_count, titles, show):
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(labels, [result[0] for result in results])
    axes[0].set_title(titles[0])
    axes[1].bar(labels, [result[1] for result in results])
    axes[1].set_title(titles[1])
    for label, result in zip(labels, results):
        axes[2].plot(result[2], label=label, linewidth=2, alpha=0.8)
    axes[2].axhline(y=object_count, color="k", linestyle="--", label="GT")
    axes[2].set_title("Liczba śladów w czasie")
    axes[2].legend()
    for axis in axes:
        axis.grid(True)
    figure.tight_layout()
    if show:
        plt.show()
    return figure


def run_experiment_2_mot_parameters(*, rng=None, seed=0, show=True):
    """Compare gating and maximum track ages on common generated inputs."""
    print("\n--- Uruchamianie Eksperymentu 2: MOT Parameters (Gate/Age) ---")
    configs = [
        {"label": "Restrykcyjny (G=9, Age=5)", "gate": 9.0, "age": 5},
        {"label": "Zrównoważony (G=16, Age=12)", "gate": 16.0, "age": 12},
        {"label": "Luźny (G=30, Age=20)", "gate": 30.0, "age": 20},
    ]
    ground_truth, _, detections = simulate_multi_scene(
        frame_count=150, n_objects=6, sigma_z=5.0, miss_prob=0.2,
        false_pos_rate=0.5, maneuver_prob=0.05, rng=rng, seed=seed,
    )
    results = []
    for config in configs:
        result = _track_scene(
            ground_truth, detections,
            max_age=config["age"], gate_d2=config["gate"],
        )
        results.append(result)
        print(f"Cfg {config['label']}: RMSE={result[0]:.2f}, IDs={result[1]}")
    return _plot_tracking_metrics(
        [config["label"] for config in configs], results, 6,
        ("RMSE (mniej = lepiej)", "ID Switches (mniej = lepiej)"), show,
    )


def run_experiment_3_noise_robustness(*, rng=None, seed=0, show=True):
    """Compare fixed tracker parameters under low and high measurement noise."""
    print("\n--- Uruchamianie Eksperymentu 3: Noise Robustness ---")
    generator = _simulation_rng(rng, seed)
    scenarios = [
        {"name": "Standard (Low Noise)", "miss": 0.1, "fp": 0.2},
        {"name": "High Noise", "miss": 0.35, "fp": 1.0},
    ]
    results = []
    for scenario in scenarios:
        ground_truth, _, detections = simulate_multi_scene(
            frame_count=150, n_objects=6, sigma_z=5.0,
            miss_prob=scenario["miss"], false_pos_rate=scenario["fp"],
            maneuver_prob=0.05, rng=generator,
        )
        result = _track_scene(ground_truth, detections, max_age=10, gate_d2=20.0)
        results.append(result)
        print(f"Scenario {scenario['name']}: RMSE={result[0]:.2f}, IDs={result[1]}")
    return _plot_tracking_metrics(
        [scenario["name"] for scenario in scenarios], results, 6,
        ("RMSE vs Noise", "ID Switches vs Noise"), show,
    )


if __name__ == "__main__":
    run_experiment_1_single_object()
    run_experiment_2_mot_parameters()
    run_experiment_3_noise_robustness()
