"""Numerical and lifecycle regressions for the synthetic Kalman example."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = Path(__file__).resolve().parents[1] / "Background modeling" / "KalmanFilter.py"
SPEC = importlib.util.spec_from_file_location("kalman_under_test", SOURCE)
KALMAN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KALMAN)


class FilterTests(unittest.TestCase):
    def test_predict_update_and_distance_preserve_equations(self):
        model = KALMAN.make_cv_model(1.0, sigma_a=0.8, sigma_z=5.0)
        kalman_filter = KALMAN.KalmanFilterLinear(
            *model, [1.0, 2.0, 3.0, 4.0], np.eye(4)
        )
        predicted_state, predicted_covariance = kalman_filter.predict()
        np.testing.assert_allclose(predicted_state.ravel(), [4, 6, 3, 4])
        np.testing.assert_allclose(predicted_covariance, [
            [2.16, 0, 1.32, 0], [0, 2.16, 0, 1.32],
            [1.32, 0, 1.64, 0], [0, 1.32, 0, 1.64],
        ])
        state, covariance = kalman_filter.update([5.0, 7.0])
        np.testing.assert_allclose(state.ravel(), [
            4.079528718703976, 6.079528718703976,
            3.04860088365243, 4.04860088365243,
        ])
        np.testing.assert_allclose(covariance, [
            [1.988217967599411, 0, 1.215022091310751, 0],
            [0, 1.988217967599411, 0, 1.215022091310751],
            [1.215022091310751, 0, 1.5758468335787925, 0],
            [0, 1.215022091310751, 0, 1.5758468335787925],
        ])
        self.assertAlmostEqual(kalman_filter.mahalanobis2([5, 7]), 0.06278794551814625)
        state[:] = 999
        covariance[:] = 999
        self.assertFalse(np.any(kalman_filter.state == 999))
        self.assertFalse(np.any(kalman_filter.covariance == 999))

    def test_singular_innovation_uses_pseudoinverse(self):
        kalman_filter = KALMAN.KalmanFilterLinear(
            np.eye(2), np.eye(2), np.zeros((2, 2)),
            np.zeros((2, 2)), [0.0, 0.0], np.diag([1.0, 0.0]),
        )
        self.assertEqual(kalman_filter.mahalanobis2([3.0, 7.0]), 9.0)
        state, covariance = kalman_filter.update([3.0, 7.0])
        np.testing.assert_array_equal(state.ravel(), [3.0, 0.0])
        np.testing.assert_array_equal(covariance, np.zeros((2, 2)))


class SimulationTests(unittest.TestCase):
    def test_single_object_repeats_with_seed_and_generator(self):
        arguments = {"frame_count": 100, "sigma_z": 5.0, "miss_prob": 0.2}
        first_truth, first_measurements = KALMAN.simulate_single_object(**arguments, seed=41)
        second_truth, second_measurements = KALMAN.simulate_single_object(
            **arguments, rng=np.random.default_rng(41)
        )
        np.testing.assert_array_equal(first_truth, second_truth)
        self.assertEqual(first_measurements, second_measurements)
        _, other_measurements = KALMAN.simulate_single_object(**arguments, seed=42)
        self.assertNotEqual(first_measurements, other_measurements)
        np.testing.assert_allclose(first_truth[49], [75.0, 50.0])
        np.testing.assert_allclose(first_truth[50], [74.0, 51.5])
        self.assertIn(None, first_measurements)

    def test_multi_scene_repeats_and_keeps_objects_inside_walls(self):
        arguments = {"frame_count": 200, "n_objects": 4, "dt": 5.0}
        first_truth, first_frames, first_detections = KALMAN.simulate_multi_scene(
            **arguments, seed=12
        )
        second_truth, second_frames, second_detections = KALMAN.simulate_multi_scene(
            **arguments, rng=np.random.default_rng(12)
        )
        np.testing.assert_array_equal(first_truth, second_truth)
        self.assertEqual(first_detections, second_detections)
        self.assertEqual(first_truth.shape, (200, 4, 2))
        self.assertEqual(len(first_frames), 200)
        self.assertEqual(len(second_frames), 200)
        self.assertEqual(first_frames[0].shape, (240, 360))
        self.assertEqual(first_frames[0].dtype, np.uint8)
        self.assertTrue(np.all(first_truth[:, :, 0] >= 10))
        self.assertTrue(np.all(first_truth[:, :, 0] <= 350))
        self.assertTrue(np.all(first_truth[:, :, 1] >= 10))
        self.assertTrue(np.all(first_truth[:, :, 1] <= 230))

    def test_empty_simulations_and_all_missing_detections(self):
        truth, measurements = KALMAN.simulate_single_object(frame_count=0)
        self.assertEqual(truth.shape, (0, 2))
        self.assertEqual(measurements, [])
        _, measurements = KALMAN.simulate_single_object(frame_count=3, miss_prob=1.0)
        self.assertEqual(measurements, [None, None, None])
        truth, frames, detections = KALMAN.simulate_multi_scene(
            frame_count=3, miss_prob=1.0, false_pos_rate=0.0
        )
        self.assertEqual(truth.shape, (3, 4, 2))
        self.assertEqual(len(frames), 3)
        self.assertEqual(detections, [[], [], []])

    def test_first_experiment_repeats_without_display_in_same_process(self):
        with patch.object(plt, "show") as display:
            first_figure = KALMAN.run_experiment_1_single_object(show=False)
            second_figure = KALMAN.run_experiment_1_single_object(show=False)
            display.assert_not_called()
        self.assertEqual(len(first_figure.axes), 3)
        for first_axis, second_axis in zip(first_figure.axes, second_figure.axes):
            for first_line, second_line in zip(first_axis.lines, second_axis.lines):
                np.testing.assert_array_equal(
                    first_line.get_xydata(), second_line.get_xydata()
                )
            np.testing.assert_array_equal(
                first_axis.collections[0].get_offsets(),
                second_axis.collections[0].get_offsets(),
            )
        plt.close(first_figure)
        plt.close(second_figure)


class AssignmentTests(unittest.TestCase):
    def _check_assignments(self, scipy_enabled):
        with patch.object(KALMAN, "SCIPY_AVAILABLE", scipy_enabled):
            for shape in [(0, 0), (0, 2), (2, 0)]:
                self.assertEqual(KALMAN.solve_assignment(np.empty(shape), 10), [])
            costs = np.array([[1.0, 99.0, 99.0], [99.0, 2.0, 99.0]])
            original = costs.copy()
            self.assertEqual(set(KALMAN.solve_assignment(costs, 10)), {(0, 0), (1, 1)})
            np.testing.assert_array_equal(costs, original)
            self.assertEqual(KALMAN.solve_assignment(np.full((2, 2), 99.0), 10), [])
            self.assertEqual(KALMAN.solve_assignment(np.array([[10.0]]), 10), [(0, 0)])

    def test_greedy_assignment_and_empty_gates(self):
        self._check_assignments(False)

    @unittest.skipUnless(KALMAN.SCIPY_AVAILABLE, "SciPy is optional")
    def test_scipy_assignment_and_empty_gates(self):
        self._check_assignments(True)


class TrackerTests(unittest.TestCase):
    def test_tracks_are_retained_until_max_age_then_expire(self):
        tracker = KALMAN.MultiObjectTracker(max_age=2, min_hits=3)
        self.assertEqual(tracker.step([]), [])
        first_output = tracker.step([[1.0, 1.0]])
        self.assertEqual(len(first_output), 1)
        first_id = first_output[0][0]
        for detection in [[[1.2, 1.2]], [[1.3, 1.3]]]:
            self.assertEqual(tracker.step(detection)[0][0], first_id)
        self.assertEqual(tracker.tracks[0].hits, 3)
        self.assertEqual(tracker.step([])[0][0], first_id)
        self.assertEqual(tracker.step([])[0][0], first_id)
        self.assertEqual(tracker.step([]), [])
        self.assertEqual(tracker.tracks, [])

    def test_unconfirmed_track_keeps_immediate_detection_behavior(self):
        tracker = KALMAN.MultiObjectTracker(max_age=2, min_hits=3)
        self.assertEqual(len(tracker.step([[1.0, 1.0]])), 1)
        self.assertEqual(tracker.step([]), [])
        self.assertEqual(len(tracker.tracks), 1)

    def test_gating_creates_new_track_and_output_positions_are_copies(self):
        tracker = KALMAN.MultiObjectTracker(max_age=1, gate_d2=0.01)
        first_id = tracker.step([[0.0, 0.0]])[0][0]
        output = tracker.step([[1000.0, 1000.0]])
        self.assertEqual(len(tracker.tracks), 2)
        self.assertNotEqual(output[0][0], first_id)
        output[0][1][:] = -999
        self.assertFalse(np.any(tracker.tracks[-1].xy() == -999))

    def test_evaluation_reports_error_and_id_changes(self):
        truth = np.zeros((3, 1, 2))
        history = [[(1, [3.0, 4.0])], [(2, [3.0, 4.0])], []]
        rmse, id_switches = KALMAN.evaluate_mot(truth, history)
        self.assertEqual(rmse, 5.0)
        self.assertEqual(id_switches, 1)
        empty_rmse, empty_switches = KALMAN.evaluate_mot(truth, [[], [], []])
        self.assertTrue(np.isnan(empty_rmse))
        self.assertEqual(empty_switches, 0)


if __name__ == "__main__":
    unittest.main()
