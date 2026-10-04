"""Synthetic regression checks for tracking and the two counting policies."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "Background modeling"


def load_demo(module_name, filename):
    specification = importlib.util.spec_from_file_location(
        module_name, SOURCE_DIRECTORY / filename
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


TRACKER_MODULE = load_demo("centroid_tracker", "centroid_tracker.py")
with patch.dict(sys.modules, {"centroid_tracker": TRACKER_MODULE}):
    COUNTING_MODULE = load_demo("counting_demo", "Counting.py")
    DIRECTIONAL_MODULE = load_demo("directional_demo", "demo_zliczanie.py")


class CentroidTrackerTests(unittest.TestCase):
    def test_empty_input_without_tracks_is_empty(self):
        tracker = TRACKER_MODULE.CentroidTracker()
        self.assertEqual({}, tracker.update([]))
        self.assertEqual(0, tracker.next_object_id)

    def test_first_detections_register_unique_ordered_identifiers(self):
        tracker = TRACKER_MODULE.CentroidTracker()
        objects = tracker.update([(10, 10), (30, 30)])
        self.assertEqual(((0, (10, 10)), (1, (30, 30))), tuple(objects.items()))
        tracker.deregister(0)
        tracker.register((50, 50))
        self.assertEqual((1, 2), tuple(tracker.objects))

    def test_nearest_matches_keep_identity_when_detection_order_changes(self):
        tracker = TRACKER_MODULE.CentroidTracker(max_distance=10)
        tracker.update([(0, 0), (100, 0)])
        objects = tracker.update([(103, 0), (2, 0)])
        self.assertEqual({0: (2, 0), 1: (103, 0)}, objects)

    def test_distance_threshold_is_inclusive(self):
        tracker = TRACKER_MODULE.CentroidTracker(max_distance=5)
        tracker.update([(0, 0)])
        self.assertEqual({0: (3, 4)}, tracker.update([(3, 4)]))
        self.assertEqual({0: (3, 4)}, tracker.update([(9, 4)]))
        self.assertEqual(1, tracker.disappeared[0])

    def test_multiple_missing_tracks_expire_without_mutating_the_iteration(self):
        tracker = TRACKER_MODULE.CentroidTracker(max_disappeared=1)
        tracker.update([(0, 0), (10, 0), (20, 0)])
        self.assertEqual(3, len(tracker.update([])))
        self.assertEqual({}, tracker.update([]))
        self.assertEqual({}, tracker.disappeared)

    def test_reappearance_resets_missing_frame_counter(self):
        tracker = TRACKER_MODULE.CentroidTracker(max_disappeared=1)
        tracker.update([(0, 0)])
        tracker.update([])
        tracker.update([(1, 0)])
        self.assertEqual(0, tracker.disappeared[0])
        self.assertEqual({0: (1, 0)}, tracker.update([]))

    def test_one_detection_is_not_assigned_to_two_existing_tracks(self):
        tracker = TRACKER_MODULE.CentroidTracker(max_distance=10)
        tracker.update([(0, 0), (8, 0)])
        tracker.update([(1, 0)])
        self.assertEqual((1, 0), tracker.objects[0])
        self.assertEqual((8, 0), tracker.objects[1])
        self.assertEqual({0: 0, 1: 1}, tracker.disappeared)

    def test_more_detections_register_unmatched_centroids(self):
        tracker = TRACKER_MODULE.CentroidTracker(max_distance=10)
        tracker.update([(0, 0)])
        objects = tracker.update([(1, 0), (100, 0)])
        self.assertEqual({0: (1, 0), 1: (100, 0)}, objects)

    def test_equal_size_unmatched_input_preserves_existing_matching_policy(self):
        tracker = TRACKER_MODULE.CentroidTracker(max_disappeared=0, max_distance=5)
        tracker.update([(0, 0)])
        self.assertEqual({}, tracker.update([(20, 0)]))
        self.assertEqual({1: (20, 0)}, tracker.update([(20, 0)]))

    def test_numpy_detections_do_not_require_boolean_array_conversion(self):
        tracker = TRACKER_MODULE.CentroidTracker()
        tracker.update(np.array([[0, 0], [10, 10]]))
        objects = tracker.update(np.array([[1, 1], [11, 11]]))
        np.testing.assert_array_equal(objects[0], [1, 1])
        np.testing.assert_array_equal(objects[1], [11, 11])


class CountingPolicyTests(unittest.TestCase):
    def test_demo_imports_do_not_open_video_or_windows(self):
        with (
            patch.dict(sys.modules, {"centroid_tracker": TRACKER_MODULE}),
            patch.object(cv2, "VideoCapture") as capture,
            patch.object(cv2, "selectROI") as select_roi,
        ):
            load_demo("counting_import_check", "Counting.py")
            load_demo("directional_import_check", "demo_zliczanie.py")
        capture.assert_not_called()
        select_roi.assert_not_called()

    def test_demos_keep_distinct_tracking_thresholds(self):
        counting = COUNTING_MODULE.create_tracker()
        directional = DIRECTIONAL_MODULE.create_tracker()
        self.assertEqual((50, 80), (counting.max_disappeared, counting.max_distance))
        self.assertEqual((40, 50), (directional.max_disappeared, directional.max_distance))
        counting.update([(0, 0)])
        directional.update([(0, 0)])
        self.assertEqual({0: (60, 0)}, counting.update([(60, 0)]))
        self.assertEqual({0: (0, 0)}, directional.update([(60, 0)]))

    def test_band_counter_counts_immediately_but_only_once(self):
        tracks = {}
        self.assertEqual(1, COUNTING_MODULE.count_tracked_people({0: (50, 250)}, tracks, 250))
        self.assertEqual(0, COUNTING_MODULE.count_tracked_people({0: (50, 240)}, tracks, 250))
        self.assertEqual(0, COUNTING_MODULE.count_tracked_people({1: (50, 275)}, tracks, 250))

    def test_directional_counter_needs_history_and_counts_each_track_once(self):
        tracks = {}
        area = np.array([[300, 150], [500, 150], [500, 350], [300, 350]])
        update = DIRECTIONAL_MODULE.count_tracked_people
        self.assertEqual((0, 0, 0), update({0: (400, 210)}, tracks, 200, area))
        self.assertEqual((1, 0, 1), update({0: (400, 190)}, tracks, 200, area))
        self.assertEqual((0, 0, 0), update({0: (400, 220)}, tracks, 200, area))
        self.assertEqual((0, 0, 0), update({1: (100, 190)}, tracks, 200, area))
        self.assertEqual((0, 1, 0), update({1: (100, 210)}, tracks, 200, area))

    def test_area_boundary_is_included_and_outside_points_are_not_counted(self):
        area = np.array([[300, 150], [500, 150], [500, 350], [300, 350]])
        track = {"counted_area": False}
        self.assertEqual(0, DIRECTIONAL_MODULE.count_area_entry(track, (299, 150), area))
        self.assertEqual(1, DIRECTIONAL_MODULE.count_area_entry(track, (300, 150), area))
        self.assertEqual(0, DIRECTIONAL_MODULE.count_area_entry(track, (400, 200), area))

    def test_debug_overlay_can_be_enabled_or_disabled_for_excluded_regions(self):
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[20:30, 20:30] = 255
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        for show_debug in (False, True):
            with self.subTest(show_debug=show_debug), patch.object(cv2, "rectangle") as draw:
                centroids = COUNTING_MODULE.detect_centroids(
                    frame, mask, (20, 20, 10, 10), show_debug
                )
            self.assertEqual([], centroids)
            self.assertEqual(int(show_debug), draw.call_count)

    def test_wide_component_is_split_into_two_centroids(self):
        mask = np.zeros((60, 60), dtype=np.uint8)
        mask[20:30, 20:40] = 255
        frame = np.zeros((60, 60, 3), dtype=np.uint8)
        centroids = COUNTING_MODULE.detect_centroids(frame, mask, (0, 0, 1, 1), False)
        self.assertEqual([(25, 25), (35, 25)], centroids)


if __name__ == "__main__":
    unittest.main()
