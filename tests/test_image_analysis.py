"""Synthetic regressions that do not require videos, datasets or model weights."""

import ast
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
BACKGROUND = REPOSITORY / 'Background modeling'


def import_script(filename):
    spec = importlib.util.spec_from_file_location('image_analysis_under_test', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def isolated_function(filename, function_name, dependencies):
    """Load an actual function without running its demo or loading a neural model."""
    tree = ast.parse(filename.read_text(encoding='utf-8-sig'), filename=str(filename))
    definition = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.Module(body=[definition], type_ignores=[])
    namespace = dict(dependencies)
    exec(compile(module, str(filename), 'exec'), namespace)
    return namespace[function_name]


class ForgeryDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.detector = import_script(REPOSITORY / 'forgery_detection.py')

    def test_zigzag_diagonal_order(self):
        block = np.arange(16).reshape(4, 4)
        expected = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15]
        np.testing.assert_array_equal(self.detector.zigzag_values(block), expected)

    def test_dct_features_and_original_block_origins(self):
        gray = np.full((10, 11), 255, dtype=np.uint8)
        descriptors = self.detector.block_descriptors(gray)
        self.assertEqual(descriptors.shape, (6, 18))
        np.testing.assert_array_equal(descriptors[:, -2:], [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]])
        np.testing.assert_array_equal(descriptors[:, 0], np.full(6, 15))
        np.testing.assert_array_equal(descriptors[:, 1:16], np.zeros((6, 15)))
        self.assertTrue(np.all(np.isfinite(descriptors)))

    def test_images_without_block_origins_produce_no_uninitialized_rows(self):
        for shape in ((8, 8), (7, 10), (10, 7)):
            with self.subTest(shape=shape):
                descriptors = self.detector.block_descriptors(np.zeros(shape, dtype=np.uint8))
                self.assertEqual(descriptors.shape, (0, 18))

    def test_matching_thresholds_and_neighbor_range(self):
        descriptors = np.zeros((12, 18))
        descriptors[:, -2] = np.arange(12) * 20
        matches = self.detector.find_similar_blocks(descriptors)
        self.assertEqual(matches.shape, (18, 6))
        np.testing.assert_array_equal(matches[0], [0, 0, 20, 0, -20, 0])
        self.assertEqual(len(self.detector.find_similar_blocks(descriptors, distance=21)), 16)
        descriptors[1:, 0] = 6
        changed_matches = self.detector.find_similar_blocks(descriptors)
        self.assertFalse(np.any(changed_matches[:, 0] == 0))

    def test_shift_filter_includes_threshold_and_rejects_noise(self):
        repeated = np.tile([5, 7, 25, 7, -20, 0], (20, 1))
        noise = np.array([[5, 7, 25, 8, -20, -1]])
        np.testing.assert_array_equal(
            self.detector.consistent_shift_matches(np.vstack((repeated, noise))), repeated
        )
        self.assertEqual(self.detector.consistent_shift_matches(repeated, vector_limit=21).shape, (0, 6))

    def test_mask_marks_both_copied_regions_and_leaves_background(self):
        matches = np.array([[10, 10, 35, 40, -25, -30]])
        mask = self.detector.forgery_mask((60, 70), matches)
        self.assertEqual(mask.dtype, np.uint8)
        np.testing.assert_array_equal(mask[10:19, 10:19], np.full((9, 9), 255))
        np.testing.assert_array_equal(mask[35:44, 40:49], np.full((9, 9), 255))
        self.assertEqual(mask[0, 0], 0)
        self.assertEqual(mask[25, 25], 0)
        self.assertFalse(np.any(self.detector.forgery_mask((12, 12), np.empty((0, 6)))))

    def test_clean_image_returns_mask_and_closes_windows(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with patch.object(cv2, 'imread', return_value=image), patch.object(cv2, 'imshow'), \
                patch.object(cv2, 'waitKey'), patch.object(cv2, 'destroyAllWindows') as close, \
                patch('builtins.print'):
            result = self.detector.copy_move_forgery_detection(Path('synthetic.png'))
        self.assertEqual(result.shape, (8, 8))
        self.assertFalse(np.any(result))
        close.assert_called_once()

    def test_complete_detector_finds_synthetic_copy_move(self):
        generator = np.random.default_rng(2026)
        gray = generator.integers(0, 256, (32, 80), dtype=np.uint8)
        gray[8:24, 56:72] = gray[8:24, 8:24]
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        with patch.object(cv2, 'imread', return_value=image), patch.object(cv2, 'imshow'), \
                patch.object(cv2, 'waitKey'), patch.object(cv2, 'destroyAllWindows'), \
                patch('builtins.print'):
            result = self.detector.copy_move_forgery_detection(Path('synthetic-copy.png'))
        np.testing.assert_array_equal(result[8:24, 8:24], np.full((16, 16), 255))
        np.testing.assert_array_equal(result[8:24, 56:72], np.full((16, 16), 255))
        self.assertEqual(result[0, 0], 0)

    def test_image_listing_filters_extensions_and_sorts(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            for name in ('z.PNG', 'a.tif', 'notes.txt', 'b.jpg'):
                (directory / name).touch()
            (directory / 'nested.png').mkdir()
            self.assertEqual(
                [path.name for path in self.detector.list_images(directory)], ['a.tif', 'b.jpg', 'z.PNG']
            )


class TrashDetectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch.object(cv2, 'imread', side_effect=AssertionError('Import must not load images')), \
                patch.object(cv2, 'imwrite', side_effect=AssertionError('Import must not save images')):
            cls.detector = import_script(BACKGROUND / 'trash_detection.py')

    def test_red_range_selects_largest_object(self):
        image = np.zeros((50, 60, 3), dtype=np.uint8)
        image[5:20, 10:30] = (0, 0, 255)
        image[35:40, 45:50] = (0, 0, 255)
        image[25:30, 5:10] = (255, 0, 0)
        _, mask, threshold, bounds = self.detector.largest_red_object(image)
        self.assertEqual(bounds, (10, 5, 20, 15))
        self.assertEqual(np.count_nonzero(mask), 325)
        self.assertEqual(mask[26, 6], 0)
        np.testing.assert_array_equal(mask, threshold)

    def test_red_threshold_boundaries_and_no_objects(self):
        hsv = np.array([[[0, 120, 120], [10, 255, 255], [11, 255, 255], [0, 119, 255]]], dtype=np.uint8)
        with patch.object(cv2, 'cvtColor', return_value=hsv):
            _, mask, _, _ = self.detector.largest_red_object(np.zeros((1, 4, 3), dtype=np.uint8))
        np.testing.assert_array_equal(mask, [[255, 255, 0, 0]])
        with self.assertRaises(ValueError):
            self.detector.largest_red_object(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_kmeans_shape_color_order_and_cluster_masking(self):
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        labels = np.array([[4], [0], [1], [4]], dtype=np.int32)
        centers = np.array([[10, 11, 12], [20, 21, 22], [30, 31, 32], [40, 41, 42], [50, 51, 52], [60, 61, 62]], dtype=np.float32)
        with patch.object(cv2, 'kmeans', return_value=(0, labels, centers)) as kmeans:
            segmented, masked = self.detector.segment_image(image, 6, cv2.KMEANS_RANDOM_CENTERS)
        np.testing.assert_array_equal(segmented, centers[labels.flatten()].astype(np.uint8).reshape(image.shape))
        np.testing.assert_array_equal(masked[0, 0], [0, 0, 255])
        np.testing.assert_array_equal(masked[1, 1], [0, 0, 255])
        np.testing.assert_array_equal(masked[0, 1], image[0, 1])
        np.testing.assert_array_equal(image, np.arange(12, dtype=np.uint8).reshape(2, 2, 3))
        self.assertEqual(kmeans.call_args.args[0].dtype, np.float32)
        self.assertEqual(kmeans.call_args.args[1], 6)
        self.assertEqual(kmeans.call_args.args[4], 10)
        self.assertEqual(kmeans.call_args.args[5], cv2.KMEANS_RANDOM_CENTERS)

    def test_masked_hsv_points_preserve_channel_values(self):
        hsv = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        mask = np.array([[0, 255, 0], [255, 0, 0]], dtype=np.uint8)
        hue, saturation, value = self.detector.hsv_points(hsv, mask)
        np.testing.assert_array_equal(hue, [3, 9])
        np.testing.assert_array_equal(saturation, [4, 10])
        np.testing.assert_array_equal(value, [5, 11])

    def test_empty_red_mask_still_saves_diagnostic_images(self):
        image = np.zeros((205, 10, 3), dtype=np.uint8)
        cropped = image[200:, :]
        with patch.object(self.detector, 'read_image', return_value=image), \
                patch.object(self.detector, 'segment_image', return_value=(cropped, cropped)), \
                patch.object(cv2, 'imwrite', return_value=True) as save, patch('builtins.print'):
            with self.assertRaises(ValueError):
                self.detector.save_second_example(Path('synthetic-output'))
        saved = {Path(call.args[0]).name: call.args[1] for call in save.call_args_list}
        self.assertEqual(set(saved), {
            'original_image3.png', 'segmented_image3.png', 'masked_image3.jpg',
            'hsv_image3.png', 'hsv_debug3.png', 'frame_threshed3.png', 'thresh_debug3.png',
        })
        self.assertEqual(saved['hsv_image3.png'].shape, (5, 10, 3))
        np.testing.assert_array_equal(saved['hsv_image3.png'], saved['hsv_debug3.png'])
        self.assertEqual(saved['frame_threshed3.png'].shape, (5, 10))
        self.assertFalse(np.any(saved['frame_threshed3.png']))
        np.testing.assert_array_equal(saved['frame_threshed3.png'], saved['thresh_debug3.png'])


class DemoHelperTests(unittest.TestCase):
    def test_flow_direction_colors_on_synthetic_vectors(self):
        draw = isolated_function(BACKGROUND / 'demo_flow3.py', 'draw_directional_flow', {'np': np, 'cv2': cv2})
        gray = np.zeros((16, 32), dtype=np.uint8)
        flow = np.zeros((16, 32, 2), dtype=np.float32)
        flow[8, 8] = (3, 0)
        flow[8, 24] = (0, 3)
        with patch.object(cv2, 'arrowedLine') as arrow:
            result = draw(gray, flow)
        self.assertEqual(result.shape, (16, 32, 3))
        self.assertEqual(arrow.call_args_list[0].args[3], (0, 255, 0))
        self.assertEqual(arrow.call_args_list[1].args[3], (0, 0, 255))
        self.assertEqual(arrow.call_args_list[0].args[2], (11, 8))

    def test_iou_aggregates_all_batch_and_image_dimensions(self):
        tensor_ops = SimpleNamespace(
            argmax=np.argmax, cast=lambda values, dtype: values.astype(dtype),
            squeeze=np.squeeze, int64=np.int64, float32=np.float32,
            reduce_sum=np.sum, reduce_mean=np.mean, where=np.where,
        )
        metric = isolated_function(BACKGROUND / 'demo_unet_iou.py', 'iou_metric', {'tf': tensor_ops})
        truth = np.array([[[[0], [1]], [[2], [2]]], [[[1], [0]], [[0], [1]]]])
        predicted_labels = np.array([[[0, 1], [1, 2]], [[1, 0], [0, 1]]])
        probabilities = np.eye(3)[predicted_labels]
        self.assertAlmostEqual(float(metric(truth, probabilities)), (1 + 0.75 + 0.5) / 3)


if __name__ == '__main__':
    unittest.main()
