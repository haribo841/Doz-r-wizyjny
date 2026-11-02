"""Synthetic regressions that do not require videos, datasets or model weights."""

import ast
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

matplotlib.use('Agg')
import matplotlib.pyplot as plt


REPOSITORY = Path(__file__).resolve().parents[1]
BACKGROUND = REPOSITORY / 'Background modeling'
NOTEBOOK = BACKGROUND / 'najlepszymodel_2.ipynb'


def notebook_namespace():
    """Load actual notebook functions, never imports, model weights or demo cells."""
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    namespace = {
        'np': np, 'cv': cv2, 'plt': plt, 'Image': Image,
        'ImageChops': ImageChops, 'ImageEnhance': ImageEnhance,
        'BytesIO': BytesIO, 'Path': Path,
    }
    code = '\n\n'.join(
        ''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code'
    )
    tree = ast.parse(code)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in {'MIN_IMAGE_SIZE', 'IMAGE_SUFFIXES'}:
                namespace[target.id] = ast.literal_eval(node.value)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
    return namespace


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


class ForgeryNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cls.functions = notebook_namespace()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(plt.close, 'all')
        self.directory = Path(self.temporary.name)

    def image(self, name, size=(32, 32)):
        filename = self.directory / name
        filename.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', size, (40, 70, 100)).save(filename)
        return filename

    def test_publication_notebook_is_clean_and_all_cells_compile(self):
        self.assertEqual(self.notebook['nbformat'], 4)
        identifiers = [cell['id'] for cell in self.notebook['cells']]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        for cell in self.notebook['cells']:
            with self.subTest(cell=cell['id']):
                self.assertFalse(cell.get('attachments'))
                if cell['cell_type'] == 'code':
                    self.assertEqual(cell['outputs'], [])
                    self.assertIsNone(cell['execution_count'])
                    compile(''.join(cell['source']), str(NOTEBOOK), 'exec')

    def test_missing_class_folder_has_configuration_hint(self):
        with self.assertRaisesRegex(FileNotFoundError, 'DATASET_DIR'):
            self.functions['list_dataset_images'](self.directory / 'missing')

    def test_empty_and_single_image_classes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'minimum dwa'):
            self.functions['list_dataset_images'](self.directory)
        self.image('only.png')
        with self.assertRaisesRegex(ValueError, 'minimum dwa'):
            self.functions['list_dataset_images'](self.directory)

    def test_listing_sorts_filters_and_validates_images(self):
        for name in ('z.PNG', 'a.tif', 'b.jpg'):
            self.image(name)
        (self.directory / 'notes.txt').write_text('not an image', encoding='utf-8')
        (self.directory / 'nested.png').mkdir()
        actual = self.functions['list_dataset_images'](self.directory)
        self.assertEqual([path.name for path in actual], ['a.tif', 'b.jpg', 'z.PNG'])

    def test_corrupt_and_missing_images_report_their_paths(self):
        corrupt = self.directory / 'corrupt.png'
        corrupt.write_bytes(b'not an image')
        for filename in (corrupt, self.directory / 'missing.png'):
            with self.subTest(filename=filename.name):
                with self.assertRaisesRegex(ValueError, filename.name):
                    self.functions['read_dataset_image'](filename)

    def test_small_images_rejected_and_32_pixel_boundary_supported(self):
        for size in ((31, 32), (32, 31)):
            filename = self.image('small.png', size)
            with self.subTest(size=size):
                with self.assertRaisesRegex(ValueError, 'small.png.*32 x 32'):
                    self.functions['read_dataset_image'](filename)
        boundary = self.image('boundary.png')
        self.assertEqual(self.functions['read_dataset_image'](boundary).shape, (32, 32, 3))

    def test_invalid_file_in_class_is_not_silently_skipped(self):
        self.image('good.png')
        (self.directory / 'bad.jpg').write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'bad.jpg'):
            self.functions['list_dataset_images'](self.directory)

    def test_paths_and_labels_keep_classes_separate_from_features(self):
        for folder in ('real', 'tampered'):
            for name in ('z.png', 'a.png'):
                self.image(f'{folder}/{name}')
        paths, labels, names = self.functions['get_image_paths_and_labels'](
            self.directory / 'real', self.directory / 'tampered'
        )
        np.testing.assert_array_equal(labels, [0, 0, 1, 1])
        self.assertEqual(names, ['a.png', 'z.png', 'a.png', 'z.png'])
        self.assertTrue(all(Path(filename).is_file() for filename in paths))

    def test_grid_handles_zero_one_and_more_than_20_images(self):
        draw = self.functions['show_image_grid']
        with patch.object(plt, 'show'), patch('builtins.print'):
            self.assertIsNone(draw([], []))
            for count in (1, 21):
                with self.subTest(count=count):
                    figure = draw([np.zeros((32, 32, 3))] * count, [str(i) for i in range(count)])
                    self.assertEqual(len(figure.axes), ((count + 4) // 5) * 5)
                    self.assertEqual(sum(bool(axis.images) for axis in figure.axes), count)
                    plt.close(figure)
        with self.assertRaises(ValueError):
            draw([np.zeros((32, 32, 3))], [])
        with self.assertRaises(ValueError):
            draw([], [], columns=0)

    def test_no_misclassifications_does_not_create_a_figure(self):
        class EmptyResults:
            empty = True

            def __init__(self):
                self.columns = {}

            def iterrows(self):
                return iter(())

            def __setitem__(self, key, values):
                self.columns[key] = values

        empty = EmptyResults()
        with patch.object(plt, 'subplots') as create, patch('builtins.print'):
            self.assertIsNone(self.functions['show_misclassifications'](empty))
        create.assert_not_called()
        cell = next(cell for cell in self.notebook['cells'] if cell.get('id') == '19da3e21')
        namespace = {'wrong_df': empty, 'Path': Path}
        exec(compile(''.join(cell['source']), str(NOTEBOOK), 'exec'), namespace)
        self.assertEqual(empty.columns['full_path'], [])

    def test_jpeg_roundtrips_are_in_memory_and_preserve_existing_files(self):
        filename = self.image('source.png')
        for name in ('temp_ela_resaved.jpg', 'temp_dct_resaved.jpg'):
            (self.directory / name).write_bytes(b'keep this existing file')
        before = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        original_save = Image.Image.save
        with patch.object(Image.Image, 'save', autospec=True, side_effect=original_save) as save:
            self.functions['perform_ela_gray'](filename)
        self.assertIsInstance(save.call_args.args[1], BytesIO)
        self.functions['perform_dct_inconsistency'](filename)
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.directory.iterdir()})

    def test_in_memory_ela_and_dct_match_file_roundtrips(self):
        filename = self.image('source.png')
        with Image.open(filename) as source_image:
            original = source_image.convert('RGB')
        ela_path = self.directory / 'comparison-ela.jpg'
        original.save(ela_path, 'JPEG', quality=90)
        with Image.open(ela_path) as compressed_image:
            expected_ela = np.array(ImageEnhance.Brightness(
                ImageChops.difference(original, compressed_image.convert('RGB'))
            ).enhance(15).convert('L'))
        _, actual_ela = self.functions['perform_ela_gray'](filename)
        np.testing.assert_array_equal(actual_ela, expected_ela)
        _, actual_dct = self.functions['perform_dct_inconsistency'](filename)
        self.assertEqual(actual_dct.shape, (32, 32))
        self.assertTrue(np.all(np.isfinite(actual_dct)))
        gray = cv2.imread(str(filename), cv2.IMREAD_GRAYSCALE)
        dct_path = self.directory / 'comparison-dct.jpg'
        self.assertTrue(cv2.imwrite(str(dct_path), gray, [cv2.IMWRITE_JPEG_QUALITY, 90]))
        success, encoded = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self.assertTrue(success)
        np.testing.assert_array_equal(
            cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE),
            cv2.imread(str(dct_path), cv2.IMREAD_GRAYSCALE),
        )
        with patch.object(cv2, 'imwrite', side_effect=AssertionError('No image may be saved')):
            self.functions['perform_dct_inconsistency'](filename)

    def test_features_are_finite_and_independent_of_names_and_classes(self):
        first = self.image('real/first.png')
        second = self.image('tampered/renamed.png')
        original = self.functions.get('extract_features_patch_cnn')
        self.functions['extract_features_patch_cnn'] = lambda _path, grid=(3, 3): (
            [np.arange(1024, dtype=np.float32)],
            [f'cnn_patch_{index}' for index in range(1024)],
            {'cnn': {'grid_y': grid[0], 'grid_x': grid[1], 'dim': 1024}},
        )
        try:
            vector, names, _ = self.functions['extract_features'](first)
            other, other_names, _ = self.functions['extract_features'](second)
        finally:
            self.functions['extract_features_patch_cnn'] = original
        self.assertEqual(vector.shape, (1048,))
        self.assertEqual(len(names), len(vector))
        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertEqual(names, other_names)
        self.assertNotIn('filename', names)
        self.assertNotIn('label', names)
        np.testing.assert_array_equal(vector, other)

    def test_loocv_clones_the_pipeline_for_each_fold(self):
        cell = next(cell for cell in self.notebook['cells'] if cell.get('id') == '89e1951d')
        tree = ast.parse(''.join(cell['source']))
        pipeline = next(
            node.value for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name) and node.value.func.id == 'make_pipeline'
        )
        pipeline_options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in pipeline.keywords}
        self.assertIn('memory', pipeline_options)
        self.assertIsNone(pipeline_options['memory'])
        classifier = pipeline.args[1]
        classifier_options = {keyword.arg: ast.literal_eval(keyword.value) for keyword in classifier.keywords}
        self.assertEqual(classifier_options, {'random_state': 42, 'learning_rate': 0.1})
        loop = next(node for node in tree.body if isinstance(node, ast.For))
        models = []

        class FakeModel:
            def fit(self, features, labels):
                self.fitted_shape = (features.shape, labels.shape)
                return self

            def predict_proba(self, _features):
                return np.array([[0.75, 0.25]])

        def clone_model(_base):
            model = FakeModel()
            models.append(model)
            return model

        folds = [(np.array([j for j in range(4) if j != i]), np.array([i])) for i in range(4)]
        namespace = {
            'clone': clone_model, 'base_model': object(),
            'loo': SimpleNamespace(split=lambda _values: iter(folds)),
            'X': np.zeros((4, 2)), 'y': np.array([0, 0, 1, 1]),
            'filenames': ['a', 'b', 'c', 'd'], 'y_true': [], 'y_pred': [], 'y_prob': [], 'test_files': [],
        }
        exec(compile(ast.Module(body=[loop], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
        self.assertEqual(len({id(model) for model in models}), 4)
        self.assertTrue(all(model.fitted_shape == ((3, 2), (3,)) for model in models))
        self.assertEqual(namespace['test_files'], ['a', 'b', 'c', 'd'])


if __name__ == '__main__':
    unittest.main()
