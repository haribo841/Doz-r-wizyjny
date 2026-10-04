# Experiment guide

## Choose an experiment

Paths below are relative to the repository root. This is a source collection; there is no single installation that configures every script.

| Files | Purpose | Input / extra requirements |
| --- | --- | --- |
| `Background modeling/KalmanFilter.py` | Synthetic single-object and multi-object tracking; noise and tracker-parameter comparisons | NumPy, Matplotlib; SciPy enables assignment optimization and a greedy fallback exists |
| `Background modeling/Background_modeling.py`, `example3.py` to `example5.py` | Mean/median backgrounds and foreground masks | OpenCV, NumPy; local video or prepared frame images |
| `Background modeling/Counting.py`, `demo_zliczanie.py` | Centroid tracking and counting | OpenCV, NumPy, imutils; `people3.mp4` |
| `Background modeling/OpticalFlow.py`, `demo_flow2.py` to `demo_flow4.py` | Motion fields, regional analysis, plots | OpenCV, NumPy, Matplotlib; some variants use pandas; local video |
| `Background modeling/Heatmap.py`, `Trajectory.py` | Activity maps and motion trajectories | OpenCV, NumPy; local video |
| `Background modeling/demo_roi.py` | Threshold/contour-based regions of interest | OpenCV, NumPy; `car.jpg` in the working directory |
| `Background modeling/demo_roi2.py` | ROI experiment with YOLO | Ultralytics and a matching model, plus an input image |
| `Background modeling/demo_sam.py`, `lab_sam_final.py` | Prompted segmentation with SAM | Segment Anything, PyTorch, Pillow, checkpoint and input image |
| `Background modeling/demo_unet.py`, `demo_unet_iou.py` | U-Net training and segmentation evaluation | TensorFlow, scikit-learn, Pillow; Oxford-IIIT Pet images and masks |
| `Background modeling/media.py`, `media2.py` | Camera-based MediaPipe experiments | MediaPipe, OpenCV; camera device 0 |
| `forgery_detection.py`, `Background modeling/trash_detection.py` | Separate image-analysis experiments | Input image folders and algorithm-specific configuration |
| `batch_export_fbx_to_obj.py` | Auxiliary asset conversion | Blender's Python runtime (`bpy`), not ordinary Python |

## Reproduce the README figure

```powershell
python -m pip install numpy matplotlib scipy
python docs/render_kalman_example.py
```

The wrapper imports the checked-in Kalman code, executes its single-object example with seed `0`, and saves `docs/images/kalman-tracking.png` without opening a window. Simulations use an explicit NumPy `Generator` instead of the global random state. The move to this generator changes the numerical samples relative to the historical example, so the README figure was regenerated. The tracking equations and scenario definitions remain the same.

The current example is validated on Windows with Python 3.13.1, NumPy 2.2.4, Matplotlib 3.10.1, and SciPy 1.16.3. All three Kalman experiments run without GUI windows. Repeated rendering is deterministic within that environment; exact plot pixels can differ between Matplotlib versions.

## Automated regression checks

```powershell
python -m pip install numpy matplotlib scipy opencv-python imutils
python -m unittest discover -s tests -v
```

The tests cover Kalman filtering and synthetic simulations, shared centroid tracking and counting, and image-analysis helpers with synthetic images. They do not open a camera, train a model, or process private recordings. The two counting demos use `centroid_tracker.py` with their original distance and disappearance thresholds.

Verification on 2026-10-04: all 45 regression tests passed, all three Kalman experiments rendered without GUI windows, and two consecutive README-figure renders produced identical files.

## Run a video/image experiment

1. Read the selected script and install only its imports.
2. Replace its hard-coded input path with a local file you are entitled to process.
3. Use the working directory expected by relative inputs such as `car.jpg` or `people3.mp4`.
4. Check output paths before running, as some scripts overwrite fixed-name result files.

Missing frames, empty detections, GPU/model compatibility, and large datasets are not handled consistently across these laboratory scripts. Do not run camera examples unintentionally; they open camera device 0. Download models only from their official publisher and review their usage terms.

## Reuse and data

There is no root license yet. The collection includes learning material and integrations whose provenance needs confirmation before a blanket license is added. Third-party code, model weights, and datasets have independent terms.

Do not commit personal recordings, faces, real identifiers, private datasets, or local notebooks merely to make an example reproducible. The documentation intentionally uses synthetic tracking data. Existing untracked datasets/notebooks are not part of this documentation update.
