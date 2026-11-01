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
| `Background modeling/najlepszymodel_2.ipynb` | ELA, DCT, noise and ResNet-18 patch features; Gradient Boosting and LOOCV | Own `real/` and `tampered/` images; NumPy, pandas, Pillow, Matplotlib, OpenCV, PyTorch, torchvision, scikit-learn and JupyterLab |
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
python -m pip install numpy matplotlib scipy opencv-python imutils pillow
python -m unittest discover -s tests -v
```

The tests cover Kalman filtering and synthetic simulations, shared centroid tracking and counting, and image-analysis helpers with synthetic images. They do not open a camera, train a model, or process private recordings. The two counting demos use `centroid_tracker.py` with their original distance and disappearance thresholds.

Verification on 2026-10-04: all 45 regression tests passed, all three Kalman experiments rendered without GUI windows, and two consecutive README-figure renders produced identical files.

Verification on 2026-11-01: all 59 regression tests passed, including 14 notebook regressions. This was checked in the existing Python environment and a separate validation environment. The new cases do not load neural-network weights or use the external dataset.

## Image forgery notebook

The notebook is an educational feature-analysis experiment, not an image-authenticity certification tool. Its Polish explanations and historical observations are retained. Published code keeps ResNet-18, ELA/DCT/noise/CNN feature definitions and ordering, `random_state=42`, LOOCV and the 0.5 decision threshold. Filenames and class-folder names are not input features.

Create a separate environment from the repository root. This CPU setup does not require CUDA:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install numpy==2.2.4 pandas==2.3.3 Pillow==13.0.0 matplotlib==3.10.1 opencv-python==4.12.0.88 scikit-learn==1.8.0 jupyterlab==4.6.5
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
cd "Background modeling"
python -m jupyterlab najlepszymodel_2.ipynb
```

The PyTorch/torchvision pair and CPU index follow the [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/). For a CUDA environment, select the corresponding official package pair rather than mixing unrelated installations. The notebook automatically uses CUDA when available, otherwise CPU.

Before running all cells, prepare images that you are entitled to process:

```text
Dataset/
  real/       # at least two original images
  tampered/   # at least two manipulated images
```

Set `DATASET_DIR = Path("Dataset")` in the first code cell, or replace it with your own data directory. Relative paths are resolved from the kernel's working directory, so launch from `Background modeling` as shown above. Supported extensions are JPG, JPEG, PNG, BMP, TIF and TIFF, case-insensitively. Each image must be readable and at least 32 x 32 pixels; missing folders, undersized classes and bad images stop execution with an explanatory path instead of silently changing the sample set. Non-image files and nested directories are skipped, and image paths are sorted.

Choose **Restart Kernel and Run All**. The first ResNet-18 initialization can download pretrained weights through torchvision into its local cache; later runs can reuse that cache. This is not model training on ImageNet. The experiment fits the small Gradient Boosting classifier separately in each leave-one-out fold. JPEG recompression for ELA and DCT is performed in memory, with no fixed temporary image files. Galleries adapt to the number of images, including an empty set of misclassifications.

The published notebook has no saved figures, tables, execution counters or model weights. The complete local `Dataset/` directory is ignored, including TIFF files. Do not commit that directory or the weight cache. Notebook tests isolate its real functions and use synthetic temporary images and a stub CNN embedding, so the ordinary regression suite does not import PyTorch or download weights.

The historical 93% accuracy describes an earlier LOOCV experiment on only 30 images (10 original, 20 manipulated). It is not a new validation report, an independent held-out benchmark, proof of generalization, or a guarantee for other sources. Correlated image variants and the small, imbalanced collection can distort evaluation; explanations of individual errors remain hypotheses.

### Local execution check, 2026-11-01

A fresh kernel completed all 23 published code cells on Windows 11 and Python 3.13.1 using CPU, PyTorch 2.9.1+cpu, torchvision 0.24.1+cpu and `ResNet18_Weights.IMAGENET1K_V1`. The remaining package versions match the setup above; execution used SciPy 1.16.3, nbclient 0.10.2 and ipykernel 7.1.0.

For the existing local collection of 10 original and 20 manipulated images, the feature matrix was 30 x 1048. LOOCV classified 28/30 images correctly (accuracy 0.933333, F1 0.95), with confusion matrix `[[9, 1], [1, 19]]` in class order `real`, `tampered`. This is a local reproduction check on the same small collection, not an independent benchmark. No accuracy target was enforced.

Executed outputs were kept outside the repository. All 30 input-file SHA-256 hashes remained unchanged, and the published notebook retained empty outputs and execution counters.

## Run a video/image experiment

1. Read the selected script and install only its imports.
2. Replace its hard-coded input path with a local file you are entitled to process.
3. Use the working directory expected by relative inputs such as `car.jpg` or `people3.mp4`.
4. Check output paths before running, as some scripts overwrite fixed-name result files.

Missing frames, empty detections, GPU/model compatibility, and large datasets are not handled consistently across these laboratory scripts. Do not run camera examples unintentionally; they open camera device 0. Download models only from their official publisher and review their usage terms.

## Reuse and data

There is no root license yet. The collection includes learning material and integrations whose provenance needs confirmation before a blanket license is added. Third-party code, model weights, and datasets have independent terms.

Do not commit personal recordings, faces, real identifiers or private datasets merely to make an example reproducible. The Kalman documentation intentionally uses synthetic tracking data. The forgery notebook is published as reviewed source only; all images, saved outputs and model weights remain outside Git. Its inclusion does not establish redistribution rights for the local dataset or add a repository-wide license.
