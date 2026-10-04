"""Save the existing deterministic Kalman demo plot without opening a GUI."""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "Background modeling" / "KalmanFilter.py"
    spec = importlib.util.spec_from_file_location("kalman_example", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the Kalman example from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    figure = module.run_experiment_1_single_object(show=False)
    try:
        output = root / "docs" / "images" / "kalman-tracking.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=130)
        print(output)
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
