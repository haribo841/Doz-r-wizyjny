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
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_show = plt.show
    plt.show = lambda: None
    try:
        module.run_experiment_1_single_object()
        output = root / "docs" / "images" / "kalman-tracking.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output, dpi=130)
        print(output)
    finally:
        plt.show = original_show
        plt.close("all")


if __name__ == "__main__":
    main()
