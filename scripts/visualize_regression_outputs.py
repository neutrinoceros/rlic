# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib>=3.10",
#     "numpy>=2.2.3",
# ]
# ///
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path(__file__).parents[1] / "tests" / "data" / "regressions"

ALL_FILES = sorted(DATA_DIR.glob("*.npy"))

UNIQUE_KERNEL_IDS = ["k0"]


def main() -> None:
    for kid in UNIQUE_KERNEL_IDS:
        files = [file for file in ALL_FILES if kid in file.stem]

        fig, axs = plt.subplots(
            nrows=2,  # rows = uv_modes
            ncols=len(files) // 2,  # cols = everything else
            sharex=True,
            sharey=True,
        )
        for (iax, ax_row), mode in zip(enumerate(axs), ["vel", "pol"], strict=True):
            mode_files = [file for file in files if mode in file.stem]

            ax_row[0].set(ylabel=mode, aspect="equal")
            for ax, file in zip(ax_row, mode_files, strict=True):
                ax.imshow(np.load(file))
                if iax == 0:
                    field_u, field_v, *_ = file.stem.split("_")
                    ax.set(title=f"{field_u}, {field_v}")

        sfile = f"/tmp/rlic_viz_kid={kid}.png"
        print(f"Saving to {sfile}")
        fig.savefig(sfile)


if __name__ == "__main__":
    main()
