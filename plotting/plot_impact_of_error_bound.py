import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


fontsize = 18
plt.rcParams.update(
    {
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": 16,
        "axes.titlesize": fontsize,
        "font.weight": "bold",
        "axes.labelweight": "bold",
    }
)


def plot_dataset(ax, filepath, title, hide_size_label=False, hide_error_label=False):
    df = pd.read_csv(filepath)
    x = df["error_bound"]
    (size_line,) = ax.plot(
        x, df["size(KB)"], color="tab:blue", marker="o", label="size (KB)"
    )
    ax.set_xlabel("error_bound")
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    if not hide_size_label:
        ax.set_ylabel("size (KB)", color=size_line.get_color())
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.tick_params(axis="y", labelcolor=size_line.get_color())

    ax2 = ax.twinx()
    (error_line,) = ax2.plot(
        x, df["avg_rel_error"], color="tab:red", marker="s", label="avg_rel_error"
    )
    if not hide_error_label:
        ax2.set_ylabel("avg_rel_error", color=error_line.get_color())
    ax2.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax2.tick_params(axis="y", labelcolor=error_line.get_color())

    ax.set_title(title)
    ax.set_xticks(x)
    ax.legend(handles=[size_line, error_line], loc="upper left")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    plot_dataset(
        axes[0],
        "results/errorbound/ccpp.csv",
        "CCPP",
        hide_error_label=True,
    )
    plot_dataset(
        axes[1],
        "results/errorbound/flights.csv",
        "Flights",
        hide_size_label=True,
    )
    fig.tight_layout()
    save_path = Path("plots")
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / "impact_of_error_bound.png")
    plt.savefig(save_path / "impact_of_error_bound.pdf")


if __name__ == "__main__":
    main()
