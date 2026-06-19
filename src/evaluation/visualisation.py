import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import src.config as config

warnings.filterwarnings("ignore")

# ── DESIGN SYSTEM ────────────────────────────────────────────────────
CRAG_COLOR   = "#4F8EF7"   # Blue
NAIVE_COLOR  = "#F76B6B"   # Red
BG_COLOR     = "#0F1117"   # Dark background
CARD_COLOR   = "#1C1F2E"   # Card background
GRID_COLOR   = "#2A2D3E"   # Grid lines
TEXT_PRIMARY = "#E8EAF6"   # Primary text
TEXT_MUTED   = "#8B90A0"   # Muted text
ACCENT_GREEN = "#43E97B"
ACCENT_AMBER = "#F9A825"

PALETTE = [CRAG_COLOR, NAIVE_COLOR]

plt.rcParams.update({
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    CARD_COLOR,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_PRIMARY,
    "axes.titlecolor":   TEXT_PRIMARY,
    "axes.grid":         True,
    "grid.color":        GRID_COLOR,
    "grid.linewidth":    0.6,
    "xtick.color":       TEXT_MUTED,
    "ytick.color":       TEXT_MUTED,
    "text.color":        TEXT_PRIMARY,
    "font.family":       "DejaVu Sans",
    "legend.facecolor":  CARD_COLOR,
    "legend.edgecolor":  GRID_COLOR,
    "legend.labelcolor": TEXT_PRIMARY,
})


def load_and_merge_data():
    crag_path  = config.PROCESSED_DIR / "evaluation_results_crag_gemini-3-1-flash-lite.csv"
    naive_path = config.PROCESSED_DIR / "evaluation_results_naive_gemini-3-1-flash-lite.csv"

    if not os.path.exists(crag_path) or not os.path.exists(naive_path):
        print("[ERROR] Could not find one or both CSV files. Check the paths.")
        return None

    df_crag  = pd.read_csv(crag_path)
    df_naive = pd.read_csv(naive_path)

    df_crag["Pipeline"]  = "CRAG"
    df_naive["Pipeline"] = "Naive"

    df = pd.concat([df_crag, df_naive], ignore_index=True)
    df = df.dropna(subset=["faithfulness", "answer_relevancy"])
    return df


def add_card_bg(ax):
    """Give each axes a subtle rounded card feel."""
    ax.set_facecolor(CARD_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)


def plot_dashboard(df):
    fig = plt.figure(figsize=(18, 22), facecolor=BG_COLOR)
    fig.suptitle(
        "CRAG  vs  Naive Baseline — Evaluation Dashboard",
        fontsize=22, fontweight="bold", color=TEXT_PRIMARY, y=0.99
    )

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.52,
        wspace=0.35,
        top=0.96, bottom=0.04,
        left=0.06, right=0.97
    )

    # ── [1] Mean Score Summary Cards (top row) ───────────────────────
    metrics = ["faithfulness", "answer_relevancy"]
    pipelines = ["CRAG", "Naive"]
    colors = {p: c for p, c in zip(pipelines, PALETTE)}

    for col_idx, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[0, col_idx])
        add_card_bg(ax)

        means = [df[df["Pipeline"] == p][metric].mean() for p in pipelines]
        bars  = ax.bar(pipelines, means, color=PALETTE, width=0.45,
                       linewidth=0, zorder=3)

        # Annotate bar tops
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02, f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=TEXT_PRIMARY
            )

        # Delta annotation
        delta = means[0] - means[1]
        delta_color = ACCENT_GREEN if delta >= 0 else NAIVE_COLOR
        delta_sign  = "+" if delta >= 0 else ""
        ax.text(
            0.98, 0.07,
            f"CRAG {delta_sign}{delta:.3f} vs Naive",
            transform=ax.transAxes,
            ha="right", fontsize=9,
            color=delta_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, edgecolor=delta_color, linewidth=1)
        )

        label = "Faithfulness" if metric == "faithfulness" else "Answer Relevancy"
        ax.set_title(f"Avg {label}", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score (0–1)", fontsize=10, color=TEXT_MUTED)
        ax.tick_params(axis="x", labelsize=11)

    # ── [2] Win / Tie / Loss breakdown (top right) ───────────────────
    ax_wtl = fig.add_subplot(gs[0, 2])
    add_card_bg(ax_wtl)

    df_crag  = df[df["Pipeline"] == "CRAG"].reset_index(drop=True)
    df_naive = df[df["Pipeline"] == "Naive"].reset_index(drop=True)
    min_len  = min(len(df_crag), len(df_naive))
    diff     = df_crag["faithfulness"].values[:min_len] - df_naive["faithfulness"].values[:min_len]

    wins  = int((diff > 0.05).sum())
    ties  = int((np.abs(diff) <= 0.05).sum())
    losses = int((diff < -0.05).sum())

    wedge_colors = [ACCENT_GREEN, ACCENT_AMBER, NAIVE_COLOR]
    wedges, texts, autotexts = ax_wtl.pie(
        [wins, ties, losses],
        labels=["CRAG Wins", "Ties", "Naive Wins"],
        autopct="%1.0f%%",
        colors=wedge_colors,
        startangle=90,
        wedgeprops=dict(linewidth=2, edgecolor=BG_COLOR),
        textprops=dict(color=TEXT_PRIMARY, fontsize=10)
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")

    ax_wtl.set_title("Faithfulness Win / Tie / Loss\n(threshold ±0.05)", fontsize=12, fontweight="bold")

    # ── [3] Violin plots — full distribution (row 2) ─────────────────
    for col_idx, metric in enumerate(metrics):
        ax = fig.add_subplot(gs[1, col_idx])
        add_card_bg(ax)

        parts = ax.violinplot(
            [df[df["Pipeline"] == p][metric].dropna().values for p in pipelines],
            positions=[0, 1],
            showmedians=True,
            showextrema=True
        )
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(PALETTE[i])
            pc.set_alpha(0.6)
        parts["cmedians"].set_color(TEXT_PRIMARY)
        parts["cmedians"].set_linewidth(2)
        parts["cbars"].set_color(GRID_COLOR)
        parts["cmaxes"].set_color(GRID_COLOR)
        parts["cmins"].set_color(GRID_COLOR)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(pipelines, fontsize=11)
        ax.set_ylim(-0.05, 1.12)
        ax.set_ylabel("Score", fontsize=10, color=TEXT_MUTED)

        label = "Faithfulness" if metric == "faithfulness" else "Answer Relevancy"
        ax.set_title(f"{label} Distribution", fontsize=13, fontweight="bold", pad=10)

        # Annotate median
        for i, p in enumerate(pipelines):
            med = df[df["Pipeline"] == p][metric].median()
            ax.text(i, med + 0.04, f"med {med:.2f}",
                    ha="center", fontsize=9, color=TEXT_PRIMARY)

    # ── [4] Score histogram overlay (row 2, col 3) ───────────────────
    ax_hist = fig.add_subplot(gs[1, 2])
    add_card_bg(ax_hist)

    for p, c in zip(pipelines, PALETTE):
        vals = df[df["Pipeline"] == p]["faithfulness"].dropna()
        ax_hist.hist(vals, bins=12, color=c, alpha=0.55, label=p, edgecolor=BG_COLOR, linewidth=0.5)

    ax_hist.set_title("Faithfulness Score Histogram", fontsize=13, fontweight="bold", pad=10)
    ax_hist.set_xlabel("Score", fontsize=10, color=TEXT_MUTED)
    ax_hist.set_ylabel("Question Count", fontsize=10, color=TEXT_MUTED)
    ax_hist.legend(fontsize=10)

    # ── [5] Scatter: Faithfulness vs Relevancy (row 3, spans 2 cols) ─
    ax_scatter = fig.add_subplot(gs[2, :2])
    add_card_bg(ax_scatter)

    for p, c, m in zip(pipelines, PALETTE, ["o", "D"]):
        sub = df[df["Pipeline"] == p]
        ax_scatter.scatter(
            sub["answer_relevancy"], sub["faithfulness"],
            c=c, label=p, alpha=0.75, s=70, marker=m, edgecolors=BG_COLOR, linewidths=0.5
        )

    # Quadrant lines
    ax_scatter.axhline(0.5, color=GRID_COLOR, linewidth=1, linestyle="--")
    ax_scatter.axvline(0.5, color=GRID_COLOR, linewidth=1, linestyle="--")
    ax_scatter.text(0.76, 0.92, "High Performance\nZone", transform=ax_scatter.transAxes,
                    fontsize=9, color=ACCENT_GREEN, ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, edgecolor=ACCENT_GREEN, alpha=0.7))

    ax_scatter.set_xlim(-0.05, 1.05)
    ax_scatter.set_ylim(-0.05, 1.05)
    ax_scatter.set_xlabel("Answer Relevancy →", fontsize=11, color=TEXT_MUTED)
    ax_scatter.set_ylabel("Faithfulness →", fontsize=11, color=TEXT_MUTED)
    ax_scatter.set_title("Performance Landscape: Faithfulness vs Relevancy", fontsize=14, fontweight="bold", pad=10)
    ax_scatter.legend(fontsize=11)

    # ── [6] Retry count impact (row 3, col 3) ────────────────────────
    ax_retry = fig.add_subplot(gs[2, 2])
    add_card_bg(ax_retry)

    df_crag_only = df[df["Pipeline"] == "CRAG"].copy()
    if "retry_count" in df_crag_only.columns:
        retry_means = df_crag_only.groupby("retry_count")["faithfulness"].mean().reset_index()
        retry_counts_col = retry_means["retry_count"].astype(str)
        ax_retry.bar(retry_counts_col, retry_means["faithfulness"],
                     color=CRAG_COLOR, alpha=0.8, linewidth=0)
        ax_retry.set_xlabel("Retry Count", fontsize=10, color=TEXT_MUTED)
        ax_retry.set_ylabel("Avg Faithfulness", fontsize=10, color=TEXT_MUTED)
        ax_retry.set_title("CRAG: Retries vs Faithfulness", fontsize=13, fontweight="bold", pad=10)
        ax_retry.set_ylim(0, 1.1)
        for i, row in retry_means.iterrows():
            ax_retry.text(i, row["faithfulness"] + 0.02, f"{row['faithfulness']:.2f}",
                          ha="center", fontsize=10, fontweight="bold", color=TEXT_PRIMARY)
    else:
        ax_retry.text(0.5, 0.5, "retry_count\nnot in data",
                      ha="center", va="center", fontsize=12, color=TEXT_MUTED)

    # ── [7] CRAG Routing Impact (row 4, full width) ──────────────────
    ax_route = fig.add_subplot(gs[3, :])
    add_card_bg(ax_route)

    df_crag_only = df[df["Pipeline"] == "CRAG"].copy()

    def routing_label(row):
        if row.get("web_search_used") == True:
            return "Web Search Triggered"
        elif row.get("retry_count", 0) > 0:
            return "Retries Used"
        else:
            return "Direct Hit (Relevant)"

    df_crag_only["Routing Action"] = df_crag_only.apply(routing_label, axis=1)

    route_summary = df_crag_only.groupby("Routing Action").agg(
        avg_faithfulness=("faithfulness", "mean"),
        avg_relevancy=("answer_relevancy", "mean"),
        count=("faithfulness", "count")
    ).reset_index()

    x      = np.arange(len(route_summary))
    width  = 0.35
    bars1  = ax_route.bar(x - width/2, route_summary["avg_faithfulness"],
                           width, label="Faithfulness", color=CRAG_COLOR, alpha=0.85, linewidth=0)
    bars2  = ax_route.bar(x + width/2, route_summary["avg_relevancy"],
                           width, label="Answer Relevancy", color=ACCENT_GREEN, alpha=0.85, linewidth=0)

    # Count labels below x axis labels
    labels_with_count = [
        f"{row['Routing Action']}\n(n={row['count']})"
        for _, row in route_summary.iterrows()
    ]
    ax_route.set_xticks(x)
    ax_route.set_xticklabels(labels_with_count, fontsize=11)

    for bar in list(bars1) + list(bars2):
        ax_route.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{bar.get_height():.2f}",
            ha="center", fontsize=9, fontweight="bold", color=TEXT_PRIMARY
        )

    ax_route.set_ylim(0, 1.18)
    ax_route.set_ylabel("Average Score", fontsize=11, color=TEXT_MUTED)
    ax_route.set_title(
        "CRAG Internal Routing: Impact on Faithfulness & Relevancy",
        fontsize=14, fontweight="bold", pad=12
    )
    ax_route.legend(fontsize=11)

    # ── Save ─────────────────────────────────────────────────────────
    plots_dir = config.DATA_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / "master_evaluation_dashboard.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    print(f"\n[SUCCESS] Dashboard saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    data = load_and_merge_data()
    if data is not None:
        plot_dashboard(data)