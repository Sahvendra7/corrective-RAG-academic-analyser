import os
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import src.config as config

warnings.filterwarnings("ignore")

# ── DESIGN SYSTEM (PROFESSIONAL LIGHT THEME) ──────────────────────────
CRAG_COLOR    = "#1F4E79"   # Muted Navy
NAIVE_COLOR   = "#95A5A6"   # Muted Slate Gray
BG_COLOR      = "#F8F9FA"   # Light Gray-Blue Background
CARD_COLOR    = "#FFFFFF"   # White Card Background
GRID_COLOR    = "#E2E8F0"   # Light Gray Gridlines
TEXT_PRIMARY  = "#1E293B"   # Deep Slate (Primary Text)
TEXT_MUTED    = "#64748B"   # Slate Gray (Muted Text)

ACCENT_GREEN  = "#2E7D32"   # Muted Forest Green (Positive Delta)
ACCENT_RED    = "#C62828"   # Muted Rust Red (Negative Delta)
ACCENT_AMBER  = "#EF6C00"   # Muted Amber (Ties)

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
    "font.family":       "sans-serif",
    "legend.facecolor":  CARD_COLOR,
    "legend.edgecolor":  GRID_COLOR,
    "legend.labelcolor": TEXT_PRIMARY,
})

# ── DATA LOADING & PRE-PROCESSING ─────────────────────────────────────

def load_data():
    crag_path  = config.PROCESSED_DIR / "evaluation_results_crag_gemini-3-1-flash-lite.csv"
    naive_path = config.PROCESSED_DIR / "evaluation_results_naive_gemini-3-1-flash-lite.csv"

    if not crag_path.exists() or not naive_path.exists():
        raise FileNotFoundError("[ERROR] Required CSV evaluation result files are missing.")

    df_crag  = pd.read_csv(crag_path)
    df_naive = pd.read_csv(naive_path)
    
    # Ensure Ragas columns exist
    for col in ["faithfulness", "answer_relevancy", "context_precision"]:
        df_crag[col] = pd.to_numeric(df_crag[col], errors='coerce')
        df_naive[col] = pd.to_numeric(df_naive[col], errors='coerce')

    return df_crag, df_naive

def segment_dataframe(df_crag, df_naive):
    from src.vectorstore.faiss_store import FAISSStore
    
    eval_json = config.PROCESSED_DIR / "eval_dataset.json"
    with open(eval_json, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    q_to_arxiv = {item["question"]: item.get("arxiv_id") for item in eval_data}
    
    store = FAISSStore()
    
    df_merged = pd.merge(
        df_crag[["question", "web_search_used", "pipeline_grade", "faithfulness", "answer_relevancy", "context_precision", "retry_count", "retrieval_ms", "generation_ms", "total_latency_ms"]],
        df_naive[["question", "faithfulness", "answer_relevancy", "context_precision", "retrieval_ms", "generation_ms", "total_latency_ms"]],
        on="question",
        suffixes=("_crag", "_naive")
    )
    
    group_labels = []
    for idx, row in df_merged.iterrows():
        question = row["question"]
        web_search = row["web_search_used"]
        expected_arxiv_id = q_to_arxiv.get(question)
        
        if not web_search:
            group_labels.append("Direct Hit (n=27)")
        else:
            # Check FAISS retrieved IDs
            results = store.search(question, top_k=5)
            top_ids = [r["arxiv_id"] for r in results]
            
            if expected_arxiv_id in top_ids:
                group_labels.append("Grader Rejection (n=16)")
            else:
                group_labels.append("Retriever Miss (n=7)")
                
    df_merged["segment"] = group_labels
    return df_merged

def format_card_axes(ax):
    ax.set_facecolor(CARD_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(GRID_COLOR)
    ax.spines['bottom'].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_MUTED)

# ── INDIVIDUAL PANEL PLOTTING FUNCTIONS ────────────────────────────────

def draw_avg_metrics(axes, df_crag, df_naive):
    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    label_map = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision"
    }
    
    for col_idx, metric in enumerate(metrics):
        ax = axes[col_idx]
        format_card_axes(ax)
        
        crag_val = df_crag[metric].mean()
        naive_val = df_naive[metric].mean()
        
        vals = [crag_val, naive_val]
        bars = ax.bar(["CRAG", "Naive"], vals, color=PALETTE, width=0.45, linewidth=0, zorder=3)
        
        # Annotate bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.02, f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=TEXT_PRIMARY
            )
            
        delta = crag_val - naive_val
        delta_color = ACCENT_GREEN if delta >= 0 else ACCENT_RED
        delta_sign = "+" if delta >= 0 else ""
        
        ax.text(
            0.95, 0.90,
            f"Delta: {delta_sign}{delta:.3f}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=9, fontweight="bold",
            color=delta_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, edgecolor=delta_color, linewidth=1)
        )
        
        ax.set_title(f"Average {label_map[metric]}", fontsize=12, fontweight="bold", pad=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=9, color=TEXT_MUTED)

def draw_violin_plots(axes, df_crag, df_naive):
    metrics = ["faithfulness", "answer_relevancy", "context_precision"]
    label_map = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision"
    }
    
    for col_idx, metric in enumerate(metrics):
        ax = axes[col_idx]
        format_card_axes(ax)
        
        crag_data = df_crag[metric].dropna().values
        naive_data = df_naive[metric].dropna().values
        
        parts = ax.violinplot(
            [crag_data, naive_data],
            positions=[0, 1],
            showmedians=True,
            showextrema=True
        )
        
        # Color bodies
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(PALETTE[i])
            pc.set_alpha(0.5)
            
        parts["cmedians"].set_color(TEXT_PRIMARY)
        parts["cmedians"].set_linewidth(2)
        parts["cbars"].set_color(GRID_COLOR)
        parts["cmaxes"].set_color(GRID_COLOR)
        parts["cmins"].set_color(GRID_COLOR)
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["CRAG", "Naive"], fontsize=10)
        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel("Score", fontsize=9, color=TEXT_MUTED)
        ax.set_title(f"{label_map[metric]} Distribution", fontsize=12, fontweight="bold", pad=8)
        
        # Annotate median values
        for i, data in enumerate([crag_data, naive_data]):
            if len(data) > 0:
                med = np.median(data)
                ax.text(i, med + 0.03, f"med {med:.2f}", ha="center", fontsize=8, color=TEXT_PRIMARY, fontweight="bold")

def draw_win_tie_loss(ax, df_merged):
    format_card_axes(ax)
    
    diff = df_merged["faithfulness_crag"] - df_merged["faithfulness_naive"]
    
    wins = int((diff > 0.05).sum())
    ties = int((np.abs(diff) <= 0.05).sum())
    losses = int((diff < -0.05).sum())
    
    wedge_colors = [ACCENT_GREEN, ACCENT_AMBER, ACCENT_RED]
    wedges, texts, autotexts = ax.pie(
        [wins, ties, losses],
        labels=[f"CRAG Wins ({wins})", f"Ties ({ties})", f"Naive Wins ({losses})"],
        autopct="%1.0f%%",
        colors=wedge_colors,
        startangle=90,
        wedgeprops=dict(linewidth=1.5, edgecolor=BG_COLOR),
        textprops=dict(color=TEXT_PRIMARY, fontsize=9)
    )
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
        at.set_color("white")
        
    ax.set_title("Faithfulness: Win / Tie / Loss\n(margin ±0.05, merged by query)", fontsize=11, fontweight="bold", pad=8)

def draw_faithfulness_histogram(ax, df_crag, df_naive):
    format_card_axes(ax)
    
    ax.hist(df_crag["faithfulness"].dropna(), bins=12, color=CRAG_COLOR, alpha=0.55, label="CRAG", edgecolor=CARD_COLOR, linewidth=0.5)
    ax.hist(df_naive["faithfulness"].dropna(), bins=12, color=NAIVE_COLOR, alpha=0.55, label="Naive", edgecolor=CARD_COLOR, linewidth=0.5)
    
    ax.set_title("Faithfulness Score Histogram", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Score", fontsize=9, color=TEXT_MUTED)
    ax.set_ylabel("Question Count", fontsize=9, color=TEXT_MUTED)
    ax.legend(fontsize=9)

def draw_retries_vs_faithfulness(ax, df_crag):
    format_card_axes(ax)
    
    if "retry_count" in df_crag.columns:
        retry_means = df_crag.groupby("retry_count")["faithfulness"].mean().reset_index()
        retry_counts_str = retry_means["retry_count"].astype(str)
        
        bars = ax.bar(retry_counts_str, retry_means["faithfulness"], color=CRAG_COLOR, alpha=0.8, width=0.4, linewidth=0)
        ax.set_xlabel("Retry Count", fontsize=9, color=TEXT_MUTED)
        ax.set_ylabel("Avg Faithfulness", fontsize=9, color=TEXT_MUTED)
        ax.set_title("CRAG: Retries vs Faithfulness", fontsize=12, fontweight="bold", pad=8)
        ax.set_ylim(0, 1.1)
        
        for bar in bars:
            val = bar.get_height()
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.02, f"{val:.2f}",
                    ha="center", fontsize=9, fontweight="bold", color=TEXT_PRIMARY
                )
    else:
        ax.text(0.5, 0.5, "retry_count column missing", ha="center", va="center", color=TEXT_MUTED)

def draw_scatter_landscape(ax, df_crag, df_naive):
    format_card_axes(ax)
    
    ax.scatter(df_naive["answer_relevancy"], df_naive["faithfulness"], c=NAIVE_COLOR, label="Naive", alpha=0.6, s=50, marker="D", edgecolors="none")
    ax.scatter(df_crag["answer_relevancy"], df_crag["faithfulness"], c=CRAG_COLOR, label="CRAG", alpha=0.7, s=60, marker="o", edgecolors="none")
    
    ax.axhline(0.5, color=GRID_COLOR, linewidth=1, linestyle="--")
    ax.axvline(0.5, color=GRID_COLOR, linewidth=1, linestyle="--")
    
    ax.text(0.78, 0.90, "High Performance\nZone", transform=ax.transAxes,
            fontsize=8, color=ACCENT_GREEN, ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, edgecolor=ACCENT_GREEN, alpha=0.8))
            
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Answer Relevancy →", fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel("Faithfulness →", fontsize=10, color=TEXT_MUTED)
    ax.set_title("Performance Landscape: Faithfulness vs Relevancy", fontsize=13, fontweight="bold", pad=8)
    ax.legend(fontsize=9, loc="lower left")

def draw_latency_comparison(ax, df_crag, df_naive):
    format_card_axes(ax)
    
    latency_metrics = ["retrieval_ms", "generation_ms", "total_latency_ms"]
    latency_labels = ["Retrieval", "Generation", "Total"]
    
    crag_latency = [df_crag[m].mean() for m in latency_metrics]
    naive_latency = [df_naive[m].mean() for m in latency_metrics]
    
    x = np.arange(len(latency_metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, crag_latency, width, label="CRAG", color=CRAG_COLOR, alpha=0.85, linewidth=0)
    bars2 = ax.bar(x + width/2, naive_latency, width, label="Naive", color=NAIVE_COLOR, alpha=0.85, linewidth=0)
    
    ax.set_xticks(x)
    ax.set_xticklabels(latency_labels, fontsize=10)
    ax.set_ylabel("Time (ms)", fontsize=9, color=TEXT_MUTED)
    ax.set_title("Average Latency Comparison", fontsize=12, fontweight="bold", pad=8)
    ax.legend(fontsize=9)
    
    max_height = max(max(crag_latency), max(naive_latency))
    ax.set_ylim(0, max_height * 1.25)
    
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (max_height * 0.02),
                f"{height:.0f}ms",
                ha="center", fontsize=8, color=TEXT_PRIMARY, fontweight="bold"
            )

def draw_routing_impact(ax, df_crag):
    format_card_axes(ax)
    
    def routing_label(row):
        if row.get("web_search_used") == True:
            return "Web Fallback"
        elif row.get("retry_count", 0) > 0:
            return "Retries Used"
        else:
            return "Direct Hit"
            
    df_c = df_crag.copy()
    df_c["Routing Action"] = df_c.apply(routing_label, axis=1)
    
    route_summary = df_c.groupby("Routing Action").agg(
        avg_faithfulness=("faithfulness", "mean"),
        avg_relevancy=("answer_relevancy", "mean"),
        avg_precision=("context_precision", "mean"),
        count=("faithfulness", "count")
    ).reset_index()
    
    # Re-order to ensure consistent presentation
    order_map = {"Direct Hit": 0, "Retries Used": 1, "Web Fallback": 2}
    route_summary["order"] = route_summary["Routing Action"].map(order_map)
    route_summary = route_summary.sort_values("order").reset_index(drop=True)
    
    x = np.arange(len(route_summary))
    width = 0.22
    
    bars1 = ax.bar(x - width, route_summary["avg_faithfulness"], width, label="Faithfulness", color=CRAG_COLOR, alpha=0.85, linewidth=0)
    bars2 = ax.bar(x, route_summary["avg_relevancy"], width, label="Answer Relevancy", color=ACCENT_GREEN, alpha=0.85, linewidth=0)
    bars3 = ax.bar(x + width, route_summary["avg_precision"], width, label="Context Precision", color=ACCENT_AMBER, alpha=0.85, linewidth=0)
    
    labels_with_count = [
        f"{row['Routing Action']}\n(n={row['count']})"
        for _, row in route_summary.iterrows()
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels_with_count, fontsize=10)
    ax.set_ylabel("Average Score", fontsize=9, color=TEXT_MUTED)
    ax.set_title("CRAG Internal Routing: Metrics Comparison", fontsize=12, fontweight="bold", pad=8)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_ylim(0, 1.15)
    
    for bar in list(bars1) + list(bars2) + list(bars3):
        h = bar.get_height()
        if not np.isnan(h):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.02,
                f"{h:.2f}",
                ha="center", fontsize=8, color=TEXT_PRIMARY, fontweight="bold"
            )

def draw_segmentation_analysis(ax_faith, ax_rel, df_segmented):
    format_card_axes(ax_faith)
    format_card_axes(ax_rel)
    
    segments = ["Retriever Miss (n=7)", "Grader Rejection (n=16)", "Direct Hit (n=27)"]
    seg_keys = ["Retriever Miss (n=7)", "Grader Rejection (n=16)", "Direct Hit (n=27)"]
    
    # Extract means
    faith_crag = []
    faith_naive = []
    rel_crag = []
    rel_naive = []
    
    for k in seg_keys:
        sub = df_segmented[df_segmented["segment"] == k]
        faith_crag.append(sub["faithfulness_crag"].mean())
        faith_naive.append(sub["faithfulness_naive"].mean())
        rel_crag.append(sub["answer_relevancy_crag"].mean())
        rel_naive.append(sub["answer_relevancy_naive"].mean())
        
    x = np.arange(len(segments))
    width = 0.35
    
    # ──── Left Plot: Faithfulness ────
    bars_f_crag = ax_faith.bar(x - width/2, faith_crag, width, label="CRAG", color=CRAG_COLOR, linewidth=0)
    bars_f_naive = ax_faith.bar(x + width/2, faith_naive, width, label="Naive", color=NAIVE_COLOR, linewidth=0)
    
    ax_faith.set_xticks(x)
    ax_faith.set_xticklabels(segments, fontsize=9, fontweight="bold")
    ax_faith.set_ylabel("Faithfulness", fontsize=9, color=TEXT_MUTED)
    ax_faith.set_title("Faithfulness by Group", fontsize=11, fontweight="bold")
    ax_faith.set_ylim(0, 1.25)
    ax_faith.legend(fontsize=9, loc="upper right")
    
    # Annotate Faithfulness
    for bar1, bar2, label_key in zip(bars_f_crag, bars_f_naive, seg_keys):
        h_crag = bar1.get_height()
        h_naive = bar2.get_height()
        
        ax_faith.text(bar1.get_x() + bar1.get_width()/2, h_crag + 0.02, f"{h_crag:.3f}", ha="center", fontsize=8, color=TEXT_PRIMARY, fontweight="bold")
        ax_faith.text(bar2.get_x() + bar2.get_width()/2, h_naive + 0.02, f"{h_naive:.3f}", ha="center", fontsize=8, color=TEXT_PRIMARY, fontweight="bold")
        
        delta = h_crag - h_naive
        d_color = ACCENT_GREEN if delta >= 0 else ACCENT_RED
        d_sign = "+" if delta >= 0 else ""
        
        # Position annotation box slightly higher than the bars
        box_y = max(h_crag, h_naive) + 0.1
        ax_faith.text(
            bar1.get_x() + width, box_y,
            f"{d_sign}{delta:.3f}",
            ha="center", fontsize=8, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=d_color, edgecolor="none", alpha=0.9)
        )
        
    # ──── Right Plot: Answer Relevancy ────
    bars_r_crag = ax_rel.bar(x - width/2, rel_crag, width, label="CRAG", color=CRAG_COLOR, linewidth=0)
    bars_r_naive = ax_rel.bar(x + width/2, rel_naive, width, label="Naive", color=NAIVE_COLOR, linewidth=0)
    
    ax_rel.set_xticks(x)
    ax_rel.set_xticklabels(segments, fontsize=9, fontweight="bold")
    ax_rel.set_ylabel("Answer Relevancy", fontsize=9, color=TEXT_MUTED)
    ax_rel.set_title("Answer Relevancy by Group", fontsize=11, fontweight="bold")
    ax_rel.set_ylim(0, 1.25)
    ax_rel.legend(fontsize=9, loc="upper right")
    
    # Annotate Relevancy
    for bar1, bar2, label_key in zip(bars_r_crag, bars_r_naive, seg_keys):
        h_crag = bar1.get_height()
        h_naive = bar2.get_height()
        
        ax_rel.text(bar1.get_x() + bar1.get_width()/2, h_crag + 0.02, f"{h_crag:.3f}", ha="center", fontsize=8, color=TEXT_PRIMARY, fontweight="bold")
        ax_rel.text(bar2.get_x() + bar2.get_width()/2, h_naive + 0.02, f"{h_naive:.3f}", ha="center", fontsize=8, color=TEXT_PRIMARY, fontweight="bold")
        
        delta = h_crag - h_naive
        d_color = ACCENT_GREEN if delta >= 0 else ACCENT_RED
        d_sign = "+" if delta >= 0 else ""
        
        # Position annotation box
        box_y = max(h_crag, h_naive) + 0.1
        ax_rel.text(
            bar1.get_x() + width, box_y,
            f"{d_sign}{delta:.3f}",
            ha="center", fontsize=8, fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=d_color, edgecolor="none", alpha=0.9)
        )

# ── INDIVIDUAL PANEL SAVERS ───────────────────────────────────────────

def save_individual_plots(df_crag, df_naive, df_segmented):
    ind_dir = config.DATA_DIR / "plots" / "individual"
    ind_dir.mkdir(parents=True, exist_ok=True)
    
    # Helper to save a single axis plot
    def save_plot(filename, draw_func, *args, figsize=(10, 6.5), is_segmented=False):
        fig = plt.figure(figsize=figsize, facecolor=BG_COLOR)
        if is_segmented:
            gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.25, bottom=0.2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            draw_func(ax1, ax2, *args)
            
            # Add Footnote
            fig.text(
                0.5, 0.05,
                "Note: Direct-Hit underperformance is driven by partial context pruning (grader rejects some but not all chunks),\n"
                "distinct from full Grader Rejection cases where all chunks are rejected and web fallback triggers.",
                ha="center", fontsize=8.5, color=TEXT_MUTED, style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=BG_COLOR, edgecolor=GRID_COLOR, linewidth=0.5)
            )
            fig.suptitle("Segmentation Analysis — Where CRAG Helps vs Hurts", fontsize=14, fontweight="bold", y=0.96)
        else:
            ax = fig.add_subplot(111)
            draw_func(ax, *args)
            
        save_path = ind_dir / filename
        plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close(fig)
        print(f"Saved individual panel: {save_path.resolve()}")

    # 1. Summary Cards (Avg Metrics)
    fig = plt.figure(figsize=(10, 5), facecolor=BG_COLOR)
    axes = [fig.add_subplot(1, 3, i) for i in range(1, 4)]
    draw_avg_metrics(axes, df_crag, df_naive)
    fig.suptitle("Aggregate Average RAGAS Metrics", fontsize=14, fontweight="bold", y=0.98)
    save_path = ind_dir / "01_avg_metrics.png"
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved individual panel: {save_path.resolve()}")
    
    # 2. Violin Distribution plots
    fig = plt.figure(figsize=(12, 6.5), facecolor=BG_COLOR)
    axes = [fig.add_subplot(1, 3, i) for i in range(1, 4)]
    draw_violin_plots(axes, df_crag, df_naive)
    fig.suptitle("Detailed Metrics Distributions (Violin Plots)", fontsize=14, fontweight="bold", y=0.98)
    save_path = ind_dir / "02_violin_distributions.png"
    plt.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"Saved individual panel: {save_path.resolve()}")
    
    # 3. Win Tie Loss Pie
    save_plot("03_win_tie_loss.png", draw_win_tie_loss, df_segmented)
    
    # 4. Faithfulness Histogram
    save_plot("04_faithfulness_histogram.png", draw_faithfulness_histogram, df_crag, df_naive)
    
    # 5. Retries vs Faithfulness
    save_plot("05_retries_vs_faithfulness.png", draw_retries_vs_faithfulness, df_crag)
    
    # 6. Scatter Landscape
    save_plot("06_performance_landscape.png", draw_scatter_landscape, df_crag, df_naive)
    
    # 7. Latency Comparison
    save_plot("07_latency_comparison.png", draw_latency_comparison, df_crag, df_naive)
    
    # 8. CRAG Internal Routing
    save_plot("08_routing_impact.png", draw_routing_impact, df_crag)
    
    # 9. Segmentation Analysis
    save_plot("09_segmentation_analysis.png", draw_segmentation_analysis, df_segmented, figsize=(13, 7), is_segmented=True)

# ── MASTER COMBINED DASHBOARD ASSEMBLY ─────────────────────────────────

def plot_dashboard(df_crag, df_naive, df_segmented):
    fig = plt.figure(figsize=(19, 30), facecolor=BG_COLOR)
    
    # Dashboard Header & Context annotation
    fig.suptitle(
        "CRAG  vs  Naive Baseline — Evaluation Dashboard",
        fontsize=24, fontweight="bold", color=TEXT_PRIMARY, y=0.985
    )
    
    # Subtitle context annotation near title
    fig.text(
        0.5, 0.963,
        "Context: Naive baseline regenerated post-corpus-fix (480 papers, 9,856 chunks).\n"
        "Aggregate CRAG underperforms Naive, but segmentation reveals CRAG's correction mechanism succeeds specifically on genuine retrieval gaps (n=7)\n"
        "while being undermined by grader false-positive rejections (n=16).",
        ha="center", va="center", fontsize=11, fontweight="medium", color=TEXT_PRIMARY,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=CARD_COLOR, edgecolor=GRID_COLOR, linewidth=1)
    )
    
    gs = gridspec.GridSpec(
        6, 3,
        figure=fig,
        hspace=0.45,
        wspace=0.30,
        top=0.94, bottom=0.03,
        left=0.06, right=0.94
    )
    
    # ── [1] Summary Cards (Row 0) ─────────────────────────────────────
    avg_axes = [fig.add_subplot(gs[0, col]) for col in range(3)]
    draw_avg_metrics(avg_axes, df_crag, df_naive)
    
    # ── [2] Violin Plots (Row 1) ──────────────────────────────────────
    violin_axes = [fig.add_subplot(gs[1, col]) for col in range(3)]
    draw_violin_plots(violin_axes, df_crag, df_naive)
    
    # ── [3, 4, 5] row 2 ───────────────────────────────────────────────
    ax_wtl = fig.add_subplot(gs[2, 0])
    draw_win_tie_loss(ax_wtl, df_segmented)
    
    ax_hist = fig.add_subplot(gs[2, 1])
    draw_faithfulness_histogram(ax_hist, df_crag, df_naive)
    
    ax_retry = fig.add_subplot(gs[2, 2])
    draw_retries_vs_faithfulness(ax_retry, df_crag)
    
    # ── [6, 7] row 3 ──────────────────────────────────────────────────
    ax_scatter = fig.add_subplot(gs[3, :2])
    draw_scatter_landscape(ax_scatter, df_crag, df_naive)
    
    ax_latency = fig.add_subplot(gs[3, 2])
    draw_latency_comparison(ax_latency, df_crag, df_naive)
    
    # ── [8] row 4 (Full Width) ────────────────────────────────────────
    ax_route = fig.add_subplot(gs[4, :])
    draw_routing_impact(ax_route, df_crag)
    
    # ── [9] row 5: Segmentation Analysis (Centerpiece) ───────────────
    # Subplot nesting for the final row
    sub_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[5, :], wspace=0.25)
    ax_seg_faith = fig.add_subplot(sub_gs[0, 0])
    ax_seg_rel = fig.add_subplot(sub_gs[0, 1])
    
    draw_segmentation_analysis(ax_seg_faith, ax_seg_rel, df_segmented)
    
    # Add footnote specifically under the segmentation centerpiece row
    fig.text(
        0.5, 0.015,
        "Note: Direct-Hit underperformance is driven by partial context pruning (grader rejects some but not all chunks), "
        "distinct from full Grader Rejection cases where all chunks are rejected and web fallback triggers.",
        ha="center", fontsize=10.5, color=TEXT_MUTED, style="italic",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=CARD_COLOR, edgecolor=GRID_COLOR, linewidth=0.5)
    )
    
    # ── Save Master combined dashboard ───────────────────────────────
    plots_dir = config.DATA_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / "master_evaluation_dashboard.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"\n[SUCCESS] Master Combined Dashboard saved to: {save_path.resolve()}\n")

# ── RUN ENTRYPOINT ────────────────────────────────────────────────────

def main():
    try:
        df_crag, df_naive = load_data()
        df_segmented = segment_dataframe(df_crag, df_naive)
        
        # Save individual panels to individual/ directory
        save_individual_plots(df_crag, df_naive, df_segmented)
        
        # Plot and save combined master dashboard
        plot_dashboard(df_crag, df_naive, df_segmented)
        
    except Exception as e:
        print(f"[FATAL ERROR] Failed to run visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()