from __future__ import annotations

import math
import os
from pathlib import Path

from src.evaluation.report import build_full_report

OUT_DIR = Path("report_assets")
OUT_DIR.mkdir(exist_ok=True)


def _try_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import pandas as pd  # type: ignore
        import seaborn as sns  # type: ignore
        return plt, pd, sns
    except Exception:
        return None


def _draw_fallback_charts():
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1600, 1000
    bg = "#ffffff"
    grid = "#e5e7eb"
    axis = "#94a3b8"
    title = "#0f172a"
    text = "#334155"
    smart = "#1f77b4"
    baseline = "#ff7f0e"

    try:
        font_regular = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 24)
        font_bold = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 28)
        font_title = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 34)
    except Exception:
        font_regular = ImageFont.load_default()
        font_bold = ImageFont.load_default()
        font_title = ImageFont.load_default()

    def draw_centered(draw, xy, txt, font, fill):
        bbox = draw.textbbox((0, 0), txt, font=font)
        tw = bbox[2] - bbox[0]
        x, y = xy
        draw.text((x - tw / 2, y), txt, font=font, fill=fill)

    def draw_chart(
        filename: str,
        heading: str,
        x_labels: list[str],
        smart_vals: list[float],
        base_vals: list[float],
        y_max: float,
        y_label: str,
    ):
        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        draw.text((70, 40), heading, font=font_title, fill=title)
        draw.text((70, 90), y_label, font=font_regular, fill=text)

        left, top, right, bottom = 120, 160, 1520, 890
        plot_w = right - left
        plot_h = bottom - top

        for i in range(6):
            y = bottom - plot_h * i / 5
            draw.line((left, y, right, y), fill=grid, width=2)
            label_val = y_max * i / 5
            label = f"{label_val:.2f}" if y_max <= 10 else f"{label_val:.0f}"
            draw.text((40, y - 12), label, font=font_regular, fill=text)

        draw.line((left, top, left, bottom), fill=axis, width=3)
        draw.line((left, bottom, right, bottom), fill=axis, width=3)

        n = len(x_labels)
        group_w = plot_w / n
        bar_w = group_w * 0.24
        gap = group_w * 0.06
        label_y = bottom + 16

        for idx, metric in enumerate(x_labels):
            gx = left + group_w * idx + group_w / 2
            smart_h = (smart_vals[idx] / y_max) * plot_h
            base_h = (base_vals[idx] / y_max) * plot_h
            smart_x0 = gx - bar_w - gap / 2
            base_x0 = gx + gap / 2
            smart_x1 = smart_x0 + bar_w
            base_x1 = base_x0 + bar_w
            smart_y0 = bottom - smart_h
            base_y0 = bottom - base_h

            draw.rectangle((smart_x0, smart_y0, smart_x1, bottom), fill=smart)
            draw.rectangle((base_x0, base_y0, base_x1, bottom), fill=baseline)

            draw_centered(draw, (smart_x0, smart_y0 - 34), f"{smart_vals[idx]:.3f}", font_bold, text)
            draw_centered(draw, (base_x0, base_y0 - 34), f"{base_vals[idx]:.3f}", font_bold, text)
            draw_centered(draw, (gx - 60, label_y), metric, font_regular, title)

        # Legend
        lx, ly = 1170, 70
        draw.rectangle((lx, ly, lx + 26, ly + 26), fill=smart)
        draw.text((lx + 38, ly - 2), "Smart", font=font_regular, fill=text)
        draw.rectangle((lx + 170, ly, lx + 196, ly + 26), fill=baseline)
        draw.text((lx + 208, ly - 2), "Baseline", font=font_regular, fill=text)

        img.save(OUT_DIR / filename)

    draw_chart(
        "ranking_quality.png",
        "Ranking Quality: Smart vs. Baseline",
        ["P@5", "R@5", "NDCG@5"],
        [0.308, 0.769, 0.686],
        [0.462, 0.846, 0.732],
        1.0,
        "Higher is better",
    )

    # Timing chart: two panels in one image.
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    draw.text((70, 40), "Performance / Timing Comparison", font=font_title, fill=title)

    panels = [
        ((120, 180, 760, 860), "Mean Time (ms)", [29.28, 2.20], 35),
        ((840, 180, 1480, 860), "Peak Memory (KB)", [127.1, 9.9], 140),
    ]
    labels = ["Smart", "Baseline"]
    colors = [smart, baseline]

    for (left, top, right, bottom), ptitle, vals, ymax in panels:
        draw.text((left, top - 52), ptitle, font=font_bold, fill=title)
        plot_w = right - left
        plot_h = bottom - top
        for i in range(6):
            y = bottom - plot_h * i / 5
            draw.line((left, y, right, y), fill=grid, width=2)
            label_val = ymax * i / 5
            draw.text((left - 80, y - 12), f"{label_val:.1f}", font=font_regular, fill=text)
        draw.line((left, top, left, bottom), fill=axis, width=3)
        draw.line((left, bottom, right, bottom), fill=axis, width=3)

        bar_w = plot_w * 0.22
        centers = [left + plot_w * 0.30, left + plot_w * 0.70]
        for idx, val in enumerate(vals):
            x0 = centers[idx] - bar_w / 2
            h = (val / ymax) * plot_h
            y0 = bottom - h
            draw.rectangle((x0, y0, x0 + bar_w, bottom), fill=colors[idx])
            draw_centered(draw, (x0, y0 - 34), f"{val:.2f}", font_bold, text)
            draw_centered(draw, (centers[idx] - 70, bottom + 16), labels[idx], font_regular, title)

    lx, ly = 1120, 70
    draw.rectangle((lx, ly, lx + 26, ly + 26), fill=smart)
    draw.text((lx + 38, ly - 2), "Smart", font=font_regular, fill=text)
    draw.rectangle((lx + 170, ly, lx + 196, ly + 26), fill=baseline)
    draw.text((lx + 208, ly - 2), "Baseline", font=font_regular, fill=text)
    img.save(OUT_DIR / "performance_timing.png")


def main():
    report = build_full_report()
    auto_summary = report["auto"]["summary"]
    auto_counts = report["auto"]["strategy_counts"]
    auto_timing = report["auto_timing"]
    retrieval_by_k = report["retrieval_by_k"]
    auto_by_k = report["auto_by_k"]

    backend = _try_matplotlib()
    if backend is None:
        _draw_fallback_charts()
        print("Saved:\n- report_assets/ranking_quality.png\n- report_assets/performance_timing.png")
        print("- report_assets/auto_routing.png")
        print("- report_assets/full_comparison.png")
        return

    plt, pd, sns = backend
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    smart_color = "#1f77b4"
    baseline_color = "#ff7f0e"

    ranking_df = pd.DataFrame(
        {
            "Metric": ["P@5", "R@5", "NDCG@5"],
            "Smart": [
                report["retrieval"]["smart"].precision_at_k,
                report["retrieval"]["smart"].recall_at_k,
                report["retrieval"]["smart"].ndcg_at_k,
            ],
            "Baseline": [
                report["retrieval"]["baseline"].precision_at_k,
                report["retrieval"]["baseline"].recall_at_k,
                report["retrieval"]["baseline"].ndcg_at_k,
            ],
        }
    )
    ranking_melt = ranking_df.melt(id_vars="Metric", var_name="Model", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=ranking_melt,
        x="Metric",
        y="Score",
        hue="Model",
        hue_order=["Smart", "Baseline"],
        palette=[smart_color, baseline_color],
        ax=ax,
    )
    ax.set_title("Ranking Quality: Smart vs. Baseline", pad=16)
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.legend(title="Model", frameon=True, loc="upper left")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ranking_quality.png", bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    counts_df = pd.DataFrame(
        {
            "Strategy": ["Baseline", "Smart"],
            "Count": [auto_counts["baseline"], auto_counts["smart"]],
        }
    )
    sns.barplot(data=counts_df, x="Strategy", y="Count", palette=[baseline_color, smart_color], ax=ax)
    ax.set_title("Auto Routing Split", pad=16)
    ax.set_xlabel("Chosen strategy")
    ax.set_ylabel("Benchmark cases")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "auto_routing.png", bbox_inches="tight")
    plt.close()

    performance_df = pd.DataFrame(
        {
            "Metric": ["Mean Time (ms)", "Peak Memory (KB)"],
            "Smart": [report["timings"]["smart"].mean_ms, report["timings"]["smart"].peak_kb],
            "Baseline": [report["timings"]["baseline"].mean_ms, report["timings"]["baseline"].peak_kb],
            "Auto": [auto_timing.mean_ms, auto_timing.peak_kb],
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, metric, title, ylabel in [
        (axes[0], "Mean Time (ms)", "Mean Time", "Milliseconds"),
        (axes[1], "Peak Memory (KB)", "Peak Memory", "KB"),
    ]:
        subset = performance_df[performance_df["Metric"] == metric].melt(id_vars="Metric", var_name="Model", value_name="Value")
        sns.barplot(data=subset, x="Model", y="Value", palette=[smart_color, baseline_color], ax=ax)
        ax.set_title(title, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(["Smart", "Baseline"])
        ax.set_ylim(0, max(subset["Value"]) * 1.35)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3, fontsize=10)
    fig.suptitle("Performance / Timing Comparison", y=1.03, fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "performance_timing.png", bbox_inches="tight")
    plt.close()

    full_fig, full_axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = full_axes[0, 0]
    sns.barplot(
        data=ranking_melt,
        x="Metric",
        y="Score",
        hue="Model",
        hue_order=["Smart", "Baseline", "Auto"],
        palette=[smart_color, baseline_color, "#2ca02c"],
        ax=ax,
    )
    ax.set_title("Ranking Quality")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(title="Model", frameon=True, loc="upper left")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    timing_long = pd.DataFrame(
        [
            {"Model": "Smart", "Metric": "Mean Time (ms)", "Value": report["timings"]["smart"].mean_ms},
            {"Model": "Baseline", "Metric": "Mean Time (ms)", "Value": report["timings"]["baseline"].mean_ms},
            {"Model": "Auto", "Metric": "Mean Time (ms)", "Value": auto_timing.mean_ms},
            {"Model": "Smart", "Metric": "Peak Memory (KB)", "Value": report["timings"]["smart"].peak_kb},
            {"Model": "Baseline", "Metric": "Peak Memory (KB)", "Value": report["timings"]["baseline"].peak_kb},
            {"Model": "Auto", "Metric": "Peak Memory (KB)", "Value": auto_timing.peak_kb},
        ]
    )

    for ax, metric in [(full_axes[0, 1], "Mean Time (ms)"), (full_axes[1, 0], "Peak Memory (KB)")]:
        subset = timing_long[timing_long["Metric"] == metric]
        sns.barplot(
            data=subset,
            x="Model",
            y="Value",
            hue="Model",
            hue_order=["Smart", "Baseline", "Auto"],
            palette=[smart_color, baseline_color, "#2ca02c"],
            ax=ax,
            legend=False,
        )
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("Value")
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)

    ax = full_axes[1, 1]
    sns.barplot(data=counts_df, x="Strategy", y="Count", palette=[baseline_color, smart_color], ax=ax)
    ax.set_title("Auto Routing Split")
    ax.set_xlabel("")
    ax.set_ylabel("Cases")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=3, fontsize=9)
    ax.text(
        0.02,
        -0.35,
        f"Auto P@5={auto_summary.precision_at_k:.3f} | R@5={auto_summary.recall_at_k:.3f} | NDCG@5={auto_summary.ndcg_at_k:.3f}",
        transform=ax.transAxes,
        fontsize=12,
        weight="bold",
    )
    full_fig.suptitle("Full Model Comparison", y=0.99, fontsize=20, fontweight="bold")
    full_fig.tight_layout()
    full_fig.savefig(OUT_DIR / "full_comparison.png", bbox_inches="tight")
    plt.close(full_fig)

    k_values = [5, 10, 15, 20]
    metric_rows = []
    for model in ["smart", "baseline", "auto"]:
        for k in k_values:
            summary = (
                auto_by_k["summaries"][f"@{k}"] if model == "auto"
                else retrieval_by_k[model][f"@{k}"]
            )
            metric_rows.append(
                {
                    "Model": model.capitalize(),
                    "k": k,
                    "P@k": summary.precision_at_k,
                    "R@k": summary.recall_at_k,
                    "NDCG@k": summary.ndcg_at_k,
                }
            )
    metric_df = pd.DataFrame(metric_rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    for ax, metric in zip(axes, ["P@k", "R@k", "NDCG@k"]):
        sns.lineplot(
            data=metric_df,
            x="k",
            y=metric,
            hue="Model",
            style="Model",
            markers=True,
            dashes=False,
            palette=[smart_color, baseline_color, "#2ca02c"],
            ax=ax,
        )
        ax.set_title(metric)
        ax.set_xlabel("k")
        ax.set_ylabel("Score")
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1.05)
        ax.legend(title="Model", frameon=True, loc="best")
    fig.suptitle("Metrics Across k", y=1.05, fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "metrics_by_k.png", bbox_inches="tight")
    plt.close()

    from PIL import Image, ImageOps

    composite_paths = [
        OUT_DIR / "ranking_quality.png",
        OUT_DIR / "performance_timing.png",
        OUT_DIR / "auto_routing.png",
        OUT_DIR / "metrics_by_k.png",
    ]
    composite_images = [Image.open(path).convert("RGB") for path in composite_paths]
    cell_w, cell_h = 900, 620
    margin = 40
    header_h = 110
    canvas = Image.new("RGB", (cell_w * 2 + margin * 3, cell_h * 2 + margin * 3 + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 40)
        sub_font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 22)
    except Exception:
        title_font = ImageFont.load_default()
        sub_font = ImageFont.load_default()
    draw.text((margin, 25), "Full Model Comparison", font=title_font, fill="#0f172a")
    draw.text((margin, 75), "Ranking quality, timing, routing split, and metrics across k", font=sub_font, fill="#334155")
    positions = [
        (margin, header_h + margin),
        (margin * 2 + cell_w, header_h + margin),
        (margin, header_h + margin * 2 + cell_h),
        (margin * 2 + cell_w, header_h + margin * 2 + cell_h),
    ]
    for img, pos in zip(composite_images, positions):
        fitted = ImageOps.contain(img, (cell_w, cell_h), method=Image.Resampling.LANCZOS)
        x, y = pos
        canvas.paste(fitted, (x + (cell_w - fitted.width) // 2, y + (cell_h - fitted.height) // 2))
    canvas.save(OUT_DIR / "full_comparison.png")

    print("Saved:\n- report_assets/ranking_quality.png\n- report_assets/performance_timing.png\n- report_assets/auto_routing.png\n- report_assets/full_comparison.png\n- report_assets/metrics_by_k.png")


if __name__ == "__main__":
    main()
