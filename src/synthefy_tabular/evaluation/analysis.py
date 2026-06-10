"""Analysis and comparison module for evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


class EvalAnalyzer:
    """Analyze and compare evaluation results across models and datasets."""

    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df.copy()

    @classmethod
    def from_csv(cls, path: str):
        return cls(pd.read_csv(path))

    @classmethod
    def merge_results(cls, *dfs):
        return cls(pd.concat(dfs, ignore_index=True))

    def _clean(self, task_type=None):
        df = self.df[self.df["error"].isna()] if "error" in self.df.columns else self.df
        if task_type:
            df = df[df["task_type"] == task_type]
        return df

    def _metric_columns(self, df):
        skip = {"n_train", "n_test", "n_features", "n_classes", "latency_ms", "throughput_sps"}
        return [c for c in df.select_dtypes(include=[np.number]).columns if c not in skip]

    @staticmethod
    def _safe_name(text: str) -> str:
        return (
            text.replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(":", "_")
        )

    @staticmethod
    def _import_matplotlib():
        try:
            import matplotlib.pyplot as plt
            return plt
        except Exception as e:
            print(f"[EvalAnalyzer] Plot generation skipped (matplotlib unavailable): {e}")
            return None

    @staticmethod
    def _save_figure(fig, output_path):
        if output_path is None:
            return None
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=220, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
        return str(out)

    @staticmethod
    def _primary_metric(task_type):
        return "auc" if task_type == "classification" else "r2"

    @staticmethod
    def _higher_is_better(metric):
        return metric not in {"rmse", "mae", "log_loss", "ece", "latency_ms"}

    @staticmethod
    def _short_model_name(name, max_len=28):
        if len(name) <= max_len:
            return name
        return name[: max_len - 3] + "..."

    def _comparison_xlim(self, means, errs, metric):
        """Choose x-limits that preserve readability for close model scores."""
        means = np.asarray(means, dtype=float)
        errs = np.asarray(errs, dtype=float)
        lo = np.nanmin(means - errs)
        hi = np.nanmax(means + errs)
        span = hi - lo if np.isfinite(hi - lo) else 0.0

        # For bounded metrics, zoom when scores are close to reveal ranking.
        if metric in {"auc", "accuracy", "f1"}:
            if span < 0.20:
                pad = max(0.01, span * 0.30)
                return max(0.0, lo - pad), min(1.0, hi + pad), True
            return 0.0, min(1.0, hi + max(0.02, span * 0.10)), False

        # R2 is often clustered; zoom by default while keeping room for labels.
        if metric == "r2":
            pad = max(0.015, span * 0.28 if span > 0 else 0.03)
            return lo - pad, hi + pad, True

        # Unbounded metrics fallback.
        pad = max(0.02, span * 0.12 if span > 0 else 0.05)
        return lo - pad, hi + pad, False

    def _pick_focus_model(self, focus_model: Optional[str] = None):
        models = sorted(self.df["model"].dropna().unique()) if "model" in self.df.columns else []
        if not models:
            return None
        if focus_model and focus_model in models:
            return focus_model
        return models[0]

    # --- Aggregate summaries ---

    def summary_by_model(self, task_type=None):
        df = self._clean(task_type)
        mcols = self._metric_columns(df)
        agg = df.groupby("model")[mcols].agg(["mean", "std", "count"])
        agg.columns = [f"{c[0]}_{c[1]}" for c in agg.columns]
        return agg.sort_values(agg.columns[0], ascending=False)

    def summary_by_source(self, model_name=None):
        df = self._clean()
        if model_name:
            df = df[df["model"] == model_name]
        return df.groupby(["model", "source"])[self._metric_columns(df)].mean().round(4)

    def per_dataset_table(self, task_type=None, metric="auc"):
        df = self._clean(task_type)
        if metric not in df.columns:
            return pd.DataFrame()
        return df.pivot_table(
            index=["source", "dataset"], columns="model",
            values=metric, aggfunc="first",
        ).sort_index()

    # --- Head-to-head comparison ---

    def head_to_head(self, model_a, model_b, metric="auc", task_type=None):
        df = self._clean(task_type)
        if metric not in df.columns:
            return {"error": f"Metric '{metric}' not found"}
        a = df[df["model"] == model_a][["dataset", "source", metric]].rename(columns={metric: f"{metric}_a"})
        b = df[df["model"] == model_b][["dataset", "source", metric]].rename(columns={metric: f"{metric}_b"})
        m = a.merge(b, on=["dataset", "source"])
        m["delta"] = m[f"{metric}_a"] - m[f"{metric}_b"]
        wins_a = int((m["delta"] > 0.001).sum())
        wins_b = int((m["delta"] < -0.001).sum())
        ties = len(m) - wins_a - wins_b
        return {
            "model_a": model_a, "model_b": model_b, "metric": metric,
            "n_datasets": len(m), "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
            "mean_delta": float(m["delta"].mean()), "std_delta": float(m["delta"].std()),
            "per_dataset": m.sort_values("delta", ascending=False).to_dict("records"),
        }

    def pairwise_comparison(self, metric="auc", task_type=None):
        df = self._clean(task_type)
        models = sorted(df["model"].unique())
        mat = pd.DataFrame(index=models, columns=models, dtype=float)
        for m_a in models:
            for m_b in models:
                if m_a == m_b:
                    mat.loc[m_a, m_b] = float("nan")
                    continue
                h = self.head_to_head(m_a, m_b, metric=metric, task_type=task_type)
                total = h["wins_a"] + h["wins_b"] + h["ties"]
                mat.loc[m_a, m_b] = h["wins_a"] / total if total > 0 else 0.5
        return mat

    # --- Checkpoint progression ---

    def checkpoint_progression(self, model_prefix, metric="auc", task_type=None):
        df = self._clean(task_type)
        if metric not in df.columns:
            return pd.DataFrame()
        sub = df[df["model"].str.startswith(model_prefix)]
        if sub.empty:
            return pd.DataFrame()
        pivot = sub.pivot_table(index=["source", "dataset"], columns="model", values=metric, aggfunc="first")
        means = pivot.mean(axis=0)
        means.name = ("MEAN", "MEAN")
        return pd.concat([pivot, means.to_frame().T])

    # --- Latency analysis ---

    def latency_summary(self):
        df = self._clean()
        cols = [c for c in ["latency_ms", "throughput_sps"] if c in df.columns]
        if not cols:
            return pd.DataFrame()
        return df.groupby("model")[cols].agg(["mean", "median", "std", "min", "max"]).round(1)

    def latency_by_dataset_size(self):
        df = self._clean()
        if "latency_ms" not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df["total_samples"] = df["n_train"] + df["n_test"]
        bins = [0, 500, 2000, 10000, 50000, float("inf")]
        labels = ["<500", "500-2K", "2K-10K", "10K-50K", ">50K"]
        df["size_bucket"] = pd.cut(df["total_samples"], bins=bins, labels=labels)
        return df.groupby(["model", "size_bucket"])["latency_ms"].agg(["mean", "count"]).round(1)

    # --- Deep diagnostics / plots ---

    @staticmethod
    def _best_other_row(row, other_cols):
        valid = []
        for col in other_cols:
            v = row[col]
            if pd.notna(v) and np.isfinite(v):
                valid.append((col, float(v)))
        if not valid:
            return pd.Series({"best_other_model": None, "best_other_score": float("nan")})
        best_model, best_score = max(valid, key=lambda x: x[1])
        return pd.Series({"best_other_model": best_model, "best_other_score": best_score})

    def comparison_vs_best_other(self, focus_model, task_type, metric=None):
        """Dataset-level comparison: focus model score vs best non-focus model."""
        metric = metric or self._primary_metric(task_type)
        df = self._clean(task_type)
        if df.empty or metric not in df.columns:
            return pd.DataFrame()

        cols = ["source", "dataset", "model", metric, "n_train", "n_test", "n_features", "n_classes"]
        cols = [c for c in cols if c in df.columns]
        sub = df[cols].copy()

        pivot = sub.pivot_table(
            index=["source", "dataset"],
            columns="model",
            values=metric,
            aggfunc="mean",
        )
        if focus_model not in pivot.columns:
            return pd.DataFrame()

        meta_cols = [c for c in ["source", "dataset", "n_train", "n_test", "n_features", "n_classes"] if c in sub.columns]
        meta = sub[sub["model"] == focus_model][meta_cols].drop_duplicates(["source", "dataset"])

        comp = pivot.reset_index().merge(meta, on=["source", "dataset"], how="left")
        comp = comp.rename(columns={focus_model: "focus_score"})
        other_cols = [c for c in pivot.columns if c != focus_model]
        if not other_cols:
            return pd.DataFrame()

        best = comp.apply(lambda row: self._best_other_row(row, other_cols), axis=1)
        comp = pd.concat([comp, best], axis=1)
        comp = comp[np.isfinite(comp["focus_score"])]
        comp["delta_vs_best_other"] = comp["focus_score"] - comp["best_other_score"]

        if "n_train" in comp.columns and "n_test" in comp.columns:
            comp["total_samples"] = comp["n_train"].fillna(0) + comp["n_test"].fillna(0)
            comp["sample_bucket"] = pd.cut(
                comp["total_samples"],
                bins=[0, 500, 2_000, 10_000, 50_000, float("inf")],
                labels=["<500", "500-2K", "2K-10K", "10K-50K", ">50K"],
            )
        if "n_features" in comp.columns:
            comp["feature_bucket"] = pd.cut(
                comp["n_features"],
                bins=[0, 20, 50, 100, 200, float("inf")],
                labels=["<=20", "21-50", "51-100", "101-200", ">200"],
            )
        if task_type == "classification" and "n_classes" in comp.columns:
            comp["class_bucket"] = pd.cut(
                comp["n_classes"],
                bins=[0, 2, 5, 10, float("inf")],
                labels=["2", "3-5", "6-10", ">10"],
            )
        return comp

    def plot_metric_leaderboard(self, task_type, metric=None, output_path=None):
        """Horizontal bar chart of mean metric by model."""
        plt = self._import_matplotlib()
        if plt is None:
            return None
        metric = metric or self._primary_metric(task_type)
        df = self._clean(task_type)
        if df.empty or metric not in df.columns:
            return None

        higher_is_better = self._higher_is_better(metric)
        agg = df.groupby("model")[metric].agg(["mean", "std", "count"])
        agg = agg.sort_values("mean", ascending=not higher_is_better)
        if agg.empty:
            return None

        agg["sem"] = agg["std"].fillna(0.0) / np.sqrt(np.maximum(agg["count"].values, 1))
        agg["ci95"] = 1.96 * agg["sem"]
        best_score = agg["mean"].iloc[0]

        fig_h = max(4.0, 0.48 * len(agg))
        fig, ax = plt.subplots(figsize=(12.5, fig_h))
        y = np.arange(len(agg))
        means = agg["mean"].values
        ci95 = agg["ci95"].fillna(0.0).values

        colors = ["#2CA02C"] + ["#4E79A7"] * (len(agg) - 1)
        ax.barh(
            y,
            means,
            xerr=ci95,
            color=colors,
            alpha=0.92,
            ecolor="#2f2f2f",
            capsize=3,
        )

        ax.set_yticks(y)
        ax.set_yticklabels([self._short_model_name(m, 30) for m in agg.index])
        ax.invert_yaxis()
        ax.set_xlabel(metric.upper())
        ax.set_title(f"{task_type.title()} leaderboard ({metric.upper()}, mean +/- 95% CI)")
        ax.grid(axis="x", alpha=0.25)

        x_lo, x_hi, zoomed = self._comparison_xlim(means, ci95, metric)
        ax.set_xlim(x_lo, x_hi)

        rng = x_hi - x_lo if np.isfinite(x_hi - x_lo) else 1.0
        pad = max(0.002, 0.015 * rng)
        for i, (_, row) in enumerate(agg.iterrows()):
            x = row["mean"] + (row["ci95"] if np.isfinite(row["ci95"]) else 0.0) + pad
            delta = row["mean"] - best_score
            if i == 0:
                label = f"{row['mean']:.4f} (best), n={int(row['count'])}"
            else:
                label = f"{row['mean']:.4f} ({delta:+.4f}), n={int(row['count'])}"
            ax.text(
                x,
                i,
                label,
                va="center",
                ha="left",
                fontsize=8.8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 0.8},
            )

        if zoomed:
            ax.text(
                0.01,
                0.01,
                "x-axis zoomed to highlight ranking differences",
                transform=ax.transAxes,
                fontsize=8,
                alpha=0.75,
            )

        fig.tight_layout()
        return self._save_figure(fig, output_path)

    def plot_source_grouped_bars(self, task_type, metric=None, output_path=None):
        """Grouped bar chart by source and model."""
        plt = self._import_matplotlib()
        if plt is None:
            return None
        metric = metric or self._primary_metric(task_type)
        df = self._clean(task_type)
        if df.empty or metric not in df.columns:
            return None

        pivot = (
            df.groupby(["source", "model"])[metric]
            .mean()
            .unstack("model")
            .sort_index()
        )
        if pivot.empty:
            return None

        models = list(pivot.columns)
        sources = list(pivot.index)
        x = np.arange(len(sources))
        width = 0.8 / max(len(models), 1)

        fig_w = max(10.0, 1.2 * len(sources))
        fig_h = max(5.0, 0.45 * len(models) + 3.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        for i, model in enumerate(models):
            offset = (i - (len(models) - 1) / 2.0) * width
            vals = pivot[model].values
            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                label=self._short_model_name(model, 24),
                alpha=0.9,
            )
            # Add exact values when the chart isn't too dense.
            if len(models) <= 7 and len(sources) <= 12:
                for b, v in zip(bars, vals):
                    if np.isfinite(v):
                        ax.text(
                            b.get_x() + b.get_width() / 2.0,
                            b.get_height(),
                            f"{v:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=7,
                            rotation=90,
                        )

        ax.set_xticks(x)
        ax.set_xticklabels(sources, rotation=25, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{task_type.title()} by source ({metric.upper()})")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, ncol=2, frameon=False)

        fig.tight_layout()
        return self._save_figure(fig, output_path)

    def plot_pairwise_heatmap(self, task_type, metric=None, output_path=None):
        """Pairwise win-rate heatmap (row model vs column model)."""
        plt = self._import_matplotlib()
        if plt is None:
            return None
        metric = metric or self._primary_metric(task_type)
        mat = self.pairwise_comparison(metric=metric, task_type=task_type)
        if mat.empty:
            return None

        fig_w = max(6.0, 1.0 * len(mat.columns) + 2.0)
        fig_h = max(5.0, 0.8 * len(mat.index) + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(mat.values, cmap="RdYlGn", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(mat.columns)))
        ax.set_xticklabels(mat.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_title(f"Pairwise win-rate heatmap ({task_type}, {metric.upper()})")

        for i in range(len(mat.index)):
            for j in range(len(mat.columns)):
                val = mat.values[i, j]
                if not np.isfinite(val):
                    text = "-"
                else:
                    text = f"{val:.0%}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color="black")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Win rate")
        fig.tight_layout()
        return self._save_figure(fig, output_path)

    def plot_focus_loss_topk(self, comp_df, focus_model, task_type, metric, top_k=25, output_path=None):
        """Top-K hardest datasets for focus model vs best non-focus model."""
        plt = self._import_matplotlib()
        if plt is None or comp_df.empty:
            return None

        data = comp_df[np.isfinite(comp_df["delta_vs_best_other"])].copy()
        if data.empty:
            return None
        data = data.sort_values("delta_vs_best_other", ascending=True).head(top_k)

        labels = [f"{s}/{d}" for s, d in zip(data["source"], data["dataset"])]
        vals = data["delta_vs_best_other"].values
        colors = ["#D62728" if v < 0 else "#2CA02C" for v in vals]

        fig_h = max(5.5, 0.35 * len(data))
        fig, ax = plt.subplots(figsize=(14, fig_h))
        y = np.arange(len(data))
        ax.barh(y, vals, color=colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0.0, color="black", linewidth=1, alpha=0.8)
        ax.set_xlabel(f"Delta = {focus_model} - best_other ({metric.upper()})")
        ax.set_title(f"{task_type.title()} hardest datasets for {focus_model} (Top {len(data)})")
        ax.grid(axis="x", alpha=0.25)

        span = np.nanmax(np.abs(vals)) if len(vals) else 1.0
        pad = max(0.01, 0.03 * span)
        for i, (_, row) in enumerate(data.iterrows()):
            x = row["delta_vs_best_other"]
            other = row["best_other_model"] if pd.notna(row["best_other_model"]) else "N/A"
            text_x = x + (pad if x >= 0 else -pad)
            ha = "left" if x >= 0 else "right"
            ax.text(text_x, i, other, va="center", ha=ha, fontsize=8)

        fig.tight_layout()
        return self._save_figure(fig, output_path)

    def plot_focus_bucket_analysis(self, comp_df, focus_model, task_type, metric, output_path=None):
        """Bucket-level mean deltas to reveal weakness regimes."""
        plt = self._import_matplotlib()
        if plt is None or comp_df.empty:
            return None

        bucket_cols = [c for c in ["sample_bucket", "feature_bucket", "class_bucket"] if c in comp_df.columns]
        if not bucket_cols:
            return None

        fig, axes = plt.subplots(1, len(bucket_cols), figsize=(6.2 * len(bucket_cols), 5.0))
        if len(bucket_cols) == 1:
            axes = [axes]

        for ax, col in zip(axes, bucket_cols):
            g = (
                comp_df.groupby(col, observed=False)["delta_vs_best_other"]
                .agg(["mean", "count"])
                .dropna()
            )
            if g.empty:
                ax.set_title(f"{col} (no data)")
                continue
            x = np.arange(len(g.index))
            means = g["mean"].values
            colors = ["#D62728" if v < 0 else "#2CA02C" for v in means]
            ax.bar(x, means, color=colors, alpha=0.9)
            ax.axhline(0.0, color="black", linewidth=1, alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels([str(v) for v in g.index], rotation=20, ha="right")
            ax.set_ylabel("Mean delta")
            ax.set_title(col.replace("_", " ").title())
            ax.grid(axis="y", alpha=0.25)
            for i, (_, row) in enumerate(g.iterrows()):
                ax.text(i, row["mean"], f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)

        fig.suptitle(f"{task_type.title()} weakness buckets for {focus_model} ({metric.upper()})", y=1.02)
        fig.tight_layout()
        return self._save_figure(fig, output_path)

    def plot_focus_scatter(self, comp_df, focus_model, task_type, metric, output_path=None):
        """Scatter diagnostics: delta vs #features and delta vs #samples."""
        plt = self._import_matplotlib()
        if plt is None or comp_df.empty:
            return None
        if "n_features" not in comp_df.columns or "total_samples" not in comp_df.columns:
            return None

        data = comp_df[np.isfinite(comp_df["delta_vs_best_other"])].copy()
        if data.empty:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        axes[0].scatter(data["n_features"], data["delta_vs_best_other"], alpha=0.65, s=24)
        axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.8)
        axes[0].set_xlabel("n_features")
        axes[0].set_ylabel("Delta vs best other")
        axes[0].set_title("Delta vs feature count")
        axes[0].grid(alpha=0.25)

        axes[1].scatter(data["total_samples"], data["delta_vs_best_other"], alpha=0.65, s=24)
        axes[1].axhline(0.0, color="black", linewidth=1, alpha=0.8)
        axes[1].set_xlabel("total_samples")
        axes[1].set_ylabel("Delta vs best other")
        axes[1].set_title("Delta vs sample count")
        axes[1].set_xscale("log")
        axes[1].grid(alpha=0.25)

        fig.suptitle(f"{task_type.title()} difficulty map for {focus_model} ({metric.upper()})", y=1.02)
        fig.tight_layout()
        return self._save_figure(fig, output_path)

    def plot_latency_overview(self, output_path=None):
        """Latency and throughput comparison bars."""
        plt = self._import_matplotlib()
        if plt is None:
            return None
        df = self._clean()
        if df.empty or "latency_ms" not in df.columns:
            return None

        agg = df.groupby("model")[["latency_ms", "throughput_sps"]].mean().sort_values("latency_ms")
        if agg.empty:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(13, max(4.8, 0.45 * len(agg))))
        y = np.arange(len(agg))

        axes[0].barh(y, agg["latency_ms"].values, color="#7F7F7F", alpha=0.9)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(agg.index)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Mean latency (ms)")
        axes[0].set_title("Latency")
        axes[0].grid(axis="x", alpha=0.25)

        axes[1].barh(y, agg["throughput_sps"].values, color="#59A14F", alpha=0.9)
        axes[1].set_yticks(y)
        axes[1].set_yticklabels(agg.index)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("Mean throughput (samples/s)")
        axes[1].set_title("Throughput")
        axes[1].grid(axis="x", alpha=0.25)

        fig.tight_layout()
        return self._save_figure(fig, output_path)

    def generate_visual_report(self, output_dir, focus_model=None, top_k=25):
        """Generate a full diagnostic plot package and return created file paths."""
        plt = self._import_matplotlib()
        if plt is None:
            return []
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except Exception:
            pass

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        created = []

        for task_type in ["classification", "regression"]:
            tt_df = self._clean(task_type)
            if tt_df.empty:
                continue
            metric = self._primary_metric(task_type)
            files = [
                self.plot_metric_leaderboard(
                    task_type, metric, out / f"{task_type}_{metric}_leaderboard.png"
                ),
                self.plot_source_grouped_bars(
                    task_type, metric, out / f"{task_type}_{metric}_by_source_bars.png"
                ),
                self.plot_pairwise_heatmap(
                    task_type, metric, out / f"{task_type}_{metric}_pairwise_heatmap.png"
                ),
            ]
            created.extend([f for f in files if f])

        latency_plot = self.plot_latency_overview(out / "latency_overview.png")
        if latency_plot:
            created.append(latency_plot)

        focus = self._pick_focus_model(focus_model)
        if focus:
            focus_safe = self._safe_name(focus)
            for task_type in ["classification", "regression"]:
                metric = self._primary_metric(task_type)
                comp = self.comparison_vs_best_other(focus, task_type, metric=metric)
                if comp.empty:
                    continue

                comp_csv = out / f"{task_type}_{focus_safe}_vs_best_other.csv"
                comp.sort_values("delta_vs_best_other").to_csv(comp_csv, index=False)
                created.append(str(comp_csv))

                hardest_csv = out / f"{task_type}_{focus_safe}_top_{top_k}_hardest.csv"
                comp.sort_values("delta_vs_best_other").head(top_k).to_csv(hardest_csv, index=False)
                created.append(str(hardest_csv))

                files = [
                    self.plot_focus_loss_topk(
                        comp, focus, task_type, metric, top_k=top_k,
                        output_path=out / f"{task_type}_{focus_safe}_hardest_topk.png",
                    ),
                    self.plot_focus_bucket_analysis(
                        comp, focus, task_type, metric,
                        output_path=out / f"{task_type}_{focus_safe}_bucket_analysis.png",
                    ),
                    self.plot_focus_scatter(
                        comp, focus, task_type, metric,
                        output_path=out / f"{task_type}_{focus_safe}_difficulty_scatter.png",
                    ),
                ]
                created.extend([f for f in files if f])

        print(f"[EvalAnalyzer] Saved {len(created)} diagnostic files to {out}")
        return created

    # --- Reports ---

    @staticmethod
    def _fmt(v, width=8, decimals=4):
        if isinstance(v, float) and np.isfinite(v):
            return f"{v:>{width}.{decimals}f}"
        return f"{'N/A':>{width}}"

    def print_report(self, show_per_dataset=True):
        print("\n" + "=" * 80)
        print("  SYNTHEFY TABULAR EVALUATION REPORT")
        print("=" * 80)

        models = sorted(self.df["model"].unique())
        sources = sorted(self.df["source"].unique()) if "source" in self.df.columns else []
        n_ds = self.df["dataset"].nunique()
        n_err = self.df["error"].notna().sum() if "error" in self.df.columns else 0
        print(f"\n  Models: {len(models)}  |  Datasets: {n_ds}  |  Sources: {', '.join(sources)}  |  Errors: {n_err}")

        # --- Classification aggregate ---
        cls_df = self._clean("classification")
        if not cls_df.empty:
            print(f"\n{'='*80}\n  CLASSIFICATION (aggregate)\n{'='*80}")
            print(f"  {'Model':<30} {'AUC':>8} {'Acc':>8} {'F1':>8} {'ECE':>8} {'LogLoss':>9} {'N':>5}")
            print(f"  {'-'*78}")
            for model in models:
                m = cls_df[cls_df["model"] == model]
                if m.empty:
                    continue
                parts = [f"  {model:<30}"]
                for c, w in [("auc",8),("accuracy",8),("f1",8),("ece",8),("log_loss",9)]:
                    v = m[c].mean() if c in m.columns else float("nan")
                    parts.append(self._fmt(v, w))
                parts.append(f"{len(m):>5}")
                print(" ".join(parts))

            # --- Per-source classification ---
            self._print_per_source(cls_df, models, "classification")

            # --- Per-dataset classification ---
            if show_per_dataset:
                self._print_per_dataset(cls_df, models, "classification")

        # --- Regression aggregate ---
        reg_df = self._clean("regression")
        if not reg_df.empty:
            print(f"\n{'='*80}\n  REGRESSION (aggregate)\n{'='*80}")
            print(f"  {'Model':<30} {'R2':>8} {'RMSE':>8} {'MAE':>8} {'N':>5}")
            print(f"  {'-'*63}")
            for model in models:
                m = reg_df[reg_df["model"] == model]
                if m.empty:
                    continue
                parts = [f"  {model:<30}"]
                for c in ["r2", "rmse", "mae"]:
                    v = m[c].mean() if c in m.columns else float("nan")
                    parts.append(f"{v:>8.4f}" if np.isfinite(v) else f"{'N/A':>8}")
                parts.append(f"{len(m):>5}")
                print(" ".join(parts))

            # --- Per-source regression ---
            self._print_per_source(reg_df, models, "regression")

            # --- Per-dataset regression ---
            if show_per_dataset:
                self._print_per_dataset(reg_df, models, "regression")

        # Latency
        if "latency_ms" in self.df.columns:
            clean = self._clean()
            print(f"\n{'='*80}\n  LATENCY\n{'='*80}")
            print(f"  {'Model':<30} {'Mean ms':>10} {'Median ms':>11} {'Throughput':>14}")
            print(f"  {'-'*67}")
            for model in models:
                m = clean[clean["model"] == model]
                if m.empty:
                    continue
                ml = m["latency_ms"].mean()
                mdl = m["latency_ms"].median()
                thr = m["throughput_sps"].mean() if "throughput_sps" in m.columns else float("nan")
                thr_s = f"{thr:>12.0f} s/s" if np.isfinite(thr) else f"{'N/A':>14}"
                print(f"  {model:<30} {ml:>10.1f} {mdl:>11.1f} {thr_s}")

        print(f"\n{'='*80}\n")

    def _print_per_source(self, df, models, task_type):
        """Print per-source aggregate table (e.g. OpenML-CC18 avg, TabArena avg)."""
        sources = sorted(df["source"].unique())
        if len(sources) < 2:
            return

        if task_type == "classification":
            metric_spec = [("auc", 8), ("accuracy", 8), ("f1", 8), ("ece", 8), ("log_loss", 9)]
        else:
            metric_spec = [("r2", 8), ("rmse", 10), ("mae", 10)]

        print(f"\n  --- By source ({task_type}) ---")
        # Header
        hdr = f"  {'Source':<20}"
        for m in models:
            hdr += f"  {m[:22]:>22}"
        print(hdr)
        print(f"  {'-' * (20 + 24 * len(models))}")

        primary = metric_spec[0][0]
        for src in sources:
            src_df = df[df["source"] == src]
            n_ds = src_df["dataset"].nunique()
            row = f"  {src + f' ({n_ds})':<20}"
            vals = []
            for m in models:
                ms = src_df[src_df["model"] == m]
                if ms.empty or primary not in ms.columns:
                    vals.append(float("nan"))
                else:
                    vals.append(ms[primary].mean())

            best_idx = -1
            finite = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
            if finite:
                best_idx = max(finite, key=lambda x: x[1])[0]

            for i, v in enumerate(vals):
                marker = " *" if i == best_idx else "  "
                if np.isfinite(v):
                    row += f"  {v:>20.4f}{marker}"
                else:
                    row += f"  {'N/A':>20}{marker}"
            print(row)

    def _print_per_dataset(self, df, models, task_type):
        """Print per-dataset results table grouped by source."""
        if task_type == "classification":
            metrics = [("auc", 8), ("accuracy", 8), ("f1", 8), ("log_loss", 9)]
        else:
            metrics = [("r2", 8), ("rmse", 10), ("mae", 10)]

        def _fmt_int(v, width):
            if pd.notna(v) and np.isfinite(v):
                return f"{int(v):>{width}d}"
            return f"{'N/A':>{width}}"

        for src in sorted(df["source"].unique()):
            src_df = df[df["source"] == src]
            datasets = sorted(src_df["dataset"].unique())

            # Compute column widths
            name_w = max(max(len(d) for d in datasets), 20)
            model_short = {m: m[:20] for m in models}
            metric_cols = [c for c, _ in metrics]
            n_models = len(models)
            meta_df = src_df.groupby("dataset")[["n_train", "n_test", "n_features"]].first()

            print(f"\n  --- {src} ({task_type}, {len(datasets)} datasets) ---")

            # Header: Dataset | model1_metric1 | model1_metric2 | ... | model2_metric1 | ...
            # Simplified: one row per dataset, one block of metrics per model
            # For readability, show primary metric per model + best marker
            primary = metric_cols[0]  # auc or r2
            hdr = (
                f"  {'Dataset':<{name_w}}"
                f"  {'Train':>8}  {'Test':>8}  {'Total':>8}  {'Feat':>6}"
            )
            for m in models:
                hdr += f"  {model_short[m]:>20}"
            print(hdr)
            table_w = name_w + 2 + 8 + 2 + 8 + 2 + 8 + 2 + 6 + (22 * n_models)
            print(f"  {'-' * table_w}")

            for ds_name in datasets:
                n_train = n_test = n_features = float("nan")
                if ds_name in meta_df.index:
                    meta = meta_df.loc[ds_name]
                    n_train = meta.get("n_train", float("nan"))
                    n_test = meta.get("n_test", float("nan"))
                    n_features = meta.get("n_features", float("nan"))
                total = (n_train + n_test) if (pd.notna(n_train) and pd.notna(n_test)) else float("nan")

                row = (
                    f"  {ds_name:<{name_w}}"
                    f"  {_fmt_int(n_train, 8)}  {_fmt_int(n_test, 8)}"
                    f"  {_fmt_int(total, 8)}  {_fmt_int(n_features, 6)}"
                )
                vals = []
                for m in models:
                    cell = src_df[(src_df["model"] == m) & (src_df["dataset"] == ds_name)]
                    if cell.empty or primary not in cell.columns:
                        vals.append(float("nan"))
                    else:
                        vals.append(cell[primary].values[0])

                best_idx = -1
                finite_vals = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
                if finite_vals:
                    best_idx = max(finite_vals, key=lambda x: x[1])[0]

                for i, v in enumerate(vals):
                    marker = " *" if i == best_idx else "  "
                    if np.isfinite(v):
                        row += f"  {v:>18.4f}{marker}"
                    else:
                        row += f"  {'ERR':>18}{marker}"
                print(row)

    def _md_per_source(self, df, models, task_type):
        """Generate per-source aggregate markdown table."""
        sources = sorted(df["source"].unique())
        if len(sources) < 2:
            return []

        primary = "auc" if task_type == "classification" else "r2"
        if task_type == "classification":
            metric_cols = ["auc", "accuracy", "f1", "ece", "log_loss"]
        else:
            metric_cols = ["r2", "rmse", "mae"]

        lines = [f"\n### By Source ({task_type.title()})\n"]
        hdr = "| Source | N |" + "|".join(f" {m} " for m in models) + "|"
        sep = "|--------|---:|" + "|".join("---:" for _ in models) + "|"
        lines.append(hdr)
        lines.append(sep)

        for src in sources:
            src_df = df[df["source"] == src]
            n_ds = src_df["dataset"].nunique()
            vals = []
            for m in models:
                ms = src_df[src_df["model"] == m]
                if ms.empty or primary not in ms.columns:
                    vals.append(float("nan"))
                else:
                    vals.append(ms[primary].mean())

            best_idx = -1
            finite = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
            if finite:
                best_idx = max(finite, key=lambda x: x[1])[0]

            row = f"| {src} | {n_ds} |"
            for i, v in enumerate(vals):
                if np.isfinite(v):
                    s = f"{v:.4f}"
                    if i == best_idx:
                        s = f"**{s}**"
                    row += f" {s} |"
                else:
                    row += " N/A |"
            lines.append(row)
        return lines

    def _md_per_dataset(self, df, models, task_type):
        """Generate per-dataset markdown tables grouped by source."""
        lines = []
        if task_type == "classification":
            primary, metrics = "auc", ["auc", "accuracy", "f1", "log_loss"]
        else:
            primary, metrics = "r2", ["r2", "rmse", "mae"]

        for src in sorted(df["source"].unique()):
            src_df = df[df["source"] == src]
            datasets = sorted(src_df["dataset"].unique())
            meta_df = src_df.groupby("dataset")[["n_train", "n_test", "n_features"]].first()

            lines.append(f"\n#### {src} ({len(datasets)} datasets)\n")
            hdr = "| Dataset | Train | Test | Total | Feat |" + "|".join(f" {m} " for m in models) + "|"
            sep = "|---------|------:|-----:|------:|-----:|" + "|".join("---:" for _ in models) + "|"
            lines.append(hdr)
            lines.append(sep)

            for ds_name in datasets:
                n_train = n_test = n_features = float("nan")
                if ds_name in meta_df.index:
                    meta = meta_df.loc[ds_name]
                    n_train = meta.get("n_train", float("nan"))
                    n_test = meta.get("n_test", float("nan"))
                    n_features = meta.get("n_features", float("nan"))
                total = (n_train + n_test) if (pd.notna(n_train) and pd.notna(n_test)) else float("nan")

                vals = []
                for m in models:
                    cell = src_df[(src_df["model"] == m) & (src_df["dataset"] == ds_name)]
                    if cell.empty or primary not in cell.columns:
                        vals.append(float("nan"))
                    else:
                        vals.append(cell[primary].values[0])

                best_idx = -1
                finite = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
                if finite:
                    best_idx = max(finite, key=lambda x: x[1])[0]

                train_s = f"{int(n_train)}" if (pd.notna(n_train) and np.isfinite(n_train)) else "N/A"
                test_s = f"{int(n_test)}" if (pd.notna(n_test) and np.isfinite(n_test)) else "N/A"
                total_s = f"{int(total)}" if (pd.notna(total) and np.isfinite(total)) else "N/A"
                feat_s = f"{int(n_features)}" if (pd.notna(n_features) and np.isfinite(n_features)) else "N/A"
                row = f"| {ds_name} | {train_s} | {test_s} | {total_s} | {feat_s} |"
                for i, v in enumerate(vals):
                    if np.isfinite(v):
                        s = f"{v:.4f}"
                        if i == best_idx:
                            s = f"**{s}**"
                        row += f" {s} |"
                    else:
                        row += " ERR |"
                lines.append(row)
        return lines

    def generate_markdown_report(self, output_path=None):
        lines = ["# Synthefy Tabular Evaluation Report\n"]
        models = sorted(self.df["model"].unique())
        sources = sorted(self.df["source"].unique()) if "source" in self.df.columns else []
        lines.append(f"**Models**: {', '.join(models)}  ")
        lines.append(f"**Datasets**: {self.df['dataset'].nunique()} across {', '.join(sources)}  ")
        lines.append(f"**Total evaluations**: {len(self.df)}\n")

        cls_df = self._clean("classification")
        if not cls_df.empty:
            lines.append("\n## Classification Results (aggregate)\n")
            lines.append("| Model | AUC | Accuracy | F1 | ECE | LogLoss | N |")
            lines.append("|-------|-----|----------|-----|-----|---------|---|")
            for model in models:
                m = cls_df[cls_df["model"] == model]
                if m.empty:
                    continue
                vals = []
                for c in ["auc", "accuracy", "f1", "ece", "log_loss"]:
                    v = m[c].mean() if c in m.columns else float("nan")
                    vals.append(f"{v:.4f}" if np.isfinite(v) else "N/A")
                lines.append(f"| {model} | {' | '.join(vals)} | {len(m)} |")

            # Per-source classification
            lines.extend(self._md_per_source(cls_df, models, "classification"))

            # Per-dataset classification
            lines.append("\n### Per-Dataset Classification (AUC)\n")
            lines.extend(self._md_per_dataset(cls_df, models, "classification"))

        reg_df = self._clean("regression")
        if not reg_df.empty:
            lines.append("\n## Regression Results (aggregate)\n")
            lines.append("| Model | R2 | RMSE | MAE | N |")
            lines.append("|-------|-----|------|-----|---|")
            for model in models:
                m = reg_df[reg_df["model"] == model]
                if m.empty:
                    continue
                vals = []
                for c in ["r2", "rmse", "mae"]:
                    v = m[c].mean() if c in m.columns else float("nan")
                    vals.append(f"{v:.4f}" if np.isfinite(v) else "N/A")
                lines.append(f"| {model} | {' | '.join(vals)} | {len(m)} |")

            # Per-source regression
            lines.extend(self._md_per_source(reg_df, models, "regression"))

            # Per-dataset regression
            lines.append("\n### Per-Dataset Regression (R2)\n")
            lines.extend(self._md_per_dataset(reg_df, models, "regression"))

        if "latency_ms" in self.df.columns:
            clean = self._clean()
            lines.append("\n## Latency\n")
            lines.append("| Model | Mean ms | Median ms | Throughput (s/s) |")
            lines.append("|-------|---------|-----------|-----------------|")
            for model in models:
                m = clean[clean["model"] == model]
                if m.empty:
                    continue
                ml = m["latency_ms"].mean()
                mdl = m["latency_ms"].median()
                thr = m["throughput_sps"].mean() if "throughput_sps" in m.columns else float("nan")
                thr_s = f"{thr:.0f}" if np.isfinite(thr) else "N/A"
                lines.append(f"| {model} | {ml:.1f} | {mdl:.1f} | {thr_s} |")

        if len(models) >= 2:
            for metric, tt in [("auc", "classification"), ("r2", "regression")]:
                tt_df = self._clean(tt)
                if tt_df.empty or metric not in tt_df.columns:
                    continue
                lines.append(f"\n## {tt.title()} Win Rates ({metric.upper()})\n")
                header = "| |" + "|".join(f" {m} " for m in models) + "|"
                sep = "|---|" + "|".join("---" for _ in models) + "|"
                lines.append(header)
                lines.append(sep)
                for m_a in models:
                    row = f"| {m_a} |"
                    for m_b in models:
                        if m_a == m_b:
                            row += " - |"
                        else:
                            h = self.head_to_head(m_a, m_b, metric=metric, task_type=tt)
                            total = h["wins_a"] + h["wins_b"] + h["ties"]
                            wr = h["wins_a"] / total if total > 0 else 0.5
                            row += f" {wr:.0%} |"
                    lines.append(row)

        report = "\n".join(lines)
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Report saved to {output_path}")
        return report
