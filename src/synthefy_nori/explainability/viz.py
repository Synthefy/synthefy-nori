"""Render a fitted EBM as a MODEL diagram: additive architecture + shape functions + data density.

An EBM (GA²M) is fully specified by ``ŷ = intercept + Σ_j f_j(x_j) + Σ_jk f_jk(x_j,x_k)`` where every
``f`` is a binned lookup table — the per-bin scores ARE the weights. This draws one dense,
self-documenting picture (readable by a human OR a VLM): inputs → main-effect shape functions
(with a grey data-density histogram so extrapolation regions are visible) → Σ (+intercept) → output,
plus the top pairwise-interaction heatmaps and an explicit how-to-read key, laid out so NO text overlaps.

    from synthefy_nori.explainability.viz import plot_ebm_model
    plot_ebm_model(ebm, feature_names, X_density=Xtrain, task="regression",
                   target_name="toxicity", skill=0.70, skill_name="R²", out_path="ebm.png")
"""

import numpy as np

from synthefy_nori.explainability._common import clip_inf_edges, shape_direction

MAX_INTERACTIONS_SHOWN = 4

# display strings for the shared shape_direction() categories
_DIR_LABEL = {
    "flat": "—",
    "negligible": "≈ flat",
    "increasing": "↑ increases target",
    "decreasing": "↓ decreases target",
    "non-monotone": "∿ mixed",
}


def _direction_label(ys):
    return _DIR_LABEL[shape_direction(ys)]


def plot_ebm_model(
    model,
    feature_names,
    *,
    X_density=None,
    task="regression",
    target_name="target",
    skill=None,
    skill_name=None,
    title=None,
    out_path=None,
    show_key=True,
    feature_ranges=None,
    class_names=None,
):
    """Draw the model diagram. Returns the matplotlib Figure (also saved to ``out_path`` if given).
    Set ``show_key=False`` to omit the bottom "how to read" explanation box.
    ``feature_ranges``: optional ``{feature_name: (lo, hi)}`` to clip each main-effect shape panel's
    x-axis (and its y-range / density / annotation) to a natural range, e.g. the 10th–90th percentile.

    Multiclass is supported: each feature then has one shape function PER CLASS, so every panel
    holds K coloured curves sharing one legend, and Σ feeds a softmax instead of a sigmoid. Pass
    ``class_names`` to label them (defaults to ``class 0..K-1``). Panels are coloured by CLASS in
    that mode — the feature identity comes from the row label and its arrow — because comparing
    classes within a panel is the whole point of the picture."""
    # deferred on purpose: keeps `import synthefy_nori.explainability` free of matplotlib
    try:  # optional dep: the explainability extra
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.patches import FancyArrowPatch
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to draw the model diagram; install it with "
            'pip install "synthefy-nori[explainability]"'
        ) from exc

    fn = list(feature_names)
    d = len(fn)
    if X_density is not None:
        X_density = np.asarray(X_density)
        if X_density.ndim != 2 or X_density.shape[1] != d:
            raise ValueError(
                f"X_density must be column-aligned with feature_names: expected {d} columns, "
                f"got shape {X_density.shape}. When the EBM was distilled on a pruned subset, "
                f"pass the SAME subset (e.g. X[:, selected_indices]), not the full table."
            )
    exp = model.explain_global()
    imp = np.asarray(model.term_importances(), dtype=float)
    intercept = float(np.ravel(model.intercept_)[0])
    order = list(np.argsort(-imp))
    rank_of = {t: i + 1 for i, t in enumerate(order)}
    # multiclass main effects carry one score column per class
    _main0 = next((exp.data(t) for t, f in enumerate(model.term_features_) if len(f) == 1), None)
    n_classes = (
        int(np.asarray(_main0["scores"], float).shape[1])
        if (_main0 is not None and np.asarray(_main0["scores"], float).ndim > 1)
        else 1
    )
    multiclass = n_classes > 1
    if multiclass and class_names is None:
        class_names = [f"class {k}" for k in range(n_classes)]
    class_names = [str(c) for c in (class_names or [])]
    if skill_name is None:
        skill_name = "macro OVR AUC" if multiclass else ("AUC" if task == "classification" else "R²")
    # audience-friendly wording: probability for classification, target units for regression
    if multiclass:
        base_desc = f"each class's score for an average case (≈ that class's overall rate)"
        y_desc = (
            f"how much that feature moves the score for EACH class — one curve per class, "
            f"so a feature can push one class up while pushing another down"
        )
        heat_desc = "n/a for multiclass"
        cbar_label = ""
    elif task == "classification":
        base_desc = f"the model's output for an average case (≈ the overall {target_name} rate)"
        y_desc = f"how much that feature moves the predicted probability of {target_name} up (+) or down (−)"
        heat_desc = f"red = higher {target_name} probability, blue = lower"
        cbar_label = f"{target_name} probability\n(red higher, blue lower)"
    else:
        base_desc = f"the average {target_name} (the output for an all-average input)"
        y_desc = f"the amount added to / subtracted from the baseline {target_name} (0 = no effect)"
        heat_desc = "red raises the prediction, blue lowers it"
        cbar_label = f"extra Δ{target_name}\n(red +, blue −)"

    terms = [(t, feats, exp.data(t)) for t, feats in enumerate(model.term_features_)]
    main_terms = [x for x in terms if len(x[1]) == 1]
    all_int = sorted([x for x in terms if len(x[1]) == 2], key=lambda x: -imp[x[0]])
    int_terms = all_int[:MAX_INTERACTIONS_SHOWN]  # only the strongest few are drawable
    n_int_hidden = len(all_int) - len(int_terms)
    n_terms = len(terms)

    # vertical layout in inches so per-row spacing is constant for ANY number of features
    nrows = len(main_terms)
    FIG_W = 20.0
    PANEL_H_IN, ROW_GAP_IN = 1.35, 1.25  # panel height + gap (room for 2-line title + x-ticks)
    TOP_MARGIN_IN, BOT_MARGIN_IN = 2.3, 0.7
    slot_in = PANEL_H_IN + ROW_GAP_IN
    FIG_H = TOP_MARGIN_IN + nrows * slot_in + BOT_MARGIN_IN
    dyi = lambda inch: inch / FIG_H  # inches -> figure-fraction (vertical)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    palette = plt.cm.tab10(np.linspace(0, 1, 10))
    fcolor = {f: palette[i % 10] for i, f in enumerate(fn)}
    ccolor = [plt.cm.tab10(np.linspace(0, 1, 10))[k % 10] for k in range(n_classes)]
    max_imp = imp.max() or 1.0
    gmax = max([np.abs(np.array(x[2]["scores"], float)).max() for x in int_terms], default=1.0) or 1.0
    inorm = Normalize(-gmax, gmax)

    ph = PANEL_H_IN / FIG_H
    yr = [1 - (TOP_MARGIN_IN + (i + 0.5) * slot_in) / FIG_H for i in range(nrows)]
    in_x, in_w = 0.05, 0.04
    mp_left, mp_w = 0.185, 0.235
    sum_x, sum_y, out_x = 0.83, 0.5, 0.915
    HDR_FS = 13.5
    HDR_Y = 1 - dyi(TOP_MARGIN_IN * 0.52)
    sup_y = 1 - dyi(TOP_MARGIN_IN * 0.16)

    fig.text(in_x, HDR_Y, "inputs  xⱼ", ha="center", fontsize=HDR_FS, weight="bold")
    in_pos = {}
    for row, (t, feats, _) in zip(yr, main_terms):
        j = feats[0]
        f = fn[j]
        fig.text(
            in_x,
            row,
            f,
            ha="center",
            va="center",
            fontsize=13.5,
            bbox=dict(boxstyle="round,pad=0.4", fc=fcolor[f], ec="black", alpha=0.85),
        )
        in_pos[t] = (in_x + in_w * 0.55, row)

    fig.text(
        mp_left + mp_w / 2, HDR_Y, "main-effect shape functions  fⱼ(xⱼ)", ha="center", fontsize=HDR_FS, weight="bold"
    )
    for row, (t, feats, dta) in zip(yr, main_terms):
        j = feats[0]
        ax = fig.add_axes([mp_left, row - ph / 2, mp_w, ph])
        xs = clip_inf_edges(dta["names"])
        ys = np.array(dta["scores"], float)
        frng = feature_ranges.get(fn[j]) if feature_ranges else None
        if frng is not None and float(frng[1]) <= float(frng[0]):
            frng = None  # degenerate (e.g. p10 == p90 on a
            # zero-inflated column): show it all
        x0, x1 = (float(frng[0]), float(frng[1])) if frng else (float(xs[0]), float(xs[-1]))
        step = len(ys) == len(xs) - 1
        # one curve per class when multiclass, else the single shape function
        curves = (
            [(ys[:, k], ccolor[k], class_names[k]) for k in range(n_classes)]
            if multiclass
            else [(ys, fcolor[fn[j]], None)]
        )
        keep = (
            (
                [i for i in range(len(ys)) if xs[i + 1] > x0 and xs[i] < x1]
                if step
                else (xs[: len(ys)] >= x0) & (xs[: len(ys)] <= x1)
            )
            if frng
            else None
        )
        visible = []
        for cy, colr, clabel in curves:
            if step:
                if not multiclass:
                    ax.stairs(cy, xs, fill=True, color=colr, alpha=0.22)
                ax.stairs(cy, xs, color=colr, lw=1.7, label=clabel)
            else:
                ax.plot(xs, cy, color=colr, lw=1.7, label=clabel)
            visible.append(cy[keep] if keep is not None else cy)
        vis = np.concatenate([v for v in visible if len(v)]) if any(len(v) for v in visible) else ys
        ys_v = np.asarray(vis).ravel()  # values visible in [x0, x1]
        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax.set_xlim(x0, x1)
        ylo, yhi = min(0.0, float(ys_v.min())), max(0.0, float(ys_v.max()))
        ypad = (yhi - ylo) * 0.08 or 0.05
        ax.set_ylim(ylo - ypad, yhi + ypad)
        if X_density is not None:
            col = X_density[:, j]
            col = col[np.isfinite(col)]
            if len(col):
                cnt, edg = np.histogram(col, bins=24, range=(x0, x1))
                axd = ax.twinx()
                axd.bar((edg[:-1] + edg[1:]) / 2, cnt, width=(edg[1] - edg[0]), color="0.55", alpha=0.30, zorder=0)
                axd.set_ylim(0, (cnt.max() or 1) * 3.4)
                axd.axis("off")
        ax.set_zorder(2)
        ax.patch.set_alpha(0)
        ax.tick_params(labelsize=11, length=3, pad=2)
        ax.set_xticks([x0, x1])
        ax.set_xticklabels([f"{x0:.2g}", f"{x1:.2g}"])
        ymin, ymax = float(ys_v.min()), float(ys_v.max())
        yrng = (ymax - ymin) or 1.0
        yt, ytl = [ymin, ymax], [f"{ymin:+.2f}", f"{ymax:+.2f}"]
        if abs(ymin) > 0.12 * yrng and abs(ymax) > 0.12 * yrng:
            yt, ytl = [ymin, 0, ymax], [f"{ymin:+.2f}", "0", f"{ymax:+.2f}"]
        ax.set_yticks(yt)
        ax.set_yticklabels(ytl)
        for sp in ax.spines.values():
            sp.set_edgecolor("0.35")
            sp.set_linewidth(0.8 + 3.2 * imp[t] / max_imp)
        tag = "  ·  NEGLIGIBLE" if imp[t] < 0.05 * max_imp else ""
        trend = f"{n_classes} curves, one per class" if multiclass else _direction_label(ys_v)
        ax.set_title(
            f"{fn[j]}   ·   imp {imp[t]:.2f} (#{rank_of[t]}/{n_terms}){tag}\n"
            f"{trend}   ·   contribution ∈ [{ymin:+.2f}, {ymax:+.2f}]",
            fontsize=12.5,
            pad=6,
            color=("0.45" if imp[t] < 0.05 * max_imp else "black"),
        )
        fig.patches.append(
            FancyArrowPatch(
                in_pos[t],
                (mp_left, row),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=9,
                lw=0.8,
                color=fcolor[fn[j]],
                alpha=0.6,
                zorder=0,
            )
        )
        fig.patches.append(
            FancyArrowPatch(
                (mp_left + mp_w, row),
                (sum_x - 0.03, sum_y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=8,
                lw=0.6,
                color="0.6",
                alpha=0.5,
                zorder=0,
                connectionstyle="arc3,rad=0.03",
            )
        )

    ix_left, ix_w, ix_h = 0.500, 0.175, dyi(2.0)
    ix_top, ix_bot = yr[0], yr[-1]  # span the same vertical extent as the shape panels
    iyr = (
        [ix_top - (ix_top - ix_bot) * (i / max(len(int_terms) - 1, 1)) for i in range(len(int_terms))]
        if len(int_terms) > 1
        else [(ix_top + ix_bot) / 2]
    )
    if int_terms:
        shown = f"top {len(int_terms)} of {len(all_int)}" if n_int_hidden else f"all {len(all_int)}"
        fig.text(
            ix_left + ix_w / 2, HDR_Y, "pairwise interactions  fⱼₖ(xⱼ,xₖ)", ha="center", fontsize=HDR_FS, weight="bold"
        )
        fig.text(
            ix_left + ix_w / 2,
            HDR_Y - dyi(0.30),
            f"({shown} — Σ below uses every one)",
            ha="center",
            fontsize=10.0,
            color="0.35",
        )
    for row, (t, feats, dta) in zip(iyr, int_terms):
        ax = fig.add_axes([ix_left, row - ix_h / 2, ix_w, ix_h])
        sc = np.array(dta["scores"], float)
        ax.imshow(sc.T, aspect="auto", origin="lower", cmap="RdBu_r", norm=inorm)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(fn[feats[0]], fontsize=11.5, labelpad=2)
        ax.set_ylabel(fn[feats[1]], fontsize=11.5, labelpad=2)
        for sp in ax.spines.values():
            sp.set_edgecolor("0.35")
            sp.set_linewidth(0.8 + 3.2 * imp[t] / max_imp)
        ax.set_title(f"{fn[feats[0]]} × {fn[feats[1]]}\nimp {imp[t]:.2f}", fontsize=10.5, pad=5)
        fig.patches.append(
            FancyArrowPatch(
                (ix_left + ix_w, row),
                (sum_x - 0.03, sum_y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=9,
                lw=0.6,
                color="0.6",
                alpha=0.5,
                zorder=0,
                connectionstyle="arc3,rad=0.03",
            )
        )
    if int_terms:
        cax = fig.add_axes([ix_left + ix_w + 0.014, ix_bot - ix_h / 2, 0.011, (ix_top - ix_bot) + ix_h])
        cb = fig.colorbar(ScalarMappable(norm=inorm, cmap="RdBu_r"), cax=cax)
        cb.ax.tick_params(labelsize=11)
        cb.set_label(cbar_label, fontsize=12)

    fig.text(
        sum_x,
        sum_y,
        "Σ",
        ha="center",
        va="center",
        fontsize=40,
        bbox=dict(boxstyle="circle,pad=0.4", fc="#ffe08a", ec="black"),
    )
    fig.text(
        sum_x,
        sum_y - dyi(2.3),
        f"intercept (baseline)\n{intercept:+.3f}",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.5"),
    )
    fig.patches.append(
        FancyArrowPatch(
            (sum_x, sum_y - dyi(1.7)),
            (sum_x, sum_y - dyi(0.6)),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=10,
            lw=0.9,
            color="0.4",
        )
    )
    if multiclass:
        # one Σ per class, then a softmax over the K sums
        out_eff = out_x + 0.02
        link_x = (sum_x + out_eff) / 2
        fig.patches.append(
            FancyArrowPatch(
                (sum_x + 0.03, sum_y),
                (link_x - 0.019, sum_y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=14,
                lw=1.6,
                color="black",
            )
        )
        fig.text(
            link_x,
            sum_y,
            "softmax",
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.34", fc="#ffd9b3", ec="black"),
        )
        fig.patches.append(
            FancyArrowPatch(
                (link_x + 0.026, sum_y),
                (out_eff - 0.032, sum_y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=14,
                lw=1.6,
                color="black",
            )
        )
        fig.text((sum_x + link_x) / 2, sum_y + dyi(0.45), f"{n_classes} scores", ha="center", fontsize=11, color="0.45")
        fig.text((link_x + out_eff) / 2, sum_y + dyi(0.45), "probabilities", ha="center", fontsize=11, color="0.45")
        out_node_x = out_eff
    elif task == "classification":
        # Σ produces log-odds; a sigmoid maps it to the predicted probability
        out_eff = out_x + 0.02
        link_x = (sum_x + out_eff) / 2
        fig.patches.append(
            FancyArrowPatch(
                (sum_x + 0.03, sum_y),
                (link_x - 0.016, sum_y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=14,
                lw=1.6,
                color="black",
            )
        )
        fig.text(
            link_x,
            sum_y,
            "σ",
            ha="center",
            va="center",
            fontsize=21,
            bbox=dict(boxstyle="circle,pad=0.34", fc="#ffd9b3", ec="black"),
        )
        fig.text(link_x, sum_y - dyi(0.95), "sigmoid", ha="center", va="center", fontsize=11, color="0.3")
        fig.patches.append(
            FancyArrowPatch(
                (link_x + 0.017, sum_y),
                (out_eff - 0.032, sum_y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=14,
                lw=1.6,
                color="black",
            )
        )
        fig.text((sum_x + link_x) / 2, sum_y + dyi(0.45), "log-odds", ha="center", fontsize=11, color="0.45")
        fig.text((link_x + out_eff) / 2, sum_y + dyi(0.45), "probability", ha="center", fontsize=11, color="0.45")
        out_node_x = out_eff
    else:
        fig.patches.append(
            FancyArrowPatch(
                (sum_x + 0.034, sum_y),
                (out_x - 0.032, sum_y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=16,
                lw=1.6,
                color="black",
            )
        )
        out_node_x = out_x
    out_label = f"ŷ\nP({target_name})\nper class" if multiclass else f"ŷ\n{target_name}"
    fig.text(
        out_node_x,
        sum_y,
        out_label,
        ha="center",
        va="center",
        fontsize=13 if multiclass else 16,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="#c7e9c0", ec="black"),
    )
    if multiclass:  # one shared legend for the per-class curves
        handles = [plt.Line2D([], [], color=ccolor[k], lw=2.4, label=class_names[k]) for k in range(n_classes)]
        fig.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.99, HDR_Y + dyi(0.5)),
            frameon=True,
            fontsize=11,
            title="class",
            ncol=1,
        )

    int_note = (
        "this model has no interaction terms (interpret's EBM does not support them for more than two classes)"
        if multiclass
        else f"the {MAX_INTERACTIONS_SHOWN} strongest of {len(all_int)} are drawn, but ALL "
        f"{len(all_int)} are in the model and in Σ"
        if n_int_hidden
        else f"all {len(all_int)} of the model's interactions are drawn"
    )
    if show_key:
        skill_txt = f"Model test {skill_name} = {skill:.3f} on held-out data.   " if skill is not None else ""
        key = (
            f"HOW TO READ  —  glass-box additive model.   Prediction = baseline + every feature's contribution.\n"
            f"• Baseline = {base_desc}.   "
            f"• Each middle graph is ONE feature's contribution: x-axis = the feature's value, "
            f"y-axis = {y_desc}.\n"
            f"• Grey bars = how many samples have each value → the curve is trustworthy where bars are tall, "
            f"EXTRAPOLATED where bars are short.   • Thicker panel border = more important feature "
            f"(rank #1 = strongest).\n"
            + (
                f"• Interactions: {int_note}.   "
                if multiclass
                else f"• Right heatmaps = pairwise interactions (effect of TWO features together beyond "
                f"their curves; {heat_desc}); {int_note}.   "
            )
            + (
                f"• One Σ per class, then a softmax over the {n_classes} sums gives the predicted probabilities.   "
                if multiclass
                else "• Σ combines the baseline and all contributions into the final prediction.   "
            )
            + f"{skill_txt}"
        )
        fig.text(
            0.5,
            0.088,
            key,
            ha="center",
            va="top",
            fontsize=9.0,
            bbox=dict(boxstyle="round,pad=0.7", fc="#f4f4f4", ec="0.6"),
        )

    tail = f", test {skill_name}={skill:.3f})" if skill is not None else ")"
    ttl = title or (f"EBM model   (d={d} features, {n_terms} terms" + tail)
    fig.suptitle(ttl, fontsize=22, weight="bold", y=sup_y)
    if out_path:
        fig.savefig(out_path, dpi=170, bbox_inches="tight", pad_inches=0.35)
    return fig
