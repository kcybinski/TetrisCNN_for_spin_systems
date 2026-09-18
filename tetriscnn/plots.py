import re
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tetriscnn.utils import *
import matplotlib.gridspec as gridspec

sns.set_theme()
sns.color_palette("Paired")
from tetriscnn.pattern_generation import *

import pickle


def save_fig(fig, folder_path, file_name):
    """
    Saves fig to path as a png file.
    """
    folder = Path(folder_path)
    save_path = folder.joinpath(file_name)
    fig.savefig(save_path)


def single_seed_plots(cf, metrics):
    if cf.save_histories:
        plot_history(metrics, cf, "history.png")
        if cf.task == "regression":
            try:
                plot_phase_transition(metrics, cf, "phase_transition2.png")
            except:
                pass

    if cf.save_final_values:
        fit_metrics = load_json(cf.logdir, "fit_metrics.json")

        if getattr(cf, "fit_branches", False) and cf.model == "tetriscnn":
            # metrics["learning_rate"] exists but stays [] whenever no scheduler ran
            # (it is only appended inside `if scheduler is not None`), so `or` -- not
            # dict.get's missing-key default -- is what falls back correctly here.
            lr_history = metrics.get("learning_rate") or [cf.learning_rate]
            plot_branch_fits(cf, fit_metrics["all_final_x"], fit_metrics["all_final_z"],
                              "branch_fits.png", final_lr=lr_history[-1])


def get_active_branches(z, final_lr, floor_margin=10.0):
    """
    Classify branches as active vs. settled on the learning-rate-set residual
    floor. Per appendices/LR_Scheduling.tex ("Origin of the activation floor"), a
    deactivated branch does not converge to exactly zero but to z_floor =
    O(alpha_t), independent of the branch's own penalty lambda_k to leading
    order; the same appendix's "Active versus deactivated branches" identifies a
    branch as active when its converged activation is well above that floor. This
    operationalizes "well above" as `floor_margin` times the final learning rate
    reached during training.

    Args:
        z: array (N, K) bottleneck activations, one column per branch.
        final_lr: float, the last learning rate reached during training
            (metrics["learning_rate"][-1] if a scheduler ran, else the fixed
            cf.learning_rate the run trained at throughout).
        floor_margin: float, how many times the floor mean|z_k| must exceed to
            count as active.

    Returns:
        active_mask: bool array (K,).
        mean_abs_z: float array (K,), the underlying ranking statistic.
    """
    mean_abs_z = np.abs(np.asarray(z)).mean(axis=0)
    floor = floor_margin * float(final_lr)
    active_mask = mean_abs_z > floor
    return active_mask, mean_abs_z


_LATEX_AVAILABLE = shutil.which("latex") is not None
_PAPER_MPLSTYLE = Path(__file__).resolve().parent / "paper.mplstyle"


def _mathtext_safe_pattern_label(latex_label):
    """Fallback rendering for a fit_activation_to_correlators pattern label, used
    only when no system LaTeX install is available (see _LATEX_AVAILABLE).

    tetriscnn.interpret always builds these with mask_to_latex_pattern(full_latex=True)
    (\\blacksquare / \\square / \\substack{...}), which requires text.usetex=True (a
    real LaTeX install) to render directly; matplotlib's built-in mathtext renderer
    does not know \\square or \\substack at all, so drawing these labels un-converted
    raises at figure-save time when LaTeX is unavailable. This swaps in the
    glyphs/formatting mathtext can actually draw; \\substack's stacked rows collapse
    to one bar-separated row, which is a legibility trade, not a numeric one.
    """
    label = latex_label.replace(r"\blacksquare", "\u25A0").replace(r"\square", "\u25A1")
    match = re.search(r"\\substack\{(.*)\}", label)
    if match:
        rows = [r.strip() for r in match.group(1).split(r"\\")]
        label = label[:match.start()] + "{" + "|".join(rows) + "}" + label[match.end():]
    return label


def _strip_math(label):
    """Drop the outer $...$ of a label so it can be nested inside a larger formula."""
    label = label.strip()
    return label[1:-1] if label.startswith("$") and label.endswith("$") else label


def _branch_fit_equation(cf, k, coefficients, patterns, r_squared, group,
                          latex=True, terms_per_line=3, max_terms=6, precision=3):
    """The fitted equation of branch k, z[P] = c_0 + sum c_j C[P_j], as a plot title.

    `coefficients` is intercept first, then one entry per `patterns` label. At most
    `max_terms` correlators are written (the largest |c|, kept in term order),
    with an ellipsis for the rest (a full two-basis fit can have hundreds). Long
    equations are broken into lines of `terms_per_line` terms, each line its own math
    span, which renders the same under usetex and under matplotlib's mathtext. Without
    LaTeX the pattern glyphs go through _mathtext_safe_pattern_label. An equivariant
    branch is linear in rotation-averaged correlators, written C_rot as in
    Figure5.ipynb.
    """
    symbol = r"C_{\mathrm{rot}}" if group is not None else "C"
    if latex:
        lhs = _strip_math(branch_label(cf, k, full_latex=True))
        glyphs = [_strip_math(p) for p in patterns]
    else:
        lhs = _strip_math(branch_label(cf, k))
        glyphs = [_strip_math(_mathtext_safe_pattern_label(p)) for p in patterns]

    coefficients = np.asarray(coefficients)
    shown = range(len(glyphs))
    if len(glyphs) > max_terms:
        shown = sorted(np.argsort(np.abs(coefficients[1:]))[::-1][:max_terms])
    chunks = [rf"{coefficients[0]:.{precision}f}"]
    for j in shown:
        c = coefficients[1 + j]
        chunks.append(rf"{'-' if c < 0 else '+'}\,{abs(c):.{precision}f}\,{symbol}[{glyphs[j]}]")
    if len(glyphs) > max_terms:
        chunks.append(r"+\,\ldots")
    chunks[-1] += rf"\quad (R^2={r_squared:.3f})"

    lines = [" ".join(chunks[i:i + terms_per_line]) for i in range(0, len(chunks), terms_per_line)]
    lines[0] = rf"z[{lhs}] \approx " + lines[0]
    return "\n".join(f"${line}$" for line in lines)


def _paris_channel_names(cf):
    """Display names for the measurement-basis channels of a Paris dataset, or
    None (fit_activation_to_correlators then falls back to ch0/ch1/...)."""
    return {
        "Paris_XY_XZ": ["X", "Z"],
        "Paris_XY_X": ["X"],
        "Paris_XY_Z": ["Z"],
        "Paris_Ising": ["Z"],
    }.get(getattr(cf, "dataset", None))


def plot_branch_fits(cf, snapshots, z, file_name, final_lr=None, max_bars=5,
                      max_terms=4, plateau_tol=0.01, plateau_floor=0.95,
                      floor_margin=1e3, branches=None, channel_names=None,
                      show_dont_save=False):
    """
    BFA-fit and plot every active branch of a trained run (see get_active_branches
    for "active"). For each, regress its activation onto the correlators of its
    own filter footprint (tetriscnn.interpret.fit_activation_to_correlators) and
    draw one panel: the fitted coefficient for every candidate correlator, with
    the best/forward-selected terms outlined and the rest faded. Generalizes
    Figure5.ipynb's manuscript figure (which hand-picks the XY branches) to
    whichever branches THIS run's own activations mark as active.

    Args:
        cf: run config (.kernels, .model, .dataset, .logdir; optionally
            .equivariant / .equivariant_group).
        snapshots: array (N, C, H, W) -- fit_metrics["all_final_x"].
        z: array (N, K) -- fit_metrics["all_final_z"].
        file_name: output filename, saved under cf.logdir.
        final_lr: the run's final learning rate; defaults to cf.learning_rate.
        max_bars: bars per panel. The selected terms always come first, then the
            rest by |coefficient|. None draws the complete coefficient spectrum of
            the full fit in term order (the appendix version of Figure5's
            figure); the title then quotes the full-fit R² instead of the pruned one.
        branches: branch indices to plot, bypassing the activity test.
        channel_names: per-channel names for the term labels; defaults to the
            Paris dataset's bases, else ch0/ch1/...
        show_dont_save: display the figure instead of writing it to cf.logdir.

    Returns:
        dict {branch_index: fit_activation_to_correlators(...) result}, or None
        if no branch cleared the activity floor.
    """
    from tetriscnn.interpret import fit_activation_to_correlators

    if cf.model != "tetriscnn":
        print(f"[plot_branch_fits] cf.model={cf.model!r} has no bottleneck branches; skipping {file_name}.")
        return None

    snapshots = np.asarray(snapshots)
    z = np.asarray(z)
    if final_lr is None:
        final_lr = cf.learning_rate

    active_mask, mean_abs_z = get_active_branches(z, final_lr, floor_margin=floor_margin)
    if branches is not None:
        active_branches = list(branches)
    else:
        active_branches = [k for k in range(len(cf.kernels)) if active_mask[k]]
    if not active_branches:
        print(f"[plot_branch_fits] no branch cleared the activity floor "
              f"({floor_margin}x final_lr={final_lr:.2e}); skipping {file_name}.")
        return None

    group = cf.equivariant_group if getattr(cf, "equivariant", False) else None
    if channel_names is None:
        channel_names = _paris_channel_names(cf)

    fits = {}
    for k in active_branches:
        n_avail = len(fit_activation_to_correlators(
            snapshots, z[:, k], cf.kernels[k], channel_names=channel_names,
            group=group, verbose=False)["patterns"])
        fits[k] = fit_activation_to_correlators(
            snapshots, z[:, k], cf.kernels[k], channel_names=channel_names,
            group=group, select_terms=min(max_terms, n_avail),
            select_strategy="forward", plateau_tol=plateau_tol,
            plateau_floor=plateau_floor, verbose=False,
        )

    # paper.mplstyle (Figure5.ipynb's manuscript-figure style) already pulls in
    # amsmath/amssymb via text.latex.preamble, which \substack and \blacksquare/\square
    # (mask_to_latex_pattern's full_latex glyphs) need -- matplotlib's default LaTeX
    # preamble does not include them and real usetex rendering fails without it. The
    # style hardcodes text.usetex: True, so it's overridden here when LaTeX is absent.
    style = [_PAPER_MPLSTYLE, {"text.usetex": _LATEX_AVAILABLE}] if _PAPER_MPLSTYLE.exists() \
        else [{"text.usetex": _LATEX_AVAILABLE}]
    # Each panel is titled with its fitted equation: the full fit when the whole
    # spectrum is drawn (max_bars=None), the pruned refit otherwise.
    equations = {}
    for k in active_branches:
        res = fits[k]
        if max_bars is None:
            terms = (res["coefficients"], res["patterns"], res["r_squared"])
        else:
            terms = (res["selected_coefficients"], res["selected_patterns"], res["selected_r_squared"])
        equations[k] = _branch_fit_equation(cf, k, *terms, group=group, latex=_LATEX_AVAILABLE)
    heights = [2.4 + 0.35 * equations[k].count("\n") for k in active_branches]

    with plt.style.context(style):
        fig, axes = plt.subplots(nrows=len(active_branches), ncols=1, squeeze=False,
                                  figsize=(7, sum(heights)),
                                  gridspec_kw={"height_ratios": heights})
        axes = axes[:, 0]

        for ax, k in zip(axes, active_branches):
            res = fits[k]
            coefs = np.asarray(res["coefficients"])
            if _LATEX_AVAILABLE:
                # res["patterns"] (tetriscnn.interpret.fit_activation_to_correlators)
                # already wraps each entry in $...$ -- do not add another pair.
                labels = [r"$\emptyset$"] + list(res["patterns"])
            else:
                labels = [r"$\emptyset$"] + [_mathtext_safe_pattern_label(p) for p in res["patterns"]]
            selected = set(res["selected_terms"] or [])

            if max_bars is not None and len(coefs) > max_bars:
                rest = [i for i in np.argsort(np.abs(coefs))[::-1] if i not in selected]
                head = sorted(sorted(selected) + rest[:max(0, max_bars - len(selected))])
            else:
                head = list(range(len(coefs)))

            idx = np.arange(len(head))
            # Full spectrum: every term drawn alike. Pruned view: the forward-selected
            # terms outlined at full opacity, the discarded ones faded.
            base = mcolors.to_rgba(branch_color(cf, k) or "C0")
            if max_bars is None:
                bar_colors = [base] * len(head)
                edges = ["none"] * len(head)
            else:
                bar_colors = [base if i in selected else (*base[:3], 0.3) for i in head]
                edges = ["0.15" if i in selected else "none" for i in head]
            ax.bar(idx, coefs[head], color=bar_colors, edgecolor=edges, linewidth=0.8)
            ax.axhline(0, color="0.3", linestyle="--")
            ax.set_xticks(idx)
            if _LATEX_AVAILABLE:
                ax.set_xticklabels([labels[i] for i in head])
            else:
                ax.set_xticklabels([labels[i] for i in head], rotation=45, ha="right")
            ax.set_ylabel(r"$c_P$")
            ax.set_title(equations[k], loc="left", fontsize=10, usetex=_LATEX_AVAILABLE)
            # Plain-text bookkeeping inside the axes, clear of a long equation.
            # usetex=False: nothing here needs LaTeX, and "|" would not survive it.
            n_kept = len(res["patterns"]) if max_bars is None else len(selected)
            ax.text(0.99, 0.96,
                    f"branch {k}, mean|z| = {mean_abs_z[k]:.3f}, {n_kept}/{len(res['patterns'])} terms",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7, usetex=False,
                    bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.85, pad=2))
            ax.grid(True, axis="y")

        plt.tight_layout()
        if show_dont_save:
            plt.show()
        else:
            save_fig(fig, cf.logdir, file_name)
        plt.close()
    return fits


def plot_parameter_sweep(swept_param_name, swept_param_values, metrics_per_value, cf, filename, scenario_labels=None):
    """
    Visualize single-parameter sweep results.

    Can create either:
    - Two-panel layout: Train/Val metrics side-by-side (default)
    - Multi-panel layout: Vertical subplots with bottleneck activations (like lambdamax)

    Args:
        swept_param_name: Name of the swept parameter
        swept_param_values: List of parameter values
        metrics_per_value: Dict mapping param values to metrics_per_seed
        cf: Config object
        filename: Output filename
        scenario_labels: Optional dict mapping swept_param_values to display labels (e.g., {"es=True-...": "(a)"})
    """
    import numpy as np

    # Prompt user for plot options
    if cf.experiment_name not in ["weighted_loss"]:
        print(f"\n=== Parameter Sweep Visualization: {swept_param_name} ===")
        log_scale = (input(f"Use logarithmic X-axis for {swept_param_name}? (y/n) [y]: ").lower().strip() or 'y') == 'y'
        moving_avg = (input("Plot moving average? (y/n) [y]: ").lower().strip() or 'y') == 'y'
        bar_plot = False
    else:
        log_scale = False
        moving_avg = False
        bar_plot = True

    plot_bottleneck = (input("Include bottleneck activation plots (lambdamax style)? (y/n) [y]: ").lower().strip() or 'y') == 'y'

    # Bottleneck normalization option
    bottleneck_norm_per_point = False
    if plot_bottleneck:
        print("\nBottleneck normalization options:")
        print("  1. Per-point: normalize each sweep point independently (dominant kernel = 1 at each point)")
        print("  2. Global: normalize by global maximum across all sweep points")
        norm_choice = input("Choose normalization (1 or 2) [1]: ").strip() or '1'
        bottleneck_norm_per_point = (norm_choice == '1')

    # Helper function for mean and standard error
    def mean_ste(arr):
        arr = np.array(arr)
        if len(arr) == 0:
            return 0, 0
        return arr.mean(axis=0), arr.std(axis=0, ddof=1) / np.sqrt(len(arr))

    # Extract metrics across parameter values
    param_vals = []
    results = {}
    all_z_data = []  # Collect all z data for global normalization

    # Use the order from swept_param_values (pre-sorted for weighted_loss scenarios)
    for param_val in swept_param_values:
        metrics_per_seed = metrics_per_value[param_val]
        results[param_val] = {}

        # Bottleneck activations (z) - compute mean/ste without normalization first
        if plot_bottleneck and "z" in metrics_per_seed and len(metrics_per_seed["z"]) > 0:
            z_data = np.array(metrics_per_seed["z"])  # shape: (n_seeds, n_kernels)
            z = np.abs(z_data)
            # Compute mean and ste of raw (unnormalized) values
            results[param_val]["z_mean_raw"], results[param_val]["z_ste_raw"] = mean_ste(z)
            all_z_data.append(results[param_val]["z_mean_raw"])  # collect mean values for global norm

        # Training goodness metric
        train_goodness = metrics_per_seed.get(f"train_{cf.goodness_str}", [])
        results[param_val]["train_goodness"], results[param_val]["train_goodness_ste"] = mean_ste(train_goodness)

        # Validation goodness metric
        val_goodness = metrics_per_seed.get(f"val_{cf.goodness_str}", [])
        results[param_val]["val_goodness"], results[param_val]["val_goodness_ste"] = mean_ste(val_goodness)

        # Training loss metric
        train_loss = metrics_per_seed.get(f"train_{cf.loss_str}", [])
        results[param_val]["train_loss"], results[param_val]["train_loss_ste"] = mean_ste(train_loss)

        # Validation loss metric
        val_loss = metrics_per_seed.get(f"val_{cf.loss_str}", [])
        results[param_val]["val_loss"], results[param_val]["val_loss_ste"] = mean_ste(val_loss)

        param_vals.append(param_val)

    param_vals = np.array(param_vals)

    # Apply bottleneck normalization after collecting all mean values
    if plot_bottleneck and all_z_data:
        if bottleneck_norm_per_point:
            # Option 1: Per-point normalization (each param value normalized independently)
            # Normalize so max kernel = 1 at each sweep point
            for param_val in param_vals:
                z_mean = results[param_val]["z_mean_raw"]
                z_ste = results[param_val]["z_ste_raw"]
                z_max = z_mean.max()  # max across kernels at this point
                if z_max > 0:
                    results[param_val]["z"] = z_mean / z_max
                    results[param_val]["z_ste"] = z_ste / z_max
                else:
                    results[param_val]["z"] = z_mean
                    results[param_val]["z_ste"] = z_ste
        else:
            # Option 2: Global normalization (normalized by max across all sweep points)
            all_z_means = np.array(all_z_data)  # shape: (n_param_vals, n_kernels)
            global_max = all_z_means.max()
            for param_val in param_vals:
                z_mean = results[param_val]["z_mean_raw"]
                z_ste = results[param_val]["z_ste_raw"]
                if global_max > 0:
                    results[param_val]["z"] = z_mean / global_max
                    results[param_val]["z_ste"] = z_ste / global_max
                else:
                    results[param_val]["z"] = z_mean
                    results[param_val]["z_ste"] = z_ste

    # Prepare display labels for X-axis
    if scenario_labels:
        # Use scenario labels for display (e.g., "(a)", "(b)", "(c)", "(d)")
        display_labels = [scenario_labels[pv] for pv in param_vals]
    else:
        # Use parameter values directly
        display_labels = None

    # Create appropriate layout
    if plot_bottleneck:
        # Lambdamax-style vertical layout
        _plot_sweep_lambdamax_style(param_vals, results, cf, filename, log_scale,
                                    param_abbrev=PARAM_ABBREVIATIONS.get(swept_param_name, swept_param_name),
                                    moving_avg=moving_avg, display_labels=display_labels, bar_plot=bar_plot)
    else:
        # Original two-panel horizontal layout
        _plot_sweep_two_panel(param_vals, results, cf, filename, log_scale, moving_avg,
                             param_abbrev=PARAM_ABBREVIATIONS.get(swept_param_name, swept_param_name),
                             display_labels=display_labels, bar_plot=bar_plot)


def _plot_sweep_two_panel(param_vals, results, cf, filename, log_scale, moving_avg, param_abbrev, display_labels=None, bar_plot=False):
    """Two-panel horizontal layout for parameter sweep (Train/Val side-by-side)."""
    import numpy as np

    # Use display labels if provided, otherwise use param values
    if display_labels is None:
        display_labels = param_vals
        x_positions = param_vals
    else:
        # For categorical labels, use integer positions
        x_positions = list(range(len(param_vals)))

    # Extract means for two-panel plot
    train_goodness_means = np.array([results[pv].get("train_goodness", 0) for pv in param_vals])
    val_goodness_means = np.array([results[pv].get("val_goodness", 0) for pv in param_vals])
    train_goodness_stdes = np.array([results[pv].get("train_goodness_ste", 0) for pv in param_vals])
    val_goodness_stdes = np.array([results[pv].get("val_goodness_ste", 0) for pv in param_vals])

    train_loss_means = np.array([results[pv].get("train_loss", 0) for pv in param_vals])
    val_loss_means = np.array([results[pv].get("val_loss", 0) for pv in param_vals])
    train_loss_stdes = np.array([results[pv].get("train_loss_ste", 0) for pv in param_vals])
    val_loss_stdes = np.array([results[pv].get("val_loss_ste", 0) for pv in param_vals])

    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Left subplot: Goodness (R² or accuracy)
    goodness_label = cf.goodness_str.upper() if 'goodness_str' in cf.keys() else 'Goodness'
    if goodness_label == "R2AGG":
        goodness_label = r"$R^2_\mathrm{agg}$"
        loss_label = r"MSE$_\mathrm{agg}$"

    ax = axs[0]

    if not bar_plot:
        # Plot both train and val
        ax.plot(x_positions, train_goodness_means, 'o--', color='red', label=f'Train {goodness_label}',
                markersize=6, linewidth=1.5, alpha=0.7)
        ax.plot(x_positions, val_goodness_means, 'o--', color='blue', label=f'Val {goodness_label}',
                markersize=6, linewidth=1.5, alpha=0.7)

        ax.fill_between(x_positions, train_goodness_means - train_goodness_stdes,
                            train_goodness_means + train_goodness_stdes, color='red', alpha=0.2)
        ax.fill_between(x_positions, val_goodness_means - val_goodness_stdes,
                            val_goodness_means + val_goodness_stdes, color='blue', alpha=0.2)

        # Moving average if requested
        if moving_avg and len(x_positions) >= 2:
            window = min(2, len(x_positions))
            train_ma = np.convolve(train_goodness_means, np.ones(window)/window, mode='valid')
            val_ma = np.convolve(val_goodness_means, np.ones(window)/window, mode='valid')
            ma_x = x_positions[window-1:] if isinstance(x_positions, list) else x_positions[window-1:]
            ax.plot(ma_x, train_ma, '-', color='darkred', linewidth=2, alpha=0.5, label='Train MA')
            ax.plot(ma_x, val_ma, '-', color='darkblue', linewidth=2, alpha=0.5, label='Val MA')
    else:
        width = 0.25  # width of the bars
        ax.bar(np.array(x_positions) - width/2, train_goodness_means, width, color='red', alpha=0.7, label=f'Train {goodness_label}', yerr=train_goodness_stdes, capsize=5)
        ax.bar(np.array(x_positions) + width/2, val_goodness_means, width, color='blue', alpha=0.7, label=f'Val {goodness_label}', yerr=val_goodness_stdes, capsize=5)
        ax.set_ylim(0.8, 1.0)  # set y-axis limits for better visibility

    if log_scale and display_labels is None:
        ax.set_xscale('log')

    # Set X-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels, fontsize=10)

    ax.set_xlabel(f'{param_abbrev}', fontsize=12)
    ax.set_ylabel(goodness_label, fontsize=12)
    ax.set_title(f'{goodness_label} vs {param_abbrev}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)

    # Right subplot: Loss (MSE or CEL)
    loss_label = cf.loss_str if 'loss_str' in cf.keys() else 'Loss'

    ax = axs[1]

    if not bar_plot:
        # Plot both train and val
        ax.plot(x_positions, train_loss_means, 'o--', color='red', label=f'Train {loss_label}',
                markersize=6, linewidth=1.5, alpha=0.7)
        ax.plot(x_positions, val_loss_means, 'o--', color='blue', label=f'Val {loss_label}',
                markersize=6, linewidth=1.5, alpha=0.7)

        ax.fill_between(x_positions, train_loss_means - train_loss_stdes,
                            train_loss_means + train_loss_stdes, color='red', alpha=0.2)
        ax.fill_between(x_positions, val_loss_means - val_loss_stdes,
                            val_loss_means + val_loss_stdes, color='blue', alpha=0.2)

        # Moving average if requested
        if moving_avg and len(x_positions) >= 2:
            window = min(2, len(x_positions))
            train_ma = np.convolve(train_loss_means, np.ones(window)/window, mode='valid')
            val_ma = np.convolve(val_loss_means, np.ones(window)/window, mode='valid')
            ma_x = x_positions[window-1:] if isinstance(x_positions, list) else x_positions[window-1:]
            ax.plot(ma_x, train_ma, '-', color='darkred', linewidth=2, alpha=0.5, label='Train MA')
            ax.plot(ma_x, val_ma, '-', color='darkblue', linewidth=2, alpha=0.5, label='Val MA')
    else:
        width = 0.25  # width of the bars
        ax.bar(np.array(x_positions) - width/2, train_loss_means, width, color='red', alpha=0.7, label=f'Train {loss_label}', yerr=train_loss_stdes, capsize=5)
        ax.bar(np.array(x_positions) + width/2, val_loss_means, width, color='blue', alpha=0.7, label=f'Val {loss_label}', yerr=val_loss_stdes, capsize=5)

    if log_scale and display_labels is None:
        ax.set_xscale('log')

    # Set X-axis ticks and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels, fontsize=10)

    ax.set_xlabel(f'{param_abbrev}', fontsize=12)
    ax.set_ylabel(loss_label, fontsize=12)
    ax.set_title(f'{loss_label} vs {param_abbrev}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    
    for ax in axs:
        ax.grid(True, alpha=0.3, linestyle='--', which='both', color='k')

    plt.tight_layout()

    # Save figure
    save_fig(fig, cf.logdir, filename)
    print(f"Saved parameter sweep plot to {cf.logdir}/{filename}")
    plt.close()


def _plot_sweep_lambdamax_style(param_vals, results, cf, filename, log_scale, param_abbrev, moving_avg=False, display_labels=None, bar_plot=False):
    """Lambdamax-style vertical layout with bottleneck activations."""
    import numpy as np

    # Use display labels if provided, otherwise use param values
    if display_labels is None:
        display_labels = param_vals
        x_positions = param_vals
    else:
        # For categorical labels, use integer positions
        x_positions = list(range(len(param_vals)))

    # Stack results into arrays (like in plot_lambda)
    def stack(metric):
        return np.array([results[pv].get(metric, 0) for pv in param_vals])

    # Define metrics to plot (similar to lambdamax)
    goodness_label = cf.goodness_str.upper() if 'goodness_str' in cf.keys() else 'Goodness'
    loss_label = cf.loss_str if 'loss_str' in cf.keys() else 'Loss'
    if goodness_label == "R2AGG":
        goodness_label = r"$R^2_\mathrm{agg}$"
        loss_label = r"MSE$_\mathrm{agg}$"

    metrics_to_plot = [
        dict(name="z", ylabel="z", title="selected branch activations", per_kernel=True),
        dict(name="goodness", ylabel=goodness_label, title=goodness_label+" (train & val)"),
        dict(name="loss", ylabel=loss_label, title=loss_label+" (train & val)", log=True),
    ]

    # Create figure with vertical subplots
    fig, ax = plt.subplots(len(metrics_to_plot), 1, figsize=(15, 5 + (len(metrics_to_plot)-1)*2),
                          sharex=True,
                          gridspec_kw={'height_ratios': [4] + [2]*(len(metrics_to_plot)-1)})

    if len(metrics_to_plot) == 1:
        ax = [ax]  # make it iterable

    for i, m in enumerate(metrics_to_plot):
        if m.get("per_kernel"):
            # Bottleneck activations: plot per kernel
            mean, ste = stack(m["name"]), stack(m["name"] + "_ste")
            if m["name"] in results[param_vals[0]]:
                for k in range(mean.shape[1]):
                    ax[i].plot(x_positions, mean[:, k], "o--",
                              label=branch_label(cf, k), color=branch_color(cf, k))
                    ax[i].fill_between(x_positions, mean[:, k] - ste[:, k],
                                      mean[:, k] + ste[:, k], alpha=.3)
                legend = ax[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left",
                                     ncol=2, fontsize=8)
                for text, line in zip(legend.get_texts(), legend.get_lines()):
                    text.set_color(line.get_color())
        else:
            # Regular metrics: plot both train and val
            train_mean = stack(f"train_{m['name']}")
            train_ste = stack(f"train_{m['name']}_ste")
            val_mean = stack(f"val_{m['name']}")
            val_ste = stack(f"val_{m['name']}_ste")

            if not bar_plot:
                ax[i].plot(x_positions, train_mean, "o--", color="red", label="train")
                ax[i].fill_between(x_positions, train_mean - train_ste, train_mean + train_ste,
                                alpha=.3, color="red")
                ax[i].plot(x_positions, val_mean, "o--", color="blue", label="val")
                ax[i].fill_between(x_positions, val_mean - val_ste, val_mean + val_ste,
                                alpha=.3, color="blue")
            else:
                width = 0.35  # width of the bars
                ax[i].bar(np.array(x_positions) - width/2, train_mean, width, color='red', alpha=0.7, label='train', yerr=train_ste, capsize=5)
                ax[i].bar(np.array(x_positions) + width/2, val_mean, width, color='blue', alpha=0.7, label='val', yerr=val_ste, capsize=5)

            if moving_avg and len(x_positions) >= 2:
                window = min(2, len(x_positions))
                train_ma = np.convolve(train_mean, np.ones(window)/window, mode='valid')
                val_ma = np.convolve(val_mean, np.ones(window)/window, mode='valid')
                ma_x = x_positions[window-1:] if isinstance(x_positions, list) else x_positions[window-1:]
                ax[i].plot(ma_x, train_ma, '-', color='darkred', linewidth=2, alpha=0.5, label='Train MA')
                ax[i].plot(ma_x, val_ma, '-', color='darkblue', linewidth=2, alpha=0.5, label='Val MA')

            ax[i].legend(loc="best", fontsize=8)

        ax[i].set_ylabel(m["ylabel"])
        ax[i].set_title(m["title"], loc="left")

        ax[i].grid(True)

        if m.get("log"):
            # Set the y-axis units to scientific notation
            ax[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1e}'))

        if log_scale and display_labels is None:
            ax[i].set_xscale("log")
            ax[i].grid(True, which="both", linestyle=":")


    ax[0].set_ylim(-0.05, 1.05)  # for z plot

    # Set X-axis ticks and labels on the bottom subplot
    ax[-1].set_xticks(x_positions)
    ax[-1].set_xticklabels(display_labels, fontsize=10)
    ax[-1].set_xlabel(param_abbrev)
    fig.suptitle(f"Parameter sweep: {param_abbrev} ({len(cf.seeds)} seeds, {len(cf.kernels)} kernels)")
    plt.tight_layout()
    save_fig(fig, cf.logdir, filename)
    print(f"Saved parameter sweep plot (lambdamax style) to {cf.logdir}/{filename}")
    plt.close()


def _handle_weighted_loss_scenarios(flat_scenarios, seed_dirs, cf):
    """
    Handle weighted_loss experiment scenario analysis.

    Extracts scenario labels from folder names, sorts by label alphabetically,
    and loads metrics for each scenario.

    Args:
        flat_scenarios: Dict of {folder_name: count}
        seed_dirs: List of seed directories
        cf: Config object

    Returns:
        Tuple of (swept_param_name, swept_param_values, metrics_per_value, scenario_label_map)
    """
    # Parse scenario labels from folder names and sort by label
    # Folder names look like: "es=False-scenarios=(c)-spc=None-wl=False"
    # Extract the scenario label and create mapping
    folder_to_label = {}
    for folder_name in flat_scenarios.keys():
        # Extract scenario label from folder name
        if "scenarios=" in folder_name:
            # Parse: find "scenarios=" and extract until next "-" or end
            start = folder_name.find("scenarios=") + len("scenarios=")
            end = folder_name.find("-", start)
            if end == -1:
                end = len(folder_name)
            label = folder_name[start:end]
            folder_to_label[folder_name] = label

    # Sort folders by their scenario labels for alphabetical display
    if folder_to_label:
        # Create list of (folder, label) tuples and sort by label
        sorted_items = sorted(folder_to_label.items(), key=lambda x: x[1])
        swept_param_values = [folder for folder, label in sorted_items]
        scenario_label_map = {folder: label for folder, label in sorted_items}
        print(f"Scenario order (sorted by label): {scenario_label_map}")
    else:
        # Fallback: use alphabetical folder order if no labels found
        swept_param_values = sorted(flat_scenarios.keys())
        scenario_label_map = None
        print("Warning: Could not extract scenario labels from folder names")

    swept_param_name = "scenario"

    # Load metrics for each scenario
    metrics_per_value = {}

    for scenario_name in swept_param_values:
        # Find seed directories for this scenario
        matching_seeds = [
            seed_dir for seed_dir in seed_dirs
            if scenario_name in str(seed_dir)
        ]

        if not matching_seeds:
            print(f"Warning: No seeds found for {scenario_name}")
            continue

        # Load metrics from each seed
        metrics_lists = {
            f"train_{cf.goodness_str}": [],
            f"val_{cf.goodness_str}": [],
            f"train_{cf.loss_str}": [],
            f"val_{cf.loss_str}": [],
            "z": [],
        }

        for seed_dir in matching_seeds:
            try:
                metrics = load_json(seed_dir, "metrics.json")

                # Extract final values for train/val metrics
                for key in [f"train_{cf.goodness_str}", f"val_{cf.goodness_str}",
                           f"train_{cf.loss_str}", f"val_{cf.loss_str}"]:
                    if key in metrics and len(metrics[key]) > 0:
                        final_val = metrics[key][-1] if isinstance(metrics[key], list) else metrics[key]
                        metrics_lists[key].append(final_val)

                # Extract bottleneck activations
                z_values = []
                for k in range(len(cf.kernels)):
                    key = f"z_{k}"
                    if key in metrics and len(metrics[key]) > 0:
                        final_z = metrics[key][-1] if isinstance(metrics[key], list) else metrics[key]
                        z_values.append(final_z)
                if z_values:
                    metrics_lists["z"].append(z_values)
            except Exception as e:
                print(f"Warning: Could not load metrics from {seed_dir}: {e}")

        # Compute statistics across seeds
        metrics_per_value[scenario_name] = {}
        for key in metrics_lists:
            if key == "z":
                # Average bottleneck values across seeds
                if metrics_lists[key]:
                    metrics_per_value[scenario_name][key] = np.mean(metrics_lists[key], axis=0).tolist()
            else:
                # Store the list of values across seeds (not the mean)
                # plot_parameter_sweep will compute mean/ste from this list
                if metrics_lists[key]:
                    metrics_per_value[scenario_name][key] = metrics_lists[key]

    return swept_param_name, swept_param_values, metrics_per_value, scenario_label_map


def generate_sweep_plot_from_logs(cf, basepath):
    """
    Generate parameter sweep plot from existing experiment logs.
    Used when remake_sweep_plot is True.

    Handles both:
    - Hierarchical naming: param1=val1/param2=val2/ (single param swept)
    - Flat naming: param1=val1-param2=val2-param3=val3 (multiple params, treated as scenarios)
    """
    # Find all seed directories
    seed_dirs = find_existing_runs(basepath, cf)

    if not seed_dirs:
        print(f"No experiment runs found in {basepath}")
        return

    # Analyze folder structure to detect swept parameter(s)
    # Look at ALL seed directories to find all parameter values
    swept_params = {}
    flat_scenarios = {}  # For flat naming: full folder name -> count

    for seed_dir in seed_dirs:
        current_path = seed_dir.parent

        # Navigate up from each seed to find parameter sweep folders
        while current_path.name != basepath.name and current_path != current_path.parent:
            folder_name = current_path.name

            # Check if this is a parameter folder (format: abbrev=value or abbrev1=val1-abbrev2=val2)
            if "=" in folder_name and not folder_name.startswith("seed_"):
                # Check if this is flat naming (contains dashes between param=value pairs)
                if "-" in folder_name and "=" in folder_name:
                    # Flat naming: treat entire folder name as a scenario
                    if folder_name not in flat_scenarios:
                        flat_scenarios[folder_name] = 0
                    flat_scenarios[folder_name] += 1
                else:
                    # Hierarchical naming: single param=value
                    try:
                        param_abbrev, value_str = folder_name.split("=", 1)

                        # Try to convert value to appropriate type
                        try:
                            value = int(value_str)
                        except ValueError:
                            try:
                                value = float(value_str)
                            except ValueError:
                                value = value_str

                        # Find full parameter name from abbreviation
                        param_name = None
                        for full_name, abbrev in PARAM_ABBREVIATIONS.items():
                            if abbrev == param_abbrev:
                                param_name = full_name
                                break

                        if param_name is None:
                            param_name = param_abbrev  # fallback to abbreviation

                        if param_name not in swept_params:
                            swept_params[param_name] = set()
                        swept_params[param_name].add(value)
                    except:
                        pass

            current_path = current_path.parent

    # Decide which mode we're in: flat scenarios or hierarchical single-param
    if flat_scenarios:
        # Flat naming mode (e.g., weighted_loss experiment)
        print(f"\nDetected flat naming with {len(flat_scenarios)} scenario combinations:")
        for scenario_name in sorted(flat_scenarios.keys()):
            print(f"  - {scenario_name}")

        # Handle weighted_loss scenarios: extract labels, sort, load metrics
        swept_param_name, swept_param_values, metrics_per_value, scenario_label_map = \
            _handle_weighted_loss_scenarios(flat_scenarios, seed_dirs, cf)

        # Generate plot
        filename = f"sweep_scenarios.png"
        plot_parameter_sweep(swept_param_name, swept_param_values, metrics_per_value, cf, filename,
                           scenario_labels=scenario_label_map)
        return

    # Hierarchical naming mode (original behavior)
    # Convert sets to sorted lists
    for param_name in swept_params:
        swept_params[param_name] = sorted(list(swept_params[param_name]))

    if not swept_params:
        print("No parameter sweeps detected in folder structure.")
        return

    if len(swept_params) != 1:
        print(f"Sweep visualization requires exactly 1 swept parameter.")
        print(f"Found {len(swept_params)} swept parameters: {list(swept_params.keys())}")
        return

    # Extract the single swept parameter
    swept_param_name = list(swept_params.keys())[0]
    swept_param_values = swept_params[swept_param_name]

    print(f"\nDetected sweep over {swept_param_name}: {swept_param_values}")

    # Load metrics for each parameter value
    metrics_per_value = {}

    for param_val in swept_param_values:
        # Find seed directories for this parameter value
        param_abbrev = PARAM_ABBREVIATIONS.get(swept_param_name, swept_param_name)
        param_folder_name = f"{param_abbrev}={param_val}"

        # Find all seeds for this parameter value
        matching_seeds = [
            seed_dir for seed_dir in seed_dirs
            if param_folder_name in str(seed_dir)
        ]

        if not matching_seeds:
            print(f"Warning: No seeds found for {param_folder_name}")
            continue

        # Load metrics from each seed
        metrics_lists = {
            f"train_{cf.goodness_str}": [],
            f"val_{cf.goodness_str}": [],
            f"train_{cf.loss_str}": [],
            f"val_{cf.loss_str}": [],
            "z": [],
        }

        for seed_dir in matching_seeds:
            try:
                metrics = load_json(seed_dir, "metrics.json")

                # Extract final values for train/val metrics
                for key in [f"train_{cf.goodness_str}", f"val_{cf.goodness_str}",
                           f"train_{cf.loss_str}", f"val_{cf.loss_str}"]:
                    if key in metrics and len(metrics[key]) > 0:
                        # Get final value (last epoch)
                        final_val = metrics[key][-1] if isinstance(metrics[key], list) else metrics[key]
                        metrics_lists[key].append(final_val)

                # Extract bottleneck activations (z values)
                # Collect final z values for each kernel
                z_values = []
                for k in range(len(cf.kernels)):
                    key = f"z_{k}"
                    if key in metrics and len(metrics[key]) > 0:
                        final_z = metrics[key][-1] if isinstance(metrics[key], list) else metrics[key]
                        z_values.append(final_z)

                if z_values:
                    metrics_lists["z"].append(z_values)
            except Exception as e:
                print(f"Warning: Could not load metrics from {seed_dir}: {e}")

        metrics_per_value[param_val] = metrics_lists

    if not metrics_per_value:
        print("No metrics loaded. Cannot generate sweep plot.")
        return

    # Set logdir to experiment level for saving plot
    cf.logdir = get_experiment_level_path(seed_dirs[0])

    # Generate the sweep plot
    filename = f"sweep_{PARAM_ABBREVIATIONS.get(swept_param_name, swept_param_name)}_remake.png"
    plot_parameter_sweep(swept_param_name, swept_param_values, metrics_per_value, cf, filename)
    print(f"\nSweep plot saved to {cf.logdir}/{filename}")


def experiment_plot(cf, basepath):
    # Ensure cf.logdir is at experiment level
    # If it's not already, navigate up from seed level
    if 'logdir' in cf.keys() and cf.logdir is not None:
        cf.logdir = get_experiment_level_path(Path(cf.logdir))
    else:
        cf.logdir = basepath

    if cf.task == "lbc":
        pass
    else:
        if cf.experiment_name in ("lambdamax", "lambdatot"):
            if cf.remake_history_and_pt_plots:
                metrics_per_lambda = load_json(cf.logdir, "metrics_per_lambda_remake.json")
                plot_lambda(metrics_per_lambda, cf, f"{cf.experiment_name}_remake.png", cf.enabled_metrics)
            else:
                filename = f"{cf.experiment_name}.png"
                if cf.remake_lambda_plot:
                    cf.logdir = basepath
                    filename = f"{cf.experiment_name}_remake.png"
                metrics_per_lambda = load_json(cf.logdir, f"metrics_per_lambda.json")
                plot_lambda(metrics_per_lambda, cf, filename, cf.enabled_metrics)


###############################################
# PLOTTING
###############################################


def plot_phase_transition(metrics, cf, file_name, fit_metrics=None):
    """"
    Plots the predicted and true tuning parameter over time on the left. 
    On the right, show the predicted tuning param as a func of the true tuning param.
    """

    MVUL_branch = np.array( [metrics[f"MVUL_{k}"] for k in range(len(cf.kernels))] ) # shape (kernels, no_unique_labels )
    MVUL_std_branch = np.array( [metrics[f"MVUL_std_{k}"] for k in range(len(cf.kernels))] ) # shape (kernels, no_unique_labels )
    MVUL_out = np.array(metrics["MVUL_out"])[-1] # shape ( no_unique_labels, 2) for Ising
    MVUL_std_out = np.array(metrics["MVUL_std_out"])[-1] # shape ( no_unique_labels, 2) for Ising
    
    unique_times = cf.val_dataset.unique_times
    if torch.is_tensor(unique_times):
        # unique_times may live on an accelerator device (cuda/mps); matplotlib
        # needs a host-side array, and .numpy() only works for CPU tensors, so
        # pull it off-device unconditionally instead of leaving this as a
        # silent CUDA-only .cpu() call (pre-existing bug, unrelated to the
        # equivariant-branches feature; fixed here because it silently dropped
        # phase-transition plots on any non-CUDA accelerator, e.g. Apple MPS).
        unique_times = unique_times.detach().cpu().numpy()

    if cf.dataset == "Paris_Ising":
        deltas = np.array(list( cf.val_dataset.processor.delta_dict.values() ))
        omegas = np.array(list( cf.val_dataset.processor.omega_dict.values() ))
    elif cf.dataset == "Paris_XY_X":
        deltas = np.array(list( cf.val_dataset.processor.delta_dict_x.values() ))
    elif cf.dataset == "Paris_XY_Z":
        deltas = np.array(list( cf.val_dataset.processor.delta_dict_z.values() ))
    elif cf.dataset == "Paris_XY_XZ":
        deltas = np.array(list( cf.val_dataset.processor.delta_dict_combined.values() ))

    handles, handles2 = [], []

    if cf.dataset == "Paris_Ising" and cf.label_param == "deltaomega":

        delta_phase_dict, omega_phase_dict = cf.val_dataset.get_phase_indicator( MVUL=MVUL_out )
        
        delta_crit_time = unique_times[(np.abs(deltas - delta_phase_dict["peak"])).argmin()]
        omega_crit_time = unique_times[(np.abs(omegas - omega_phase_dict["peak"])).argmin()]

        branch_delta_phase_dicts, branch_omega_phase_dicts = [], []
        branch_delta_crit_times, branch_omega_crit_times = [], []

        for k in range(len(cf.kernels)):
            branch_delta_phase_dicts.append( cf.val_dataset.get_phase_indicator( MVUL=MVUL_branch[k], output=False )[0] )
            branch_omega_phase_dicts.append( cf.val_dataset.get_phase_indicator( MVUL=MVUL_branch[k], output=False )[1] )

            branch_delta_crit_times.append( unique_times[(np.abs(deltas - branch_delta_phase_dicts[-1]["peak"])).argmin()] )
            branch_omega_crit_times.append( unique_times[(np.abs(omegas - branch_omega_phase_dicts[-1]["peak"])).argmin()] )

        fig = plt.figure(figsize=(18,  8))
        gs = gridspec.GridSpec(2,3, width_ratios=[2,1,1]) 

    elif cf.dataset == "Paris_Ising" and cf.label_param == "delta":

        delta_phase_dict = cf.val_dataset.get_phase_indicator( MVUL=MVUL_out )
        delta_crit_time = unique_times[(np.abs(deltas - delta_phase_dict["peak"])).argmin()]

        branch_phase_dicts = []
        branch_delta_crit_times = []
        for k in range(len(cf.kernels)):
            branch_phase_dicts.append( cf.val_dataset.get_phase_indicator( MVUL=MVUL_branch[k], output=False ) )
            branch_delta_crit_times.append( unique_times[(np.abs(deltas - branch_phase_dicts[-1]["peak"])).argmin()] )
        
        fig = plt.figure(figsize=(14,  8))
        gs = gridspec.GridSpec(2,2, width_ratios=[2,1]) 

    elif cf.dataset in ["Paris_XY_X", "Paris_XY_Z", "Paris_XY_XZ"] and cf.label_param == "delta":

        delta_phase_dict = cf.val_dataset.get_phase_indicator( MVUL=MVUL_out )
        delta_crit_time = unique_times[(np.abs(deltas - delta_phase_dict["peak"])).argmin()]
        
        branch_phase_dicts = []
        branch_delta_crit_times = []
        for k in range(len(cf.kernels)):
            branch_phase_dicts.append( cf.val_dataset.get_phase_indicator( MVUL=MVUL_branch[k], output=False ) )
            branch_delta_crit_times.append( unique_times[(np.abs(deltas - branch_phase_dicts[-1]["peak"])).argmin()] )

        fig = plt.figure(figsize=(14,  8))
        gs = gridspec.GridSpec(2, 2 , width_ratios=[2,1]) 

    else:

        phase_dict = cf.val_dataset.get_phase_indicator( MVUL=MVUL_out )
        branch_phase_dicts = []
        for k in range(len(cf.kernels)):
            branch_phase_dicts.append( cf.val_dataset.get_phase_indicator( MVUL=MVUL_branch[k], output=False ) )

        fig = plt.figure(figsize=(14,  8))
        gs = gridspec.GridSpec(2, 2 , width_ratios=[2,1]) 

    # delta, omega as a function of time
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])

    ax10 = fig.add_subplot(gs[1,0])

    ax00.set_ylabel(r"$\delta$ ",)
    ax00.set_xlabel(r"time (ns)") 
    ax00.tick_params(axis="y", colors="darkblue") 

    ax10.set_ylabel(r"$a_k$")
    ax10.set_xlabel(r"time (ns)")

    ax00.grid(True) 
    ax01.grid(True)
    ax10.grid(True)

    # top plot, always present
    line_delta, = ax00.plot(unique_times, deltas, marker='o', markersize=2, color='darkblue', label=r'$\delta$', alpha = 0.3)
    handles += [line_delta]

    if cf.dataset == "Paris_Ising":
        ax00b = ax00.twinx()
        ax00b.set_ylabel(r"$\Omega$ ", color='darkred')
        ax00b.tick_params(axis="y", colors="darkred")
        ax00b.grid(False)         # disable grid on omega-axis

        line_omega, = ax00b.plot(unique_times, omegas, marker='o', markersize=2, color='darkred', label=r'$\Omega$', alpha = 0.3)

    # predicted label_param vs. true label_param
    if cf.label_param in ["deltaomega", "delta"]:
        ax11 = fig.add_subplot(gs[1,1]) 

        line_delta_crit = ax00.axvline(x=delta_crit_time, color='darkblue', linestyle='dotted',linewidth=2, label=r"$\arg\max_\delta\; d \hat\delta/d \delta$")

        ax01.plot( delta_phase_dict["tuning_true"], delta_phase_dict["pred"], marker='o', markersize=2, color='darkblue', label=r'$\hat\delta(\delta)$')
        ax01.axvline(x=delta_phase_dict["peak"], color='darkblue', linestyle='dotted',linewidth=2, label=r"$\arg\max_\delta\; d \hat\delta/d \delta$")

        ax01.set_ylabel(r"$\hat\delta$ ", color='darkblue')
        ax01.set_xlabel(r"$\delta$ ") 

        ax11.set_ylabel(r"$ a_k$")
        ax11.set_xlabel(r"$\delta$ ") 
        ax11.grid(True)

    if cf.label_param == "delta":
        line_hat_delta, = ax00.plot(unique_times, MVUL_out, marker='o', markersize=2, color='darkblue', label=r'$\hat \delta$')
        ax00.fill_between(unique_times, MVUL_out - MVUL_std_out, MVUL_out + MVUL_std_out, color='darkblue', alpha=0.1)

        handles += [line_hat_delta, line_delta_crit]
        ax01.plot( delta_phase_dict["tuning_true"], delta_phase_dict["pred"], marker='o', markersize=2, color='darkblue', label=r'$\hat\delta(\delta)$')
    
        for k in range(len(cf.kernels)):
            l, = ax10.plot( unique_times, MVUL_branch[k], marker='o', markersize=2, label=f"{branch_label(cf, k)}", color=branch_color(cf, k))
            ax10.fill_between(unique_times, MVUL_branch[k] - MVUL_std_branch[k], MVUL_branch[k] + MVUL_std_branch[k], color=l.get_color(), alpha=0.1)

            ax10.axvline(x=branch_delta_crit_times[k], linestyle='dotted',linewidth=2, color=l.get_color())
            ax11.plot( branch_phase_dicts[k]["tuning_true"], branch_phase_dicts[k]["pred"], marker='o', markersize=2, color=l.get_color())
            ax11.axvline(x=branch_phase_dicts[k]["peak"], linestyle='dotted',linewidth=2, color=l.get_color())
            handles2 += [l]

        legend = ax01.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=12)
        legend2 = ax11.legend(handles=handles2, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=12)

    if cf.label_param == "deltaomega":
        ax02 = fig.add_subplot(gs[0,2])
        ax12 = fig.add_subplot(gs[1,2])

        line_hat_delta, = ax00.plot(unique_times, MVUL_out[:,0], marker='o', markersize=2, color='darkblue', label=r'$\hat \delta$')
        ax00.fill_between(unique_times, MVUL_out[:,0] - MVUL_std_out[:,0], MVUL_out[:,0] + MVUL_std_out[:,0], color='darkblue', alpha=0.1)

        line_hat_omega, = ax00b.plot(unique_times, MVUL_out[:,1], marker='o', markersize=2, color='darkred', label=r'$\hat \Omega$')
        ax00b.fill_between(unique_times, MVUL_out[:,1] - MVUL_std_out[:,1], MVUL_out[:,1] + MVUL_std_out[:,1], color='darkred', alpha=0.1)

        line_omega_crit = ax00b.axvline(x=omega_crit_time, color='darkred', linestyle='dashed',linewidth=2, label=r"$\arg\max_\Omega\; d \hat\Omega/d \Omega$")

        ax02.plot( omega_phase_dict["tuning_true"], omega_phase_dict["pred"], marker='o', markersize=2, color='darkred', label=r'$\hat\Omega(\Omega)$')
        ax02.axvline(x=omega_phase_dict["peak"], color='darkred', linestyle='dashed', label =r"$\arg\max_\Omega\; d \hat\Omega/d \Omega$",linewidth=2) # type: ignore

        for k in range(len(cf.kernels)):
            l, = ax10.plot( unique_times, MVUL_branch[k], marker='o', markersize=2,label =f"{branch_label(cf, k)}", color=branch_color(cf, k) )
            ax10.fill_between(unique_times, MVUL_branch[k] - MVUL_std_branch[k], MVUL_branch[k] + MVUL_std_branch[k], color=l.get_color(), alpha=0.1)

            ax10.axvline(x=branch_delta_crit_times[k], linestyle='dotted',linewidth=2, color=l.get_color())
            ax10.axvline(x=branch_omega_crit_times[k], linestyle='dashed',linewidth=2, color=l.get_color())
            handles2 += [l]

            ax11.plot( branch_delta_phase_dicts[k]["tuning_true"], branch_delta_phase_dicts[k]["pred"], marker='o', markersize=2, label=f"{branch_label(cf, k)}", color=branch_color(cf, k) )
            ax11.axvline(x=branch_delta_phase_dicts[k]["peak"], linestyle='dotted',linewidth=2, color=l.get_color())

            ax12.plot( branch_omega_phase_dicts[k]["tuning_true"], branch_omega_phase_dicts[k]["pred"], marker='o', markersize=2, label=f"{branch_label(cf, k)}", color=branch_color(cf, k) )
            ax12.axvline(x=branch_omega_phase_dicts[k]["peak"], linestyle='dashed',linewidth=2, color=l.get_color())

        ax02.set_ylabel(r"$\hat\Omega$ ", color='darkred')
        ax02.set_xlabel(r"$\Omega$ ")
        ax02.grid(True)
       
        ax12.set_ylabel(r"$ a_k $")
        ax12.set_xlabel(r"$\Omega$ ")
        ax12.grid(True)
        
        handles += [line_hat_delta, line_delta_crit, line_omega, line_hat_omega, line_omega_crit]

        legend = ax02.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=12)
        legend2 = ax12.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=12)

    if cf.label_param == "t":
        l, = ax01.plot( phase_dict["tuning_true"], phase_dict["pred"], marker='o', markersize=2, color='black', label=r'$\hat t$')
        handles += [l]

        l = ax01.axvline(x=phase_dict["peak"], color='black', linestyle='dotted',linewidth=2, label=r"$\arg\max_t \; d \hat t/d t$")
        handles += [l]
        
        max_val = max( ax01.get_xlim()[1], ax01.get_ylim()[1] )
        l, = ax01.plot( [0, max_val], [0, max_val], color='gray', linestyle='dotted', label=r"$\hat t = t$" )
        handles += [l]
        for k in range(len(cf.kernels)):
            l, = ax10.plot(unique_times, MVUL_branch[k], marker='o', markersize=2,label =f"{branch_label(cf, k)}", color=branch_color(cf, k) )
            ax10.fill_between(unique_times, MVUL_branch[k] - MVUL_std_branch[k], MVUL_branch[k] + MVUL_std_branch[k], color=l.get_color(), alpha=0.1)

            ax10.axvline(x=branch_phase_dicts[k]["peak"], linestyle='dotted',linewidth=2, color=l.get_color())

            handles2 += [l]

        ax01.set_xlim(ax00.get_xlim()[0], max_val)
        ax01.set_ylabel(r"$\hat t$ ", color='black')
        ax01.set_xlabel(r"$t$ ") 
        legend = ax01.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=12)
        legend2 = ax10.legend(handles=handles2, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=12)


    for text, line in zip(legend.get_texts(), legend.get_lines()):
        text.set_color(line.get_color())
    for text, line in zip(legend2.get_texts(), legend2.get_lines()):
        text.set_color(line.get_color())

    ax00.set_title(r"$\mathbf{tuning\;parameters}$", loc='left', fontsize=12)
    ax10.set_title(r"$\mathbf{branch\;outputs}$", loc='left', fontsize=12)
    plt.tight_layout()
    save_fig(fig, cf.logdir, file_name)
    plt.close()


def plot_history(metrics, cf, file_name, show_dont_save=False):
    """
    Plots the per-epoch training curves selected by cf.enabled_metrics (default
    {"z", "loss", "goodness"} if unset): the task loss plus its L1 sparsity term
    ("loss", tetriscnn only shows the l1 companion panel), the goodness metric
    ("goodness"), the bottleneck activations ("z", tetriscnn only), and the phase
    indicator over epochs ("pt", regression only, when tracked). The figure grid is
    sized to however many of these are both requested and applicable, so asking for
    fewer metrics yields a smaller figure rather than leaving empty panels.
    """
    enabled = set(getattr(cf, "enabled_metrics", None) or {"z", "loss", "goodness"})
    is_tetriscnn = cf.model == "tetriscnn"

    metric1, metric2 = f"train_{cf.loss_str}", "train_l1"
    metric3, metric4 = f"val_{cf.loss_str}", "val_l1"
    metric5 = cf.goodness_str

    panels = []  # draw(ax) callables, in display order

    if "loss" in enabled:
        def _draw_loss(ax):
            ax.plot(metrics[metric1], label='train', color="red")
            ax.plot(metrics[metric3], label='val', color="blue")
            if not is_tetriscnn:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_yscale("log")
            ax.set_xlabel("epoch")
            ax.set_ylabel(f"{cf.loss_str}")
            ax.set_title(fr"$\mathbf{{{cf.loss_str}}}$, train= {metrics[metric1][-1]:.2e}, val={metrics[metric3][-1]:.2e}", loc='left')
        panels.append(_draw_loss)

        if is_tetriscnn:
            def _draw_l1(ax):
                ax.plot(metrics[metric2], label='train', color="red")
                ax.plot(metrics[metric4], label='val', color="blue")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.set_yscale("log")
                ax.set_xlabel("epoch")
                ax.set_ylabel("l1")
                ax.set_title(
                    fr"$\mathbf{{l1}}$, train= {metrics[metric2][-1]:.2e}, val={metrics[metric4][-1]:.2e}",
                    loc='left'
                )
            panels.append(_draw_l1)

    if "goodness" in enabled:
        def _draw_goodness(ax):
            ax.plot(metrics[f"val_"+metric5], label=metric5, color="blue")
            ax.plot(metrics[f"train_"+metric5], label=f"train_{metric5}", color="red")
            if cf.task == "regression":
                ax.plot(metrics["train_r2agg"], label="train_r2agg", color="red", linestyle='--')
                ax.plot(metrics["val_r2agg"], label="val_r2agg", color="blue", linestyle='--')
            ax.set_xlabel("epoch")
            ax.set_ylabel(cf.goodness_str)
            ax.set_title(fr"$\mathbf{{{cf.goodness_str}}}$, train= {metrics[f'train_{metric5}'][-1]:.2f}, val={metrics[f'val_{metric5}'][-1]:.2f}", loc='left')
            lower_limit = 0 if cf.task == "regression" else 0.49
            ax.set_ylim(lower_limit, 1.01)
        panels.append(_draw_goodness)

    if "z" in enabled and is_tetriscnn:
        z = np.array([metrics[f'z_{k}'] for k in range(len(cf.kernels))])  # (no_kernels, epochs)

        def _draw_z(ax):
            for i in range(len(cf.kernels)):
                ax.plot(np.abs(z[i]), label=branch_label(cf, i), color=branch_color(cf, i))
            if "learning_rate" in metrics.keys():
                ax.plot(metrics["learning_rate"], label="LR", color="black", linestyle='--', alpha=0.75)
            legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
            for text, line in zip(legend.get_texts(), legend.get_lines()):
                text.set_color(line.get_color())
            ax.set_xlabel("epoch")
            ax.set_ylabel("abs(z)")
            ax.set_title("bottleneck activations")
            ax.set_yscale("log")
        panels.append(_draw_z)

    if "pt" in enabled and cf.task == "regression" and "pt" in metrics:
        def _draw_pt(ax):
            pt = np.array(metrics["pt"])  # (epochs, 2): [deriv, peak], see train.py
            ax.plot(pt[:, 0], label="deriv", color="magenta")
            ax.plot(pt[:, 1], label=f"peak ({cf.label_param})", color="darkorange")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.set_xlabel("epoch")
            ax.set_ylabel("phase indicator")
            ax.set_title("phase indicator (val)", loc='left')
        panels.append(_draw_pt)

    if not panels:
        print(f"[plot_history] cf.enabled_metrics={enabled!r} selects nothing plottable "
              f"for this run (model={cf.model}, task={cf.task}); skipping {file_name}.")
        return

    ncols = 2 if len(panels) > 1 else 1
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 4.5 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    for ax, draw in zip(flat_axes, panels):
        draw(ax)
        ax.grid(True)
    for ax in flat_axes[len(panels):]:  # unused cells in the last row, if any
        ax.axis("off")

    plt.tight_layout()
    if show_dont_save:
        plt.show()
    else:
        save_fig(fig, cf.logdir, file_name)
    plt.close()


def plot_lambda(all_metrics, cf, file_name, enabled_metrics=None):
    """
    Plot the effect of lambda on validation metrics and activations.
    Extensible: new metrics can be added via `metrics_to_plot`.
    Parameters:
        all_metrics: dict of dicts, where each key is a lambda value.
        cf: configuration object with experiment settings.
        file_name: name of the file to save the plot.
        enabled_metrics: dict of metric names to plot; if None, plot all predefined metrics, 
                            e.g. enabled_metrics = {"z", "loss", "pt"}
    """
    lambdas = np.array(cf.lambdas)

    print(f"Plotting lambda for {len(lambdas)} values: {lambdas}")
    def mean_ste(arr):
        arr = np.array(arr)
        return arr.mean(axis=0), arr.std(axis=0, ddof=1) / np.sqrt(len(arr))

    # --- aggregate results ---
    results = {lam: {} for lam in lambdas}

    for lam in lambdas:
        if lam != 1.0:
            lam = int(lam)
        lam_str = f"{lam}"
        print(f"all_metrics[lam_str].keys() = {all_metrics[lam_str].keys()}")
        # z special handling
        temp = np.array(all_metrics[lam_str])
        print(f"temp: {temp}")
        print(f"lambda={lam}: z before normalization = {temp}")
        if cf.experiment_name in {"lambdamax", "singlerun"}:
            z = np.abs(all_metrics[lam_str]["z"])
            z = z / z.max(axis=1, keepdims=True) 
        else:
            z = np.array(all_metrics[lam_str]["z"])
        results[lam]["z"], results[lam]["z_ste"] = mean_ste(z)

        # goodness
        g_mean, g_ste = mean_ste(all_metrics[lam_str][f"val_{cf.goodness_str}"])
        results[lam]["goodness"], results[lam]["goodness_ste"] = g_mean, g_ste

        # loss
        l_mean, l_ste = mean_ste(all_metrics[lam_str][f"val_{cf.loss_str}"])
        results[lam]["loss"], results[lam]["loss_ste"] = l_mean, l_ste

        # phase indicator + epochs
        for key in ["pt", "pt2", "epochs", "net2_norm"]:
            if key in all_metrics[lam_str]:  # only aggregate if present
                if key == "pt" or key == "pt2":
                    if cf.task == "regression":
                        deriv = np.array([x for x, _ in all_metrics[lam_str][key]]) 
                        peak = np.array([y for _, y in all_metrics[lam_str][key]]) 

                        deriv, deriv_s = mean_ste(deriv) # deriv
                        results[lam][f"{key}_deriv"], results[lam][f"{key}_deriv_ste"] = deriv, deriv_s
                    elif cf.task == "lbc":
                        peak = np.array([y for y in all_metrics[lam_str][key]]) 

                    m, s = mean_ste(peak) # peak
                    results[lam][key], results[lam][f"{key}_ste"] = m, s
                else:
                    print(f"Aggregating {key} for lambda={lam}")
                    m, s = mean_ste(all_metrics[lam_str][key])
                    results[lam][key], results[lam][f"{key}_ste"] = m, s

        # regression-specific
        if cf.task == "regression":
            a_mean, a_ste = mean_ste(all_metrics[lam_str]["val_r2agg"])
            results[lam]["r2agg"], results[lam]["r2agg_ste"] = a_mean, a_ste

    # convert to arrays for plotting
    def stack(metric):
        try:
            return np.array([results[lam][metric] for lam in lambdas])
        except KeyError: # takes into account the case when pt2 is not included
            return None

    if cf.label_param == 'deltaomega':
        pt_label = rf'$\delta$'
        pt_label2 = rf'$\Omega$'
        deriv_label1 = rf'$d \hat \delta/d \delta$'
        deriv_label2 = rf'$d \hat \Omega/d \Omega$'
    elif cf.label_param == 'delta':
        pt_label = rf'$\delta$'
        pt_label2 = '-'
        deriv_label1 = rf'$d \hat \delta/d \delta$'
    elif cf.label_param == 'omega':
        pt_label = rf'$\Omega$'
        pt_label2 = '-'
        deriv_label1 = rf'$d \hat \Omega/d \Omega$'
    elif cf.label_param == 't':
        pt_label = rf'$t$'
        pt_label2 = '-'
        deriv_label1 = rf'$d \hat t/d t$'
    
    # --- define metrics to plot ---
    metrics_to_plot = [
        dict(name="z", ylabel="z", title="selected branch activations", per_kernel=True),
        dict(name="goodness", ylabel=f"{cf.goodness_str}", title=f"validation {cf.goodness_str}"),
        dict(name="loss", ylabel=cf.loss_str, title=f"validation {cf.loss_str}", log=True),
        dict(name="pt", ylabel=pt_label, title=f"phase indicator {pt_label}", color="magenta"),
        dict(name="pt_deriv", ylabel=deriv_label1, title=deriv_label1),
        dict(name="pt_deriv_ste", ylabel=f"{deriv_label1} (ste)", title=f"{deriv_label1} (ste)"),
        dict(name="net2_norm", ylabel="net2_norm", title="net2_norm", color="black"),
        dict(name="epochs", ylabel="epochs", title="epochs", color="black"),
    ]

    if cf.label_param == "deltaomega":
        metrics_to_plot.extend([
            dict(name="pt2", ylabel=pt_label2, title=f"phase indicator {pt_label2}", color="orange"),
            dict(name="pt2_deriv", ylabel=deriv_label2, title=deriv_label2),
            dict(name="pt2_deriv_ste", ylabel=f"{deriv_label2} (ste)", title=f"{deriv_label2} (ste)"),
        ])

    
    # select a subset of all metrics to plot for presentation purposes
    if enabled_metrics:
        active_metrics = [m for m in metrics_to_plot if m["name"] in enabled_metrics]
    else:
        active_metrics = metrics_to_plot
    print(f"ACTIVE METRICS FOR LAMBDA PLOTTING: {[m['name'] for m in active_metrics]}")
    # --- plotting ---
    fig, ax = plt.subplots(len(active_metrics), 1, figsize=(9, 3+(len(active_metrics)-1)*2 ), sharex=True,
                           gridspec_kw={'height_ratios': [5] + [2]*(len(active_metrics)-1)})

    norm_pt = None      # for (pt_deriv, pt_deriv_ste)
    norm_pt2 = None     # for (pt2_deriv, pt2_deriv_ste)

    for i, m in enumerate(active_metrics):
        mean, ste = stack(m["name"]), stack(m["name"] + "_ste")
        if mean is None:
            print(f"Skipping plot for {m['name']} as no data is available, index {i}")
            ax[i].axis("off")
            continue
        else:
            mean = mean.squeeze()
            if ste is not None:
                ste = ste.squeeze()
        if m.get("per_kernel"):  # z special handling
            for k in range(mean.shape[1]):
                ax[i].plot(lambdas, mean[:, k], marker="o", label=branch_label(cf, k), color=branch_color(cf, k))
                if ste is not None:
                    ax[i].fill_between(lambdas, mean[:, k] - ste[:, k], mean[:, k] + ste[:, k], alpha=.3)
            legend = ax[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2, fontsize=8)
            for text, line in zip(legend.get_texts(), legend.get_lines()):
                text.set_color(line.get_color())

        elif m.get("name") in ["pt_deriv", "pt2_deriv", "pt_deriv_ste", "pt2_deriv_ste"]:

            if m["name"] in ["pt_deriv", "pt_deriv_ste"]:

                if norm_pt is None:  # first time we see this pair → set scale
                    norm_pt = Normalize(vmin=mean.min(), vmax=mean.max())
                norm = norm_pt

            elif m["name"] in ["pt2_deriv", "pt2_deriv_ste"]:

                if norm_pt2 is None:
                    norm_pt2 = Normalize(vmin=mean.min(), vmax=mean.max())
                norm = norm_pt2

            im = ax[i].imshow(mean.T, aspect='auto', cmap='viridis',
                            interpolation='nearest',
                            extent=[lambdas[0], lambdas[-1], 0, 1],
                            origin="lower",
                            norm=norm)

            divider = make_axes_locatable(ax[i])
            cax = divider.append_axes("right", size="3%", pad=0.05)
            cbar = fig.colorbar(ax[i].images[0], cax=cax)
            cbar.set_label(deriv_label1)
            
 
            # ylabs  = np.array(unique_labels)[idx]
            # turn ticks off
            ax[i].set_yticks([])
            ax[i].grid(False, which="both")

        else:
            ax[i].plot(lambdas, mean, marker="o", color=m.get("color", "blue"))
            ax[i].fill_between(lambdas, mean - ste, mean + ste, alpha=.3, color=m.get("color", "blue"))
            ax[i].grid(True)
        ax[i].set_title(m["title"], loc="left")
        ax[i].set_ylabel(m["ylabel"])
        if m.get("log"):
            ax[i].set_yscale("log")

    ax[-1].set_xlabel(r"$\lambda_{\max}$")
    ax[-1].set_xticks(lambdas)
    fig.suptitle(f"{cf.experiment_name} experiment: {len(cf.seeds)} seeds, {len(cf.kernels)} kernels")
    plt.tight_layout()
    save_fig(fig, cf.logdir, file_name)
    plt.close()


def plot_lbc(metrics_per_seed, cf, file_name, enabled_metrics=None, show=False):
    """
    Plot Learning by Confusion (LBC) results aggregated over seeds.
    """
    print(f"\nPLOTTING LBC with metrics_per_seed keys: {metrics_per_seed.keys()}")
    if cf.remake_lambda_plot:
        unique_labels = parse_array_field(cf.unique_labels)
    else:
        unique_labels = np.array(cf.val_dataset.processor.unique_labels)
        
    partitions = (unique_labels[:-1] + unique_labels[1:]) / 2

    metrics_to_plot = [
        dict(name="z", ylabel="abs(z)", title="final bottleneck values", per_kernel=True),
        dict(name="acc", ylabel="accuracy", title="acc"),
        dict(name="loss", ylabel="loss", title="loss", log=True),
    ]
    if enabled_metrics:
        active_metrics = [m for m in metrics_to_plot if m["name"] in enabled_metrics]
    else:
        active_metrics = metrics_to_plot

    fig, ax = plt.subplots(
        len(active_metrics), 1,
        figsize=(9, 3 + (len(active_metrics) - 1) * 2),
        sharex=True,
        gridspec_kw={'height_ratios': [5] + [2] * (len(active_metrics) - 1)}
    )
    if len(active_metrics) == 1:
        ax = [ax]

    seeds = list(metrics_per_seed.keys())
    for i, m in enumerate(active_metrics):
        metric_name = m["name"]

        # collect data across seeds
        def stack(metric_key):
            arrs = []
            for seed in seeds:
                arrs.append(np.array(metrics_per_seed[seed][metric_key]))
            return np.stack(arrs)  # shape: (n_seeds, n_partitions, ...)
        
        if metric_name == "z":
            data = stack("z")  # (seeds, partitions, kernels) or (seeds, partitions)
            if data.ndim == 3:  # multiple kernels
                for k in range(data.shape[2]):
                    mean = np.mean(np.abs(data[:, :, k]), axis=0)
                    std = np.std(np.abs(data[:, :, k]), axis=0)
                    ax[i].plot(partitions, mean, marker="o", linestyle="-", markersize=4,
                               label=branch_label(cf, k), color=branch_color(cf, k))
                    ax[i].fill_between(partitions, mean - std, mean + std, alpha=0.2)
            else:  # single kernel
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                ax[i].plot(partitions, mean, marker="o", linestyle="-", markersize=4)
                ax[i].fill_between(partitions, mean - std, mean + std, alpha=0.2)

            ax[i].set_yscale("log")

        elif metric_name == "acc":
            for split, color in [("train_acc", "red"), ("val_acc", "blue")]:
                data = stack(split)  # (seeds, partitions)
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                accs = mean
                ax[i].plot(partitions, mean, marker="o", linestyle="-", markersize=4, label=split.split("_")[0], color=color)
                ax[i].fill_between(partitions, mean - std, mean + std, color=color, alpha=0.2)

        elif metric_name == "loss":
            for split, color in [("train_CEL", "red"), ("val_CEL", "blue")]:
                data = stack(split)
                mean = np.mean(data, axis=0)
                std = np.std(data, axis=0)
                ax[i].plot(partitions, mean, marker="o", linestyle="-", markersize=4, label=split.split("_")[0], color=color)
                ax[i].fill_between(partitions, mean - std, mean + std, color=color, alpha=0.2)
            if m.get("log"):
                ax[i].set_yscale("log")

        ax[i].set_ylabel(m["ylabel"])
        ax[i].set_title(m["title"], loc="left")
        ax[i].grid(True)


    tc, peak_id = find_lbc_transition(accs, partitions)
    print(f"Identified transition point at tc={tc:.2f}, partition index={peak_id}, acc={accs[peak_id]:.4f}")

    new_metrics_per_seed = {"z":[], f"val_{cf.loss_str}":[], f"train_{cf.loss_str}":[],
                            f"val_{cf.goodness_str}":[], f"train_{cf.goodness_str}":[], "pt":[]}
    for seed in seeds:
        metrics_per_partition = metrics_per_seed[seed]
        
        if peak_id == 0:
            for key in metrics_per_partition.keys():
                new_metrics_per_seed[key].append( 0 )
            new_metrics_per_seed["pt"].append( 0 )
        else:
            for key in metrics_per_partition.keys():
                new_metrics_per_seed[key].append( metrics_per_partition[key][peak_id] )
            new_metrics_per_seed["pt"].append( partitions[peak_id] )

    # plot a vertical line for tc in all subplots
    for i, m in enumerate(active_metrics):
        ax[i].axvline(x=tc, color="black", linestyle="--", label=fr"$t_c=${tc:.2f}")

        if m["name"] == "z":
            legend = ax[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=2, fontsize=8)
            for text, line in zip(legend.get_texts(), legend.get_lines()):
                text.set_color(line.get_color())
        elif m["name"] == "acc":
            ax[i].legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")

    # # obtain the values of the metrics at the transition point

    ax[-1].set_xlabel(r'$t$')
    ax[-1].tick_params(axis='x', rotation=45)
    ax[-1].set_xticks(partitions)

    fig.suptitle(rf"learning by confusion: {cf.dataset}, {f'{len(cf.kernels)} kernels' if 'kernels' in cf.keys() else ''}, {len(cf.seeds)} seeds, $\lambda_\max={cf.lam}$")
    plt.tight_layout()
    save_fig(fig, cf.logdir, file_name)
    with open(Path(cf.logdir) / f"{file_name.split('.')[0]}_plot_data_new_metrics_per_seed.pkl", "wb") as f:
        pickle.dump(new_metrics_per_seed, f)
    with open(Path(cf.logdir) / f"{file_name.split('.')[0]}_plot_data_metrics_per_seed.pkl", "wb") as f:
        pickle.dump(metrics_per_seed, f)
    if show:
        plt.show()
    plt.close()
    return new_metrics_per_seed
