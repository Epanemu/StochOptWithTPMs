import itertools
import logging
import os
import time
from typing import Any, List, Tuple, cast

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from stochopt.data.DataHandler import DataHandler
from stochopt.data.Features import Binary, Categorical, Contiguous
from stochopt.data.Types import DataLike

# Import TPM trainers
try:
    from stochopt.tpms.cnet_tpm import CNetTPM
    from stochopt.tpms.spn_tpm import SpnTPM
    from stochopt.tpms.tpm import TPM
    from stochopt.tpms.TreeTPM.tree_tpm import TreeTPM
except ImportError:
    logging.warning("Could not import TPM modules. TPM training will fail.")

try:
    from stochopt.tpms.nn_pm import NNPM
except ImportError:
    logging.warning("Could not import NNPM. NN PM training will fail.")

log = logging.getLogger(__name__)


def train_tpm(cfg: DictConfig, data: DataLike, data_handler: DataHandler) -> TPM:
    """
    Train a TPM (SPN, CNet, or Tree) on the provided data.
    """
    tpm_cfg = cfg.method
    tpm_name = tpm_cfg.name

    encoded_data = data_handler.encode(data, normalize=False, one_hot=False)

    if tpm_name == "spn":
        log.info("Training SPN...")
        tpm: TPM = SpnTPM(data_handler)
        tpm.train(
            encoded_data,
            min_instances_slice=tpm_cfg.min_instances_slice,
            n_clusters=tpm_cfg.n_clusters,
        )
        return tpm

    elif tpm_name == "cnet":
        log.info("Training CNet...")
        tpm = CNetTPM(data_handler)
        # Extract CNet specific params
        tpm.train(
            encoded_data,
            min_instances_slice=tpm_cfg.min_instances_slice,
            max_depth=tpm_cfg.max_depth,
            discretization_method=tpm_cfg.discretization_method,
            n_bins=tpm_cfg.n_bins,
        )
        return tpm

    elif tpm_name == "tree":
        log.info(f"Training TreeTPM (Learner: {tpm_cfg.learner})...")
        tpm = TreeTPM(data_handler)
        if tpm_cfg.learner == "greedy":
            tpm.train_greedy_top_down(
                encoded_data,
                min_samples=tpm_cfg.min_samples,
                max_depth=tpm_cfg.max_depth,
                val_ratio=tpm_cfg.val_ratio,
                alpha=tpm_cfg.alpha,
                max_branches=tpm_cfg.max_branches,
            )
        elif tpm_cfg.learner == "cnet":
            tpm.train(
                encoded_data,
                min_instances_slice=tpm_cfg.min_instances_slice,
                max_depth=tpm_cfg.max_depth,
                n_bins=tpm_cfg.n_bins,
            )
        else:
            raise ValueError(f"Unknown Tree learner: {tpm_cfg.learner}")
        return tpm

    else:
        raise ValueError(f"Unknown TPM name: {tpm_name}")


def _plot_empirical_pairplot(
    tpm_data: np.ndarray,
    feat_names: list[str],
    data_handler: DataHandler,
    title: str = "Empirical distribution pairplot (sat=1)",
    n_bins: int = 30,
) -> None:
    """
    Log a pairplot of empirical frequencies for TPM training data (sat=1 rows only)
    to MLflow.

    Off-diagonal cells show 2-D heatmaps of counts; diagonal cells show 1-D histograms.
    """
    sat_idx = feat_names.index("sat")
    sat_mask = tpm_data[:, sat_idx] == 1
    data = tpm_data[sat_mask]

    all_plot_names = [nm for nm in feat_names if nm != "sat"]
    all_plot_cols = [i for i, nm in enumerate(feat_names) if nm != "sat"]

    # Limit to 10 dimensions for readability
    plot_names = all_plot_names[:10]
    plot_cols = all_plot_cols[:10]

    data = data[:, plot_cols]
    n = len(plot_names)
    cell_size = 4.0
    dpi = 100

    fig, axes = plt.subplots(
        n,
        n,
        figsize=(cell_size * n, cell_size * n),
        squeeze=False,
    )

    for i in range(n):
        xi = data[:, i]
        feat_i = data_handler.features[i]
        if isinstance(feat_i, Contiguous) and feat_i.discrete:
            bins_i = np.arange(feat_i.bounds[0], feat_i.bounds[1] + 1) - 0.5
        else:
            bins_i = np.linspace(xi.min(), xi.max(), n_bins + 1)
        for j in range(n):
            ax = axes[i, j]
            xj = data[:, j]
            feat_j = data_handler.features[j]
            if isinstance(feat_j, Contiguous) and feat_j.discrete:
                bins_j = np.arange(feat_j.bounds[0], feat_j.bounds[1] + 1) - 0.5
            else:
                bins_j = np.linspace(xj.min(), xj.max(), n_bins + 1)
            if i == j:
                ax.hist(xi, bins=bins_i, color="steelblue", edgecolor="none")
            else:
                h, xedges, yedges = np.histogram2d(
                    xj, xi, bins=cast(Any, [bins_j, bins_i])
                )
                ax.imshow(
                    h.T,
                    origin="lower",
                    aspect="auto",
                    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                    cmap="viridis",
                )
            # Outer-edge axis labels; hide tick labels on inner cells.
            if j == 0:
                ax.set_ylabel(
                    plot_names[i],
                    fontsize=14,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=10,
                )
            else:
                ax.set_yticklabels([])
            if i == n - 1:
                ax.set_xlabel(plot_names[j], fontsize=14)
            else:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=10)

    fig.suptitle(title, fontsize=18)
    fig.tight_layout()
    mlflow.log_figure(fig, "tpm_empirical_pairplot.png", save_kwargs={"dpi": dpi})
    plt.close(fig)


def _get_tpm_grid_distribution(
    data_handler: DataHandler,
    tpm: TPM,
    active_cols: List[int],
    sat_val: float = 1.0,
    n_bins: int = 30,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Generate a full joint grid for the active columns, evaluate the TPM
    (with sat=sat_val), and return the grid points, their volumes (for density),
    and the probabilities.
    """
    feat_names = data_handler.feature_names
    sat_idx = feat_names.index("sat")

    feature_vals = []
    feature_bins = []
    for col in active_cols:
        feat = data_handler.features[col]
        if isinstance(feat, Contiguous):
            if feat.discrete:
                vals = np.arange(feat.bounds[0], feat.bounds[1] + 1).astype(float)
                bins = vals - 0.5
                bins = np.append(bins, bins[-1] + 1)
            else:
                edges = np.linspace(feat.bounds[0], feat.bounds[1], n_bins + 1)
                vals = (edges[:-1] + edges[1:]) / 2
                bins = edges
        elif isinstance(feat, (Categorical, Binary)):
            vals = np.arange(len(feat.orig_vals)).astype(float)
            bins = np.append(vals - 0.5, vals[-1] + 0.5)
        else:
            raise ValueError(f"Unknown feature type: {type(feat)}")
        feature_vals.append(vals)
        feature_bins.append(bins)

    # Calculate grid size
    grid_shape = [len(v) for v in feature_vals]
    total_points = np.prod(grid_shape)
    grid_size_str = f"joint grid of size {total_points} (active vars: {active_cols})"
    log.info(f"Evaluating TPM on {grid_size_str}")

    # Generate full grid
    # Using np.indices and broadcasting might be faster for large grids?
    # But itertools is easier to read.
    grid_points = np.array(list(itertools.product(*feature_vals)))

    # Evaluate TPM
    # We must pad with None for marginalized vars and set sat=sat_val
    probs = []
    for p in grid_points:
        sample: list[float | None] = [None] * data_handler.n_features
        for i, col in enumerate(active_cols):
            sample[col] = p[i]
        sample[sat_idx] = sat_val
        probs.append(tpm.log_probability(np.array(sample, dtype=object)))

    np_probs = np.array(probs).reshape(grid_shape)
    return np_probs, feature_vals, feature_bins


def _plot_tpm_pairplot(
    data_handler: DataHandler,
    tpm: TPM,
    title: str = "TPM modeled log-probability pairplot (sat=1)",
    n_bins: int = 30,
) -> None:
    """
    Log a pairplot of TPM modeled log-probabilities (with sat=1) to MLflow.
    Evaluates the TPM on the full joint grid.
    """
    feat_names = data_handler.feature_names
    all_plot_names = [nm for nm in feat_names if nm != "sat"]
    all_plot_cols = [i for i, nm in enumerate(feat_names) if nm != "sat"]

    # Limit to 10 dimensions for readability
    plot_names = all_plot_names[:10]
    plot_cols = all_plot_cols[:10]
    n = len(plot_names)
    cell_size = 8.0 if n == 1 else 4.0
    dpi = 100

    # 1. Get full joint log-density on grid: log f(x, sat=1)
    log_density, feat_vals, feat_bins = _get_tpm_grid_distribution(
        data_handler, tpm, plot_cols, sat_val=1.0, n_bins=n_bins
    )
    joint_density = np.exp(log_density)

    # Precompute bin widths for all dimensions
    widths = []
    for idx_in_plot, col in enumerate(plot_cols):
        feat = data_handler.features[col]
        is_cont = isinstance(feat, Contiguous) and not feat.discrete
        widths.append(
            np.diff(feat_bins[idx_in_plot])
            if is_cont
            else np.ones(len(feat_vals[idx_in_plot]))
        )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("red")

    # Use constrained_layout for automatic margin/colorbar management
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(cell_size * n, cell_size * n),
        squeeze=False,
        layout="constrained",
    )

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            # Ensure consistent axis limits for each feature
            ax.set_xlim(feat_bins[j][0], feat_bins[j][-1])
            if i != j:
                ax.set_ylim(feat_bins[i][0], feat_bins[i][-1])

            if i == j:
                # 1D Marginal Density: integrate out all axes except i
                # f(x_i) = sum_{k != i} f(x) * prod_{k != i} delta_x_k
                axes_to_integrate = [k for k in range(n) if k != i]
                f_integrand = joint_density
                for k in axes_to_integrate:
                    w = widths[k]
                    w_shape = [1] * n
                    w_shape[k] = len(w)
                    f_integrand = f_integrand * w.reshape(w_shape)

                marginal_density = np.sum(f_integrand, axis=tuple(axes_to_integrate))
                ax.bar(
                    feat_bins[i][:-1],
                    marginal_density,
                    width=np.diff(feat_bins[i]),
                    align="edge",
                    color="steelblue",
                    edgecolor="none",
                )
            else:
                # 2D Marginal Density: integrate out all axes except i and j
                # f(x_i, x_j) = sum_{k != i,j} f(x) * prod_{k != i,j} delta_x_k
                axes_to_integrate = [k for k in range(n) if k not in [i, j]]
                f_integrand = joint_density
                for k in axes_to_integrate:
                    w = widths[k]
                    w_shape = [1] * n
                    w_shape[k] = len(w)
                    f_integrand = f_integrand * w.reshape(w_shape)

                marginal_density_2d = np.sum(f_integrand, axis=tuple(axes_to_integrate))
                # Transpose handling: imshow(M) -> Y=axis0, X=axis1
                plot_m = marginal_density_2d if i < j else marginal_density_2d.T

                log_density_2d = np.log(plot_m + 1e-12)
                masked_m = np.ma.masked_where(plot_m <= 1e-12, log_density_2d)
                ax.imshow(
                    masked_m,
                    origin="lower",
                    aspect="auto",
                    extent=[
                        feat_bins[j][0],
                        feat_bins[j][-1],
                        feat_bins[i][0],
                        feat_bins[i][-1],
                    ],
                    cmap=cmap,
                )

    # Styling and labeling
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            # Ticks on all sides
            ax.tick_params(
                labelleft=(j == 0),
                labelright=(j == n - 1),
                labelbottom=(i == n - 1),
                labeltop=(i == 0),
                left=True,
                right=True,
                bottom=True,
                top=True,
                labelsize=10,
            )

            # Left side labels (Feature name or Mass for diagonal)
            if j == 0:
                y_lbl = "Marginal Density" if i == j else plot_names[i]
                ax.set_ylabel(y_lbl, fontsize=14, rotation=90, labelpad=15)

            # Right side labels - use figure coordinates to avoid n-dependence
            if j == n - 1:
                y_lbl = "Marginal Density" if i == j else plot_names[i]
                # Right labels: constrained_layout will make room for this text!
                ax.text(
                    1.2,
                    0.5,
                    y_lbl,
                    transform=ax.transAxes,
                    fontsize=14,
                    rotation=270,
                    ha="left",
                    va="center",
                )

            # X labels
            if i == n - 1:
                ax.set_xlabel(plot_names[j], fontsize=14)
            if i == 0:
                ax.set_title(plot_names[j], fontsize=14, pad=20)

    # Shared colorbar for all 2D plots
    im = None
    for i in range(n):
        for j in range(n):
            if i != j:
                images = axes[i, j].get_images()
                if images:
                    im = images[0]
                    break
        if im:
            break

    if im:
        # Automatic placement via constrained_layout
        fig.colorbar(im, ax=axes, location="right", format="%.2g", shrink=0.8)

    fig.suptitle(title, fontsize=18)
    mlflow.log_figure(fig, "tpm_modeled_pairplot.png", save_kwargs={"dpi": dpi})
    plt.close(fig)


def _plot_marginal_conditional_pairplot(
    data_handler: DataHandler,
    tpm: TPM,
    log_px: float,
    risk_level: float,
    active_cols: List[int],
    title_prefix: str = "Marginalized",
    n_bins: int = 30,
) -> None:
    """
    Grid-based marginalized conditional pairplots.
    """
    feat_names = data_handler.feature_names
    # Limit to 10 dimensions for readability
    active_cols = active_cols[:10]
    plot_names = [feat_names[col] for col in active_cols]
    n = len(plot_names)
    cell_size = 8.0 if n == 1 else 4.0
    dpi = 100

    # 1. Get joint density of active columns (sat=1)
    log_density_joint, feat_vals, feat_bins = _get_tpm_grid_distribution(
        data_handler, tpm, active_cols, sat_val=1.0, n_bins=n_bins
    )
    joint_density = np.exp(log_density_joint)

    # Precompute widths
    widths = []
    for i, col in enumerate(active_cols):
        feat = data_handler.features[col]
        is_cont = isinstance(feat, Contiguous) and not feat.discrete
        widths.append(np.diff(feat_bins[i]) if is_cont else np.ones(len(feat_vals[i])))

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("red")

    for plot_type in ["log_joint", "conditional_prob"]:
        # Use constrained_layout for automatic margin/colorbar management
        fig, axes = plt.subplots(
            n,
            n,
            figsize=(cell_size * n, cell_size * n),
            squeeze=False,
            layout="constrained",
        )

        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                # Consistent axis limits
                ax.set_xlim(feat_bins[j][0], feat_bins[j][-1])
                if i != j:
                    ax.set_ylim(feat_bins[i][0], feat_bins[i][-1])

                if i == j:
                    # Marginalize to 1D
                    axes_to_integrate_1d = [k for k in range(n) if k != i]
                    f_integrand = joint_density
                    for k in axes_to_integrate_1d:
                        w = widths[k]
                        w_shape = [1] * n
                        w_shape[k] = len(w)
                        f_integrand = f_integrand * w.reshape(w_shape)

                    m1d_density = np.sum(f_integrand, axis=tuple(axes_to_integrate_1d))

                    if plot_type == "conditional_prob":
                        # P(sat=1 | x_i) = f(x_i, sat=1) / f(x_i)
                        # f_x_i is the marginal density of x_i
                        vals = m1d_density / np.exp(log_px)
                        ax.bar(
                            feat_bins[i][:-1],
                            vals,
                            width=np.diff(feat_bins[i]),
                            align="edge",
                            color="steelblue",
                        )
                        ax.axhline(
                            1 - risk_level, color="red", linestyle="--", alpha=0.7
                        )
                        ax.set_ylim([0, 1.1])
                    else:
                        log_density_1d = np.log(m1d_density + 1e-12)
                        ax.bar(
                            feat_bins[i][:-1],
                            log_density_1d,
                            width=np.diff(feat_bins[i]),
                            align="edge",
                            color="steelblue",
                        )
                else:
                    # Marginalize to 2D
                    axes_to_integrate = [k for k in range(n) if k not in [i, j]]
                    f_integrand = joint_density
                    for k in axes_to_integrate:
                        w = widths[k]
                        w_shape = [1] * n
                        w_shape[k] = len(w)
                        f_integrand = f_integrand * w.reshape(w_shape)

                    m2d_density = np.sum(f_integrand, axis=tuple(axes_to_integrate))
                    plot_m = m2d_density if i < j else m2d_density.T

                    if plot_type == "conditional_prob":
                        grid_v = plot_m / np.exp(log_px)
                        masked_v = np.ma.masked_where(grid_v <= 1e-12, grid_v)
                        ax.imshow(
                            masked_v,
                            origin="lower",
                            aspect="auto",
                            extent=[
                                feat_bins[j][0],
                                feat_bins[j][-1],
                                feat_bins[i][0],
                                feat_bins[i][-1],
                            ],
                            cmap=cmap,
                            vmin=0,
                            vmax=1,
                        )
                    else:
                        log_density_2d = np.log(plot_m + 1e-12)
                        masked_h = np.ma.masked_where(plot_m <= 1e-12, log_density_2d)
                        ax.imshow(
                            masked_h,
                            origin="lower",
                            aspect="auto",
                            extent=[
                                feat_bins[j][0],
                                feat_bins[j][-1],
                                feat_bins[i][0],
                                feat_bins[i][-1],
                            ],
                            cmap=cmap,
                        )

        # Labels and Styling
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                ax.tick_params(
                    labelleft=(j == 0),
                    labelright=(j == n - 1),
                    labelbottom=(i == n - 1),
                    labeltop=(i == 0),
                    left=True,
                    right=True,
                    bottom=True,
                    top=True,
                    labelsize=10,
                )

                if j == 0:
                    cond_prob_m = i == j and plot_type == "conditional_prob"
                    log_joint_m = i == j and plot_type == "log_joint"
                    y_lbl = (
                        "Cond Prob"
                        if cond_prob_m
                        else "Log Density"
                        if log_joint_m
                        else plot_names[i]
                    )
                    ax.set_ylabel(y_lbl, fontsize=14, rotation=90, labelpad=15)

                if j == n - 1:
                    y_lbl = (
                        "Cond Prob"
                        if (i == j and plot_type == "conditional_prob")
                        else "Log Density"
                        if (i == j and plot_type == "log_joint")
                        else plot_names[i]
                    )
                    # Right labels: constrained_layout will make room for this text!
                    ax.text(
                        1.1,
                        0.5,
                        y_lbl,
                        transform=ax.transAxes,
                        fontsize=14,
                        rotation=270,
                        ha="left",
                        va="center",
                    )

                if i == n - 1:
                    ax.set_xlabel(plot_names[j], fontsize=14)
                if i == 0:
                    ax.set_title(plot_names[j], fontsize=14, pad=20)

        # Shared colorbar for all 2D plots
        im = None
        for i_c in range(n):
            for j_c in range(n):
                if i_c != j_c:
                    images = axes[i_c, j_c].get_images()
                    if images:
                        im = images[0]
                        break
            if im:
                break

        if im:
            # Automatic placement via constrained_layout
            fig.colorbar(im, ax=axes, location="right", format="%.2g", shrink=0.8)

        fig.suptitle(f"{title_prefix} {plot_type}", fontsize=18)
        mlflow.log_figure(
            fig, f"tpm_marginal_{plot_type}.png", save_kwargs={"dpi": dpi}
        )
        plt.close(fig)


def _plot_nnpm_conditional_pairplot(
    data_handler: DataHandler,
    nnpm: NNPM,
    title: str = "NNPM predicted probability pairplot",
    n_bins: int = 30,
    plot_names: list[str] | None = None,
) -> None:
    """
    Log a pairplot of NNPM predicted satisfaction probabilities P(sat=1 | x)
    to MLflow. Direct inference of the NN.
    """
    feat_names = data_handler.feature_names
    # We only care about x variables (decisions)
    # Decisions are all but 'sat'
    if plot_names is None:
        plot_names = feat_names[:10]
        plot_cols = list(range(len(plot_names)))
    else:
        plot_names = plot_names[:10]
        plot_cols = [i for i, nm in enumerate(feat_names) if nm in plot_names]
    n = len(plot_names)
    cell_size = 8.0 if n == 1 else 4.0
    dpi = 100

    # We use a simplified grid distribution for the NN, or sampling
    # if too many dimensions
    probs, feat_vals, feat_bins, is_grid = _get_nnpm_grid_probs(
        data_handler, nnpm, plot_cols, n_bins=n_bins
    )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("red")

    fig, axes = plt.subplots(
        n,
        n,
        figsize=(cell_size * n, cell_size * n),
        squeeze=False,
        layout="constrained",
    )

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.set_xlim(feat_bins[j][0], feat_bins[j][-1])
            if i != j:
                ax.set_ylim(feat_bins[i][0], feat_bins[i][-1])

            if i == j:
                if is_grid:
                    # 1D Marginal Prob: mean over all other axes expect i
                    axes_to_mean = [k for k in range(n) if k != i]
                    marginal_prob = np.mean(probs, axis=tuple(axes_to_mean))
                    ax.bar(
                        feat_bins[i][:-1],
                        marginal_prob,
                        width=np.diff(feat_bins[i]),
                        align="edge",
                        color="steelblue",
                        edgecolor="none",
                    )
                else:
                    # In sampling mode, feat_vals[0] is the points array
                    points = feat_vals[0]
                    h_sum, _ = np.histogram(
                        points[:, i], bins=feat_bins[i], weights=probs
                    )
                    h_cnt, _ = np.histogram(points[:, i], bins=feat_bins[i])
                    # Avoid division by zero
                    marginal_prob = np.divide(
                        h_sum, h_cnt, out=np.zeros_like(h_sum), where=h_cnt > 0
                    )
                    ax.bar(
                        feat_bins[i][:-1],
                        marginal_prob,
                        width=np.diff(feat_bins[i]),
                        align="edge",
                        color="steelblue",
                        edgecolor="none",
                    )
                ax.set_ylim([0, 1.1])
            else:
                if is_grid:
                    # 2D Marginal Prob: mean over all axes except i and j
                    axes_to_mean = [k for k in range(n) if k not in [i, j]]
                    marginal_prob_2d = np.mean(probs, axis=tuple(axes_to_mean))
                    # Transpose handling: imshow(M) -> Y=axis0, X=axis1
                    plot_m = marginal_prob_2d if i < j else marginal_prob_2d.T
                else:
                    points = feat_vals[0]
                    h_sum, _, _ = np.histogram2d(
                        points[:, j],
                        points[:, i],
                        bins=[feat_bins[j], feat_bins[i]],
                        weights=probs,
                    )
                    h_cnt, _, _ = np.histogram2d(
                        points[:, j],
                        points[:, i],
                        bins=[feat_bins[j], feat_bins[i]],
                    )
                    plot_m = np.divide(
                        h_sum, h_cnt, out=np.zeros_like(h_sum), where=h_cnt > 0
                    )

                ax.imshow(
                    plot_m,
                    origin="lower",
                    aspect="auto",
                    extent=[
                        feat_bins[j][0],
                        feat_bins[j][-1],
                        feat_bins[i][0],
                        feat_bins[i][-1],
                    ],
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                )

    # Styling and labeling
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            ax.tick_params(
                labelleft=(j == 0),
                labelright=(j == n - 1),
                labelbottom=(i == n - 1),
                labeltop=(i == 0),
                left=True,
                right=True,
                bottom=True,
                top=True,
                labelsize=10,
            )

            if j == 0:
                y_lbl = "P(sat=1)" if i == j else plot_names[i]
                ax.set_ylabel(y_lbl, fontsize=14, rotation=90, labelpad=15)

            if j == n - 1:
                y_lbl = "P(sat=1)" if i == j else plot_names[i]
                ax.text(
                    1.2,
                    0.5,
                    y_lbl,
                    transform=ax.transAxes,
                    fontsize=14,
                    rotation=270,
                    ha="left",
                    va="center",
                )

            if i == n - 1:
                ax.set_xlabel(plot_names[j], fontsize=14)
            if i == 0:
                ax.set_title(plot_names[j], fontsize=14, pad=20)

    # Shared colorbar
    im = None
    for i in range(n):
        for j in range(n):
            if i != j:
                images = axes[i, j].get_images()
                if images:
                    im = images[0]
                    break
        if im:
            break

    if im:
        fig.colorbar(im, ax=axes, location="right", format="%.2g", shrink=0.8)

    fig.suptitle(title, fontsize=18)
    mlflow.log_figure(fig, "nn_conditional_pairplot.png", save_kwargs={"dpi": dpi})
    plt.close(fig)


def _get_nnpm_grid_probs(
    data_handler: DataHandler,
    nnpm: NNPM,
    plot_cols: List[int],
    n_bins: int = 30,
    n_samples: int = 100000,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], bool]:
    """
    Generate P(sat=1 | x) on a joint grid OR via sampling for NNPM.
    Returns (probs, feature_vals_or_points, feature_bins, is_grid).
    """
    # 1. Determine all decision variables (those the NN was trained on)
    # NNPM.n_x is the expected number of features.
    # We assume the NN was trained on the variables provided in the DataHandler.
    # In run_experiment, these were x_names.
    # However, _get_nnpm_grid_probs only gets plot_cols.
    # If len(plot_cols) < nnpm.n_x, we MUST use sampling to marginalize out the others.

    n_plot = len(plot_cols)
    use_sampling = n_plot < nnpm.n_x or (n_plot > 4)  # threshold for joint grid

    # We still need the bins for the variables we ARE plotting
    plot_feat_vals = []
    plot_feat_bins = []
    for col in plot_cols:
        feat = data_handler.features[col]
        if isinstance(feat, Contiguous):
            if feat.discrete:
                vals = np.arange(feat.bounds[0], feat.bounds[1] + 1).astype(float)
                bins = vals - 0.5
                bins = np.append(bins, bins[-1] + 1)
            else:
                edges = np.linspace(feat.bounds[0], feat.bounds[1], n_bins + 1)
                vals = (edges[:-1] + edges[1:]) / 2
                bins = edges
        elif isinstance(feat, (Categorical, Binary)):
            vals = np.arange(len(feat.orig_vals)).astype(float)
            bins = np.append(vals - 0.5, vals[-1] + 0.5)
        else:
            raise ValueError(f"Unknown feature type: {type(feat)}")
        plot_feat_vals.append(vals)
        plot_feat_bins.append(bins)

    if not use_sampling:
        grid_shape = [len(v) for v in plot_feat_vals]
        total_points = np.prod(grid_shape)
        log.info(f"Evaluating NNPM on joint grid of size {total_points}")
        grid_points = np.array(list(itertools.product(*plot_feat_vals)))
        probs = nnpm.predict_prob(grid_points).reshape(grid_shape)
        return probs, plot_feat_vals, plot_feat_bins, True
    else:
        # Sampling approach
        log.info(
            f"Evaluating NNPM via sampling ({n_samples} points) "
            f"across {nnpm.n_x} dimensions"
        )

        # We need to know which features in the DataHandler are the decision variables.
        # Assumption: The NN was trained on a contiguous block of decision variables.
        # In run_experiment, feat_names = xi_names + x_names + [sat_name].
        # We'll assume the decision variables are the ones NOT named 'sat' and NOT in
        # xi_names. Since we don't know xi_names here, we'll assume the last nnpm.n_x
        # features (excluding 'sat') are the decision variables.

        feat_names = data_handler.feature_names
        decision_cols = [i for i, nm in enumerate(feat_names) if nm != "sat"]
        # If there are more than nnpm.n_x, take the last ones (typical for xi + x)
        if len(decision_cols) > nnpm.n_x:
            decision_cols = decision_cols[-nnpm.n_x :]

        # Generate random points for ALL decision variables
        sampled_points = np.zeros((n_samples, nnpm.n_x))
        for i, col in enumerate(decision_cols):
            feat = data_handler.features[col]
            if isinstance(feat, Contiguous):
                sampled_points[:, i] = np.random.uniform(
                    feat.bounds[0], feat.bounds[1], n_samples
                )
            elif isinstance(feat, (Categorical, Binary)):
                sampled_points[:, i] = np.random.choice(len(feat.orig_vals), n_samples)

        probs = nnpm.predict_prob(sampled_points)

        # We return the points projected onto the plot_cols
        # We need to map plot_cols indices to their position in decision_cols
        plot_indices_in_decision = []
        for pc in plot_cols:
            plot_indices_in_decision.append(decision_cols.index(pc))

        projected_points = sampled_points[:, plot_indices_in_decision]

        return probs, [projected_points], plot_feat_bins, False


def run_experiment(cfg: DictConfig) -> None:
    """
    Main experiment execution function.
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run():
        try:
            mlflow.set_tag("mlflow.runName", cfg.mlflow.experiment_name)
            if cfg.method.name == "tree":
                mlflow.set_tag("method", f"tree_{cfg.method.learner}")
            else:
                mlflow.set_tag("method", cfg.method.name)
            mlflow.set_tag(
                "n_products", str(cfg.problem.n_products)
            )  # String for categorical grouping
            mlflow.set_tag("samples.opt", str(cfg.samples.opt))
            mlflow.set_tag("samples.train", str(cfg.samples.train))
            mlflow.set_tag("samples.train_decisions", str(cfg.samples.train_decisions))
            mlflow.set_tag("problem.type", cfg.problem.get("name", "unknown"))
            mlflow.set_tag("problem.distro", cfg.problem.demand_dist)
            mlflow.set_tag("problem.corr", cfg.problem.correlated)
            mlflow.set_tag("status", "RUNNING")

            slurm_job_id = os.environ.get("SLURM_JOB_ID")
            mlflow.set_tag("slurm_job_id", slurm_job_id)

            slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
            slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
            if slurm_array_task_id is not None:
                slurm_id = f"{slurm_array_job_id}_{slurm_array_task_id}"
                # mlflow.set_tag("slurm_array_task_id", slurm_array_task_id)
            else:
                slurm_id = str(slurm_job_id)
            mlflow.set_tag("slurm_id", slurm_id)  # combined, for quick lookup

            # Add date tag for grouping batches of experiments
            import datetime

            today = datetime.datetime.now().strftime("%Y-%m-%d")
            mlflow.set_tag("benchmark_date", today)

            # Log all parameters
            mlflow.log_params(dict(cfg))

            # 1. Instantiate Problem
            log.info(f"Instantiating problem: {cfg.problem._target_}")
            problem = instantiate(cfg.problem, solver=cfg.solver, _convert_="all")

            # 2. Generate Data
            log.info(f"Generating {cfg.samples.train} training samples...")
            train_samples = problem.generate_samples(
                n_samples=cfg.samples.train,
                seed=cfg.seed,
            )

            # 3. Setup DataHandler and TPM / NNPM
            method_type = cfg.method.get("type", cfg.method.name)

            if method_type == "nn_pm":
                # ----- Neural Network PM (not a classical TPM) -----
                log.info("Training NNPM...")
                nn_cfg = cfg.method

                # get working directory of this hydra run
                local_run_dir = HydraConfig.get().runtime.output_dir

                nnpm = NNPM()
                tpm_start_time = time.time()
                nnpm.train(
                    problem,
                    train_samples,
                    epochs=nn_cfg.epochs,
                    batch_size=nn_cfg.batch_size,
                    lr=nn_cfg.lr,
                    hidden_size_factors=nn_cfg.get("hidden_size_factors", None),
                    min_hidden_size=nn_cfg.get("min_hidden_size", 5),
                    max_hidden_size=nn_cfg.get("max_hidden_size", 100),
                    val_size=nn_cfg.get("val_size", 10000),
                    log_every=nn_cfg.get("log_every", 10),
                    seed=cfg.seed,
                    loss_type=nn_cfg.get("loss_type", "bolt"),
                    focal_gamma=nn_cfg.get("focal_gamma", 2.0),
                    folder=local_run_dir,
                )
                tpm_train_duration = time.time() - tpm_start_time
                mlflow.log_metric("tpm_train_duration", tpm_train_duration)

                tpm = None

                # Setup DataHandler for plotting
                xi_names, x_names, sat_name = problem.get_feature_names()
                feat_names = xi_names + x_names + [sat_name]
                categ_map = problem.get_categ_map()
                discrete_features = problem.get_discrete()

                # Generate a small amount of TPM data just for the DataHandler to grasp bounds/types
                tpm_data_mock, _ = problem.generate_tpm_data(
                    n_decisions=cfg.samples.train_decisions,
                    train_samples=train_samples,
                    seed=cfg.seed,
                )
                data_handler = DataHandler(
                    tpm_data_mock,
                    feature_names=feat_names,
                    discrete=discrete_features,
                    categ_map=categ_map,
                )
                tpm_data = None

                # Plotting for NN
                _nn_title = (
                    f"NNPM: {nn_cfg.get('name', 'nn_pm')} | "
                    f"{cfg.problem.get('name', 'unknown')}, {cfg.problem.n_products}d"
                )
                _plot_nnpm_conditional_pairplot(
                    data_handler, nnpm, title=_nn_title, plot_names=x_names
                )

                # Plot the empirical distribution of the training data
                _empirical_title = (
                    f"Empirical: {cfg.problem.get('name', 'unknown')}, {tpm_data_mock.shape[1]-1}D | "
                    f"{cfg.samples.train_decisions} samples (assuming sat=1)"
                )
                _plot_empirical_pairplot(
                    tpm_data_mock, feat_names, data_handler, title=_empirical_title
                )

            elif method_type == "tpm":
                log.info("Generating TPM training data...")
                tpm_data, feat_names = problem.generate_tpm_data(
                    n_decisions=cfg.samples.train_decisions,
                    train_samples=train_samples,
                    cartesian_product=cfg.samples.get("cartesian_product", False),
                    seed=cfg.seed,
                )

                categ_map = problem.get_categ_map()
                discrete_features = problem.get_discrete()

                data_handler = DataHandler(
                    tpm_data,
                    y=None,
                    feature_names=feat_names,
                    discrete=discrete_features,
                    categ_map=categ_map,
                )

                # Train TPM
                log.info("Training TPM...")
                tpm_start_time = time.time()
                tpm = train_tpm(cfg, tpm_data, data_handler)
                tpm_train_duration = time.time() - tpm_start_time
                mlflow.log_metric("tpm_train_duration", tpm_train_duration)

                # evaluate prob of training set and log the mean logprob to mlflow
                log.info("Evaluating TPM on training set...")
                probs = [tpm.log_probability(row) for row in tpm_data]
                mlflow.log_metric("tpm_train_mean_logprob", float(np.mean(probs)))

                # visualize the fit
                _tpm_method = cfg.method.name
                if _tpm_method == "tree":
                    _tpm_method = f"tree/{cfg.method.learner}"
                _problem_name = cfg.problem.get("name", "unknown")
                _n_products = cfg.problem.get("n_products", "?")

                # 1. Empirical Pairplot
                _empirical_title = (
                    f"Empirical: {_problem_name}, {tpm_data.shape[1]-1}D | "
                    f"{cfg.samples.train_decisions} samples (assuming sat=1)"
                )
                _plot_empirical_pairplot(
                    tpm_data, feat_names, data_handler, title=_empirical_title
                )

                # 2. TPM Modeled Pairplot
                _tpm_title = (
                    f"TPM: {_tpm_method} | "
                    f"{_problem_name}, {_n_products}d | "
                    f"n_train={cfg.samples.train_decisions} (sat=1)"
                )
                _plot_tpm_pairplot(data_handler, tpm, title=_tpm_title)

            else:
                tpm = None
                data_handler = None
                tpm_data = None
                tpm_train_duration = 0.0

            # 4. Build and Solve Model
            log.info(f"Building model for method: {cfg.method.name}")

            # For robust/sample average, we need scenarios
            if cfg.method.name in ["robust", "sample_average"]:
                # Use a subset of training samples
                opt_samples = train_samples[: cfg.samples.opt]
            else:
                opt_samples = None

            build_start_time = time.time()

            build_method_name = cfg.method.name
            if method_type == "tpm":
                build_method_name = "tpm"
            elif method_type == "nn_pm":
                build_method_name = "nn_pm"

            problem.build_model(
                method=build_method_name,
                tpm=tpm if method_type != "nn_pm" else nnpm,
                data_handler=data_handler,
                scenarios=opt_samples,
                risk_level=cfg.risk_level,
            )
            build_duration = time.time() - build_start_time
            mlflow.log_metric("build_duration", build_duration)

            if tpm is not None and data_handler is not None:
                log.info("Generating marginalized conditional plots...")
                log.info(f"Uniform log density of x: {problem.x_log_density}")
                log.info(f"Uniform density of x: {np.exp(problem.x_log_density)}")
                # active columns are those not marginalized out (the products)
                # and NOT the sat variable itself
                n_marg = train_samples.shape[1]
                active_cols = list(range(n_marg, data_handler.n_features - 1))
                _plot_marginal_conditional_pairplot(
                    data_handler,
                    tpm,
                    problem.x_log_density,
                    risk_level=cfg.risk_level,
                    active_cols=active_cols,
                    title_prefix=f"Marginalized ({_tpm_method})",
                )

            log.info("Solving model...")
            solve_start_time = time.time()
            result = problem.solve(time_limit=cfg.time_limit)
            solve_duration = time.time() - solve_start_time

            log.info(f"Result: {result}")
            mlflow.log_metric("solve_duration", solve_duration)
            mlflow.log_metric(
                "total_duration", build_duration + solve_duration + tpm_train_duration
            )

            # Log solver status
            solver_status = str(result.get("status", "UNKNOWN"))
            mlflow.set_tag("solver_status", solver_status)

            if result["objective"] is not None:
                mlflow.log_metric("objective", result["objective"])

            # If solver failed (infeasible/unbounded), mark run as failed
            if solver_status.lower() not in ["optimal", "ok", "success"]:
                log.warning(f"Solver status is {solver_status}, marking run as failed.")
                mlflow.set_tag("status", "FAILED")
                mlflow.set_tag("error_type", "SOLVER_FAILURE")
                mlflow.set_tag(
                    "error_message", f"Solver returned status: {solver_status}"
                )
                if result.get("objective") is None:
                    return

            # 5. Evaluation / Verification
            log.info("Evaluating solution...")

            # Get solution vector
            try:
                x_sol = problem.get_solution()
                # Log solution vector to mlflow
                solution_dict = {f"x_{i}": float(x_sol[i]) for i in range(len(x_sol))}
                mlflow.log_dict(solution_dict, "solution.json")
                log.info(f"Solution: {x_sol}")
            except ValueError as e:
                log.warning(f"Could not retrieve solution: {e}")
                mlflow.set_tag("status", "FAILED")
                mlflow.set_tag("error_type", "NO_SOLUTION")
                mlflow.set_tag("error_message", str(e))
                return

            # Validation on new samples
            val_seed = cfg.seed + 1  # Different seed
            n_val = cfg.samples.get("validation", cfg.samples.test)
            validation_samples = problem.generate_samples(
                n_samples=n_val, seed=val_seed
            )

            val_satisfied = problem.check_satisfaction(x_sol, validation_samples)
            val_prob_satisfied = np.mean(val_satisfied)

            log.info(f"Validation Satisfaction Probability: {val_prob_satisfied}")
            mlflow.log_metric("val_prob_satisfied", val_prob_satisfied)
            mlflow.log_metric("val_violation_prob", 1 - val_prob_satisfied)

            # Training set violation (in-sample)
            train_satisfied = problem.check_satisfaction(x_sol, train_samples)
            train_prob_satisfied = np.mean(train_satisfied)
            mlflow.log_metric("train_prob_satisfied", train_prob_satisfied)
            mlflow.log_metric("train_violation_prob", 1 - train_prob_satisfied)

            # If TPM exists, log the probability of the solution
            if method_type == "nn_pm":
                # NNPM: predict P(sat | x) directly
                log.info("Calculating P(satisfied | x_sol) from NNPM...")
                p_sat = float(nnpm.predict_prob(x_sol.reshape(1, -1))[0])
                mlflow.log_metric("nn_prob_satisfied", p_sat)
                log.info(f"P(satisfied | x_sol) from NNPM: {p_sat}")

            elif tpm is not None:
                log.info("Calculating P(satisfied | x_sol) from TPM...")
                # pad x_sol with None for marginalized variables
                # and 1 for the satisfied constraint
                n_marg = tpm.data_handler.n_features - len(x_sol) - 1
                x_sol = np.array([None] * n_marg + list(x_sol) + [1])
                p_x_sol = tpm.log_probability(x_sol)
                mlflow.log_metric(
                    "true_tpm_logprob_satisfied",
                    p_x_sol - problem.x_log_density,
                )
                mlflow.log_metric(
                    "true_tpm_prob_satisfied",
                    np.exp(p_x_sol - problem.x_log_density),
                )
                log.info(
                    "P(satisfied | x_sol) from true TPM: "
                    + f"{np.exp(p_x_sol - problem.x_log_density)}"
                )
                # TODO make this somehow neat? also above, passing cfg is not ideal
                # TODO check also the division by p(x) if not uniform
                p_x_sol_approx = tpm.log_probability_approx(x_sol, **cfg.method)
                mlflow.log_metric(
                    "approx_tpm_logprob_satisfied",
                    p_x_sol_approx - problem.x_log_density,
                )
                mlflow.log_metric(
                    "approx_tpm_prob_satisfied",
                    np.exp(p_x_sol_approx - problem.x_log_density),
                )
                log.info(
                    "P(satisfied | x_sol) from approx TPM: "
                    + f"{np.exp(p_x_sol_approx - problem.x_log_density)}"
                )

            # Mark as successful
            mlflow.set_tag("status", "SUCCESS")
            log.info("Experiment completed successfully")

        except MemoryError as e:
            log.error(f"Out of memory error: {e}")
            mlflow.set_tag("status", "OOM")
            mlflow.set_tag("error_type", "OUT_OF_MEMORY")
            mlflow.set_tag("error_message", str(e))
            raise

        except TimeoutError as e:
            log.error(f"Timeout error: {e}")
            mlflow.set_tag("status", "TIMEOUT")
            mlflow.set_tag("error_type", "TIMEOUT")
            mlflow.set_tag("error_message", str(e))
            raise

        except Exception as e:
            log.error(f"Experiment failed with error: {e}", exc_info=True)
            mlflow.set_tag("status", "FAILED")
            mlflow.set_tag("error_type", type(e).__name__)
            mlflow.set_tag("error_message", str(e))
            # Log full traceback
            import traceback

            mlflow.log_text(traceback.format_exc(), "error_traceback.txt")
            raise
