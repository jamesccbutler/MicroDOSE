# Core
import os
import numpy as np
import pandas as pd
from itertools import combinations
import glob
import re
from matplotlib.lines import Line2D

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from statannotations.Annotator import Annotator

# Scikit-bio for PERMANOVA & DistanceMatrix
from skbio.stats.distance import permanova, DistanceMatrix
from scipy.spatial.distance import pdist, squareform

# Stats & Metrics
from scipy.stats import mannwhitneyu, rankdata
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from matplotlib.patches import Ellipse, Patch
from matplotlib.colors import to_rgb, to_hex
import colorsys
from itertools import combinations

# GPU stuff
import cupy as cp
from cupyx.scipy.sparse.linalg import svds

# Machine Learning
import xgboost as xgb

# Deep Learning
import torch

# Optimization
import optuna
from statannotations.Annotator import Annotator
from matplotlib.cm import get_cmap


def generate_critic_checkpoint_configs(model_path, model_name, exp_name):
    pattern = os.path.join(model_path, f"{model_name}_*_critic_pvalue_reached.pth")
    files = sorted(glob.glob(pattern))

    checkpoint_configs = []

    for file in files:
        match = re.search(rf"{model_name}_(\d+)_critic_pvalue_reached\.pth", file)
        if match:
            epoch = int(match.group(1))
            checkpoint_configs.append({
                "step": epoch,
                "type": "pvalue_reached",
                "exp_name": exp_name
            })

    return checkpoint_configs

def generate_pvalue_checkpoint_configs(model_path, model_name, exp_name):
    pattern = os.path.join(model_path, f"{model_name}_*_generator_pvalue_reached.pth")
    files = sorted(glob.glob(pattern))

    checkpoint_configs = []

    for file in files:
        match = re.search(rf"{model_name}_(\d+)_generator_pvalue_reached\.pth", file)
        if match:
            epoch = int(match.group(1))
            checkpoint_configs.append({
                "step": epoch,
                "type": "pvalue_reached",
                "exp_name": exp_name
            })

    return checkpoint_configs

def ensemble_gan_samples_optimized(generators, num_samples, labels=None, use_ema=True):

    # Generate one batch of noise and labels (shared across all models)
    generator_ref = generators[0]  # Reference model to get latent size
    noise, shared_labels = generator_ref.G.sample_latent(num_samples, labels=labels)

    # Generate all outputs in one pass for all models
    generated_samples = torch.stack([
        (gan.ema.ema_model if use_ema else gan.G)(noise, shared_labels)
        for gan in generators
    ], dim=0)  

    aggregated_samples = torch.mean(generated_samples, dim=0)  # Average across GANs

    return aggregated_samples, shared_labels

def generate_dual_gan_samples(
    gan_path,
    exp_name,
    num_samples,
    set_masks,
    num_channels,
    num_classes,
    feature_size,
    timesteps,
    trainer,
    gen_path,
    label_path,
    score_path,
    mask_path=None,
    batch_size=10000,
    top_k_ratio=0.5,
    device="cuda"
):
    def process_gan_generation(gan_index):
        x_file = f"X_gen_{exp_name}_{num_samples}_{gan_index}"
        y_file = f"y_gen_{exp_name}_{num_samples}_{gan_index}"
        x_gan = x_file
        y_gan = y_file

        gen_masks = torch.load(f'{gan_path}{x_gan}').unsqueeze(1)
        binary_masks = torch.from_numpy(np.where(gen_masks > 0.5, 1, 0))
        gen_labels = torch.tensor(torch.load(f'{gan_path}/{y_gan}'))

        samples = []
        scores = []
        labels = []
        masks_collected = []

        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)
            current_batch_size = end - start

            print(f"\n🚀 GAN {gan_index} - Generating batch {i+1}/{num_batches} ({current_batch_size} samples)")

            if set_masks or num_channels == 2:
                batch_masks = binary_masks[start:end]
                batch_labels = gen_labels[start:end]
            else:
                batch_masks = None
                batch_labels = torch.randint(0, num_classes, (current_batch_size,))

            gen_batch, score_batch = trainer.generate_samples_likelihood(
                num_samples=current_batch_size,
                num_steps=timesteps,
                data_shape=(1, feature_size),
                class_labels=batch_labels.to(device),
                inpaint_masks=batch_masks
            )

            num_top = int(current_batch_size * top_k_ratio)
            top_indices = torch.topk(score_batch, num_top).indices
            score_cutoff = score_batch[top_indices[-1]]
            print(f"✔ Score cutoff: {score_cutoff.item():.4f}")

            keep_mask = score_batch >= score_cutoff

            samples.append(gen_batch[keep_mask])
            scores.append(score_batch[keep_mask])
            labels.append(batch_labels[keep_mask])
            if batch_masks is not None:
                masks_collected.append(batch_masks[keep_mask])

        final_samples = torch.cat(samples, dim=0)
        final_scores = torch.cat(scores, dim=0)
        final_labels = torch.cat(labels, dim=0)
        final_masks = torch.cat(masks_collected, dim=0) if masks_collected else None

        return final_samples, final_labels, final_scores, final_masks

    # Generate from both GANs
    samples1, labels1, scores1, masks1 = process_gan_generation(1)
    samples2, labels2, scores2, masks2 = process_gan_generation(2)

    # Concatenate
    final_samples = torch.cat([samples1, samples2], dim=0)[:num_samples]
    final_labels = torch.cat([labels1, labels2], dim=0)[:num_samples]
    final_scores = torch.cat([scores1, scores2], dim=0)[:num_samples]
    final_masks = torch.cat([masks1, masks2], dim=0)[:num_samples] if (masks1 is not None and masks2 is not None) else None

    # Save
    print(f"\n✅ Final total sample count: {final_samples.shape[0]}")
    torch.save(final_samples, gen_path)
    torch.save(final_labels, label_path)
    torch.save(final_scores, score_path)
    if final_masks is not None and mask_path is not None:
        torch.save(final_masks, mask_path)

def ensemble_critic_score(samples, labels, critics):
    scores = []
    for gan in critics:
        with torch.no_grad():
            score = gan.D(samples.to(gan.device), labels.to(gan.device)).squeeze().cpu()
            scores.append(score)
    mean_score = torch.stack(scores).mean(dim=0)
    return mean_score

def generate_with_critic_ensemble(
    generators, critics, num_desired_samples, 
    top_k_ratio=0.10, use_ema=True, labels=None):
    collected_samples = []
    collected_labels = []
    collected_scores = []

    generator_ref = generators[0]
    nclasses = generator_ref.nclasses
    device = generator_ref.device

    score_cutoff = None
    iteration = 0

    total_collected = 0
    while total_collected < num_desired_samples:
        remaining = num_desired_samples - len(collected_samples)

        # Create labels if not provided
        if labels is None:
            batch_labels = torch.randint(0, nclasses, (remaining,))
        else:
            if isinstance(labels, int):
                batch_labels = torch.full((remaining,), labels)
            else:
                batch_labels = labels[:remaining]
        batch_labels = batch_labels.to(device)

        # Generate samples with your existing ensemble generator
        aggregated_samples, shared_labels = ensemble_gan_samples_optimized(
            generators, remaining, labels=batch_labels, use_ema=use_ema
        )

        # Score using critic ensemble
        critic_scores = ensemble_critic_score(aggregated_samples, shared_labels, critics)

        # Set score cutoff based on first batch
        if score_cutoff is None:
            num_top = int(remaining * top_k_ratio)
            top_indices = torch.topk(critic_scores, num_top).indices
            score_cutoff = critic_scores[top_indices[-1]]
            print(f"✔ Score cutoff set at: {score_cutoff.item():.4f}")

        # Keep samples >= cutoff
        keep_mask = critic_scores >= score_cutoff
        if keep_mask.any():
            kept_samples = aggregated_samples[keep_mask]
            kept_labels = shared_labels[keep_mask]
            kept_scores = critic_scores[keep_mask]

            collected_samples.append(kept_samples)
            collected_labels.append(kept_labels)
            collected_scores.append(kept_scores)

            # Update total
            total_collected += kept_samples.size(0)

        print(f"🔁 Kept {keep_mask.sum().item()} samples")

    if len(collected_samples) == 0:
        raise RuntimeError("❌ No samples met the critic score cutoff.")

    final_samples = torch.cat(collected_samples, dim=0)[:num_desired_samples]
    final_labels = torch.cat(collected_labels, dim=0)[:num_desired_samples]
    final_scores = torch.cat(collected_scores, dim=0)[:num_desired_samples]

    print(f"✅ Finished generation. Final sample count: {final_samples.shape[0]}")
    return final_samples, final_labels, final_scores


# Function to determine significance stars
def get_p_value_star(p_value):
    if p_value < 0.0001:
        return "****"
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"

    
# ✅ Function: Shannon Diversity Calculation
def shannon_diversity(data):
    proportions = data / np.sum(data, axis=1, keepdims=True)
    return -np.sum(proportions * np.log(proportions + 1e-10), axis=1)

# ✅ Custom CLR Function (Handles Zero Values)
def custom_clr(matrix, epsilon=1e-10):
    """
    Compute Centered Log-Ratio (CLR) transformation with zero handling.
    """
    matrix = np.where(matrix == 0, epsilon, matrix)  # Replace zeros with small constant
    geometric_mean = np.exp(np.mean(np.log(matrix), axis=1, keepdims=True))  # Compute geometric mean
    return np.log(matrix / geometric_mean)  # Apply CLR transformation

# Manual normalization
def normalize_to_minus1_1(X):
    X_min = np.min(X)
    X_max = np.max(X)
    X_norm = 2 * (X - X_min) / (X_max - X_min) - 1
    return X_norm, X_min, X_max

# Manual reverse
def reverse_manual_normalize(X_norm, X_min, X_max):
    return ((X_norm + 1) / 2) * (X_max - X_min) + X_min

# ✅ Compute Beta Diversity on GPU
def compute_beta_diversity_gpu(data_combined, already_clr, metric="braycurtis"):
    if metric == "braycurtis":
        distance_matrix = cp.asarray(squareform(pdist(data_combined, metric="braycurtis")))

    elif metric == "jaccard":
        binary_data = (data_combined > 0).astype(float)
        distance_matrix = cp.asarray(squareform(pdist(binary_data, metric="jaccard")))

    elif metric == "aitchison":
        if already_clr == False:
            safe_data = np.where(data_combined <= 0, 1e-10, data_combined)
            clr_transformed = custom_clr(safe_data)
        else:
            clr_transformed = data_combined
        distance_matrix = cp.asarray(squareform(pdist(clr_transformed, metric="euclidean")))

    else:
        raise ValueError("Unknown metric")

    return distance_matrix

def adjust_lightness(color, amount=1.0):
    """Adjust color lightness. amount < 1 is darker, >1 is lighter."""
    c = to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*c)
    new_color = colorsys.hls_to_rgb(h, max(0, min(1, l * amount)), s)
    return to_hex(new_color)

def draw_confidence_ellipse(x, y, ax, n_std=2.0, color='gray', linestyle='-'):
    if len(x) < 2 or len(y) < 2:
        return
    cov = np.cov(x, y)
    if np.linalg.det(cov) == 0:
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse((np.mean(x), np.mean(y)), width, height, angle=theta,
                      facecolor='none', edgecolor=color, linestyle=linestyle,
                      linewidth=1.5, alpha=1.0)
    ax.add_patch(ellipse)

def plot_pcoa_with_violin_split_by_class(pc_df, explained_var, measure, save_path, quality_folder):
    sns.set(style="white", font_scale=1.2)

    # === Set color by Class (blue/red) ===
    unique_classes = sorted(pc_df["Class"].unique()) if "Class" in pc_df.columns else sorted(pc_df["Source"].unique())
    default_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    class_palette = {cls: default_colors[i % len(default_colors)] for i, cls in enumerate(unique_classes)}
    # class_palette = {
    #     "Healthy": "#1f77b4",  # blue
    #     "CRC": "#d62728"       # red
    # }

    # === Set marker shape by Source ===
    source_markers = {
        "Real": "o",
        "Synthetic": "x",
        "Validation": "s"
    }

    # === Update Label column ===
    pc_df["Label"] = pc_df["Source"] + " " + pc_df["Class"]
    labels_order = sorted(pc_df["Label"].unique())
    pairs = list(combinations(labels_order, 2))

    # === Create layout ===
    fig = plt.figure(figsize=(20, 11))
    gs = GridSpec(3, 3, width_ratios=[1.2, 0.2, 6], height_ratios=[6, 0.2, 2],
                  wspace=0.05, hspace=0.05)

    ax_main = fig.add_subplot(gs[0, 2])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_bottom = fig.add_subplot(gs[2, 2], sharex=ax_main)

    # === Scatter plot ===
    for (src, cls), group in pc_df.groupby(["Source", "Class"]):
        label = f"{src} {cls}"
        
        if src == "Synthetic":
            ax_main.scatter(
                group["PC1"], group["PC2"],
                color=class_palette[cls],
                marker='x',
                linewidth=2.0,
                edgecolor='none',
                alpha=1.0,
                s=100,
                label=label
            )
        else:
            ax_main.scatter(
                group["PC1"], group["PC2"],
                color=class_palette[cls],
                marker=source_markers[src],
                edgecolor='k',
                linewidth=0.3,
                alpha=0.9,
                s=60,
                label=label
            )

        linestyle = "--" if src == "Synthetic" else "-"
        draw_confidence_ellipse(group["PC1"].values, group["PC2"].values, ax=ax_main,
                                color=class_palette[cls], linestyle=linestyle)

    ax_main.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)", fontsize=12)
    ax_main.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)", fontsize=12)
    for spine in ax_main.spines.values():
        spine.set_visible(False)

    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()

    # === Left violin (PC2) ===
    violin_left = sns.violinplot(data=pc_df, y="PC2", x="Label",
                                 order=labels_order,
                                 palette=[class_palette[l.split(" ")[1]] for l in labels_order],
                                 ax=ax_left, orient="v", linewidth=1, width=0.45)

    for patch, label in zip(ax_left.collections[::2], labels_order):
        if "Synthetic" in label:
            patch.set_linestyle((0, (5, 5)))
            patch.set_linewidth(2.0)
            patch.set_alpha(0.6)
            patch.set_edgecolor("#555555")

    ax_left.set_ylim(ylim)
    ax_left.set_xlabel("")
    ax_left.set_ylabel("")
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.tick_params(left=False, bottom=False)
    sns.despine(ax=ax_left, left=True, bottom=True)

    annotator_y = Annotator(ax_left, pairs, data=pc_df, x="Label", y="PC2", orient="v", order=labels_order)
    annotator_y.configure(test='Mann-Whitney', text_format='star', loc='outside',
                          line_offset_to_group=0.02, line_height=0.01, text_offset=0.005)
    annotator_y.apply_and_annotate()

    # === Bottom violin (PC1) ===
    violin_bottom = sns.violinplot(data=pc_df, x="PC1", y="Label",
                                   order=labels_order,
                                   palette=[class_palette[l.split(" ")[1]] for l in labels_order],
                                   ax=ax_bottom, orient="h", linewidth=1, width=0.6)

    for patch, label in zip(ax_bottom.collections[::2], labels_order):
        if "Synthetic" in label:
            patch.set_linestyle((0, (5, 5)))
            patch.set_linewidth(2.0)
            patch.set_alpha(0.6)
            patch.set_edgecolor("#555555")

    ax_bottom.set_xlim(xlim)
    ax_bottom.set_xlabel("")
    ax_bottom.set_ylabel("")
    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([])
    ax_bottom.tick_params(left=False, bottom=False)
    sns.despine(ax=ax_bottom, left=True, bottom=True)

    annotator_x = Annotator(ax_bottom, pairs, data=pc_df, x="PC1", y="Label", orient="h", order=labels_order)
    annotator_x.configure(test='Mann-Whitney', text_format='star', loc='outside',
                          line_offset_to_group=0.02, line_height=0.01, text_offset=0.005)
    annotator_x.apply_and_annotate()

    # === Custom legend (add line style explanation) ===
    handles, labels = ax_main.get_legend_handles_labels()
    solid_line = Line2D([], [], color='black', linestyle='-', label='Real (solid)')
    dashed_line = Line2D([], [], color='black', linestyle='--', label='Synthetic (dashed)')
    handles += [solid_line, dashed_line]

    ax_main.legend(handles=handles, title="", loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False, fontsize=9)

    # === Title and save ===
    ax_main.set_title(f"PCoA of Beta Diversity ({measure.capitalize()})", fontsize=14, pad=15)
    fig.subplots_adjust(top=0.9, right=0.78)

    plot_path = os.path.join(save_path, quality_folder, f"pcoa_{measure}_by_class.png")
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close()

# def draw_confidence_ellipse(x, y, ax, n_std=2.0, color='gray'):
#     if len(x) < 2 or len(y) < 2:
#         return
#     cov = np.cov(x, y)
#     if np.linalg.det(cov) == 0:
#         return
#     vals, vecs = np.linalg.eigh(cov)
#     order = vals.argsort()[::-1]
#     vals, vecs = vals[order], vecs[:, order]
#     theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
#     width, height = 2 * n_std * np.sqrt(vals)
#     ellipse = Ellipse((np.mean(x), np.mean(y)), width, height, angle=theta,
#                       facecolor=color, edgecolor=color, alpha=0.2, linewidth=1.5)
#     ax.add_patch(ellipse)
#     edge = Ellipse((np.mean(x), np.mean(y)), width, height, angle=theta,
#                    facecolor='none', edgecolor=color, linewidth=1.5)
#     ax.add_patch(edge)

def plot_pcoa_with_violin(pc_df, explained_var, measure, save_path, quality_folder):
    sns.set(style="white", font_scale=1.2)

    # Use distinct colorblind-friendly palette
    custom_palette = [
        "#007BFF",  # bright blue
        "#28A745",  # bright green
        "#DC3545",  # bright red
        "#FFC107",  # bright amber
        "#6610F2",  # bright violet
        "#17A2B8"   # bright cyan
    ]

    labels_order = sorted(pc_df["Label"].unique())
    palette = dict(zip(labels_order, custom_palette[:len(labels_order)]))
    pairs = list(combinations(labels_order, 2))

    fig = plt.figure(figsize=(20, 11))
    gs = GridSpec(3, 3, width_ratios=[1.2, 0.2, 6], height_ratios=[6, 0.2, 2],
                  wspace=0.05, hspace=0.05)

    ax_main = fig.add_subplot(gs[0, 2])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_bottom = fig.add_subplot(gs[2, 2], sharex=ax_main)

    # PCoA Scatter
    sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue="Label",
                    palette=palette, ax=ax_main, s=60,
                    alpha=0.9, edgecolor='k', linewidth=0.3)

    for label, group in pc_df.groupby("Label"):
        draw_confidence_ellipse(group["PC1"].values, group["PC2"].values, ax=ax_main, color=palette[label])

    ax_main.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)", fontsize=12)
    ax_main.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)", fontsize=12)
    ax_main.legend(title="", loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, frameon=False)
    for spine in ax_main.spines.values():
        spine.set_visible(False)

    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()

    # Violin plot: PC2 (left)
    sns.violinplot(data=pc_df, y="PC2", x="Label", order=labels_order,
                   palette=palette, ax=ax_left, orient="v", linewidth=1, width=0.45)
    ax_left.set_ylim(ylim)
    ax_left.set_xlabel("")
    ax_left.set_ylabel("")
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.tick_params(left=False, bottom=False)
    sns.despine(ax=ax_left, left=True, bottom=True)

    # Filter valid PC2 pairs and compute corrected p-values
    valid_pairs_y = []
    pvals_y = []
    for label1, label2 in pairs:
        group1 = pc_df[pc_df["Label"] == label1]["PC2"].dropna()
        group2 = pc_df[pc_df["Label"] == label2]["PC2"].dropna()
        if len(group1) > 1 and len(group2) > 1:
            stat, p = mannwhitneyu(group1, group2)
            valid_pairs_y.append((label1, label2))
            pvals_y.append(p)
    corrected_y = np.minimum(np.array(pvals_y) * len(pvals_y), 1.0)

    annotator_y = Annotator(ax_left, valid_pairs_y, data=pc_df, x="Label", y="PC2", orient="v", order=labels_order)
    annotator_y.configure(test=None, text_format='star', loc='outside',
                          line_offset_to_group=0.02, line_height=0.01, text_offset=0.005)
    annotator_y.set_pvalues_and_annotate(corrected_y)

    # Violin plot: PC1 (bottom)
    sns.violinplot(data=pc_df, x="PC1", y="Label", order=labels_order,
                   palette=palette, ax=ax_bottom, orient="h", linewidth=1, width=0.6)
    ax_bottom.set_xlim(xlim)
    ax_bottom.set_xlabel("")
    ax_bottom.set_ylabel("")
    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([])
    ax_bottom.tick_params(left=False, bottom=False)
    sns.despine(ax=ax_bottom, left=True, bottom=True)

    # Filter valid PC1 pairs and compute corrected p-values
    valid_pairs_x = []
    pvals_x = []
    for label1, label2 in pairs:
        group1 = pc_df[pc_df["Label"] == label1]["PC1"].dropna()
        group2 = pc_df[pc_df["Label"] == label2]["PC1"].dropna()
        if len(group1) > 1 and len(group2) > 1:
            stat, p = mannwhitneyu(group1, group2)
            valid_pairs_x.append((label1, label2))
            pvals_x.append(p)
    corrected_x = np.minimum(np.array(pvals_x) * len(pvals_x), 1.0)

    annotator_x = Annotator(ax_bottom, valid_pairs_x, data=pc_df, x="PC1", y="Label", orient="h", order=labels_order)
    annotator_x.configure(test=None, text_format='star', loc='outside',
                          line_offset_to_group=0.02, line_height=0.01, text_offset=0.005)
    annotator_x.set_pvalues_and_annotate(corrected_x)

    # Final touches
    ax_main.set_title(f"PCoA of Beta Diversity ({measure.capitalize()})", fontsize=14, pad=15)
    fig.subplots_adjust(top=0.88, right=0.78)

    plot_path = os.path.join(save_path, quality_folder, f"pcoa_{measure}_by_label.png")
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close()


# def adjust_lightness(color, amount=1.0):
#     c = to_rgb(color)
#     c = colorsys.rgb_to_hls(*c)
#     return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

# # === Helper function to adjust lightness ===
# def adjust_lightness(color, amount=1.0):
#     c = to_rgb(color)
#     c = colorsys.rgb_to_hls(*c)
#     return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

# # === Main plot function ===
# def plot_pcoa_with_violin_split_by_class(pc_df, explained_var, measure, save_path, quality_folder):
#     sns.set(style="white", font_scale=1.2)

#     # === Set color by Class ===
#     class_palette = {
#         "Healthy": "#1b9e77",  # green
#         "CRC": "#d95f02"       # orange
#     }

#     # === Set marker shape by Source ===
#     source_markers = {
#         "Real": "o",
#         "Synthetic": "^",
#         "Validation": "s"
#     }

#     # === Update Label column to be readable (no underscore) ===
#     pc_df["Label"] = pc_df["Source"] + " " + pc_df["Class"]
#     labels_order = sorted(pc_df["Label"].unique())
#     pairs = list(combinations(labels_order, 2))

#     # === Create layout ===
#     fig = plt.figure(figsize=(20, 11))
#     gs = GridSpec(3, 3, width_ratios=[1.2, 0.2, 6], height_ratios=[6, 0.2, 2],
#                   wspace=0.05, hspace=0.05)

#     ax_main = fig.add_subplot(gs[0, 2])
#     ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
#     ax_bottom = fig.add_subplot(gs[2, 2], sharex=ax_main)

#     # === Main scatter plot ===
#     for (src, cls), group in pc_df.groupby(["Source", "Class"]):
#         label = f"{src} {cls}"
#         ax_main.scatter(
#             group["PC1"], group["PC2"],
#             color=class_palette[cls],
#             marker=source_markers[src],
#             edgecolor='k',
#             linewidth=0.3,
#             alpha=0.9,
#             s=60,
#             label=label
#         )
#         draw_confidence_ellipse(group["PC1"].values, group["PC2"].values, ax=ax_main, color=class_palette[cls])

#     ax_main.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)", fontsize=12)
#     ax_main.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)", fontsize=12)
#     for spine in ax_main.spines.values():
#         spine.set_visible(False)

#     ax_main.legend(title="", loc='upper left', bbox_to_anchor=(1.05, 1), frameon=False, fontsize=9)

#     xlim = ax_main.get_xlim()
#     ylim = ax_main.get_ylim()

#     # === Left violin (PC2) ===
#     sns.violinplot(data=pc_df, y="PC2", x="Label",
#                    order=labels_order,
#                    palette=[class_palette[label.split(" ")[1]] for label in labels_order],
#                    ax=ax_left, orient="v", linewidth=1, width=0.45)
#     ax_left.set_ylim(ylim)
#     ax_left.set_xlabel("")
#     ax_left.set_ylabel("")
#     ax_left.set_xticks([])
#     ax_left.set_yticks([])
#     ax_left.tick_params(left=False, bottom=False)
#     sns.despine(ax=ax_left, left=True, bottom=True)

#     annotator_y = Annotator(ax_left, pairs, data=pc_df, x="Label", y="PC2", orient="v", order=labels_order)
#     annotator_y.configure(test='Mann-Whitney', text_format='star', loc='outside',
#                           line_offset_to_group=0.02, line_height=0.01, text_offset=0.005)
#     annotator_y.apply_and_annotate()

#     # === Bottom violin (PC1) ===
#     sns.violinplot(data=pc_df, x="PC1", y="Label",
#                    order=labels_order,
#                    palette=[class_palette[label.split(" ")[1]] for label in labels_order],
#                    ax=ax_bottom, orient="h", linewidth=1, width=0.6)
#     ax_bottom.set_xlim(xlim)
#     ax_bottom.set_xlabel("")
#     ax_bottom.set_ylabel("")
#     ax_bottom.set_xticks([])
#     ax_bottom.set_yticks([])
#     ax_bottom.tick_params(left=False, bottom=False)
#     sns.despine(ax=ax_bottom, left=True, bottom=True)

#     annotator_x = Annotator(ax_bottom, pairs, data=pc_df, x="PC1", y="Label", orient="h", order=labels_order)
#     annotator_x.configure(test='Mann-Whitney', text_format='star', loc='outside',
#                           line_offset_to_group=0.02, line_height=0.01, text_offset=0.005)
#     annotator_x.apply_and_annotate()

#     # === Title and save ===
#     ax_main.set_title(f"PCoA of Beta Diversity ({measure.capitalize()})", fontsize=14, pad=15)
#     fig.subplots_adjust(top=0.9, right=0.78)

#     plot_path = os.path.join(save_path, quality_folder, f"pcoa_{measure}_by_class.png")
#     plt.savefig(plot_path, dpi=600, bbox_inches="tight")
#     plt.close()


def gpu_pcoa_truncated(distance_matrix_gpu, k=3):
    """
    Perform Principal Coordinate Analysis (PCoA) using truncated SVD on GPU.
    Returns:
        coords: Principal coordinates (NumPy array)
        explained_var: Percentage of variance explained by each PC (NumPy array)
    """
    # Convert CuPy array to NumPy for compatibility if needed
    if isinstance(distance_matrix_gpu, cp.ndarray):
        distance_matrix_cpu = cp.asnumpy(distance_matrix_gpu)
    else:
        distance_matrix_cpu = distance_matrix_gpu

    # Double-Centering
    n = distance_matrix_cpu.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ distance_matrix_cpu @ H  # Double-centered matrix

    # Convert to CuPy
    B_gpu = cp.asarray(B)

    # Truncated SVD
    U, S, Vt = svds(B_gpu, k=k)

    # Sort components by descending singular values
    sorted_idx = cp.argsort(S)[::-1]
    S = S[sorted_idx]
    U = U[:, sorted_idx]

    # Coordinates
    coords_gpu = U @ cp.diag(cp.sqrt(S))
    coords = cp.asnumpy(coords_gpu)

    # Compute percentage of variance explained
    total_variance = cp.sum(cp.abs(cp.linalg.eigvalsh(B_gpu)))
    explained_var = cp.asnumpy((S / total_variance) * 100)

    return coords, explained_var


def process_beta_diversity(data_combined, labels, measure, already_clr, save_path, quality_folder):
    print(f"Processing {measure} on GPU...")

    permanova_results = []

    # Compute beta diversity on the full dataset (GPU-based)
    beta_matrix_gpu = compute_beta_diversity_gpu(data_combined, already_clr, measure)

    # Move to CPU & Convert to DistanceMatrix for PERMANOVA
    beta_matrix_cpu = cp.asnumpy(beta_matrix_gpu)

    # ✅ Ensure Unique Sample Labels
    unique_labels = [f"{labels[i]}_{i}" for i in range(len(labels))]

    # ✅ Create DistanceMatrix with Unique IDs
    distance_matrix_skbio = DistanceMatrix(beta_matrix_cpu, ids=unique_labels)

    # Perform overall PERMANOVA
    overall_permanova = permanova(distance_matrix_skbio, labels, permutations=199)
    overall_pval = overall_permanova['p-value']
    overall_significance = get_p_value_star(overall_pval)
    
    permanova_results.append([measure, "Overall", overall_permanova['test statistic'], overall_pval, overall_significance])

    print(f"Overall PERMANOVA Results for {measure}: {overall_permanova}")

    # Perform pairwise PERMANOVA
    unique_groups = np.unique(labels)
    pairwise_results = []
    
    for group1, group2 in combinations(unique_groups, 2):
        # Get indices for the pairwise test
        idx = np.where((labels == group1) | (labels == group2))[0]
        sub_distance_matrix = DistanceMatrix(beta_matrix_cpu[np.ix_(idx, idx)], ids=[unique_labels[i] for i in idx])
        
        # Run PERMANOVA for this pair
        pairwise_permanova = permanova(sub_distance_matrix, labels[idx], permutations=199)
        pairwise_pval = pairwise_permanova['p-value']
        pairwise_significance = get_p_value_star(pairwise_pval)

        # Store pairwise result
        pairwise_results.append(f"{group1} vs. {group2}: {pairwise_significance} (p={pairwise_pval:.3g})")
        permanova_results.append([measure, f"{group1} vs. {group2}", pairwise_permanova['test statistic'], pairwise_pval, pairwise_significance])

        print(f"Pairwise PERMANOVA {group1} vs {group2} ({measure}): {pairwise_permanova}")
    
    # Perform GPU-accelerated PCoA using Truncated SVD
    pcoa_transformed, explained_var = gpu_pcoa_truncated(beta_matrix_gpu, k=3)

    # Prepare data
    pc_df = pd.DataFrame(pcoa_transformed, columns=["PC1", "PC2", "PC3"])
    pc_df["Label"] = labels

    plot_pcoa_with_violin(pc_df, explained_var, measure, save_path, quality_folder)

    return permanova_results

def process_beta_diversity_class(data_combined, group_labels, measure, already_clr, save_path, quality_folder):
    print(f"Processing {measure} on GPU...")

    permanova_results = []

    # Compute beta diversity on the full dataset (GPU-based)
    beta_matrix_gpu = compute_beta_diversity_gpu(data_combined, already_clr, measure)

    # Move to CPU & Convert to DistanceMatrix for PERMANOVA
    beta_matrix_cpu = cp.asnumpy(beta_matrix_gpu)

    # ✅ Ensure Unique Sample Labels
    unique_labels = [f"{group_labels[i]}_{i}" for i in range(len(group_labels))]

    # ✅ Create DistanceMatrix with Unique IDs
    distance_matrix_skbio = DistanceMatrix(beta_matrix_cpu, ids=unique_labels)

    # Perform overall PERMANOVA
    overall_permanova = permanova(distance_matrix_skbio, group_labels, permutations=199)
    overall_pval = overall_permanova['p-value']
    overall_significance = get_p_value_star(overall_pval)

    permanova_results.append([measure, "Overall", overall_permanova['test statistic'], overall_pval, overall_significance])

    print(f"Overall PERMANOVA Results for {measure}: {overall_permanova}")

    # Perform pairwise PERMANOVA
    unique_groups = np.unique(group_labels)
    for group1, group2 in combinations(unique_groups, 2):
        idx = np.where((group_labels == group1) | (group_labels == group2))[0]
        sub_distance_matrix = DistanceMatrix(beta_matrix_cpu[np.ix_(idx, idx)], ids=[unique_labels[i] for i in idx])
        pairwise_permanova = permanova(sub_distance_matrix, group_labels[idx], permutations=199)
        pairwise_pval = pairwise_permanova['p-value']
        pairwise_significance = get_p_value_star(pairwise_pval)
        permanova_results.append([measure, f"{group1} vs. {group2}", pairwise_permanova['test statistic'], pairwise_pval, pairwise_significance])
        print(f"Pairwise PERMANOVA {group1} vs {group2} ({measure}): {pairwise_permanova}")

    # Perform GPU-accelerated PCoA using Truncated SVD
    pcoa_transformed, explained_var = gpu_pcoa_truncated(beta_matrix_gpu, k=3)

    # Prepare data
    pc_df = pd.DataFrame(pcoa_transformed, columns=["PC1", "PC2", "PC3"])
    pc_df["Source"] = [g.split("_")[0] for g in group_labels]
    pc_df["Class"] = [g.split("_")[1] for g in group_labels]

    plot_pcoa_with_violin_split_by_class(pc_df, explained_var, measure, save_path, quality_folder)

    return permanova_results

# ✅ Step 1: Get the Top 10% Most Abundant Features
def get_top_abundant_features(real_data, percentage=10):
    num_features = real_data.shape[1]
    top_n = max(1, int((percentage / 100) * num_features))  # Ensure at least 1 feature
    feature_sums = real_data.sum(dim=0)  # Sum across all samples
    top_features = torch.topk(feature_sums, top_n).indices  # Get top feature indices
    return top_features

# ✅ Step 3: Compute Pairwise Spearman Correlations
def compute_spearman_matrix(real_data, synthetic_data, top_features):
    real_selected = real_data[:, top_features].cpu().numpy()
    synthetic_selected = synthetic_data[:, top_features].cpu().numpy()

    # Rank-transform each feature (convert to ranks before computing correlation)
    real_ranked = np.apply_along_axis(rankdata, axis=0, arr=real_selected)
    synthetic_ranked = np.apply_along_axis(rankdata, axis=0, arr=synthetic_selected)

    # Compute Spearman correlation (which is Pearson on ranked data)
    real_corr_matrix = np.corrcoef(real_ranked.T)
    synthetic_corr_matrix = np.corrcoef(synthetic_ranked.T)

    # Extract upper triangle for scatter plot comparison
    real_corr_values = real_corr_matrix[np.triu_indices(len(top_features), k=1)]
    synthetic_corr_values = synthetic_corr_matrix[np.triu_indices(len(top_features), k=1)]

    return real_corr_matrix, synthetic_corr_matrix, real_corr_values, synthetic_corr_values

# ✅ Step 4: Compute Proportionality (ϕ) Using Lovell et al. (2015)
def compute_proportionality(real_data, synthetic_data, top_features, already_clr):
    real_selected = real_data[:, top_features].cpu().numpy()
    synthetic_selected = synthetic_data[:, top_features].cpu().numpy()

    if not already_clr:
        real_log = np.log(real_selected + 1e-10)
        synthetic_log = np.log(synthetic_selected + 1e-10)
    else:
        real_log = real_selected
        synthetic_log = synthetic_selected

    real_phi_list, synthetic_phi_list = [], []

    for m, k in combinations(range(real_selected.shape[1]), 2):  # All feature pairs
        real_phi_list.append(np.var(real_log[:, m] - real_log[:, k]) / np.var(real_log[:, m]))
        synthetic_phi_list.append(np.var(synthetic_log[:, m] - synthetic_log[:, k]) / np.var(synthetic_log[:, m]))

    return np.array(real_phi_list), np.array(synthetic_phi_list)

# ✅ Step 5: Compute R² and MSE for Similarity Metrics
def compute_correlation_metrics(real_values, synthetic_values):
    r2 = r2_score(real_values, synthetic_values)
    mse = mean_squared_error(real_values, synthetic_values)
    return r2, mse

# ✅ Step 6: Plot Spearman Correlation Matrix
def plot_correlation_matrix(corr_matrix, title, save_path, quality_folder):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, square=True, annot=False)
    plt.title(title, pad=15)
    plt.xlabel("")  # Optional: remove if you want axes labeled
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{save_path}/{quality_folder}/{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
    plt.close()


# ✅ Step 7: Scatter Plot - MB-GAN Style Comparison
def plot_mbgan_comparison(real_values, synthetic_values, metric_name, r2, mse, save_path, quality_folder):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=real_values, y=synthetic_values, alpha=0.7, edgecolor=None)

    # Identity line
    plt.axline((0, 0), slope=1, color="red", linestyle="dashed")

    # Axis labels with padding
    plt.xlabel(f"Real {metric_name}", labelpad=10)
    plt.ylabel(f"Synthetic {metric_name}", labelpad=10)

    # Title with spacing
    plt.title(f"{metric_name} Comparison\n\nR² = {r2:.4f}, MSE = {mse:.4f}", pad=15)

    # Save
    plt.tight_layout()
    plt.savefig(f"{save_path}/{quality_folder}/{metric_name.lower().replace(' ', '_')}_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

# ✅ Step 8: Compute Everything & Generate MB-GAN Style Plots
def analyze_mbgan(real_data, synthetic_data, save_path, quality_folder, already_clr, percentage=10):
    top_features = get_top_abundant_features(real_data, percentage)

    # Compute Spearman correlation matrices and scatter plot values
    real_corr_matrix, synthetic_corr_matrix, real_corr_values, synthetic_corr_values = compute_spearman_matrix(
        real_data, synthetic_data, top_features
    )

    # Compute Proportionality (ϕ)
    real_phi, synthetic_phi = compute_proportionality(real_data, synthetic_data, top_features, already_clr)

    # Compute similarity metrics
    r2_spearman, mse_spearman = compute_correlation_metrics(real_corr_values, synthetic_corr_values)
    r2_phi, mse_phi = compute_correlation_metrics(real_phi, synthetic_phi)

    # Generate correlation matrix heatmaps
    plot_correlation_matrix(real_corr_matrix, "Spearman Correlation (Real)", save_path, quality_folder)
    plot_correlation_matrix(synthetic_corr_matrix, "Spearman Correlation (Synthetic)", save_path, quality_folder)

    # Generate scatter plots
    plot_mbgan_comparison(real_corr_values, synthetic_corr_values, "Spearman Correlation", r2_spearman, mse_spearman, save_path, quality_folder)
    plot_mbgan_comparison(real_phi, synthetic_phi, "Proportionality", r2_phi, mse_phi, save_path, quality_folder)

    return top_features.cpu().numpy(), real_corr_values, synthetic_corr_values, r2_spearman, mse_spearman, real_phi, synthetic_phi, r2_phi, mse_phi

def sample_hyperparameter_grid(n_trials=500, seed=42):
    def objective(trial):
        return {
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "eta": trial.suggest_float("eta", 0.001, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 0.8),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            "gamma": trial.suggest_float("gamma", 1e-2, 10**1.5, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "alpha": trial.suggest_float("alpha", 1e-10, 10, log=True),
            "lambda": trial.suggest_float("lambda", 1e-10, 10, log=True)
        }

    np.random.seed(seed)
    sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(sampler=sampler, direction="maximize")  # we won't actually optimize anything
    trials = [study.ask() for _ in range(n_trials)]
    param_dicts = [objective(t) for t in trials]
    return pd.DataFrame(param_dicts)

def plot_auc_curves(logistic_results, xgb_results, rf_results, svm_results, save_path, quality_folder):

    # Sort and unpack
    logistic_results = sorted(logistic_results)
    xgb_results = sorted(xgb_results)
    rf_results = sorted(rf_results)
    svm_results = sorted(svm_results)

    log_sizes, auc_log = zip(*logistic_results)
    xgb_sizes, auc_xgb = zip(*xgb_results)
    rf_sizes, auc_rf = zip(*rf_results)
    svm_sizes, auc_svm = zip(*svm_results)

    log_base_idx = 0
    xgb_base_idx = 0
    rf_base_idx = 0
    svm_base_idx = 0

    log_best_idx = int(np.nanargmax(auc_log))
    xgb_best_idx = int(np.nanargmax(auc_xgb))
    rf_best_idx = int(np.nanargmax(auc_rf))
    svm_best_idx = int(np.nanargmax(auc_svm))

    # AUC improvements
    log_diff = auc_log[log_best_idx] - auc_log[log_base_idx]
    xgb_diff = auc_xgb[xgb_best_idx] - auc_xgb[xgb_base_idx]
    rf_diff = auc_rf[rf_best_idx] - auc_rf[rf_base_idx]
    svm_diff = auc_svm[svm_best_idx] - auc_svm[svm_base_idx]

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid", context="talk", font_scale=1.0)

    plt.plot(log_sizes, auc_log, marker='o', label='Logistic Regression', color='C0')
    plt.plot(xgb_sizes, auc_xgb, marker='s', label='XGBoost', color='C1')
    plt.plot(rf_sizes, auc_rf, marker='^', label='Random Forest', color='C2')
    plt.plot(svm_sizes, auc_svm, marker='d', label='SVM', color='C3')

    plt.xlabel("Number of Synthetic Samples", labelpad=12)
    plt.ylabel("AUC Score", labelpad=12)
    plt.title("Model AUC vs. Synthetic Data Augmentation", pad=20)
    plt.grid(True, linestyle="--", alpha=0.6)

    ax = plt.gca()
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)

    # Display AUC improvements below legend
    ax.text(
        1.02, 0.60,
        f"▲ LogReg: {log_diff:.2f} (Best: {np.nanmax(auc_log):.2f})\n"
        f"▲ XGBoost: {xgb_diff:.2f} (Best: {np.nanmax(auc_xgb):.2f})\n"
        f"▲ RF: {rf_diff:.2f} (Best: {np.nanmax(auc_rf):.2f})\n"
        f"▲ SVM: {svm_diff:.2f} (Best: {np.nanmax(auc_svm):.2f})",
        transform=ax.transAxes, ha='left', va='top', fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="black")
    )

    output_path = os.path.join(save_path, quality_folder, "model_comparison_auc_vs_synthetic_size.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    print(f"📊 Final plot saved at: {output_path}")

def get_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

def plot_metric_with_significance(real_vals, real_labels, synth_vals, synth_labels, label_dict,
                                   metric_name="Metric", save_path="plot.png", title=None, synth_label="Synthetic"):

    # Convert labels using label_dict
    real_labels_named = [label_dict[int(l)] for l in real_labels]
    synth_labels_named = [label_dict[int(l)] for l in synth_labels]

    # Build DataFrame
    df = pd.DataFrame({
        metric_name: list(real_vals) + list(synth_vals),
        "Class": real_labels_named + synth_labels_named,
        "Group": ["Real"] * len(real_vals) + [synth_label] * len(synth_vals)
    })

    # Initialize plot
    plt.figure(figsize=(8, 6))
    ax = sns.boxplot(data=df, x="Class", y=metric_name, hue="Group", palette="Set2", showfliers=False)

    # Define pairwise comparisons: (("ClassA", "Real"), ("ClassA", "Synthetic")), ...
    class_names = sorted(set(df["Class"]))
    pairs = [((cls, "Real"), (cls, synth_label)) for cls in class_names]

    # Add significance annotations
    annotator = Annotator(ax, pairs, data=df, x="Class", y=metric_name, hue="Group")
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside', verbose=0)
    annotator.apply_and_annotate()

    # Title logic
    if title is None:
        title = f"Comparison of {metric_name} Between Real and {synth_label} Samples"
    ax.set_title(title, fontsize=13)

    # Set legend outside
    ax.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Final layout
    ax.set_ylabel(metric_name)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def stratified_sample_three_groups(real_data, real_labels,
                                   synth_data, synth_labels,
                                   val_data, val_labels,
                                   samples_per_class=100,
                                   seed=1337):
    """
    Stratified sampling per class:
    - Try to sample `samples_per_class` per class for each group.
    - If a group lacks enough samples for a class, use all available.
    - Groups are sampled independently — no limiting by weakest group.

    Returns:
        real_sampled, real_labels_sampled,
        synth_sampled, synth_labels_sampled,
        val_sampled, val_labels_sampled
    """
    torch.manual_seed(seed)

    def sample_group(data, labels, label_set, name):
        sampled, sampled_labels = [], []
        for lbl in label_set:
            idx = (labels == lbl).nonzero(as_tuple=True)[0]
            n = samples_per_class if len(idx) >= samples_per_class else len(idx)
            if n == 0:
                print(f"⚠️ No samples available for class {lbl.item()} in {name}")
                continue
            sel = idx[torch.randperm(len(idx))[:n]]
            sampled.append(data[sel])
            sampled_labels.append(labels[sel])
        return torch.cat(sampled), torch.cat(sampled_labels)

    # Use all classes seen in any dataset
    unique_labels = torch.unique(torch.cat([real_labels, synth_labels, val_labels]))

    real_data_out, real_labels_out = sample_group(real_data, real_labels, unique_labels, "real")
    synth_data_out, synth_labels_out = sample_group(synth_data, synth_labels, unique_labels, "synthetic")
    val_data_out, val_labels_out = sample_group(val_data, val_labels, unique_labels, "validation")

    return real_data_out, real_labels_out, synth_data_out, synth_labels_out, val_data_out, val_labels_out

def analyze_processed_data(processed_data, gen_label, num_samples, real_data, real_labels, validation_data, validation_labels, label_dict, save_path, quality_folder, already_clr = False):

    gen_path_final = os.path.join(save_path, quality_folder, f"best_filtered_generated_{num_samples}.pt")
    genlab_path_final = os.path.join(save_path, quality_folder, f"best_filtered_generated_labels_{num_samples}.pt")

    if not isinstance(processed_data, torch.Tensor):
        processed_data = torch.tensor(processed_data, dtype=torch.float32)

    if os.path.exists(gen_path_final):
            print(f"Skipping analysis, already exists in {os.path.join(save_path, quality_folder)}")
    else:
        print(f"Analyzing generated samples...")
        os.makedirs(os.path.join(save_path, quality_folder), exist_ok=True)

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(processed_data.cpu().numpy().flatten(), bins=50, alpha=0.7, color="purple", edgecolor="black")
    plt.xlabel("Generated Sample Values (> 0)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Generated Samples (> 0)")

    # Save the plot
    generated_samples_hist_path = os.path.join(save_path, quality_folder, "generated_samples_reverse_normalization.png")
    plt.savefig(generated_samples_hist_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved histogram to: {generated_samples_hist_path}")

    # Compute row sums
    row_sums = processed_data.cpu().numpy().sum(axis=1)

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(row_sums, bins=50, alpha=0.7, color="blue", edgecolor="black")
    plt.xlabel("Sum of Rows")
    plt.ylabel("Frequency")
    plt.title("Histogram of Row Sums in final_clamp")

    # Save the plot
    histogram_path = os.path.join(save_path, quality_folder, "final_clamp_row_sums.png")
    plt.savefig(histogram_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Histogram saved at: {histogram_path}")

    real_data_cpu = real_data.squeeze(1).cpu()
    synthetic_data_cpu = processed_data.squeeze(1).cpu()
    validation_data_cpu = validation_data.squeeze(1).cpu()

    real_labels_cpu = real_labels.cpu().flatten()
    synth_labels_cpu = gen_label.cpu().flatten()
    validation_labels_cpu = validation_labels.cpu().flatten()

    real_data_plot, real_labels_plot, synthetic_data_plot, synthetic_labels_plot, validation_data_plot, validation_labels_plot = stratified_sample_three_groups(
        real_data_cpu, real_labels_cpu,
        synthetic_data_cpu, synth_labels_cpu,
        validation_data_cpu, validation_labels_cpu,
        samples_per_class=250,
        seed=42
    )

    print(f"real_data_plot shape:        {real_data_plot.shape}")
    print(f"real_labels_plot shape:      {real_labels_plot.shape}")
    print(f"synthetic_data_plot shape:   {synthetic_data_plot.shape}")
    print(f"synthetic_labels_plot shape: {synthetic_labels_plot.shape}")
    print(f"validation_data_plot shape:  {validation_data_plot.shape}")
    print(f"validation_labels_plot shape:{validation_labels_plot.shape}")

    sparsity_real = np.mean((real_data_plot == 0).numpy(), axis=1)
    sparsity_synth = np.mean((synthetic_data_plot == 0).numpy(), axis=1)

    # Labels
    real_labels_arr = real_labels_plot.cpu().numpy()
    synth_labels_arr = synthetic_labels_plot.cpu().numpy()
    val_labels_arr = validation_labels_plot.cpu().numpy()

    # Plot Sparsity
    plot_metric_with_significance(
        sparsity_real, real_labels_arr,
        sparsity_synth, synth_labels_arr,
        label_dict,
        metric_name="Sparsity",
        save_path=os.path.join(save_path, quality_folder, "sparsity_boxplot.png"),
        title="Sparsity Distribution Comparison"
    )

    # =========================
    # === Alpha Diversity =====
    # =========================
    if not already_clr:

        shannon_real = shannon_diversity(real_data_plot.numpy())
        shannon_synth = shannon_diversity(synthetic_data_plot.numpy())
        # Plot Alpha Diversity
        plot_metric_with_significance(
            shannon_real, real_labels_arr,
            shannon_synth, synth_labels_arr,
            label_dict,
            metric_name="Shannon Diversity",
            save_path=os.path.join(save_path, quality_folder, "alpha_diversity_boxplot.png"),
            title="Alpha Diversity Distribution Comparison",

        )

        shannon_real = shannon_diversity(real_data_plot.numpy())
        shannon_synth = shannon_diversity(validation_data_plot.numpy())
        # Plot Alpha Diversity
        plot_metric_with_significance(
            shannon_real, real_labels_arr,
            shannon_synth, val_labels_arr,
            label_dict,
            metric_name="Shannon Diversity",
            save_path=os.path.join(save_path, quality_folder, "alpha_diversity_boxplot_validation.png"),
            title="Alpha Diversity Distribution Comparison",
            synth_label="Validation"
        )
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    synthetic_data_tensor = torch.tensor(synthetic_data_plot, dtype=torch.float32, device=device)
    real_data_tensor = torch.tensor(real_data_plot, dtype=torch.float32, device=device)
    validation_data_tensor = torch.tensor(validation_data_plot, dtype=torch.float32, device=device)

    print(f"Shape: real_data_tensor = {real_data_tensor.shape}")
    print(f"Shape: synthetic_data_tensor = {synthetic_data_tensor.shape}")
    print(f"Shape: validation_data_tensor = {validation_data_tensor.shape}")

    # ✅ Combine all data & source labels
    data_combined = torch.cat([real_data_tensor, validation_data_tensor, synthetic_data_tensor], dim=0).cpu().numpy()
    labels = np.array(
        ["Real"] * real_data_plot.shape[0] +
        ["Validation"] * validation_data_plot.shape[0] +
        ["Synthetic"] * synthetic_data_plot.shape[0]
    )

    print(f"Shape: data_combined = {data_combined.shape}")
    print(f"Shape: labels = {labels.shape}")

    # ✅ Combine class labels to match shape of labels
    combined_class_labels = np.concatenate([
        real_labels_plot.cpu().numpy().flatten(),
        validation_labels_plot.cpu().numpy().flatten(),
        synthetic_labels_plot.cpu().numpy().flatten()
    ])

    print(f"Shape: combined_class_labels = {combined_class_labels.shape}")

    # ✅ Subset Real + Synthetic
    keep_mask = np.isin(labels, ["Real", "Synthetic"])
    data_subset = data_combined[keep_mask]
    source_subset = labels[keep_mask]
    class_subset = combined_class_labels[keep_mask]

    print(f"Shape: data_subset = {data_subset.shape}")
    print(f"Shape: source_subset = {source_subset.shape}")
    print(f"Shape: class_subset = {class_subset.shape}")

    # ✅ Convert class indices to string labels
    class_subset = class_subset.astype(int)
    class_subset = np.array([label_dict[c] for c in class_subset])

    # ✅ Build group labels (e.g., "Real_CRC")
    group_labels = np.array([f"{src}_{cls}" for src, cls in zip(source_subset, class_subset)])
    print(f"Shape: group_labels = {group_labels.shape}")
    print(f"Group label counts:\n{pd.Series(group_labels).value_counts()}")

    # ✅ Process Beta Diversity & PERMANOVA for Three Datasets
    beta_measures = ["braycurtis", "jaccard", "aitchison"]
    permanova_results_grouped = []

    for measure in beta_measures:
        print(f"▶ Processing {measure} (Real vs Synthetic by class)...")
        permanova_results_grouped += process_beta_diversity_class(
            data_subset, group_labels, measure, already_clr, save_path, quality_folder
        )

    # === Save Results ===
    permanova_df = pd.DataFrame(permanova_results_grouped, columns=["Measure", "Comparison", "Statistic", "p-value", "Significance"])
    permanova_stats_path = os.path.join(save_path, quality_folder, "permanova_stats_train_vs_synth_by_class.csv")
    permanova_df.to_csv(permanova_stats_path, index=False)

    # ✅ Run Beta Diversity Analysis on all three groups
    permanova_results = []
    for measure in beta_measures:
        print(f"▶ Processing {measure} (All Groups)...")
        permanova_results += process_beta_diversity(
            data_combined, labels, measure, already_clr, save_path, quality_folder
        )

    # ✅ Save PERMANOVA results to a CSV file
    permanova_df = pd.DataFrame(permanova_results, columns=["Measure", "Comparison", "Statistic", "p-value", "Significance"])
    permanova_stats_path = os.path.join(save_path, quality_folder, "permanova_stats_gpu.csv")
    permanova_df.to_csv(permanova_stats_path, index=False)
    print(f"✅ PERMANOVA statistical test results saved to: {permanova_stats_path}")

    # ✅ Step 9: MB-GAN similarity analysis
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_data_tensor = torch.tensor(real_data_plot, dtype=torch.float32, device=device)
    synthetic_data_tensor = torch.tensor(synthetic_data_plot, dtype=torch.float32, device=device)

    print(f"▶ Running MB-GAN analysis on {real_data_tensor.shape[0]} real and {synthetic_data_tensor.shape[0]} synthetic samples")

    top_features, real_corr_values, synthetic_corr_values, r2_spearman, mse_spearman, real_phi, synthetic_phi, r2_phi, mse_phi = analyze_mbgan(
        real_data_tensor, synthetic_data_tensor, save_path, quality_folder, already_clr, percentage=10)

    # ✅ Step 10: Print Results
    print("Top 10% Most Abundant Features (Indices):", top_features)
    print("R² for Spearman Correlation Fit:", r2_spearman)
    print("MSE for Spearman Correlation Fit:", mse_spearman)
    print("R² for Proportionality Fit:", r2_phi)
    print("MSE for Proportionality Fit:", mse_phi)

    torch.save(processed_data, gen_path_final)
    torch.save(gen_label, genlab_path_final)

    # Ensure PyTorch uses GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train = real_data.squeeze(1).cpu()
    y_train = real_labels.cpu().flatten()
    X_val = validation_data.squeeze(1).cpu()
    y_val = validation_labels.cpu().flatten()
    X_synth = processed_data.squeeze(1).cpu()
    y_synth = gen_label.cpu().flatten()

    # Define synthetic sample sizes
    gen_data_sizes = np.unique(np.linspace(1, X_synth.shape[0], num=10, dtype=int))
    logistic_results = []
    xgb_results = []

    # Logistic base model
    log_reg_base = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
    log_reg_base.fit(X_train, y_train)
    base_auc_logistic = roc_auc_score(y_val, log_reg_base.predict_proba(X_val)[:, 1])
    logistic_results.append((0, base_auc_logistic))
    print(f"Base AUC (Logistic Regression): {base_auc_logistic:.4f}")

    # Logistic regression with augmentation
    for size in gen_data_sizes[1:]:
        idx = np.random.choice(X_synth.shape[0], size=size, replace=False)
        X_aug = np.vstack([X_train, X_synth[idx]])
        y_aug = np.concatenate([y_train, y_synth[idx].cpu().numpy()])

        clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
        clf.fit(X_aug, y_aug)
        auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])
        logistic_results.append((size, auc))
        print(f"LogReg with {size} synthetic → AUC: {auc:.4f}")

    rf_results = []

    # Random Forest evaluation
    for size in gen_data_sizes:
        print(f"\n🌲 Testing Random Forest with {size} synthetic samples")
        if size > 0:
            idx = np.random.choice(X_synth.shape[0], size=size, replace=False)
            X_aug = np.vstack([X_train, X_synth[idx]])
            y_aug = np.concatenate([y_train, y_synth[idx].cpu().numpy()])
        else:
            X_aug = X_train
            y_aug = y_train

        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_aug, y_aug)
        auc = roc_auc_score(y_val, rf_clf.predict_proba(X_val)[:, 1])
        rf_results.append((size, auc))
        print(f"Random Forest with {size} synthetic → AUC: {auc:.4f}")

    # Save Random Forest results
    rf_df = pd.DataFrame(rf_results, columns=["synthetic_size", "rf_auc"])
    rf_path = os.path.join(save_path, quality_folder, "rf_auc_results.csv")
    rf_df.to_csv(rf_path, index=False)
    print(f"🌲 RF AUC results saved: {rf_path}")

    svm_results = []

    # Use fixed value for C
    fixed_C = 1.0

    for size in gen_data_sizes:
        print(f"\n🧠 Testing SVM with {size} synthetic samples")

        # Augment training data
        if size > 0:
            idx = np.random.choice(X_synth.shape[0], size=size, replace=False)
            X_aug = np.vstack([X_train, X_synth[idx]])
            y_aug = np.concatenate([y_train, y_synth[idx].cpu().numpy()])
        else:
            X_aug = X_train
            y_aug = y_train

        base_svm = LinearSVC(C=fixed_C, max_iter=10000, random_state=42)
        calibrated_svm = CalibratedClassifierCV(base_svm, method='sigmoid', cv=5)
        calibrated_svm.fit(X_aug, y_aug)

        auc = roc_auc_score(y_val, calibrated_svm.predict_proba(X_val)[:, 1])
        svm_results.append((size, auc))
        print(f"SVM (C={fixed_C}) with {size} synthetic → AUC: {auc:.4f}")

    svm_df = pd.DataFrame(svm_results, columns=["synthetic_size", "svm_auc"])
    svm_path = os.path.join(save_path, quality_folder, "svm_auc_results.csv")
    svm_df.to_csv(svm_path, index=False)
    print(f"🧠 SVM AUC results saved: {svm_path}")

    # XGBoost setup
    xgb_param_grid_df = sample_hyperparameter_grid(n_trials=25)
    synth_sizes = np.insert(gen_data_sizes, 0, 0)
    cv_folds = 5
    max_boost_rounds = 200
    early_stopping_rounds = 200
    random_state = 42

    # XGBoost evaluation
    for synth_size in synth_sizes:
        print(f"\n🔁 Testing with {synth_size} synthetic samples")
        if synth_size > 0:
            idx = np.random.choice(X_synth.shape[0], size=synth_size, replace=False)
            X_synth_sampled = X_synth[idx]
            y_synth_sampled = y_synth[idx]

        best_auc = -np.inf
        best_params = None
        best_iteration = None
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

        for _, row in xgb_param_grid_df.iterrows():
            params = row.to_dict()
            params.update({
                "booster": "gbtree",
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "tree_method": "gpu_hist",
                "device": "cuda",
                "max_depth": int(params["max_depth"]),
                "min_child_weight": int(params["min_child_weight"])
            })

            fold_aucs = []
            for train_idx, val_idx in skf.split(X_train, y_train):
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_val_fold = X_train[val_idx]
                y_val_fold = y_train[val_idx]

                if synth_size > 0:
                    X_fold_train = np.vstack([X_fold_train, X_synth_sampled])
                    y_fold_train = np.concatenate([y_fold_train, y_synth_sampled.cpu().numpy()])


                dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
                dval = xgb.DMatrix(X_val_fold, label=y_val_fold)

                model = xgb.train(
                    params, dtrain, num_boost_round=max_boost_rounds,
                    evals=[(dval, "val")], early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False
                )

                preds = model.predict(dval)
                fold_aucs.append(roc_auc_score(y_val_fold, preds))

            mean_auc = np.mean(fold_aucs)
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = params
                best_iteration = model.best_iteration

        print(f"✅ Best XGBoost CV AUC: {best_auc:.4f}")
        final_train = np.vstack([X_train, X_synth_sampled]) if synth_size > 0 else X_train
        final_label = np.concatenate([y_train, y_synth_sampled.cpu().numpy()]) if synth_size > 0 else y_train

        dtrain_final = xgb.DMatrix(final_train, label=final_label)
        dval_final = xgb.DMatrix(X_val, label=y_val)

        final_model = xgb.train(best_params, dtrain_final, num_boost_round=best_iteration)
        final_auc = roc_auc_score(y_val, final_model.predict(dval_final))
        xgb_results.append((synth_size, final_auc))

        print(f"🎯 Final XGBoost AUC: {final_auc:.4f}")

        # Save logistic results
        log_df = pd.DataFrame(logistic_results, columns=["synthetic_size", "logistic_auc"])
        log_path = os.path.join(save_path, quality_folder, "logistic_auc_results.csv")
        log_df.to_csv(log_path, index=False)

        # Save XGBoost results
        xgb_df = pd.DataFrame(xgb_results, columns=["synthetic_size", "xgboost_auc"])
        xgb_path = os.path.join(save_path, quality_folder, "xgboost_auc_results.csv")
        xgb_df.to_csv(xgb_path, index=False)

        print(f"✅ AUC results saved:\n- {log_path}\n- {xgb_path}")

    

    plot_auc_curves(logistic_results, xgb_results, rf_results, svm_results, save_path, quality_folder)