# Core
import os
import numpy as np
import pandas as pd
from itertools import combinations
import glob
import re

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
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

def draw_confidence_ellipse(x, y, ax, n_std=2.0, color='gray'):
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
                      facecolor=color, edgecolor=color, alpha=0.2, linewidth=1.5)
    ax.add_patch(ellipse)
    edge = Ellipse((np.mean(x), np.mean(y)), width, height, angle=theta,
                   facecolor='none', edgecolor=color, linewidth=1.5)
    ax.add_patch(edge)


def plot_pcoa_with_violin(pc_df, explained_var, measure, save_path, quality_folder):
    sns.set(style="white", font_scale=1.2)

    # Setup
    labels_order = pc_df["Label"].unique()
    palette = sns.color_palette("Set2", n_colors=len(labels_order))
    label_to_color = dict(zip(labels_order, palette))
    pairs = list(combinations(labels_order, 2))

    # Make figure taller to accommodate top stars
    fig = plt.figure(figsize=(20, 11))
    gs = GridSpec(3, 3, width_ratios=[1.2, 0.2, 6], height_ratios=[6, 0.2, 2],
                  wspace=0.05, hspace=0.05)

    ax_main = fig.add_subplot(gs[0, 2])
    ax_left = fig.add_subplot(gs[0, 0], sharey=ax_main)
    ax_bottom = fig.add_subplot(gs[2, 2], sharex=ax_main)

    # PCoA Scatter
    sns.scatterplot(data=pc_df, x="PC1", y="PC2", hue="Label",
                    palette=label_to_color, ax=ax_main, s=60,
                    alpha=0.9, edgecolor='k', linewidth=0.3)

    for label, group in pc_df.groupby("Label"):
        color = label_to_color[label]
        draw_confidence_ellipse(group["PC1"].values, group["PC2"].values, ax=ax_main, color=color)

    ax_main.set_xlabel(f"PC1 ({explained_var[0]:.1f}%)", fontsize=12)
    ax_main.set_ylabel(f"PC2 ({explained_var[1]:.1f}%)", fontsize=12)
    for spine in ax_main.spines.values():
        spine.set_visible(False)

    # Legend just to the right
    ax_main.legend(title="", loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, frameon=False)

    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()

    # Left violin
    sns.violinplot(data=pc_df, y="PC2", x="Label", palette=label_to_color,
                   ax=ax_left, orient="v", linewidth=1, width=0.45)
    ax_left.set_ylim(ylim)
    ax_left.set_xlabel("")
    ax_left.set_ylabel("")
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.tick_params(left=False, bottom=False)
    sns.despine(ax=ax_left, left=True, bottom=True)

    annotator_y = Annotator(ax_left, pairs, data=pc_df, x="Label", y="PC2", orient="v")
    annotator_y.configure(test='t-test_ind', text_format='star', loc='outside',
                          line_offset_to_group=0.02, line_height=0.01, text_offset=0.005)
    annotator_y.apply_and_annotate()

    # Bottom violin
    sns.violinplot(data=pc_df, x="PC1", y="Label", palette=label_to_color,
                   ax=ax_bottom, orient="h", linewidth=1, width=0.6)
    ax_bottom.set_xlim(xlim)
    ax_bottom.set_xlabel("")
    ax_bottom.set_ylabel("")
    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([])
    ax_bottom.tick_params(left=False, bottom=False)
    sns.despine(ax=ax_bottom, left=True, bottom=True)

    annotator_x = Annotator(ax_bottom, pairs, data=pc_df, x="PC1", y="Label", orient="h")
    annotator_x.configure(test='t-test_ind', text_format='star', loc='outside',
                          line_offset_to_group=0.02, line_height=0.01, text_offset=0.005)
    annotator_x.apply_and_annotate()

    # Slightly elevated title
    ax_main.set_title(f"PCoA of Beta Diversity ({measure.capitalize()})", fontsize=14, pad=15)

    # Add more margin above to ensure stars don't get clipped
    fig.subplots_adjust(top=0.86)

    # Save (no bbox_inches to avoid cutting anything off)
    pcoa_plot_path = os.path.join(save_path, quality_folder, f"pcoa_{measure}_publication.png")
    plt.savefig(pcoa_plot_path, dpi=600)
    plt.close()

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

def analyze_processed_data(processed_data, gen_label, num_samples, real_data, real_labels, validation_data, validation_labels, save_path, quality_folder, already_clr = False):

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

    real_data_np = real_data.squeeze(1).numpy()
    validation_data_np = validation_data.squeeze(1).numpy()
    synthetic_data_np = processed_data.squeeze(1).numpy()

    # Compute sparsity for each dataset
    sparsity_train = np.mean(real_data_np == 0, axis=1)
    sparsity_validation = np.mean(validation_data_np == 0, axis=1)
    sparsity_synthetic = np.mean(synthetic_data_np == 0, axis=1)

    df = pd.DataFrame({
        "Sparsity": np.concatenate([sparsity_train, sparsity_validation, sparsity_synthetic]),
        "Group": (["Real Train"] * len(sparsity_train)) +
                (["Real Validation"] * len(sparsity_validation)) +
                (["Synthetic"] * len(sparsity_synthetic))
    })

    # === Plot ===
    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(x="Group", y="Sparsity", data=df, width=0.5)
    ax.set_xlabel("") 
    plt.title("Sparsity Comparison")

    # Define group pairs
    pairs = [
        ("Real Train", "Real Validation"),
        ("Real Validation", "Synthetic"),
        ("Real Train", "Synthetic")
    ]

    # Annotate significance (precomputed stars) and store stats
    p_values = []
    test_stats = []

    for g1, g2 in pairs:
        values1 = df[df["Group"] == g1]["Sparsity"]
        values2 = df[df["Group"] == g2]["Sparsity"]
        stat, pval = mannwhitneyu(values1, values2, alternative="two-sided")
        pval_corr = min(pval * len(pairs), 1.0)  # Bonferroni correction
        
        # Annotate
        if pval_corr <= 1e-4:
            star = "****"
        elif pval_corr <= 1e-3:
            star = "***"
        elif pval_corr <= 1e-2:
            star = "**"
        elif pval_corr <= 0.05:
            star = "*"
        else:
            star = "ns"
            
        p_values.append(star)
        test_stats.append((stat, pval))  # Save stats for CSV

    # Add annotations with statannotations

    annotator = Annotator(ax, pairs, data=df, x="Group", y="Sparsity")
    annotator.set_custom_annotations(p_values)
    annotator.configure(test=None)
    annotator.annotate()

    # === Add full p-value legend ===
    summary_lines = [
        "p-value annotation legend:",
        "     ns: 5.00e-02 < p ≤ 1.00e+00",
        "      *: 1.00e-02 < p ≤ 5.00e-02",
        "     **: 1.00e-03 < p ≤ 1.00e-02",
        "    ***: 1.00e-04 < p ≤ 1.00e-03",
        "   ****: p ≤ 1.00e-04",
        ""
    ]

    # Add comparison-specific results
    for (g1, g2), star in zip(pairs, p_values):
        values1 = df[df["Group"] == g1]["Sparsity"]
        values2 = df[df["Group"] == g2]["Sparsity"]
        stat, pval = mannwhitneyu(values1, values2, alternative="two-sided")
        pval_corr = min(pval * len(pairs), 1.0)
        summary_lines.append(
            f"{g1} vs. {g2}: Mann-Whitney U test (2-sided, Bonferroni), "
            f"P = {pval_corr:.3e}, U = {stat:.3e} ({star})"
        )

    legend_text = "\n".join(summary_lines)

    plt.text(
        x=1.03, y=0.98, s=legend_text,
        transform=ax.transAxes,
        fontsize=9,
        va='top', ha='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85)
    )

    # === Save Plot ===
    sparsity_plot_path = os.path.join(save_path, quality_folder, "sparsity_distribution.png")
    plt.savefig(sparsity_plot_path, dpi=300, bbox_inches="tight")
    plt.close()


    # Save statistical test results
    # Save statistical test results
    stats_results = pd.DataFrame({
        "Comparison": [
            "Train vs Validation",
            "Validation vs Synthetic",
            "Train vs Synthetic"
        ],
        "Mann-Whitney U Statistic": [s[0] for s in test_stats],
        "p-value": [s[1] for s in test_stats]
    })

    stats_file_path = os.path.join(save_path, quality_folder, "sparsity_stats_wilcoxon.csv")
    stats_results.to_csv(stats_file_path, index=False)

    if not already_clr:

        # ✅ Compute Shannon Diversity
        shannon_real = shannon_diversity(real_data_np)
        shannon_validation = shannon_diversity(validation_data_np)
        shannon_synthetic = shannon_diversity(synthetic_data_np)

        # ✅ Create DataFrame
        alpha_diversity_df = pd.DataFrame({
            "Alpha Diversity": np.concatenate([shannon_real, shannon_validation, shannon_synthetic]),
            "Dataset": (["Real Train"] * len(shannon_real)) + 
                    (["Real Validation"] * len(shannon_validation)) + 
                    (["Synthetic"] * len(shannon_synthetic))
        })

        # ✅ Define comparisons
        comparisons = [
            ("Real Train", "Real Validation"),
            ("Real Train", "Synthetic"),
            ("Real Validation", "Synthetic")
        ]

        # ✅ Compute p-values and significance stars
        p_values = []
        for g1, g2 in comparisons:
            stat, pval = mannwhitneyu(
                alpha_diversity_df[alpha_diversity_df["Dataset"] == g1]["Alpha Diversity"],
                alpha_diversity_df[alpha_diversity_df["Dataset"] == g2]["Alpha Diversity"],
                alternative="two-sided"
            )
            pval_corr = min(pval * len(comparisons), 1.0)  # Bonferroni correction
            p_values.append(pval_corr)

        # Generate significance stars
        def get_star(p):
            if p <= 1e-4: return "****"
            elif p <= 1e-3: return "***"
            elif p <= 1e-2: return "**"
            elif p <= 0.05: return "*"
            else: return "ns"

        sig_stars = [get_star(p) for p in p_values]

        # ✅ Plot with statannotations
        plt.figure(figsize=(8, 5))
        ax = sns.boxplot(x="Dataset", y="Alpha Diversity", data=alpha_diversity_df,
                        width=0.5, palette=["blue", "orange", "green"])
        ax.set_xlabel("")  # Remove x-axis label
        plt.title("Alpha Diversity")
        plt.ylabel("Shannon Diversity Index")

        # Annotate
        annotator = Annotator(ax, comparisons, data=alpha_diversity_df, x="Dataset", y="Alpha Diversity")
        annotator.set_custom_annotations(sig_stars)
        annotator.configure(test=None)
        annotator.annotate()

        # ✅ Save Plot
        alpha_diversity_plot_path = os.path.join(save_path, quality_folder, "alpha_diversity_distribution.png")
        plt.savefig(alpha_diversity_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # ✅ Save Stats to CSV
        alpha_diversity_stats = pd.DataFrame({
            "Comparison": [f"{g1} vs {g2}" for (g1, g2) in comparisons],
            "p-value (Bonferroni)": p_values,
            "Significance": sig_stars
        })
        alpha_diversity_stats_path = os.path.join(save_path, quality_folder, "alpha_diversity_stats.csv")
        alpha_diversity_stats.to_csv(alpha_diversity_stats_path, index=False)

        print(f"Alpha Diversity plot saved to: {alpha_diversity_plot_path}")
        print(f"Statistical test results saved to: {alpha_diversity_stats_path}")

    # Shuffle gen dataset
    indices = torch.randperm(len(synthetic_data_np))  # Generate random indices
    gen_shuffled = synthetic_data_np[indices]
    gen_label_shuffled = gen_label[indices]

    # Shuffle gen dataset
    indices = torch.randperm(len(real_data_np))  # Generate random indices
    real_shuffled = real_data_np[indices]
    real_label_shuffled = real_labels[indices]

    # Shuffle gen dataset
    indices = torch.randperm(len(validation_data_np))  # Generate random indices
    validation_shuffled = validation_data_np[indices]
    validation_label_shuffled = validation_labels[indices]

    # Sample to match the size of X_train
    sample_size = 500
    synthetic_data_plot = gen_shuffled[:sample_size]
    synthetic_labels_plot = gen_label_shuffled[:sample_size]

    real_data_plot = real_shuffled[:sample_size]
    real_labels_plot = real_label_shuffled[:sample_size]

    validation_data_plot = validation_shuffled[:sample_size]
    validation_labels_plot = validation_label_shuffled[:sample_size]

    # ✅ Move Data to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ✅ Convert Data to Torch & Move to GPU
    synthetic_data_tensor = torch.tensor(synthetic_data_plot, dtype=torch.float32, device=device)
    real_data_tensor = torch.tensor(real_data_plot, dtype=torch.float32, device=device)
    validation_data_tensor = torch.tensor(validation_data_plot, dtype=torch.float32, device=device)

    # ✅ Combine all datasets & assign labels
    data_combined = torch.cat([real_data_tensor, validation_data_tensor, synthetic_data_tensor], dim=0).cpu().numpy()
    labels = np.array(
        ["Real"] * real_data_plot.shape[0] +
        ["Validation"] * validation_data_plot.shape[0] +
        ["Synthetic"] * synthetic_data_plot.shape[0]
    )


    # ✅ Process Beta Diversity & PERMANOVA for Three Datasets
    beta_measures = ["braycurtis", "jaccard", "aitchison"]

    # ✅ Run Beta Diversity Analysis
    for measure in beta_measures:
        permanova_results = process_beta_diversity(data_combined, labels, measure, already_clr, save_path, quality_folder)

    # ✅ Save PERMANOVA results to a CSV file
    permanova_df = pd.DataFrame(permanova_results, columns=["Measure", "Comparison", "Statistic", "p-value", "Significance"])
    permanova_stats_path = os.path.join(save_path, quality_folder, "permanova_stats_gpu.csv")
    permanova_df.to_csv(permanova_stats_path, index=False)

    print(f"PERMANOVA statistical test results saved to: {permanova_stats_path}")

    # ✅ Step 9: Run the Analysis on Tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_data_tensor = torch.tensor(real_data_np, dtype=torch.float32, device=device)
    synthetic_data_tensor = torch.tensor(synthetic_data_np, dtype=torch.float32, device=device)

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

    # Prepare data
    X_train = real_data_np
    y_train = real_labels
    X_val = validation_data_np
    y_val = validation_labels
    X_synth = synthetic_data_np
    y_synth = gen_label

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