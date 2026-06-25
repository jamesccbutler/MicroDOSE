import torch
import torch.optim as optim
from dataloaders import get_micro_dataloaders
from models import Generator_Binary, Discriminator_Binary, Classifier
from training import Trainer_Binary
from numpy.random import seed
seed(0)
import numpy as np
import sys
import os
from itertools import product, combinations
# Add the parent directory to sys.path
import data_loader
from experimentor import Experimentor
from models import Generator_dbg, Critic_dbg
seed(0)
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd
import glob
import re
import seaborn as sns

from minimal_diffusion.conditional_diffusion_loglikelihood import *
from minimal_diffusion.unet_1d_class_cond import *
from minimal_diffusion.analysis_functions_experiment import *
import copy
import gc
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
from skbio.diversity import alpha_diversity
from skbio.diversity import beta_diversity
from skbio.stats.distance import permanova
from skbio.stats.ordination import pcoa
from scipy.spatial.distance import braycurtis, jaccard
from skbio.stats.composition import clr
from joblib import Parallel, delayed
import cupy as cp
from scipy.spatial.distance import pdist, squareform
from cupyx.scipy.sparse.linalg import svds
from skbio import DistanceMatrix
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import kruskal, mannwhitneyu
from sklearn.metrics import roc_auc_score
import optuna
from scipy.stats import ks_2samp
# time recording
import time
import platform
import psutil
import csv
from datetime import datetime

def get_system_info():
    info = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "cpu": platform.processor(),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 2)
    else:
        info["gpu"] = "CPU"
        info["gpu_memory_gb"] = 0
    return info

# After
FIXED_FIELDS = [
    "timestamp", "exp_name", "phase", "epochs", "elapsed_seconds",
    "cpu", "ram_gb", "gpu", "gpu_memory_gb",
    "device", "batch_size", "lambda_cls", "num_channels", "timesteps", "batch_samples", "num_steps"
]

def log_run(log_path, phase, exp_name, epochs, elapsed_seconds, extra=None):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys_info = get_system_info()
    row = {f: None for f in FIXED_FIELDS}
    row.update({
        "timestamp": sys_info["timestamp"],
        "exp_name": exp_name,
        "phase": phase,
        "epochs": epochs,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "cpu": sys_info["cpu"],
        "ram_gb": sys_info["ram_gb"],
        "gpu": sys_info["gpu"],
        "gpu_memory_gb": sys_info["gpu_memory_gb"],
    })
    if extra:
        row.update(extra)

    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIXED_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[LOG] {phase} logged to {log_path}")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

date = "160625"
nclasses = 2
img_cols = 940

# Didn't work "PRJNA531273", "PRJEB27928"
#Other: "PRJDB4176", "PRJEB10878", "PRJEB6070", "PRJEB7774", "PRJNA389927", "PRJNA447983", "PRJNA429097", "PRJNA763023", "PRJNA731589", "PRJNA961076"
# Parameter grids
validation_options = ["PRJEB12449"]
dlc_options = [0]
lambda_cls_options = [0]
batch_size_options = [64]
log_likelihood_options = ["no_log_likelihood"] 

for dlc, validation, lambda_cls, batch_size, log_likelihood, in product(dlc_options, validation_options, lambda_cls_options, batch_size_options, log_likelihood_options):

    data = data_loader.load_FeMAI(validation=validation)
    label_dict = data.label_dict
    exp = Experimentor(data=data, select_feature=False, phylo_clust=False)
    exp_name = f'FeMAI_binary_val_{validation}_{date}_lambda_{lambda_cls}_bs_{batch_size}_{log_likelihood}'
    run_log_path = f'logs/{exp_name}/run_log.csv'

    train_loader, test_loader, X_train, y_train, X_test, y_test = get_micro_dataloaders(exp, batch_size=batch_size)

    generator = Generator_Binary(latent_dim=100, img_cols=img_cols, nclasses=nclasses, min_val=0.0005, max_val=0.9995).to(device)
    discriminator = Discriminator_Binary(img_cols=img_cols, nclasses=nclasses).to(device)
    classifier = Classifier(input_dim=940, nclasses=2).to(device)

    lr = 1e-4
    betas = (.5, .9)
    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    cls_optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-5)

    epochs = 10000
    trainer = Trainer_Binary(generator, discriminator, classifier, G_optimizer, D_optimizer, cls_optimizer, X_train, y_train, X_test, y_test, nclasses,
                                f'Result_Files/{exp_name}',
                                f'Model_Files/{exp_name}',
                                "binary_gan",
                                label_dict,
                                lambda_cls=lambda_cls,
                                dlc = dlc,
                                device = device)

    # Path to the result file
    ext_scores_path = f'Result_Files/{exp_name}/ext_scores.csv'

    # Check if training should be skipped
    skip_training = False
    if os.path.exists(ext_scores_path):
        try:
            df_scores = pd.read_csv(ext_scores_path)
            if 'epoch' in df_scores.columns and epochs in df_scores['epoch'].values:
                print(f"[SKIP] Epoch {epochs} already exists in {ext_scores_path}. Skipping training.")
                skip_training = True
        except Exception as e:
            print(f"[WARNING] Could not read {ext_scores_path}: {e}")

    # Run training only if it hasn't already been done
    if not skip_training:
        start_time_gan = time.perf_counter()
        trainer.train(train_loader, epochs, sample_every=100)
        elapsed_time_gan = time.perf_counter() - start_time_gan
        log_run(run_log_path, "gan_training", exp_name, epochs, elapsed_time_gan, extra={"device": str(trainer.device), "batch_size": batch_size, "lambda_cls": lambda_cls})

    data = data_loader.load_FeMAI(validation=validation)
    label_dict = data.label_dict
    exp = Experimentor(data=data, select_feature=False, phylo_clust = False)
    nclasses = 2
    img_cols = 940

    train_loader, test_loader, X_train, y_train, X_test, y_test = get_micro_dataloaders(exp, batch_size=64)

    generator = Generator_Binary(latent_dim = 100, img_cols = img_cols, nclasses = nclasses, min_val=0.0005, max_val=0.9995)
    discriminator = Discriminator_Binary(img_cols = img_cols, nclasses = nclasses)
    classifier = Classifier(input_dim=img_cols, nclasses=nclasses)

    # Initialize optimizers
    lr = 1e-4
    betas = (.5, .9)
    G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=betas)
    cls_optimizer = optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-5)

    model_path = f'Model_Files/{exp_name}'
    model_name = "binary_gan"

    checkpoint_configs = generate_pvalue_checkpoint_configs(model_path, model_name, exp_name)

    print(checkpoint_configs)

    # Load multiple GAN models from saved checkpoints
    generators = []
    for config in checkpoint_configs:
        step = config["step"]
        model_type = config["type"]  # 'loss' or 'perf'
        exp_name_run = config["exp_name"]

        trainer = Trainer_Binary(generator, discriminator, classifier, G_optimizer, D_optimizer, cls_optimizer, 
                                X_train, y_train, X_test, y_test, nclasses, 
                                f'Result_Files/{exp_name_run}', 
                                f'Model_Files/{exp_name_run}', 
                                "binary_gan", 
                                label_dict)
        
        trainer.load_model(step, type=model_type)  # Load model based on step & type
        generators.append(trainer)  # Add the trained model to the ensemble

    critic_configs = generate_critic_checkpoint_configs(model_path, model_name, exp_name)

    critics = []
    for config in critic_configs:
        step = config["step"]
        exp_name_run = config["exp_name"]

        # Reuse the Trainer_Binary structure but only keep critic
        trainer = Trainer_Binary(generator, discriminator, classifier, G_optimizer, D_optimizer, cls_optimizer, 
                                X_train, y_train, X_test, y_test, nclasses, 
                                f'Result_Files/{exp_name_run}', 
                                f'Model_Files/{exp_name_run}', 
                                "binary_gan", 
                                label_dict)

        trainer.load_critic_only(step, type='pvalue_reached')
        critics.append(trainer)
    
    num_samples = 30000  # Number of generated samples
    
    top_k_ratio = 0.10

    x_file = f"X_gen_{exp_name}_{num_samples}_1"
    y_file = f"y_gen_{exp_name}_{num_samples}_1"
    if os.path.exists(f"/home/jbutler/DiffMicro/GAN_bin_masks/binary_gan_femai/{x_file}"):
        print(f"Skipping generation, already exists in gan folder")
    else:
        print(f"Generating samples...")

        if log_likelihood == "no_log_likelihood":
            print("Generating GAN samples without log-likelihood")

            aggregated_samples, generated_labels = ensemble_gan_samples_optimized(
                generators, num_samples, use_ema=True)

            X_generated = torch.where(aggregated_samples  > .5, 1, 0)
            generated_labels = generated_labels

            x_file = f"X_gen_{exp_name}_{num_samples}_1"
            y_file = f"y_gen_{exp_name}_{num_samples}_1"
            torch.save(X_generated, f"/home/jbutler/DiffMicro/GAN_bin_masks/binary_gan_femai/{x_file}")
            torch.save(generated_labels, f"/home/jbutler/DiffMicro/GAN_bin_masks/binary_gan_femai/{y_file}")
        
        elif log_likelihood == "log_likelihood":

            print("Generating GAN samples with log_likelihood")
            aggregated_samples, generated_labels, filtered_scores = generate_with_critic_ensemble(
                generators=generators,
                critics=critics,
                num_desired_samples=num_samples,
                top_k_ratio=top_k_ratio,
                use_ema=True
            )

            X_train = exp.X_train_binary.squeeze(1)
            train_labels = exp.y_train
            X_val = exp.X_test_binary.squeeze(1)
            val_labels = exp.y_test
            X_generated = torch.where(aggregated_samples  > .5, 1, 0)
            generated_labels = generated_labels
            x_file = f"X_gen_{exp_name}_{num_samples}_1"
            y_file = f"y_gen_{exp_name}_{num_samples}_1"
            torch.save(X_generated, f"/home/jbutler/DiffMicro/GAN_bin_masks/binary_gan_femai/{x_file}")
            torch.save(generated_labels, f"/home/jbutler/DiffMicro/GAN_bin_masks/binary_gan_femai/{y_file}")

            aggregated_samples, generated_labels, filtered_scores = generate_with_critic_ensemble(
                generators=generators,
                critics=critics,
                num_desired_samples=num_samples,
                top_k_ratio=top_k_ratio,
                use_ema=True
            )

            X_generated = torch.where(aggregated_samples  > .5, 1, 0)
            generated_labels = generated_labels
            x_file = f"X_gen_{exp_name}_{num_samples}_2"
            y_file = f"y_gen_{exp_name}_{num_samples}_2"
            torch.save(X_generated, f"/home/jbutler/DiffMicro/GAN_bin_masks/binary_gan_femai/{x_file}")
            torch.save(generated_labels, f"/home/jbutler/DiffMicro/GAN_bin_masks/binary_gan_femai/{y_file}")
        else:
            print("No valid option for log-likelihood found...")
    
    # Define hyperparameter space
    hyperparam_space = {
        "num_channels": [2],
        "inpaint": [True],
        "batch_size": [64],
        "lr": [1e-4],
        "timesteps": [500],
        "use_film": [True]
    }

    # Generate all combinations of hyperparameters
    all_experiments = list(product(*hyperparam_space.values()))
    print(f"Total Experiments to Run: {len(all_experiments)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = "femai"
    val_study = validation
    date = date
    study = f'{ds}_{val_study}_{date}'
    # for use if using an older model
    model_study = study
    os.makedirs(study, exist_ok=True)

    # Load masks
    num_samples = num_samples
    gan_path = "/home/jbutler/DiffMicro/GAN_bin_masks/binary_gan_femai/"
    x_gan = x_file
    y_gan = y_file

    gen_masks = torch.load(f'{gan_path}{x_gan}').unsqueeze(1)
    binary_masks = torch.from_numpy(np.where(gen_masks > .5, 1, 0))
    gen_labels = torch.tensor(torch.load(f'{gan_path}/{y_gan}'))

    num_classes = 2
    epochs = 200
    feature_size = 940
    epsilon = 1e-10
    log_min = torch.log(torch.tensor(epsilon))
    log_max = 0

    # Load dataset once to avoid reloading every run
    data = data_loader.load_FeMAI(validation = validation)
    exp = Experimentor(data=data, select_feature=False, phylo_clust=False)

    t_dataset = exp.X_train
    train_labels = exp.y_train
    v_dataset = exp.X_test
    val_labels = exp.y_test
    val_size = exp.X_test.shape[0]

    # Normalize train dataset
    t_bin_chan = torch.where(t_dataset == 0, torch.tensor(-1), torch.tensor(1))
    t_binary = torch.where(t_dataset == 0, torch.tensor(0), torch.tensor(1))
    print(t_binary)
    log_t = torch.log(t_dataset + epsilon)
    scaled_t = 2 * (log_t - log_min) / (log_max - log_min) - 1
    t_2_chan = torch.cat((scaled_t, t_bin_chan), dim=1)

    # Normalize validation dataset
    v_bin_chan = torch.where(v_dataset == 0, torch.tensor(-1), torch.tensor(1))
    v_binary = torch.where(v_dataset == 0, torch.tensor(0), torch.tensor(1))
    log_v = torch.log(v_dataset + epsilon)
    scaled_v = 2 * (log_v - log_min) / (log_max - log_min) - 1
    v_2_chan = torch.cat((scaled_v, v_bin_chan), dim=1)

    real_data_np = t_binary.squeeze(1).numpy()
    validation_data_np = v_binary.squeeze(1).numpy()
    synthetic_data_np = binary_masks.squeeze(1).numpy()

    ######### Get plots for binary data
    print("Getting Binary Data Plots....")

    bin_10000 = binary_masks.squeeze(1)[:10000]
    gen_labels_10000 = gen_labels[:10000]
    x_train_binary = t_binary.squeeze(1)
    print(x_train_binary)
    x_test_binary = v_binary.squeeze(1)
    train_lab = train_labels
    test_lab = val_labels
    print(train_lab)

    save_path_clipped = f"{study}/raw_gan"
    quality_folder = "gan_binary_data"
    analysis_dir = os.path.join(save_path_clipped, quality_folder)
    

    if not os.path.exists(save_path_clipped):
        os.makedirs(save_path_clipped, exist_ok = True)

        X_size = x_train_binary.shape[0]
        indices = np.random.choice(binary_masks.shape[0], size=X_size, replace=False)
        X_fake_xsize = binary_masks.squeeze(1)[indices]
        y_fake_xsize = gen_labels[indices]
        sparsity_train = np.mean(x_train_binary.numpy() == 0, axis=1)
        sparsity_fake = np.mean(X_fake_xsize.numpy() == 0, axis=1)
        print(sparsity_fake)
        print(sparsity_train)

        num_randm = X_size
        num_features = x_train_binary.shape[1]
        X_random = np.random.randint(2, size=(num_randm, num_features))
        sparsity_random = np.mean(X_random == 0, axis=1)

        # Third plot: Histograms by disease type
        unique_diseases = np.unique(train_lab)
        label_mapping = label_dict
        for disease in unique_diseases:
            fig_disease, ax_disease = plt.subplots(1, 1, figsize=(10, 8))
            
            mask_real = (train_lab == disease)
            mask_fake = (y_fake_xsize == disease)
            
            sparsity_train_disease = sparsity_train[mask_real]
            sparsity_fake_disease = sparsity_fake[mask_fake]
            sparsity_random_disease = sparsity_random[mask_real]  # Assuming random data uses the same mask
            
            flat_real_disease = sparsity_train_disease.flatten()
            flat_fake_disease = sparsity_fake_disease.flatten()
            flat_random_disease = sparsity_random_disease.flatten()

            ks_stat, ks_pvalue = ks_2samp(flat_real_disease, flat_fake_disease)
            
            # Plot histograms
            ax_disease.hist(flat_real_disease, bins=100, alpha=0.5, label='Real Data')
            ax_disease.hist(flat_fake_disease, bins=100, alpha=0.5, label='Synthetic Data')
            ax_disease.hist(flat_random_disease, bins=100, alpha=0.5, label='Random Sparsity Values')

            # Title and labels
            ax_disease.set_title(f'Histograms of Real, Synthetic, and Random Sparsity for {label_mapping[disease]}', fontsize=16)
            ax_disease.set_xlabel('Sparsity', fontsize=14)
            ax_disease.set_ylabel('Frequency', fontsize=14)
            ax_disease.tick_params(axis='both', labelsize=12)

            # Legend
            ax_disease.legend(loc='upper right', fontsize=10)

            # Add p-value at the top center
            ax_disease.text(
                0.3, 0.98,
                f'KS-test p-value (Real vs Synthetic): {ks_pvalue:.2e}',
                transform=ax_disease.transAxes,
                fontsize=12,
                ha='center',
                va='top',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
            )

            plt.tight_layout()

            # Optionally, save the figure
            file_path_disease = os.path.join(save_path_clipped, f'sparsity_plots_disease_{label_mapping[disease]}.png')
            plt.savefig(file_path_disease)

        # Range of generated data sizes to test
        gen_data_sizes = np.unique(np.geomspace(1, bin_10000.shape[0], num=100, dtype=int))

        # Store AUC scores
        auc_scores = []

        # Store AUC scores and corresponding data sizes
        auc_scores = []
        data_sizes = []

        # Initial model without augmentation
        log_reg = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
        log_reg.fit(x_train_binary, train_lab)
        probs_val_initial = log_reg.predict_proba(x_test_binary)[:, 1]  # Probability for positive class
        initial_auc = roc_auc_score(test_lab, probs_val_initial)
        best_auc = initial_auc

        auc_scores.append(initial_auc)
        data_sizes.append(0)  # No additional generated data

        # Models with increasing amounts of randomly sampled generated data
        for size in gen_data_sizes[1:]:  # Skip 0 since we already have the initial AUC
            if size == 0:
                continue  # Avoid empty sample case

            # Randomly sample 'size' examples from generated data
            indices = np.random.choice(bin_10000.shape[0], size=size, replace=False)
            X_augmented = np.vstack((x_train_binary, bin_10000[indices]))
            labels_augmented = np.concatenate((train_lab, gen_labels_10000[indices]))

            log_reg_augmented = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)
            log_reg_augmented.fit(X_augmented, labels_augmented)

            probs_val_augmented = log_reg_augmented.predict_proba(x_test_binary)[:, 1]
            augmented_auc = roc_auc_score(test_lab, probs_val_augmented)
            if augmented_auc > best_auc:
                best_auc = augmented_auc

            auc_scores.append(augmented_auc)
            data_sizes.append(size)
            
        # Plot and save
        plt.figure(figsize=(8, 6))
        plt.plot(data_sizes, auc_scores, marker='o', linestyle='-', color='b')
        plt.xlabel("Number of Synthetic Samples")
        plt.ylabel("AUC Score")
        plt.title(f"Logistic AUC vs. Synthetic Data Augmentation")
        plt.grid(True)

        # Annotate initial and best AUC in top-right corner
        textstr = f'Initial AUC: {initial_auc:.2f}\nBest AUC: {best_auc:.2f}'
        plt.text(
            0.98, 0.98, textstr,
            transform=plt.gca().transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
        )

        # Define filename and full path
        fig_path = os.path.join(save_path_clipped, "auc_vs_synthetic_data.png")

        # Save the figure
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')  # Use before plt.show()

        if not os.path.exists(analysis_dir):
            analyze_processed_data(binary_masks.squeeze(1), gen_labels, num_samples, x_train_binary, train_lab, x_test_binary, test_lab, label_dict, save_path_clipped, quality_folder, already_clr = False)
        else:
            print(f"skipping unnormalized clipped analysis")

    else:
        print("skipping binary GAN analysis")

    # Compute sparsity for each dataset
    sparsity_train = np.mean(real_data_np == 0, axis=1)
    sparsity_validation = np.mean(validation_data_np == 0, axis=1)
    sparsity_synthetic = np.mean(synthetic_data_np == 0, axis=1)

    # Perform pairwise statistical tests
    mw_train_val = mannwhitneyu(sparsity_train, sparsity_validation, alternative="two-sided")
    mw_train_synth = mannwhitneyu(sparsity_train, sparsity_synthetic, alternative="two-sided")
    mw_val_synth = mannwhitneyu(sparsity_validation, sparsity_synthetic, alternative="two-sided")

    # Get significance stars for comparisons
    p_star_train_val = get_p_value_star(mw_train_val.pvalue)
    p_star_train_synth = get_p_value_star(mw_train_synth.pvalue)
    p_star_val_synth = get_p_value_star(mw_val_synth.pvalue)

    # Boxplot with significance annotations
    plt.figure(figsize=(8, 5))
    ax = sns.boxplot(data=[sparsity_train, sparsity_validation, sparsity_synthetic], width=0.5)
    plt.xticks([0, 1, 2], ["Train", "Validation", "Synthetic"])
    plt.ylabel("Sparsity (Fraction of absent features)")
    plt.title("Sparsity Distribution")

    # Add significance annotations
    x1, x2, x3 = 0, 1, 2  # Position of groups
    y_max = max(max(sparsity_train), max(sparsity_validation), max(sparsity_synthetic)) + 0.02  # Adjust y position

    # Train vs Validation
    plt.plot([x1, x1, x2, x2], [y_max, y_max + 0.02, y_max + 0.02, y_max], lw=1.5, c="black")
    plt.text((x1 + x2) / 2, y_max + 0.03, p_star_train_val, ha='center', va='bottom', fontsize=12)

    # Train vs Synthetic
    plt.plot([x1, x1, x3, x3], [y_max + 0.05, y_max + 0.07, y_max + 0.07, y_max + 0.05], lw=1.5, c="black")
    plt.text((x1 + x3) / 2, y_max + 0.08, p_star_train_synth, ha='center', va='bottom', fontsize=12)

    # Validation vs Synthetic
    plt.plot([x2, x2, x3, x3], [y_max + 0.10, y_max + 0.12, y_max + 0.12, y_max + 0.10], lw=1.5, c="black")
    plt.text((x2 + x3) / 2, y_max + 0.13, p_star_val_synth, ha='center', va='bottom', fontsize=12)

    quality_folder = f"{study}/check_quality"
    os.makedirs(quality_folder, exist_ok=True)

    # Save the plot
    sparsity_plot_path = os.path.join(quality_folder, "sparsity_distribution.png")
    plt.savefig(sparsity_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Save statistical test results
    stats_results = pd.DataFrame({
        "Comparison": ["Train vs Validation", "Train vs Synthetic", "Validation vs Synthetic"],
        "Mann-Whitney U Statistic": [mw_train_val.statistic, mw_train_synth.statistic, mw_val_synth.statistic],
        "p-value": [mw_train_val.pvalue, mw_train_synth.pvalue, mw_val_synth.pvalue]
    })

    stats_file_path = os.path.join(quality_folder, "sparsity_stats_wilcoxon.csv")
    stats_results.to_csv(stats_file_path, index=False)

    # Filter values
    scaled_t_filtered = scaled_t[scaled_t > -1].cpu().numpy().flatten()
    t_dataset_filtered = t_dataset[t_dataset > 0].cpu().numpy().flatten()

    # Plot and save histogram for scaled_t
    plt.figure(figsize=(8, 5))
    plt.hist(scaled_t_filtered, bins=50, alpha=0.7, color="blue", edgecolor="black")
    plt.xlabel("Scaled_t Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Norm. Train Values (> -1)")
    scaled_t_hist_path = os.path.join(quality_folder, "scaled_t_histogram.png")
    plt.savefig(scaled_t_hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram to: {scaled_t_hist_path}")

    # Plot and save histogram for t_dataset
    plt.figure(figsize=(8, 5))
    plt.hist(t_dataset_filtered, bins=50, alpha=0.7, color="green", edgecolor="black")
    plt.xlabel("T_dataset Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Train Values (> 0)")
    t_dataset_hist_path = os.path.join(quality_folder, "t_dataset_histogram.png")
    plt.savefig(t_dataset_hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram to: {t_dataset_hist_path}")

    norm_min_train = torch.min(scaled_t[scaled_t > -1])
    raw_min_train = torch.min(t_dataset[t_dataset > 0])

    scaled_v_filtered = scaled_v[scaled_v > -1].cpu().numpy().flatten()
    v_dataset_filtered = v_dataset[v_dataset > 0].cpu().numpy().flatten()

    # ✅ Plot and save histogram for scaled_v
    plt.figure(figsize=(8, 5))
    plt.hist(scaled_v_filtered, bins=50, alpha=0.7, color="red", edgecolor="black")
    plt.xlabel("Scaled_v Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Norm. Validation Values (> -1)")
    scaled_v_hist_path = os.path.join(quality_folder, "scaled_v_histogram.png")
    plt.savefig(scaled_v_hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram to: {scaled_v_hist_path}")

    # ✅ Plot and save histogram for v_dataset
    plt.figure(figsize=(8, 5))
    plt.hist(v_dataset_filtered, bins=50, alpha=0.7, color="purple", edgecolor="black")
    plt.xlabel("V_dataset Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Validation Values (> 0)")
    v_dataset_hist_path = os.path.join(quality_folder, "v_dataset_histogram.png")
    plt.savefig(v_dataset_hist_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved histogram to: {v_dataset_hist_path}")

    # Run each experiment
    for i, hp_values in enumerate(all_experiments):

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        hp_dict = dict(zip(hyperparam_space.keys(), hp_values))

        num_channels = hp_dict["num_channels"]
        inpaint = hp_dict["inpaint"]
        batch_size = hp_dict["batch_size"]
        lr = hp_dict["lr"]
        timesteps = hp_dict["timesteps"]
        use_film = hp_dict["use_film"]

        out_channels = 1
        set_masks = inpaint
        train_d = t_2_chan if num_channels > 1 else scaled_t
        validation_d = v_2_chan if num_channels > 1 else scaled_v

        train_dataset = Dataset1D(train_d, train_labels, return_data=True, return_labels=True)
        val_dataset = Dataset1D(validation_d, val_labels, return_data=True, return_labels=True)

        save_run = f"exp_{i}_E{epochs}_{num_channels}ch_{inpaint}inpaint_{batch_size}bs_{lr}lr_{timesteps}ts_{use_film}UseFilm_{log_likelihood}"
        save_path = os.path.join(study, save_run)

        # Check if experiment folder exists
        if os.path.exists(save_path):
            print(f"Skipping experiment {i}: Folder {save_path} already exists.")
            continue  # Skip to the next experiment

        os.makedirs(save_path, exist_ok=True)

        print(f"Starting Experiment {i + 1}/{len(all_experiments)}: {hp_dict}")

        # Initialize model
        model = UNet1DModel(
            sample_size=feature_size,
            in_channels=num_channels,
            out_channels=out_channels,
            num_classes=num_classes,
            use_film=use_film
        )

        # Initialize trainer
        trainer = UNet1DTrainer(
            model, train_dataset, masks=None, inpaint=inpaint,
            val_dataset=val_dataset, batch_size=batch_size,
            lr=lr, timestep_range=(0, timesteps), save_path=save_path
        )

        # Train
        start_time_diff_train = time.perf_counter()
        trainer.train(epochs=epochs)
        elapsed_time_diff_train = time.perf_counter() - start_time_diff_train
        log_run(run_log_path, "diffusion_training", exp_name, epochs, elapsed_time_diff_train, extra={"device": str(trainer.device), "num_channels": num_channels, "timesteps": timesteps})

        print(f"Experiment {i + 1}/{len(all_experiments)} completed. Results saved in {save_path}\n")


    best_experiments_list = []

    # Run each experiment
    for i, hp_values in enumerate(all_experiments):

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        hp_dict = dict(zip(hyperparam_space.keys(), hp_values))

        num_channels = hp_dict["num_channels"]
        inpaint = hp_dict["inpaint"]
        batch_size = hp_dict["batch_size"]
        lr = hp_dict["lr"]
        timesteps = hp_dict["timesteps"]
        use_film = hp_dict["use_film"]

        out_channels = 1
        set_masks = inpaint
        train_d = t_2_chan if num_channels > 1 else scaled_t
        validation_d = v_2_chan if num_channels > 1 else scaled_v

        train_dataset = Dataset1D(train_d, train_labels, return_data=True, return_labels=True)
        val_dataset = Dataset1D(validation_d, val_labels, return_data=True, return_labels=True)

        save_run = f"exp_{i}_E{epochs}_{num_channels}ch_{inpaint}inpaint_{batch_size}bs_{lr}lr_{timesteps}ts_{use_film}UseFilm_{log_likelihood}"
        save_path = os.path.join(study, save_run)

        # Use film applies FiLM (Feature-wise Linear Modulation) in ResidualTemporalBlock1D instead of directly adding class to time embedding
        model = UNet1DModel(sample_size = feature_size, in_channels = num_channels, out_channels = out_channels, num_classes = num_classes, use_film = use_film)

        # use masks variable if inpainting from GAN binary generated samples
        # Initialize trainer
        trainer = UNet1DTrainer(model, train_dataset, masks = None, inpaint = inpaint, val_dataset=val_dataset, batch_size=batch_size, lr=lr, timestep_range=(0, timesteps), save_path=save_path)
        
        model.load_state_dict(torch.load(os.path.join(save_path, 'Models', 'best_unet1d.pth')))
        trainer.model.to(trainer.device)

        num_samples = num_samples

        gen_path = os.path.join(save_path, f"generated_{num_samples}.pt")
        label_path = os.path.join(save_path, f"generated_labels_{num_samples}.pt")
        score_path = os.path.join(save_path, f"generated_scores_{num_samples}.pt")
        mask_path = os.path.join(gan_path, f"masks_{exp_name}_{num_samples}_1")

        if os.path.exists(gen_path):
            print(f"Skipping generation, already exists in {save_path}")
        else:
            print(f"Generating samples...")

            if log_likelihood == "no_log_likelihood":

                gan_path = "/home/jbutler/DiffMicro/GAN_bin_masks/binary_gan_femai/"
                x_file = f"X_gen_{exp_name}_{num_samples}_1"
                y_file = f"y_gen_{exp_name}_{num_samples}_1"
                x_gan = x_file
                y_gan = y_file

                gen_masks = torch.load(f'{gan_path}{x_gan}').unsqueeze(1)
                binary_masks = torch.from_numpy(np.where(gen_masks > .5, 1, 0))
                gen_labels = torch.tensor(torch.load(f'{gan_path}/{y_gan}'))

                masks = binary_masks[:num_samples] if set_masks or num_channels == 2 else None

                if set_masks == True or num_channels == 2:
                    class_labels = gen_labels[:num_samples]
                else:
                    class_labels = torch.randint(0, num_classes, (num_samples,))

                # Prepare files
                if os.path.exists(gen_path):
                    os.remove(gen_path)
                if os.path.exists(label_path):
                    os.remove(label_path)
                if os.path.exists(score_path):
                    os.remove(score_path)
                if os.path.exists(mask_path):
                    os.remove(mask_path)

                batch_size = 10000
                num_batches = (num_samples + batch_size - 1) // batch_size  # ceil division

                for i in range(num_batches):
                    start = i * batch_size
                    end = min(start + batch_size, num_samples)
                    current_batch_size = end - start
                    print(f"\n🚀 Generating batch {i+1}/{num_batches} ({current_batch_size} samples)")

                    # Slice class labels and masks for this batch
                    batch_labels = class_labels[start:end].to(trainer.device)
                    batch_masks = masks[start:end] if masks is not None else None

                    # Generate samples
                    
                    start_time_diff_gen = time.perf_counter()

                    gen_batch, score_batch = trainer.generate_samples_likelihood(
                        num_samples=current_batch_size,
                        num_steps=timesteps,
                        data_shape=(1, feature_size),
                        class_labels=batch_labels,
                        inpaint_masks=batch_masks
                    )

                    elapsed_time_diff_gen = time.perf_counter() - start_time_diff_gen
                    log_run(run_log_path, f"diffusion_generation_batch_{i+1}", exp_name, None, elapsed_time_diff_gen,
                            extra={"device": str(trainer.device), "batch_samples": current_batch_size, "num_steps": timesteps})

                    print(f"✅ Batch {i+1} complete. Samples: {gen_batch.shape[0]}")

                    # Save or append to disk
                    if i == 0:
                        torch.save(gen_batch, gen_path)
                        torch.save(batch_labels, label_path)
                        torch.save(score_batch, score_path)
                        if batch_masks is not None:
                            torch.save(batch_masks, mask_path)
                    else:
                        # Load previous
                        prev_gen = torch.load(gen_path)
                        prev_score = torch.load(score_path)
                        prev_labels = torch.load(label_path)
                        torch.save(torch.cat([prev_gen, gen_batch], dim=0), gen_path)
                        torch.save(torch.cat([prev_score, score_batch], dim=0), score_path)
                        torch.save(torch.cat([prev_labels, batch_labels], dim=0), label_path)

                        if batch_masks is not None:
                            prev_masks = torch.load(mask_path)
                            torch.save(torch.cat([prev_masks, batch_masks], dim=0), mask_path)

            elif log_likelihood == "log_likelihood":

                generate_dual_gan_samples(
                    gan_path="/home/jbutler/DiffMicro/GAN_bin_masks/binary_gan_femai/",
                    exp_name= exp_name,
                    num_samples= num_samples,
                    set_masks= set_masks,
                    num_channels= num_channels,
                    num_classes= num_classes,
                    feature_size= feature_size,
                    timesteps= timesteps,
                    trainer=trainer,
                    gen_path= gen_path,
                    label_path= label_path,
                    score_path= score_path,
                    mask_path= os.path.join(save_path, f"allmasks_{num_samples}.pt"),  # Can be None if not saving masks
                )

                #gan_path = "/home/jbutler/DiffMicro/GAN_bin_masks/binary_gan_femai/"
                #x_file = f"X_gen_{exp_name}_{num_samples}_1"
                #y_file = f"y_gen_{exp_name}_{num_samples}_1"
                #x_gan = x_file
                #y_gan = y_file

                #gen_masks = torch.load(f'{gan_path}{x_gan}').unsqueeze(1)
                #binary_masks = torch.from_numpy(np.where(gen_masks > .5, 1, 0))
                #gen_labels = torch.tensor(torch.load(f'{gan_path}/{y_gan}'))

                #masks = binary_masks[:num_samples] if set_masks or num_channels == 2 else None

                #if set_masks == True or num_channels == 2:
                    #class_labels = gen_labels[:num_samples]
                #else:
                    #class_labels = torch.randint(0, num_classes, (num_samples,))

                #top_k_ratio = 0.5

                #collected_samples = []
                #collected_scores = []
                #collected_labels = []
                #collected_masks = []

                # Step 1: First generation - determine score cutoff
            #     initial_generated_samples, initial_scores = trainer.generate_samples_likelihood(
            #         num_samples=num_samples,
            #         num_steps=timesteps,
            #         data_shape=(1, feature_size),
            #         class_labels=class_labels.to(trainer.device),
            #         inpaint_masks=masks
            #     )

            #     # Determine cutoff based on top-k
            #     num_top = int(num_samples * top_k_ratio)
            #     top_indices = torch.topk(initial_scores, num_top).indices
            #     score_cutoff = initial_scores[top_indices[-1]]
            #     print(f"✔ Score cutoff set at: {score_cutoff.item():.4f}")

            #     # Filter samples meeting or exceeding the cutoff
            #     keep_mask = initial_scores >= score_cutoff
            #     kept_samples = initial_generated_samples[keep_mask]
            #     kept_scores = initial_scores[keep_mask]
            #     kept_labels = class_labels[keep_mask]
            #     kept_masks = masks[keep_mask]

            #     collected_samples.append(kept_samples)
            #     collected_scores.append(kept_scores)
            #     collected_labels.append(kept_labels)
            #     collected_masks.append(kept_masks)

            #     total_collected = kept_samples.size(0)
            #     print(f"✔ Initial collected: {total_collected} / {num_samples}")

            #     x_file = f"X_gen_{exp_name}_{num_samples}_2"
            #     y_file = f"y_gen_{exp_name}_{num_samples}_2"
            #     x_gan = x_file
            #     y_gan = y_file

            #     gen_masks = torch.load(f'{gan_path}{x_gan}').unsqueeze(1)
            #     binary_masks = torch.from_numpy(np.where(gen_masks > .5, 1, 0))
            #     gen_labels = torch.tensor(torch.load(f'{gan_path}/{y_gan}'))

            #     masks = binary_masks[:num_samples] if set_masks or num_channels == 2 else None

            #     if set_masks == True or num_channels == 2:
            #         class_labels = gen_labels[:num_samples]
            #     else:
            #         class_labels = torch.randint(0, num_classes, (num_samples,))

            #     # Step 1: First generation - determine score cutoff
            #     initial_generated_samples, initial_scores = trainer.generate_samples_likelihood(
            #         num_samples=num_samples,
            #         num_steps=timesteps,
            #         data_shape=(1, feature_size),
            #         class_labels=class_labels.to(trainer.device),
            #         inpaint_masks=masks
            #     )

            #     # Determine cutoff based on top-k
            #     num_top = int(num_samples * top_k_ratio)
            #     top_indices = torch.topk(initial_scores, num_top).indices
            #     score_cutoff = initial_scores[top_indices[-1]]
            #     print(f"✔ Score cutoff set at: {score_cutoff.item():.4f}")

            #     # Filter samples meeting or exceeding the cutoff
            #     keep_mask = initial_scores >= score_cutoff
            #     kept_samples = initial_generated_samples[keep_mask]
            #     kept_scores = initial_scores[keep_mask]
            #     kept_labels = class_labels[keep_mask]
            #     kept_masks = masks[keep_mask]

            #     collected_samples.append(kept_samples)
            #     collected_scores.append(kept_scores)
            #     collected_labels.append(kept_labels)
            #     collected_masks.append(kept_masks)

            #     total_collected = kept_samples.size(0)
            #     print(f"✔ Initial collected: {total_collected} / {num_samples}")

            #     # Final concatenation and trimming to exactly `num_samples`
            #     final_samples = torch.cat(collected_samples, dim=0)[:num_samples]
            #     final_scores = torch.cat(collected_scores, dim=0)[:num_samples]
            #     final_labels = torch.cat(collected_labels, dim=0)[:num_samples]
            #     final_masks = torch.cat(collected_masks, dim=0)[:num_samples]

            #     print(f"\n✅ Finished generation. Final sample count: {final_samples.shape[0]}")

            #     torch.save(final_samples, gen_path)
            #     torch.save(final_labels, label_path)
            #     torch.save(final_scores, score_path)

            # else:
            #     print("No value for log-likelihood found...")
                
        generated_samples = torch.load(gen_path)
        class_labels_og = torch.load(label_path)

        quality_folder = "log_data"

        os.makedirs(os.path.join(save_path, quality_folder), exist_ok=True)

        # Convert tensor to numpy and flatten
        generated_samples_np = generated_samples.cpu().numpy().flatten()
        generated_samples_filtered = generated_samples_np[generated_samples_np > -1]

        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.hist(generated_samples_np, bins=50, alpha=0.7, color="purple", edgecolor="black")
        plt.xlabel("Generated Sample Values (> -1)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Generated Samples (> -1)")

        # Save the plot
        generated_samples_hist_path = os.path.join(save_path, quality_folder, "generated_samples_histogram_filtered.png")
        plt.savefig(generated_samples_hist_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved histogram to: {generated_samples_hist_path}")

        generated_samples_np = generated_samples.cpu().numpy().flatten()
        generated_samples_filtered = generated_samples_np[(generated_samples_np > -1) & (generated_samples_np < norm_min_train.numpy())]

        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.hist(generated_samples_filtered, bins=50, alpha=0.7, color="orange", edgecolor="black")
        plt.xlabel("Generated Sample Values (-1 to Norm Min Train)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Generated Samples (-1 to Norm Min Train)")

        # Save the plot
        generated_samples_hist_path = os.path.join(save_path, quality_folder, "generated_samples_histogram_filtered_range.png")
        plt.savefig(generated_samples_hist_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved histogram to: {generated_samples_hist_path}")

        generated_samples_final = generated_samples

        #generated_samples_final = torch.where(generated_samples < norm_min_train, -1, generated_samples)

        save_path_clipped = f"{save_path}/clipped_data"
        os.makedirs(save_path_clipped, exist_ok=True)
        os.makedirs(os.path.join(save_path_clipped, quality_folder), exist_ok=True)

        # Convert tensor to numpy and flatten
        generated_samples_np = generated_samples_final.cpu().numpy().flatten()
        generated_samples_filtered = generated_samples_np[generated_samples_np > -1]

        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.hist(generated_samples_filtered, bins=50, alpha=0.7, color="purple", edgecolor="black")
        plt.xlabel("Clipped Generated Sample Values (> -1)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Generated Samples (> -1)")

        # Save the plot
        generated_samples_hist_path = os.path.join(save_path_clipped, quality_folder, "clipped_generated_samples_histogram_filtered.png")
        plt.savefig(generated_samples_hist_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved histogram to: {generated_samples_hist_path}")

        generated_samples_np = generated_samples_final.cpu().numpy().flatten()
        generated_samples_filtered = generated_samples_np[(generated_samples_np > -1) & (generated_samples_np < norm_min_train.numpy())]

        # Plot histogram
        plt.figure(figsize=(8, 5))
        plt.hist(generated_samples_filtered, bins=50, alpha=0.7, color="orange", edgecolor="black")
        plt.xlabel("Clipped Generated Sample Values (-1 to Norm Min Train)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Generated Samples (-1 to Norm Min Train)")

        # Save the plot
        generated_samples_hist_path = os.path.join(save_path_clipped, quality_folder, "clipped_generated_samples_histogram_filtered_range.png")
        plt.savefig(generated_samples_hist_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved histogram to: {generated_samples_hist_path}")

        data = generated_samples_final.clone()
        # data.add_(1).div_(2).mul_(log_max - log_min).add_(log_min)  # Normalize & Scale
        # data.exp_().sub_(epsilon)  # Exponentiate & Adjust
        # # This clip ensures values that should be -1 remain -1
        # data.clamp_(min=1e-16)  # In-place operation to ensure minimum value
        # data.sub_(1e-16).relu_()  # Set values below threshold to 0
        # data.squeeze_()  # In-place squeeze

        min_nonzero = torch.min(t_dataset[t_dataset > 0])

        # 2. Recover original scale from generated samples
        log_recovered = ((generated_samples + 1) / 2) * (log_max - log_min) + log_min
        t_recovered = torch.exp(log_recovered) - epsilon  # same shape as generated_samples

        t_recovered = torch.where(
            generated_samples < -0.5,
            torch.tensor(0.0, device=t_recovered.device, dtype=t_recovered.dtype),
            torch.clamp(t_recovered, min=min_nonzero)
        )

        processed_data = t_recovered.squeeze(1)

        real_data = exp.X_train.squeeze(1)
        real_labels = exp.y_train
        validation_data = exp.X_test.squeeze(1)
        validation_labels = exp.y_test
        gen = processed_data.squeeze(1)
        gen_label = class_labels_og

        quality_folder = "unnormalized_data"
        analysis_dir = os.path.join(save_path_clipped, quality_folder)
        if not os.path.exists(analysis_dir):
            analyze_processed_data(processed_data, gen_label, num_samples, real_data, real_labels, validation_data, validation_labels, label_dict, save_path_clipped, quality_folder, already_clr = False)
        else:
            print(f"skipping unnormalized clipped analysis")

        quality_folder = "check_quality_clr"
        analysis_dir = os.path.join(save_path_clipped, quality_folder)
        if not os.path.exists(analysis_dir):
            clr = custom_clr(processed_data)
            real_data_clr = torch.tensor(custom_clr(real_data))
            validation_data_clr = torch.tensor(custom_clr(validation_data))
            analyze_processed_data(clr, gen_label, num_samples, real_data_clr, real_labels, validation_data_clr, validation_labels, label_dict, save_path_clipped, quality_folder, already_clr = True)
        else:
            print(f"skipping clr clipped analysis")

        quality_folder = "check_quality_force_ra"
        analysis_dir = os.path.join(save_path_clipped, quality_folder)
        if not os.path.exists(analysis_dir):
            rel_ab = processed_data / processed_data.sum(dim=1, keepdim=True)
            analyze_processed_data(rel_ab, gen_label, num_samples, real_data, real_labels, validation_data, validation_labels, label_dict, save_path_clipped, quality_folder, already_clr = False)
        else:
            print(f"skipping force ra clipped analysis")
        
        quality_folder = "check_quality_force_ra_clr"
        analysis_dir = os.path.join(save_path_clipped, quality_folder)
        if not os.path.exists(analysis_dir):
            rel_ab = processed_data / processed_data.sum(dim=1, keepdim=True)
            ra_clr = custom_clr(rel_ab)
            real_data_clr = torch.tensor(custom_clr(real_data))
            validation_data_clr = torch.tensor(custom_clr(validation_data))
            analyze_processed_data(ra_clr, gen_label, num_samples, real_data_clr, real_labels, validation_data_clr, validation_labels, label_dict, save_path_clipped, quality_folder, already_clr = True)
        else:
            print(f"skipping force ra clr clipped analysis")

        