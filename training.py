import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

import os
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans, AgglomerativeClustering
from skbio.stats.ordination import pcoa
from skbio.diversity import beta_diversity
from collections import deque
from minimal_diffusion.analysis_functions_experiment import *


from scipy.spatial.distance import pdist, squareform, cdist

from ema_pytorch import EMA

from experimentor import Experimentor

# For random number generation
from numpy.random import randn, randint, seed

seed(0)

# Importing the custom functions
from scipy.stats import ks_2samp
import time

num_threads = 20
print(num_threads)
torch.set_num_threads(num_threads)

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper


class FileManager:
    def __init__(self, result_path, label_dict):
        self.result_path = result_path
        self.label_dict = label_dict
        self.paths = {
            'sample': None,
            'plot_overall': None,
            'pcoa_overall': None,
            'label': None,
        }
        self.paths.update({f'plot_{value}': None for value in label_dict.values()})
        self.paths.update({f'pcoa_{value}': None for value in label_dict.values()})

    def create_paths(self, epoch):
        self.paths['sample'] = f'{self.result_path}/X_fake_epoch_{epoch}.npy'
        self.paths['plot_overall'] = f'{self.result_path}/sparsity_plots_overall_{epoch}.png'
        self.paths['pcoa_overall'] = f'{self.result_path}/pcoa_plot_overall_{epoch}.png'
        self.paths['label'] = f'{self.result_path}/y_fake_epoch_{epoch}.npy'
        
        for value in self.label_dict.values():
            self.paths[f'plot_{value}'] = f'{self.result_path}/sparsity_plots_disease_{value}_{epoch}.png'
            self.paths[f'pcoa_{value}'] = f'{self.result_path}/pcoa_plot_{value}_{epoch}.png'

    def delete_existing_files(self):
        for path in self.paths.values():
            if path and os.path.exists(path):
                os.remove(path)

class Trainer():
    def __init__(self, generator, discriminator, classifier, gen_optimizer, dis_optimizer, cls_optimizer, X_train, y_train, X_test, y_test, nclasses, result_path, model_path, model_name, label_dict,
                 gp_weight=10, critic_iterations=5, print_every=5, auc_weight = 1.0, lambda_cls=0.1, dlc = 0):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.C = classifier
        self.D_opt = dis_optimizer
        self.C_opt = cls_optimizer
        self.losses = {'G': [], 'G_adv': [], 'D': [], 'GP': [], 'gradient_norm': [], 'G_cls': []}
        self.num_steps = 0
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.latent_dim = 100
        self.nclasses = nclasses
        self.result_path = result_path
        self.model_path = model_path
        self.model_name = model_name
        self.highest_score = 0
        self.lowest_loss = 10
        self.highest_perf = 0
        self.auc_weight = auc_weight
        self.lambda_cls = lambda_cls
        self.dlc = dlc

        self.X_train = X_train
        self.X_train_binary = np.where(self.X_train > 0.5, 1, 0)
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_binary = np.where(self.X_test > 0.5, 1, 0)

        self.p_value_save_count = 0

        # Pass only these buffers to EMA

        self.ema = EMA(self.G, beta=0.995, update_every=10)

        #self.auc_zf = self.compute_auc_zero(self.X_train_binary, self.y_train, self.X_test_binary, self.y_test)

        self.last_saved_crit = None
        self.last_saved_gen = None
        self.last_saved_gopt  = None
        self.last_saved_copt = None
        self.last_saved_ema = None

        self.last_saved_gen_loss = None
        self.last_saved_crit_loss = None
        self.last_saved_gopt_loss = None
        self.last_saved_copt_loss = None
        self.last_saved_ema_loss = None

        self.last_saved_gen_perf = None
        self.last_saved_crit_perf = None
        self.last_saved_gopt_perf = None
        self.last_saved_copt_perf = None
        self.last_saved_ema_perf = None

        self.start_epoch = 0
        self.label_dict = label_dict

        self.file_manager = FileManager(self.result_path, label_dict)

    def _critic_train_iteration(self, data):
            """ """

            data, label = data
            data = data.to(self.device)
            label = label.to(self.device)

            label = label.long()

            # Get generated data
            batch_size = data.size(0)
            generated_data, gen_label = self.sample_generator(batch_size, labels = label, use_ema=False)

            generated_data = generated_data.detach().cpu()

            # Calculate probabilities on real and generated data
            
            d_real = self.D(data, label)
            
            d_generated = self.D(generated_data, gen_label)

            # Get gradient penalty
            gradient_penalty = self._gradient_penalty(data, label, generated_data)
            self.losses['GP'].append(gradient_penalty.item())

            # Create total loss and optimize
            self.D_opt.zero_grad()
            d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
            d_loss.backward()

            self.D_opt.step()

            # Record loss
            self.losses['D'].append(d_loss.item())

            self.C_opt.zero_grad()
            cls_preds = self.C(data)
            c_loss = nn.CrossEntropyLoss()(cls_preds, label)
            c_loss.backward()
            self.C_opt.step()

    def _generator_train_iteration(self, data):
        """ """
        self.G_opt.zero_grad()

        data, label = data
        #data_binary = np.where(data.detach().cpu().numpy() > 0.5, 1, 0)
        #label_binary = label.detach().cpu().numpy()
        batch_size = data.size(0)
        label = label.long()

        #ata_binary = self.X_train_binary
        #label_binary = self.y_train

        # Get generated data
        # batch_size = data_binary.shape[0]

        generated_data, gen_label = self.sample_generator(batch_size, use_ema = False)

        # Compute AUC score
        #X_synth_binary = np.where(generated_data.detach().cpu().numpy() > 0.5, 1, 0)
        y_synth = gen_label.detach().cpu().numpy()

        # Ensure y_synth is in the correct format for the AUC calculation
        y_synth = y_synth.flatten()

        # auc_score = self.compute_auc_diff(data_binary, label_binary, self.X_test_binary, self.y_test, X_synth_binary, y_synth)
        # weighted_auc = self.auc_weight * auc_score

        # Calculate loss and optimize
        d_generated = self.D(generated_data, gen_label)
        g_loss_adv = -d_generated.mean()

        # Compute Classification Loss (Encourage Useful Data)
        cls_preds = self.C(generated_data)  # Classifier predictions
        g_loss_cls = nn.CrossEntropyLoss()(cls_preds, gen_label)

        # Combine Losses
        g_loss = g_loss_adv + self.lambda_cls * g_loss_cls

        g_loss.backward()
        self.G_opt.step()

        with open("model_parameters_and_buffers.txt", "w") as f:
            f.write("=== Checking G parameters ===\n")
            for name, param in self.G.named_parameters():
                f.write(f"{name}: {param.dtype}\n")
            f.write("=== Checking G buffers ===\n")
            for name, buf in self.G.named_buffers():
                f.write(f"{name}: {buf.dtype}\n")

            f.write("=== Checking D parameters ===\n")
            for name, param in self.D.named_parameters():
                f.write(f"{name}: {param.dtype}\n")
            f.write("=== Checking D buffers ===\n")
            for name, buf in self.D.named_buffers():
                f.write(f"{name}: {buf.dtype}\n")
        
        self.ema.update()

        # Record loss
        self.losses['G'].append(g_loss.item())
        self.losses['G_cls'].append(g_loss_cls.item())
        self.losses['G_adv'].append(g_loss_adv.item())

    def _gradient_penalty(self, real_data, label, generated_data):
        batch_size = real_data.size(0)

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated.requires_grad_(True)

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated, label)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs= torch.ones(prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
    
    @timeit
    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1

            img = data[0]
            label = data[1]

            # Assuming data[0] is the input and should be on the same device as the model
            inputs = img, label

            # Perform a critic (discriminator) training iteration
            self._critic_train_iteration(inputs)

            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(inputs)

            
    def train(self, data_loader, epochs, sample_every):

        for epoch in range(self.start_epoch, epochs):
            #print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

            d_loss_last = self.losses['D'][-1]
            g_loss_last = self.losses['G_adv'][-1]

            if (epoch + 1) % sample_every == 0:
                self.sample_images(epoch + 1, 10000, d_loss = d_loss_last, g_loss = g_loss_last)
                
                print(f"End of epoch: Processed {self.num_steps} steps")
                print(f"D: {self.losses['D'][-1]}")
                print(f"GP: {self.losses['GP'][-1]}")
                print(f"Gradient norm: {self.losses['gradient_norm'][-1]}")
                print(f"G: {self.losses['G'][-1]}")

    def sample_generator(self, num_samples, labels = None, use_ema=True):
        generator = self.ema.ema_model if use_ema else self.G
        noise, labels = generator.sample_latent(num_samples, labels=labels)
        generated_data = generator(noise, labels)
        return generated_data, labels
    

    def compute_auc_zero(self, X_train, y_train, X_test, y_test):
        X_train_in = np.asarray(X_train).squeeze()
        y_train_in = np.asarray(y_train).squeeze()
        X_test_in = np.asarray(X_test).squeeze()
        y_test_in = np.asarray(y_test).squeeze()

        if len(np.unique(y_train_in)) > 1:
            y_train_binarized = label_binarize(y_train_in, classes=np.unique(y_train_in))
            y_test_binarized = label_binarize(y_test_in, classes=np.unique(y_test_in))

            unique_labels = np.unique(y_train_in)
            if self.nclasses == 2:
                model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                model.fit(X_train_in, y_train_in)
                y_pred_proba = model.predict_proba(X_test_in)
                macro_avg_score = roc_auc_score(y_test_in, y_pred_proba[:, 1])
            else:
                model = LogisticRegression(penalty='l1', solver='saga', multi_class='ovr', max_iter=500)
                model.fit(X_train_in, y_train_in)
                y_pred_proba = model.predict_proba(X_test_in)
                macro_avg_score = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr', average='macro')

            score = macro_avg_score
        else:
            score = np.nan
        
        return score

    def compute_auc_diff(self, X_train, y_train, X_test, y_test, X_synth, y_synth, prop=1):
        X_train_in = np.asarray(X_train).squeeze()
        y_train_in = np.asarray(y_train).squeeze()
        X_test_in = np.asarray(X_test).squeeze()
        y_test_in = np.asarray(y_test).squeeze()
        X_synth_in = np.asarray(X_synth).squeeze()
        y_synth_in = np.asarray(y_synth).squeeze()

        scores_dict = {}

        nsynth = int(len(X_synth_in) * prop)
        synth_indices = np.random.choice(len(X_synth_in), nsynth, replace=False)
        X_synth_sub = X_synth_in[synth_indices]
        y_synth_sub = y_synth_in[synth_indices]
        X_train_comb = np.concatenate((X_train_in, X_synth_sub))
        y_train_comb = np.concatenate((y_train_in, y_synth_sub))

        if len(np.unique(y_train_comb)) > 1:
            y_train_binarized = label_binarize(y_train_comb, classes=np.unique(y_train_comb))
            y_test_binarized = label_binarize(y_test_in, classes=np.unique(y_test_in))

            unique_labels = np.unique(y_train_comb)
            if self.nclasses == 2:
                model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                model.fit(X_train_comb, y_train_comb)
                y_pred_proba = model.predict_proba(X_test_in)
                macro_avg_score = roc_auc_score(y_test_in, y_pred_proba[:, 1])
            else:
                model = LogisticRegression(penalty='l1', solver='saga', multi_class='ovr', max_iter=500)
                model.fit(X_train_comb, y_train_comb)
                y_pred_proba = model.predict_proba(X_test_in)
                macro_avg_score = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr', average='macro')

            score = macro_avg_score
        else:
            score = np.nan

        # Compute the difference between the scores for proportions 1 and 0
        diff = score - self.auc_zf
        
        return diff


    def load_model(self, epoch, type = None, load_ema=False):

        if type is None:

            generator_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_generator.pth'
            ema_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_generator_ema.pth'
            discriminator_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_critic.pth'
            G_optimizer_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_G_opt.pth'
            D_optimizer_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_D_opt.pth'
        
        else:

            generator_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_generator_{type}.pth'
            ema_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_generator_{type}_ema.pth'
            discriminator_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_critic_{type}.pth'
            G_optimizer_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_G_opt_{type}.pth'
            D_optimizer_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_D_opt_{type}.pth'


        self.G.load_state_dict(torch.load(generator_path))
        self.D.load_state_dict(torch.load(discriminator_path))
        self.G_opt.load_state_dict(torch.load(G_optimizer_path))
        self.D_opt.load_state_dict(torch.load(D_optimizer_path))
        self.ema.ema_model.load_state_dict(torch.load(ema_path))
        self.start_epoch = epoch + 1

    @timeit
    def sample_images(self, epoch, n_samples, d_loss, g_loss):

        save_by_p_value = False
        
        current_loss_score = g_loss
        if current_loss_score < self.lowest_loss:
            self.lowest_loss = current_loss_score  # Update the highest score

            #if self.last_saved_gen_loss and os.path.exists(self.last_saved_gen_loss):
                #os.remove(self.last_saved_gen_loss)
            
            #if self.last_saved_crit_loss and os.path.exists(self.last_saved_crit_loss):
                #os.remove(self.last_saved_crit_loss)

            #if self.last_saved_gopt_loss and os.path.exists(self.last_saved_gopt_loss):
                #os.remove(self.last_saved_gopt_loss)
            
            #if self.last_saved_copt_loss and os.path.exists(self.last_saved_copt_loss):
                #os.remove(self.last_saved_copt_loss)

            os.makedirs(self.model_path, exist_ok=True)
            gen_path_loss = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_loss.pth")
            crit_path_loss = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_critic_loss.pth") 
            torch.save(self.G.state_dict(), gen_path_loss)
            torch.save(self.D.state_dict(), crit_path_loss)

            
            gen_ema_path_loss = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_loss_ema.pth")
            torch.save(self.ema.ema_model.state_dict(), gen_ema_path_loss)

            gopt_path_loss = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_G_opt_loss.pth")
            copt_path_loss = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_D_opt_loss.pth") 
            torch.save(self.G_opt.state_dict(), gopt_path_loss)
            torch.save(self.D_opt.state_dict(), copt_path_loss)

            self.last_saved_gen_loss = gen_path_loss
            self.last_saved_crit_loss = crit_path_loss
            self.last_saved_gopt_loss = gopt_path_loss
            self.last_saved_copt_loss = copt_path_loss
            self.last_saved_ema_loss = gen_ema_path_loss

            gen_imgs, labels = self.sample_generator(n_samples)

            gen_imgs = gen_imgs.detach().cpu()
            labels = labels.detach().cpu()

            X_fake, y_fake = gen_imgs.reshape((-1, self.X_train.shape[1])), labels.cpu().numpy()

            os.makedirs(self.result_path, exist_ok=True)
            np.save(f'{self.result_path}/X_fake_epoch_{epoch}.npy', X_fake)
            np.save(f'{self.result_path}/y_fake_epoch_{epoch}.npy', y_fake)
        
        ### ADDED CODE HERE#####
        losses = np.array([epoch, d_loss, g_loss])
        
        loss_cols = ["epoch", "d_loss", "g_loss"]
                 # Check if the file exists
        
        os.makedirs(self.result_path, exist_ok=True)

        if not os.path.exists(self.result_path + '/loss.csv'):
            # Open the file in write mode, creating a new file if it does not exist
            with open(self.result_path + '/loss.csv', 'w', newline='') as csv_file:
                # Create a writer object
                writer = csv.writer(csv_file)
                # Write the column names as the first row of the CSV file
                writer.writerow(loss_cols)
                
        with open(self.result_path + '/loss.csv', 'a', newline='') as csv_file:
            # Create a writer object
            writer = csv.writer(csv_file)
            # Write the array as a row in the CSV file
            writer.writerow(losses)
        
        dfl= pd.read_csv(self.result_path + '/loss.csv')
        # Use EMA weights to generate images
        gen_imgs, labels = self.sample_generator(n_samples)

        gen_imgs = gen_imgs.detach().cpu()
        labels = labels.detach().cpu()

        X_fake, y_fake = gen_imgs.reshape((-1, self.X_train.shape[1])), labels.cpu().numpy()
        X_train_binary = np.where(self.X_train > .5, 1, 0)
        X_fake_binary = np.where(X_fake > 0.5, 1, 0)
        X_test_binary = np.where(self.X_test > 0.5, 1, 0)

        X_fake_xsize = X_fake_binary[:X_train_binary.shape[0]]
        y_fake_xsize = y_fake[:X_fake_xsize.shape[0]]

        unique_classes, counts = np.unique(y_fake, return_counts=True)

        if len(unique_classes) == self.nclasses and all(count >= 2 for count in counts):

            # see how well logistic can distinguish between real and fake

            track_df = self.classify(X_train_binary, self.y_train, X_fake_xsize, y_fake_xsize, X_test_binary, self.y_test, model='logi')

            track_df['epoch'] = epoch

            # Specify your file path
            file_path = os.path.join(self.result_path, 'track_results.csv')

            if os.path.exists(file_path):
                # File exists, load previous data, append new data, and save
                previous_data = pd.read_csv(file_path, index_col=0)
                updated_data = pd.concat([previous_data, track_df], axis=0)
                updated_data.to_csv(file_path, index=True)
            else:
                # First iteration or file doesn't exist, save current DataFrame directly
                track_df.to_csv(file_path, index=True)

            df = pd.read_csv(file_path, index_col=0)
            df['.25Diff'] = df['ROC AUC Score 0.25'] - df['ROC AUC Score 0']
            df['1Diff'] = df['ROC AUC Score 1'] - df['ROC AUC Score 0']
            score_metrics = ['Real_Val', 'RF_Val', 'Real_Fake', '.25Diff', '1Diff'], ['Real vs. Validation', 'R. v. F. Validation', 'Real vs. Fake', f'AUC ({X_train_binary.shape[0]} R {X_fake_xsize.shape[0] // 2} F) - AUC No Aug', f'AUC ({X_train_binary.shape[0]} R {X_fake_xsize.shape[0] * 2} F) - AUC No Aug']
            
            perf_score = df['ROC AUC Score 1'].iloc[-1]
            if perf_score > self.highest_perf:
                self.highest_perf = perf_score  # Update the highest score

                #if self.last_saved_gen_perf and os.path.exists(self.last_saved_gen_perf):
                 #   os.remove(self.last_saved_gen_perf)
                
                #if self.last_saved_crit_perf and os.path.exists(self.last_saved_crit_perf):
                 #   os.remove(self.last_saved_crit_perf)

                #if self.last_saved_gopt_perf and os.path.exists(self.last_saved_gopt_perf):
                 #   os.remove(self.last_saved_gopt_perf)
                
                #if self.last_saved_copt_perf and os.path.exists(self.last_saved_copt_perf):
                 #   os.remove(self.last_saved_copt_perf)

                os.makedirs(self.model_path, exist_ok=True)
                gen_path_perf = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_perf.pth")
                crit_path_perf = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_critic_perf.pth") 
                torch.save(self.G.state_dict(), gen_path_perf)
                torch.save(self.D.state_dict(), crit_path_perf)

                gen_ema_path_perf = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_perf_ema.pth")
                torch.save(self.ema.ema_model.state_dict(), gen_ema_path_perf)

                gopt_path_perf = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_G_opt_perf.pth")
                copt_path_perf = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_D_opt_perf.pth") 
                torch.save(self.G_opt.state_dict(), gopt_path_perf)
                torch.save(self.D_opt.state_dict(), copt_path_perf)

                self.last_saved_gen_perf = gen_path_perf
                self.last_saved_crit_perf = crit_path_perf
                self.last_saved_gopt_perf = gopt_path_perf
                self.last_saved_copt_perf = copt_path_perf
                self.last_saved_ema_perf = gen_ema_path_perf
            
            folder_name = self.result_path
            # Folder structure setup
            base_folder = os.path.join(self.result_path, 'Track_Performance')
            folder_names = df.index.unique()

            for folder_name in folder_names:
                # Create directory for each index
                path = os.path.join(base_folder, str(folder_name))
                os.makedirs(path, exist_ok=True)
                
                # Filter data for the current index
                current_df = df[df.index == folder_name]
                
                # Function to plot and save figures
                def plot_and_save(metrics, title, y_label, file_name, y_2 = False):
                    plt.figure(figsize=(10, 6))
                    metric_names, metric_labels = metrics
                    for metric, label in zip(metric_names, metric_labels):
                        plt.plot(current_df['epoch'].to_numpy(), current_df[metric].to_numpy(), label=label, marker='o')
                    plt.title(title)
                    plt.xlabel('epoch')
                    plt.ylabel(y_label)

                    #if y_2:
                        #plt.gca().twinx().set_ylabel('AUC Real vs. Fake Train & Validation')
                    
                    plt.legend()
                    plt.grid(True)
                    plt.savefig(os.path.join(path, file_name))
                    plt.close()
                
                # Plot for each category
                #plot_and_save(shannon_metrics, 'Shannon Metrics', 'Value','shannon_metrics.png')
                #plot_and_save(richness_metrics, 'Richness Metrics', 'Value', 'richness_metrics.png')
                #plot_and_save(beta_metrics, 'Beta Diversity Metrics', 'Value', 'beta_diversity_metrics.png')
                plot_and_save(score_metrics, 'Performance AUC Real vs. Fake and AUC Disease Augmented', 'AUC', 'scores_and_real_fake.png', y_2 = True)

        else:
            print("Sample classes are not represented or not enough samples, so tracking is skipped")

        threshold_range = np.arange(0.5, 1.0, 0.01)
        ks_statistics, p_values, avg_sparsity_trains, avg_sparsity_fakes, high_p_threshold, high_p_value = self.analyze_sparsity(X_train_binary, X_fake, threshold_range)

        if high_p_value > 0.05 and self.p_value_save_count < 5:  # ✅ Save if p_value > 0.05
            save_by_p_value = True
            self.p_value_save_count += 1

        if save_by_p_value:
            os.makedirs(self.model_path, exist_ok=True)

            # ✅ Define paths for saving
            gen_pv = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_pvalue.pth")
            crit_pv = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_critic_pvalue.pth")
            gen_ema_pv = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_ema_pvalue.pth")
            gopt_pv = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_G_opt_pvalue.pth")
            copt_pv = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_D_opt_pvalue.pth")

            # ✅ Save models and optimizers
            torch.save(self.G.state_dict(), gen_pv)
            torch.save(self.D.state_dict(), crit_pv)
            torch.save(self.ema.ema_model.state_dict(), gen_ema_pv)
            torch.save(self.G_opt.state_dict(), gopt_pv)
            torch.save(self.D_opt.state_dict(), copt_pv)
        
        fake_thresh = 0.5
        X_fake_binary = np.where(X_fake > fake_thresh, 1, 0) # otherwise set to high_p_threshold
        
        X_fake_xsize = X_fake_binary[0:X_train_binary.shape[0]]
        y_fake_xsize = y_fake[0:X_fake_xsize.shape[0]]
        
        sparsity_train = np.mean(X_train_binary == 0, axis=1)
        sparsity_fake = np.mean(X_fake_xsize == 0, axis=1)
        
        ###################### calculate synthetic same as X_train
        
        sparsity_scores = np.array([high_p_value])
        
        #np.array([epoch, d_loss, g_loss])
        
        scores = sparsity_scores
        
        all_scores = np.append(epoch, scores)
        
        score_cols = ["epoch", "p_value"]
        
        # Check if the file exists
        if not os.path.exists(self.result_path + '/ext_scores.csv'):
            # Open the file in write mode, creating a new file if it does not exist
            with open(self.result_path + '/ext_scores.csv', 'w', newline='') as csv_file:
                # Create a writer object
                writer = csv.writer(csv_file)
                # Write the column names as the first row of the CSV file
                writer.writerow(score_cols)

        with open(self.result_path + '/ext_scores.csv', 'a', newline='') as csv_file:
            # Create a writer object
            writer = csv.writer(csv_file)
            # Write the array as a row in the CSV file
            writer.writerow(all_scores)
        
        dfe = pd.read_csv(self.result_path + '/ext_scores.csv')

        self.store_best_generated_data(dfe, epoch, X_fake, y_fake, threshold_range, ks_statistics, p_values, sparsity_train, sparsity_fake, avg_sparsity_trains, avg_sparsity_fakes, X_train_binary, self.y_train, X_fake_xsize, y_fake_xsize, fake_thresh)
    

        dfep = dfe.set_index("epoch")
        dfepl = dfep.transpose().values.tolist()

        dflp = dfl.set_index("epoch")
        dflpl = dflp.transpose().values.tolist()

        score_cols = ["p_value"]

        fig = plt.figure(figsize=(18, 18), dpi=300)
        
        ax1 = plt.subplot2grid((4, 5), (0, 0), colspan=5)

        x = dfep.index
        lines = dfepl
        labels  = ["p_value"]

        # fig1 = plt.figure()
        for i, l in zip(lines, labels):  
            ax1.plot(np.array(x), np.array(i), label='l')
            ax1.legend(labels, loc="lower left")
            ax1.set_title("sparsity check")

        ax2 = plt.subplot2grid((4, 5), (1, 0), colspan=5)

        x = dflp.index
        lines = dflpl
        labels  = dflp.columns

        # fig1 = plt.figure()
        for i, l in zip(lines, labels):  
            ax2.plot(np.array(x),np.array(i), label='l')
            ax2.legend(labels, loc="lower left")
            ax2.set_title("loss")
        
        plt.savefig(self.result_path + "/track.png")
        plt.clf()
            
        #plt.savefig("images/mnist_%d.png" % epoch)
        #plt.close()

    @timeit
    def _pcoa_manual(self, distance_matrix):
        """Perform PCoA on a given distance matrix."""
        # Number of samples
        n_samples = distance_matrix.shape[0]

        # Center the distance matrix (Gower's centering)
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        B = -0.5 * H.dot(distance_matrix ** 2).dot(H)

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(B)

        # Sort eigenvectors and eigenvalues in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Select the principal coordinates
        principal_coords = eigvecs * np.sqrt(np.maximum(eigvals, 0))

        return eigvals, principal_coords

    @timeit
    def _project_new_data(self, new_data, original_data, principal_coords_original):
        """Project new data onto existing PCoA space."""
        # Calculate Jaccard distance from new points to original points
        new_distance_matrix = cdist(new_data, original_data, metric='jaccard')

        # Center the new distance matrix using the centering matrix H from the original data
        num_samples = original_data.shape[0]
        H = np.eye(num_samples) - np.ones((num_samples, num_samples)) / num_samples
        new_B = -0.5 * H.dot(new_distance_matrix ** 2).dot(H)

        # Project new points onto the PCoA space using the original eigenvectors
        projected_new_coords = np.linalg.lstsq(principal_coords_original, new_B.T, rcond=None)[0].T

        return projected_new_coords
    
    @timeit
    def store_best_generated_data(self, dfe, epoch, X_fake, y_fake, threshold_range, ks_statistics, p_values, sparsity_train, sparsity_fake, avg_sparsity_trains, avg_sparsity_fakes, X_train_binary, y_train, X_fake_xsize, y_fake_xsize, fake_thresh):
        if epoch in dfe['epoch'].values:
            row = dfe[dfe['epoch'] == epoch].iloc[0]
            current_score = row['p_value']

            # Check if the current score is higher than the highest score
            if current_score > self.highest_score:
                self.highest_score = current_score  # Update the highest score

                #if self.last_saved_gen and os.path.exists(self.last_saved_gen):
                    #os.remove(self.last_saved_gen)
                
                #if self.last_saved_crit and os.path.exists(self.last_saved_crit):
                    #os.remove(self.last_saved_crit)

                #if self.last_saved_gopt and os.path.exists(self.last_saved_gopt):
                    #os.remove(self.last_saved_gopt)
                
                #if self.last_saved_copt and os.path.exists(self.last_saved_copt):
                    #os.remove(self.last_saved_copt)

                #os.makedirs(self.model_path, exist_ok=True)
                #gen_path = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator.pth")
                #crit_path = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_critic.pth") 
                #torch.save(self.G.state_dict(), gen_path)
                #torch.save(self.D.state_dict(), crit_path)

                #gopt_path = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_G_opt.pth")
                #copt_path = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_D_opt.pth") 
                #torch.save(self.G_opt.state_dict(), gopt_path)
                #torch.save(self.D_opt.state_dict(), copt_path)

                #self.last_saved_gen = gen_path
                #self.last_saved_crit = crit_path
                #self.last_saved_gopt = gopt_path
                #self.last_saved_copt = copt_path

                # Convert to PyTorch tensors and save
                #np.save(f'{self.result_path}/X_fake_epoch_{epoch}.npy', X_fake)
                #np.save(f'{self.result_path}/y_fake_epoch_{epoch}.npy', y_fake)
                self.plot_sparsity(epoch, threshold_range, ks_statistics, p_values, sparsity_train, sparsity_fake, avg_sparsity_trains, avg_sparsity_fakes, X_train_binary, y_train, X_fake_xsize, y_fake_xsize, fake_thresh)
                

                # Define random binary matrix and labels
                def stratified_sample(X, y, n_samples_per_class):
                    X_sampled, y_sampled = [], []
                    unique_labels = np.unique(y)
                    for label in unique_labels:
                        # Get indices for the current label
                        label_indices = np.where(y == label)[0]
                        
                        # Sample from this label's data
                        sampled_indices = np.random.choice(label_indices, n_samples_per_class, replace=False)
                        X_sampled.append(X[sampled_indices])
                        y_sampled.append(y[sampled_indices])
                    
                    return np.vstack(X_sampled), np.hstack(y_sampled)

                n_samples_per_class = 50 

                # Sample real, fake, and random data
                X_train_sampled, y_train_sampled = stratified_sample(X_train_binary, y_train, n_samples_per_class)
                X_fake_sampled, y_fake_sampled = stratified_sample(X_fake_xsize, y_fake_xsize, n_samples_per_class)

                n_dimensions = 2

                # Define random binary matrix and labels
                num_random_samples = X_train_sampled.shape[0]  # Adjust this to the desired number of random samples
                num_features = X_train_binary.shape[1]  # Assuming the number of features is the same

                X_random = np.random.randint(2, size=(num_random_samples, num_features))
                y_random = np.random.choice([0, 1, 2], size=num_random_samples)  # Randomly assign labels

                real_distance_matrix = squareform(pdist(X_train_sampled, metric='jaccard'))

                # Perform PCoA on real data
                eigvals_real, principal_coords_real = self._pcoa_manual(real_distance_matrix)

                # Project fake samples onto the PCoA space of the real data
                projected_fake_coords = self._project_new_data(X_fake_sampled, X_train_sampled, principal_coords_real)

                # Project random samples onto the PCoA space of the real data
                projected_random_coords = self._project_new_data(X_random, X_train_sampled, principal_coords_real)


                # Select the top principal coordinates
                principal_coords_real = principal_coords_real[:, :n_dimensions]
                projected_fake_coords = projected_fake_coords[:, :n_dimensions]
                projected_random_coords = projected_random_coords[:, :n_dimensions]

                # Combine data for plotting
                combined_coords = np.vstack((principal_coords_real, projected_fake_coords, projected_random_coords))
                combined_labels = np.concatenate((y_train_sampled, y_fake_sampled, y_random))
                combined_types = ['real'] * len(y_train_sampled) + ['fake'] * len(y_fake_sampled) + ['random'] * len(y_random)

                pcoa_df = pd.DataFrame(combined_coords, columns=['PC1', 'PC2'])
                pcoa_df['label'] = combined_labels
                pcoa_df['type'] = combined_types


                # Replace numeric labels with textual ones
                label_mapping = self.label_dict
                pcoa_df['label'] = pcoa_df['label'].map(label_mapping)

                pcoa_df_sampled = pcoa_df

                predefined_colors = [
                    '#FF6347',  # tomato red
                    '#4682B4',  # steel blue
                    '#32CD32',  # lime green
                    '#FFD700',  # gold
                    '#FF4500',  # orange red
                    '#1E90FF',  # dodger blue
                    '#228B22',  # forest green
                    '#8A2BE2',  # blue violet
                    '#D2691E',  # chocolate
                    '#DC143C'   # crimson
                ]

                def generate_palettes(label_mapping):
                    palette_overall = {}
                    palette_label = {}
                    palette_group = {}

                    for i, (index, label) in enumerate(label_mapping.items()):
                        color = predefined_colors[i % len(predefined_colors)]
                        palette_overall[label] = color
                        palette_label[f'real_{label}'] = color
                        palette_label[f'fake_{label}'] = predefined_colors[(i + 1) % len(predefined_colors)]
                    
                    # Add 'random' color
                    palette_overall['random'] = '#FFD700'  # gold
                    palette_label['random'] = '#FFD700'    # gold
                    palette_group['random'] = '#FFD700'    # gold
                    
                    # Default colors for 'real' and 'fake' groups
                    palette_group['real'] = '#FF6347'  # tomato red
                    palette_group['fake'] = '#4682B4'  # steel blue
                    
                    return palette_overall, palette_label, palette_group

                palette_overall, palette_label, palette_group = generate_palettes(label_mapping)

                # Create a combined column to handle both 'label' and 'type'
                pcoa_df_sampled['combined'] = pcoa_df_sampled.apply(
                    lambda row: f"{row['type']}_{row['label']}" if row['type'] != 'random' else 'random', axis=1
                )

                pcoa_df_sampled['label_rand'] = pcoa_df_sampled.apply(
                    lambda row: 'random' if row['type'] == 'random' else row['label'], axis=1
                )

                # File paths
                plot_path_overall = f'{self.result_path}/pcoa_plot_overall_{epoch}.png'

                # Delete previous files if they exist
                if plot_path_overall and os.path.exists(plot_path_overall):
                    os.remove(plot_path_overall)

                # Create and save the overall plot
                plt.figure(figsize=(10, 8))
                sns.scatterplot(data=pcoa_df_sampled, x='PC1', y='PC2', hue='label_rand', style='type', s=100, palette=palette_overall,
                                markers={"real": "o", "fake": "^", "random": "s"})
                plt.title('PCoA of Real, Fake, and Random Samples by Label')
                plt.xlabel('Principal Coordinate 1')
                plt.ylabel('Principal Coordinate 2')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.savefig(plot_path_overall)
                plt.show()

                # Create and save individual plots for each label
                for label, group_df in pcoa_df_sampled.groupby('label'):
                    plot_path_label = f'{self.result_path}/pcoa_plot_{label}_{epoch}.png'
                    
                    # Delete previous files if they exist
                    if plot_path_label and os.path.exists(plot_path_label):
                        os.remove(plot_path_label)
                    
                    plt.figure(figsize=(10, 8))
                    sns.scatterplot(data=group_df, x='PC1', y='PC2', hue='type', s=100, palette=palette_group)
                    plt.title(f'PCoA of Real, Fake, and Random Samples by Label: {label}')
                    plt.xlabel('Principal Coordinate 1')
                    plt.ylabel('Principal Coordinate 2')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.savefig(plot_path_label)
                    plt.show()

                #self.file_manager.delete_existing_files()

                self.file_manager.create_paths(epoch)  
    
    @timeit
    def classify(self, X_train, y_train, X_synth, y_synth, X_test, y_test, model='logi'):

        results = []
        results_val = []
        results_rv = []

        X_train_in = np.asarray(X_train).squeeze()
        y_train_in = np.asarray(y_train).squeeze()
        X_test_in = np.asarray(X_test).squeeze()
        y_test_in = np.asarray(y_test).squeeze()
        X_synth_in = np.asarray(X_synth).squeeze()
        y_synth_in = np.asarray(y_synth).squeeze()
            
        unique_labels = np.unique(y_synth_in)
        real_labels = np.zeros_like(y_train_in)
        fake_labels = np.ones_like(y_synth_in)
        
        real_labels_val = np.zeros_like(y_test_in)

        comb_x = np.concatenate((X_train_in, X_synth_in))
        comb_y_dis = np.concatenate((y_train_in, y_synth_in))
        comb_y = np.concatenate((real_labels, fake_labels))

        comb_x_val = np.concatenate((X_test_in, X_synth_in))
        comb_y_dis_val = np.concatenate((y_test_in, y_synth_in))
        comb_y_val = np.concatenate((real_labels_val, fake_labels))

        real_labels_val_1 = np.ones_like(y_test_in)

        comb_x_rv = np.concatenate((X_test_in, X_train_in))
        comb_y_dis_rv = np.concatenate((y_test_in, y_train_in))
        comb_y_rv = np.concatenate((real_labels_val_1, real_labels))

        X_train, X_test, y_train, y_test = train_test_split(comb_x, comb_y, test_size=0.3, random_state=42)
        X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(comb_x_val, comb_y_val, test_size=0.3, random_state=42)
        X_train_rv, X_test_rv, y_train_rv, y_test_rv = train_test_split(comb_x_rv, comb_y_rv, test_size=0.3, random_state=42)

        if len(np.unique(y_train)) > 1: 

            model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
            model.fit(X_train, y_train)
            gen_pred = model.predict_proba(X_test)[:,1]
            overall_score = roc_auc_score(y_test, gen_pred)
            results.append({'Label': 'Overall', 'ROC AUC Score': overall_score})

            model_val = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
            model_val.fit(X_train_val, y_train_val)
            gen_pred_val = model_val.predict_proba(X_test_val)[:,1]
            overall_score_val = roc_auc_score(y_test_val, gen_pred_val)
            results_val.append({'Label': 'Overall', 'ROC AUC Score': overall_score_val})

            model_rv = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
            model_rv.fit(X_train_rv, y_train_rv)
            gen_pred_rv = model_rv.predict_proba(X_test_rv)[:,1]
            overall_score_rv = roc_auc_score(y_test_rv, gen_pred_rv)
            results_rv.append({'Label': 'Overall', 'ROC AUC Score': overall_score_rv})
        else:
            results.append({'Label': 'Overall', 'ROC AUC Score': np.nan})
            results_val.append({'Label': 'Overall', 'ROC AUC Score': np.nan})
            results_rv.append({'Label': 'Overall', 'ROC AUC Score': np.nan})

        for label in unique_labels:
            label_indices = (comb_y_dis == label)

            X_train, X_test, y_train, y_test = train_test_split(comb_x[label_indices], comb_y[label_indices], test_size=0.3, random_state=42)

            label_indices_val = (comb_y_dis_val == label)

            X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(comb_x_val[label_indices_val], comb_y_val[label_indices_val], test_size=0.3, random_state=42)

            label_indices_rv = (comb_y_dis_rv == label)

            X_train_rv, X_test_rv, y_train_rv, y_test_rv = train_test_split(comb_x_rv[label_indices_rv], comb_y_rv[label_indices_rv], test_size=0.3, random_state=42)

            if (len(X_train) > 0 and len(X_test) > 0) and len(np.unique(y_train)) > 1:

                model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                model.fit(X_train, y_train)
                gen_pred = model.predict_proba(X_test)[:,1]
                overall_score = roc_auc_score(y_test, gen_pred)
                results.append({'Label': label, 'ROC AUC Score': overall_score})

                model_val = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                model_val.fit(X_train_val, y_train_val)
                gen_pred_val = model_val.predict_proba(X_test_val)[:,1]
                overall_score_val = roc_auc_score(y_test_val, gen_pred_val)
                results_val.append({'Label': label, 'ROC AUC Score': overall_score_val})

                model_rv = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                model_rv.fit(X_train_rv, y_train_rv)
                gen_pred_rv = model_rv.predict_proba(X_test_rv)[:,1]
                overall_score_rv = roc_auc_score(y_test_rv, gen_pred_rv)
                results_rv.append({'Label': label, 'ROC AUC Score': overall_score})

            else:
                results.append({'Label': label, 'ROC AUC Score': np.nan})
                results_val.append({'Label': label, 'ROC AUC Score': np.nan})
                results_rv.append({'Label': label, 'ROC AUC Score': np.nan})

        bindf = pd.DataFrame(results).set_index('Label', inplace = False)
        bindf.index.name = None
        bindfv = pd.DataFrame(results_val).set_index('Label', inplace = False)
        bindfv.index.name = None
        bindfrv = pd.DataFrame(results_rv).set_index('Label', inplace = False)
        bindfrv.index.name = None
        
        ########

        proportions = [0, .25, 1]
        scores_df = {'Label': ['Overall'] + list(range(len(set(y_train_in))))}

        for prop in proportions:

            nsynth = int(len(X_synth_in) * prop)
            synth_indices = np.random.choice(len(X_synth_in), nsynth, replace=False)
            X_synth_sub = X_synth_in[synth_indices]
            y_synth_sub = y_synth_in[synth_indices]
            X_train_comb = np.concatenate((X_train_in, X_synth_sub))
            y_train_comb = np.concatenate((y_train_in, y_synth_sub))

            if len(np.unique(y_train_comb)) > 1:

                
                y_train_binarized = label_binarize(y_train_comb, classes=np.unique(y_train_comb))
                y_test_binarized = label_binarize(y_test_in, classes=np.unique(y_test_in))
            
                if len(unique_labels) == 2:
                    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                else:
                    model = LogisticRegression(penalty='l1', solver='saga', multi_class='ovr', max_iter=500)

                model.fit(X_train_comb, y_train_comb)
                y_pred_proba = model.predict_proba(X_test_in)

                # Initialize list to store ROC AUC scores for each class
                class_scores = []

                if len(unique_labels) == 2:
                    for i in range(len(unique_labels)):
                        # Binary classification case
                        roc_auc = roc_auc_score(y_test_in, y_pred_proba[:, 1])
                        class_scores.append(roc_auc)
                    macro_avg_score = roc_auc_score(y_test_in, y_pred_proba[:, 1])
                else:
                    # Multi-class classification case
                    for i in range(y_test_binarized.shape[1]):
                        # Calculate the ROC AUC score for each class
                        roc_auc = roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i])
                        class_scores.append(roc_auc)
                        
                    macro_avg_score = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr', average='macro')

                # Combine class scores and macro average into a list with the overall score first
                scores = [macro_avg_score] + class_scores

                # Assuming you want to store these scores in a DataFrame similar to your XGBoost example
                scores_df[f'ROC AUC Score {prop}'] = scores
                
            else:
                scores_df[f'ROC AUC Score {prop}'] = np.nan
        
        scores_df = pd.DataFrame(scores_df).set_index('Label', inplace = False)
        scores_df.index.name = None

        scores_df['Real_Fake'] = bindf
        scores_df['RF_Val'] = bindfv
        scores_df['Real_Val'] = bindfrv

        return(scores_df)

    @timeit
    def plot_sparsity(self, epoch, threshold_range, ks_statistics, p_values, sparsity_train, sparsity_fake, avg_sparsity_trains, avg_sparsity_fakes, X_train, y_train, X_fake, y_fake_xsize, fake_thresh):
    
        # Create a figure with 2 rows and 1 column for the first two plots
        fig, axs = plt.subplots(2, 1, figsize=(10, 15))

        # First row: KS Statistic, Average Sparsity, and P-Value
        axs[0].plot(threshold_range, ks_statistics, label='KS Statistic')
        axs[0].plot(threshold_range, avg_sparsity_trains, label='Average Sparsity Train')
        axs[0].plot(threshold_range, avg_sparsity_fakes, label='Average Sparsity Fake')
        axs[0].set_xlabel('Threshold')
        axs[0].set_ylabel('Values')
        axs[0].set_title('KS Statistic, Average Sparsity, and P-Value')
        axs[0].legend(loc='upper left')

        # Secondary axis for P-Value on the first row
        ax2 = axs[0].twinx()
        ax2.plot(threshold_range, p_values, color='red', label='P-Value')
        ax2.set_ylabel('P-Value')
        ax2.legend(loc='upper right')

        num_randm = X_train.shape[0]
        num_features = X_train.shape[1]

        X_random = np.random.randint(2, size=(num_randm, num_features))

        sparsity_random = np.mean(X_random == 0, axis=1)

        # Second row: Overlapping histograms of Real and Fake data
        flat_real = sparsity_train.flatten()
        flat_fake = sparsity_fake.flatten()
        flat_random = sparsity_random.flatten()
        axs[1].hist(flat_real, bins=100, alpha=0.5, label='Real')
        axs[1].hist(flat_fake, bins=100, alpha=0.5, label=f'Fake Thresh = {fake_thresh}')
        axs[1].hist(flat_random, bins=100, alpha=0.5, label='Random')
        axs[1].set_title(f'Histograms of Real, Fake and Random {epoch}')
        axs[1].legend()

        plt.tight_layout()

        # Optionally, save the figure
        file_path = os.path.join(self.result_path, f'sparsity_plots_overall_{epoch}.png')
        plt.savefig(file_path)
        plt.show()

        # Third plot: Histograms by disease type
        unique_diseases = np.unique(y_train)
        label_mapping = self.label_dict
        for disease in unique_diseases:
            fig_disease, ax_disease = plt.subplots(1, 1, figsize=(10, 8))
            
            mask_real = (y_train == disease)
            mask_fake = (y_fake_xsize == disease)
            
            sparsity_train_disease = sparsity_train[mask_real]
            sparsity_fake_disease = sparsity_fake[mask_fake]
            sparsity_random_disease = sparsity_random[mask_real]  # Assuming random data uses the same mask
            
            flat_real_disease = sparsity_train_disease.flatten()
            flat_fake_disease = sparsity_fake_disease.flatten()
            flat_random_disease = sparsity_random_disease.flatten()
            
            # Plotting histograms
            ax_disease.hist(flat_real_disease, bins=100, alpha=0.5, label='Real')
            ax_disease.hist(flat_fake_disease, bins=100, alpha=0.5, label=f'Fake Thresh = {fake_thresh}')
            ax_disease.hist(flat_random_disease, bins=100, alpha=0.5, label='Random')
            ax_disease.set_title(f'Histograms of Real, Fake, and Random for Disease Type {label_mapping[disease]} (Epoch {epoch})')
            ax_disease.set_xlabel('Sparsity')
            ax_disease.set_ylabel('Frequency')
            ax_disease.legend()

            plt.tight_layout()

            # Optionally, save the figure
            file_path_disease = os.path.join(self.result_path, f'sparsity_plots_disease_{label_mapping[disease]}_{epoch}.png')
            plt.savefig(file_path_disease)
            plt.show()

    @timeit
    def analyze_sparsity(self, X_train, X_fake, threshold_range):
        high_p_value = 0
        high_p_threshold = 0
        ks_statistics = []
        p_values = []
        avg_sparsity_trains = []
        avg_sparsity_fakes = []
        sparsity_diffs = []

        for threshold in threshold_range:
            # Binarize X_fake based on the threshold
            X_fake_binary = np.where(X_fake > threshold, 1, 0)
            fake_xsize = X_fake_binary[:X_train.shape[0]]

            # Calculate sparsity
            sparsity_train = np.mean(X_train == 0, axis=1)
            sparsity_fake = np.mean(fake_xsize == 0, axis=1)

            # Calculate average sparsity
            avg_sparsity_train = np.mean(sparsity_train)
            avg_sparsity_fake = np.mean(sparsity_fake)

            # Calculate sparsity difference
            spars_diff = abs(avg_sparsity_train - avg_sparsity_fake)

            # KS Test
            ks_statistic, p_value = stats.ks_2samp(sparsity_train, sparsity_fake)

            if p_value > high_p_value:
                high_p_value = p_value
                high_p_threshold = threshold

            # Append results to lists
            ks_statistics.append(ks_statistic)
            p_values.append(p_value)
            avg_sparsity_trains.append(avg_sparsity_train)
            avg_sparsity_fakes.append(avg_sparsity_fake)
            sparsity_diffs.append(spars_diff)
        
        return ks_statistics, p_values, avg_sparsity_trains, avg_sparsity_fakes, high_p_threshold, high_p_value


class Trainer_Binary():
    def __init__(self, generator, discriminator, classifier, gen_optimizer, dis_optimizer, cls_optimizer, X_train, y_train, X_test, y_test, nclasses, result_path, model_path, model_name, label_dict,
                 gp_weight=10, critic_iterations=5, print_every=5, auc_weight = 1.0, lambda_cls=2, dlc = 0, device = None):
        self.device = device if device else torch.device("cpu")
        self.G = generator.to(self.device)
        self.D = discriminator.to(self.device)
        self.C = classifier.to(self.device)
        self.G_opt = gen_optimizer
        self.D_opt = dis_optimizer
        self.C_opt = cls_optimizer
        self.losses = {'G': [], 'G_adv': [], 'D': [], 'GP': [], 'gradient_norm': [], 'G_cls': [], 'C_real': [], 'C_synth': []}
        self.num_steps = 0
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every
        self.latent_dim = 100
        self.nclasses = nclasses
        self.result_path = result_path
        self.model_path = model_path
        self.model_name = model_name
        self.highest_score = 0
        self.lowest_loss = 10
        self.lowest_real_cls_loss = 10
        self.highest_perf = 0
        self.auc_weight = auc_weight
        self.lambda_cls = lambda_cls
        self.dlc = dlc

        self.X_train = torch.tensor(np.clip(X_train, 0.0005, 0.9995), dtype=torch.float32).to(self.device)
        self.y_train = torch.tensor(y_train, dtype=torch.long).to(self.device)

        self.X_test = torch.tensor(np.clip(X_test, 0.0005, 0.9995), dtype=torch.float32).to(self.device)
        self.y_test = torch.tensor(y_test, dtype=torch.long).to(self.device)

        self.X_train_binary = (self.X_train > 0.5).int()
        self.X_test_binary = (self.X_test > 0.5).int()

        self.p_value_save_count = 0
        self.p_value_reached = False
        self.stop_saving_models = False
        self.max_pvalue_saves = 10

        # Pass only these buffers to EMA

        self.ema = EMA(self.G, beta=0.995, update_every=10)

        #self.auc_zf = self.compute_auc_zero(self.X_train_binary, self.y_train, self.X_test_binary, self.y_test)

        self.saved_crit_real = None
        self.saved_gen_real = None
        self.saved_gopt_real  = None
        self.saved_copt_real = None
        self.saved_ema_real = None

        self.saved_gen_loss = None
        self.saved_crit_loss = None
        self.saved_gopt_loss = None
        self.saved_copt_loss = None
        self.saved_ema_loss = None

        self.saved_gen_perf = None
        self.saved_crit_perf = None
        self.saved_gopt_perf = None
        self.saved_copt_perf = None
        self.saved_ema_perf = None

        self.generated_buffer = []  # Stores past generated data
        self.generated_labels = [] 
        


        self.start_epoch = 0
        self.label_dict = label_dict

        self.file_manager = FileManager(self.result_path, label_dict)

    def _critic_train_iteration(self, data):
            """ """

            data, label = data
            data = data.to(self.device)
            label = label.to(self.device)

            label = label.long()

            # Get generated data
            batch_size = data.size(0)
            generated_data, gen_label = self.sample_generator(batch_size, labels = label, use_ema=False)

            generated_data = generated_data.to(self.device)
            # Calculate probabilities on real and generated data
            
            d_real = self.D(data, label)
            
            d_generated = self.D(generated_data, gen_label)

            # Get gradient penalty
            gradient_penalty = self._gradient_penalty(data, label, generated_data)
            self.losses['GP'].append(gradient_penalty.item())

            # Create total loss and optimize
            self.D_opt.zero_grad()
            d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
            d_loss.backward()

            self.D_opt.step()

            # Record loss
            self.losses['D'].append(d_loss.item())

    def _generator_train_iteration(self, data):
        data, label = data

        data = data.to(self.device)
        label = label.to(self.device)
        label = label.long()
        batch_size = data.size(0)

        # Generate synthetic data (for classifier only - detached)
        generated_data_cls, gen_label_cls = self.sample_generator(batch_size, labels=label, use_ema=False)
        
        # Generate synthetic data (for generator training - attached)
        generated_data_g, gen_label_g = self.sample_generator(batch_size, labels=label, use_ema=False)

        # ---------- Train Classifier (C) ----------
        self.C_opt.zero_grad()

        # Sample from the replay buffer
        if len(self.generated_buffer) > batch_size:
            buffer_idxs = torch.randint(0, len(self.generated_buffer), (batch_size,))
            buffered_samples = torch.stack([self.generated_buffer[i] for i in buffer_idxs])
            buffered_labels = torch.stack([self.generated_labels[i] for i in buffer_idxs])
        else:
            # If buffer isn't full yet, use only new synthetic data
            buffered_samples = generated_data_cls
            buffered_labels = gen_label_cls

    
        # Combine new and buffered samples for classifier training
        cls_inputs = torch.cat([buffered_samples, generated_data_cls], dim=0)
        cls_targets = torch.cat([buffered_labels, gen_label_cls], dim=0)

        # Classifier prediction and loss
        cls_preds = self.C(cls_inputs)
        loss_cls_synth = nn.CrossEntropyLoss()(cls_preds, cls_targets)

        # Total classifier loss
        loss_C_total = loss_cls_synth
        loss_C_total.backward()
        self.C_opt.step()

        # ---------- Train Generator (G) ----------
        self.G_opt.zero_grad()

        # GAN Adversarial loss
        d_generated = self.D(generated_data_g, gen_label_g)
        g_loss_adv = -d_generated.mean()

        cls_preds_real = self.C(data)
        g_loss_cls = nn.CrossEntropyLoss()(cls_preds_real, label)

        cosine_sim = torch.nn.functional.cosine_similarity(generated_data_g.unsqueeze(1), generated_data_g.unsqueeze(0), dim=2)
        diversity_loss = cosine_sim.mean()

        # Total Generator loss
        g_loss = g_loss_adv + self.lambda_cls * g_loss_cls - self.dlc * diversity_loss

        self.losses['G_adv'].append(g_loss_adv.item())
        self.losses['G_cls'].append(g_loss_cls.item())
        self.losses['G'].append(g_loss.item())
        self.losses['C_real'].append(g_loss_cls.item())
        self.losses['C_synth'].append(loss_cls_synth.item())

        g_loss.backward()
        self.G_opt.step()

        # Store new samples correctly
        for i in range(batch_size):
            self.generated_buffer.append(generated_data_g[i].detach())  
            self.generated_labels.append(gen_label_g[i].detach())

        if len(self.generated_buffer) > 10000:
            self.generated_buffer.pop(0)
            self.generated_labels.pop(0)

        with open("model_parameters_and_buffers.txt", "w") as f:
            f.write("=== Checking G parameters ===\n")
            for name, param in self.G.named_parameters():
                f.write(f"{name}: {param.dtype}\n")
            f.write("=== Checking G buffers ===\n")
            for name, buf in self.G.named_buffers():
                f.write(f"{name}: {buf.dtype}\n")

            f.write("=== Checking D parameters ===\n")
            for name, param in self.D.named_parameters():
                f.write(f"{name}: {param.dtype}\n")
            f.write("=== Checking D buffers ===\n")
            for name, buf in self.D.named_buffers():
                f.write(f"{name}: {buf.dtype}\n")
        
        self.ema.update()

    def _gradient_penalty(self, real_data, label, generated_data):
        batch_size = real_data.size(0)
        device = real_data.device

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_data)
        interpolated = alpha * real_data + (1 - alpha) * generated_data
        interpolated.requires_grad_(True)

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated, label)
        grad_outputs = torch.ones_like(prob_interpolated, device=device)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs= grad_outputs,
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()
    
    @timeit
    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            self.num_steps += 1

            img = data[0].to(self.device)
            label = data[1].to(self.device)
            inputs = (img, label)

            # Perform a critic (discriminator) training iteration
            self._critic_train_iteration(inputs)

            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(inputs)

            
    def train(self, data_loader, epochs, sample_every):

        for epoch in range(self.start_epoch, epochs):
            #print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

            d_loss_last = self.losses['D'][-1]
            g_loss_last = self.losses['G_adv'][-1]
            g_cls_last = self.losses['G_cls'][-1]
            c_real = self.losses['C_real'][-1]
            c_synth = self.losses['C_synth'][-1]

            if (epoch + 1) % sample_every == 0:
                self.sample_images(epoch + 1, 10000, d_loss = d_loss_last, g_loss = g_loss_last, g_cls_last = g_cls_last, c_real = c_real, c_synth = c_synth)
                
                print(f"End of epoch: Processed {self.num_steps} steps")
                print(f"D: {self.losses['D'][-1]}")
                print(f"GP: {self.losses['GP'][-1]}")
                print(f"Gradient norm: {self.losses['gradient_norm'][-1]}")
                print(f"G: {self.losses['G'][-1]}")

    def sample_generator(self, num_samples, labels = None, use_ema=True):
        generator = self.ema.ema_model if use_ema else self.G
        noise, labels = generator.sample_latent(num_samples, labels=labels)
        noise = noise.to(self.device)
        labels = labels.to(self.device)
        generated_data = generator(noise, labels)
        return generated_data, labels
    

    def compute_auc_zero(self, X_train, y_train, X_test, y_test):
        X_train_in = np.asarray(X_train).squeeze()
        y_train_in = np.asarray(y_train).squeeze()
        X_test_in = np.asarray(X_test).squeeze()
        y_test_in = np.asarray(y_test).squeeze()

        if len(np.unique(y_train_in)) > 1:
            y_train_binarized = label_binarize(y_train_in, classes=np.unique(y_train_in))
            y_test_binarized = label_binarize(y_test_in, classes=np.unique(y_test_in))

            unique_labels = np.unique(y_train_in)
            if self.nclasses == 2:
                model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                model.fit(X_train_in, y_train_in)
                y_pred_proba = model.predict_proba(X_test_in)
                macro_avg_score = roc_auc_score(y_test_in, y_pred_proba[:, 1])
            else:
                model = LogisticRegression(penalty='l1', solver='saga', multi_class='ovr', max_iter=500)
                model.fit(X_train_in, y_train_in)
                y_pred_proba = model.predict_proba(X_test_in)
                macro_avg_score = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr', average='macro')

            score = macro_avg_score
        else:
            score = np.nan
        
        return score

    def compute_auc_diff(self, X_train, y_train, X_test, y_test, X_synth, y_synth, prop=1):
        X_train_in = np.asarray(X_train).squeeze()
        y_train_in = np.asarray(y_train).squeeze()
        X_test_in = np.asarray(X_test).squeeze()
        y_test_in = np.asarray(y_test).squeeze()
        X_synth_in = np.asarray(X_synth).squeeze()
        y_synth_in = np.asarray(y_synth).squeeze()

        scores_dict = {}

        nsynth = int(len(X_synth_in) * prop)
        synth_indices = np.random.choice(len(X_synth_in), nsynth, replace=False)
        X_synth_sub = X_synth_in[synth_indices]
        y_synth_sub = y_synth_in[synth_indices]
        X_train_comb = np.concatenate((X_train_in, X_synth_sub))
        y_train_comb = np.concatenate((y_train_in, y_synth_sub))

        if len(np.unique(y_train_comb)) > 1:
            y_train_binarized = label_binarize(y_train_comb, classes=np.unique(y_train_comb))
            y_test_binarized = label_binarize(y_test_in, classes=np.unique(y_test_in))

            unique_labels = np.unique(y_train_comb)
            if self.nclasses == 2:
                model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                model.fit(X_train_comb, y_train_comb)
                y_pred_proba = model.predict_proba(X_test_in)
                macro_avg_score = roc_auc_score(y_test_in, y_pred_proba[:, 1])
            else:
                model = LogisticRegression(penalty='l1', solver='saga', multi_class='ovr', max_iter=500)
                model.fit(X_train_comb, y_train_comb)
                y_pred_proba = model.predict_proba(X_test_in)
                macro_avg_score = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr', average='macro')

            score = macro_avg_score
        else:
            score = np.nan

        # Compute the difference between the scores for proportions 1 and 0
        diff = score - self.auc_zf
        
        return diff

    def load_critic_only(self, epoch, type="pvalue_reached"):
        path = f'{self.model_path}/{self.model_name}_{epoch:06d}_critic_{type}.pth'
        self.D.load_state_dict(torch.load(path))

    def load_model(self, epoch, type = None, load_ema=False):

        # Default path structure
        generator_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_generator.pth'
        ema_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_generator_ema.pth'
        discriminator_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_critic.pth'
        G_optimizer_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_G_opt.pth'
        D_optimizer_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_D_opt.pth'

        # If model_type is specified, modify paths accordingly
        if type is not None:
            generator_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_generator_{type}.pth'
            ema_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_generator_{type}_ema.pth'
            discriminator_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_critic_{type}.pth'
            G_optimizer_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_G_opt_{type}.pth'
            D_optimizer_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_D_opt_{type}.pth'

        # Special case for 'pvalue' model type if necessary
        if type == 'pvalue' or type == 'pvalue_reached':  # Compare with the string 'pvalue' directly
            generator_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_generator_{type}.pth'
            ema_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_generator_ema_{type}.pth'
            discriminator_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_critic_{type}.pth'
            G_optimizer_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_G_opt_{type}.pth'
            D_optimizer_path = f'{self.model_path}/{self.model_name}_{epoch:06d}_D_opt_{type}.pth'

        self.G.load_state_dict(torch.load(generator_path))
        self.D.load_state_dict(torch.load(discriminator_path))
        self.G_opt.load_state_dict(torch.load(G_optimizer_path))
        self.D_opt.load_state_dict(torch.load(D_optimizer_path))
        self.ema.ema_model.load_state_dict(torch.load(ema_path))
        self.start_epoch = epoch + 1

    @timeit
    def sample_images(self, epoch, n_samples, d_loss, g_loss, g_cls_last, c_real, c_synth):

        save_by_p_value = False

        # Initialize deques to store the last three checkpoints
        self.saved_gen_real = deque(maxlen=3)
        self.saved_crit_real = deque(maxlen=3)
        self.saved_gopt_real = deque(maxlen=3)
        self.saved_copt_real = deque(maxlen=3)
        self.saved_ema_real = deque(maxlen=3)

        current_real_score = c_real
        if current_real_score < self.lowest_real_cls_loss and not self.stop_saving_models:
            self.lowest_real_cls_loss = current_real_score

            os.makedirs(self.model_path, exist_ok=True)

            # Paths for current saves
            gen_path_real = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_real.pth")
            crit_path_real = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_critic_real.pth")
            gen_ema_path_real = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_real_ema.pth")
            gopt_path_real = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_G_opt_real.pth")
            copt_path_real = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_D_opt_real.pth")

            # Save current models
            torch.save(self.G.state_dict(), gen_path_real)
            torch.save(self.D.state_dict(), crit_path_real)
            torch.save(self.ema.ema_model.state_dict(), gen_ema_path_real)
            torch.save(self.G_opt.state_dict(), gopt_path_real)
            torch.save(self.D_opt.state_dict(), copt_path_real)

            # Append new paths and remove oldest if over limit
            for dq, path in zip(
                [self.saved_gen_real, self.saved_crit_real, self.saved_gopt_real, self.saved_copt_real, self.saved_ema_real],
                [gen_path_real, crit_path_real, gopt_path_real, copt_path_real, gen_ema_path_real]
            ):
                if len(dq) == dq.maxlen:
                    old_path = dq.popleft()
                    if os.path.exists(old_path):
                        os.remove(old_path)
                dq.append(path)
        
        # Initialize deques to store the last three checkpoints
        self.saved_gen_loss = deque(maxlen=3)
        self.saved_crit_loss = deque(maxlen=3)
        self.saved_gopt_loss = deque(maxlen=3)
        self.saved_copt_loss = deque(maxlen=3)
        self.saved_ema_loss = deque(maxlen=3)

        current_loss_score = g_loss
        if current_loss_score < self.lowest_loss and not self.stop_saving_models:
            self.lowest_loss = current_loss_score

            os.makedirs(self.model_path, exist_ok=True)

            # Paths for current saves
            gen_path_loss = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_loss.pth")
            crit_path_loss = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_critic_loss.pth")
            gen_ema_path_loss = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_loss_ema.pth")
            gopt_path_loss = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_G_opt_loss.pth")
            copt_path_loss = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_D_opt_loss.pth")

            # Save current models
            torch.save(self.G.state_dict(), gen_path_loss)
            torch.save(self.D.state_dict(), crit_path_loss)
            torch.save(self.ema.ema_model.state_dict(), gen_ema_path_loss)
            torch.save(self.G_opt.state_dict(), gopt_path_loss)
            torch.save(self.D_opt.state_dict(), copt_path_loss)

            # Append new paths and remove oldest if over limit
            for dq, path in zip(
                [self.saved_gen_loss, self.saved_crit_loss, self.saved_gopt_loss, self.saved_copt_loss, self.saved_ema_loss],
                [gen_path_loss, crit_path_loss, gopt_path_loss, copt_path_loss, gen_ema_path_loss]
            ):
                if len(dq) == dq.maxlen:
                    old_path = dq.popleft()
                    if os.path.exists(old_path):
                        os.remove(old_path)
                dq.append(path)
        
        ### ADDED CODE HERE#####
        losses = np.array([epoch, d_loss, g_loss, g_cls_last, c_real, c_synth])
        
        loss_cols = ["epoch", "d_loss", "g_loss", "g_cls_last", "C_real", "C_synth"]
                 # Check if the file exists
        
        os.makedirs(self.result_path, exist_ok=True)

        if not os.path.exists(self.result_path + '/loss.csv'):
            # Open the file in write mode, creating a new file if it does not exist
            with open(self.result_path + '/loss.csv', 'w', newline='') as csv_file:
                # Create a writer object
                writer = csv.writer(csv_file)
                # Write the column names as the first row of the CSV file
                writer.writerow(loss_cols)
                
        with open(self.result_path + '/loss.csv', 'a', newline='') as csv_file:
            # Create a writer object
            writer = csv.writer(csv_file)
            # Write the array as a row in the CSV file
            writer.writerow(losses)
        
        dfl= pd.read_csv(self.result_path + '/loss.csv')
        # Use EMA weights to generate images
        gen_imgs, labels = self.sample_generator(n_samples)

        gen_imgs = gen_imgs.detach().cpu()
        labels = labels.detach().cpu()

        X_fake, y_fake = gen_imgs.reshape((-1, self.X_train.shape[1])), labels.cpu().numpy()
        
        X_real_np = self.X_train.cpu().numpy()
        X_test_np = self.X_test.cpu().numpy()
        X_fake_np = gen_imgs.reshape((-1, self.X_train.shape[1])).cpu().numpy()
        y_fake = labels.cpu().numpy()

        # Apply binarization using NumPy
        X_train_binary = np.where(X_real_np > 0.5, 1, 0)
        X_fake_binary = np.where(X_fake_np > 0.5, 1, 0)
        X_test_binary = np.where(X_test_np > 0.5, 1, 0)

        X_fake_xsize = X_fake_binary[:X_train_binary.shape[0]]
        y_fake_xsize = y_fake[:X_fake_xsize.shape[0]]

        unique_classes, counts = np.unique(y_fake, return_counts=True)

        if len(unique_classes) == self.nclasses and all(count >= 2 for count in counts):

            # see how well logistic can distinguish between real and fake

            track_df = self.classify(
                X_train_binary,
                self.y_train.cpu().numpy(),    # ensure y_train is on CPU and NumPy
                X_fake_xsize,
                y_fake_xsize,
                X_test_binary,
                self.y_test.cpu().numpy(),     # ensure y_test is on CPU and NumPy
                model='logi'
            )

            track_df['epoch'] = epoch

            # Specify your file path
            file_path = os.path.join(self.result_path, 'track_results.csv')

            if os.path.exists(file_path):
                # File exists, load previous data, append new data, and save
                previous_data = pd.read_csv(file_path, index_col=0)
                updated_data = pd.concat([previous_data, track_df], axis=0)
                updated_data.to_csv(file_path, index=True)
            else:
                # First iteration or file doesn't exist, save current DataFrame directly
                track_df.to_csv(file_path, index=True)

            df = pd.read_csv(file_path, index_col=0)
            df['.25Diff'] = df['ROC AUC Score 0.25'] - df['ROC AUC Score 0']
            df['1Diff'] = df['ROC AUC Score 1'] - df['ROC AUC Score 0']
            score_metrics = ['Real_Val', 'RF_Val', 'Real_Fake', '.25Diff', '1Diff'], ['Real vs. Validation', 'R. v. F. Validation', 'Real vs. Fake', f'AUC ({X_train_binary.shape[0]} R {X_fake_xsize.shape[0] // 2} F) - AUC No Aug', f'AUC ({X_train_binary.shape[0]} R {X_fake_xsize.shape[0] * 2} F) - AUC No Aug']
            
            # Initialize deques to store the last three checkpoints
            self.saved_gen_perf = deque(maxlen=3)
            self.saved_crit_perf = deque(maxlen=3)
            self.saved_gopt_perf = deque(maxlen=3)
            self.saved_copt_perf = deque(maxlen=3)
            self.saved_ema_perf = deque(maxlen=3)

            perf_score = df['ROC AUC Score 1'].iloc[-1]
            if perf_score > self.highest_perf:
                self.highest_perf = perf_score

                os.makedirs(self.model_path, exist_ok=True)

                # Paths for current saves
                gen_path_perf = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_perf.pth")
                crit_path_perf = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_critic_perf.pth")
                gen_ema_path_perf = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_perf_ema.pth")
                gopt_path_perf = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_G_opt_perf.pth")
                copt_path_perf = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_D_opt_perf.pth")

                # Save current models
                torch.save(self.G.state_dict(), gen_path_perf)
                torch.save(self.D.state_dict(), crit_path_perf)
                torch.save(self.ema.ema_model.state_dict(), gen_ema_path_perf)
                torch.save(self.G_opt.state_dict(), gopt_path_perf)
                torch.save(self.D_opt.state_dict(), copt_path_perf)

                # Append new paths and remove oldest if over limit
                for dq, path in zip(
                    [self.saved_gen_perf, self.saved_crit_perf, self.saved_gopt_perf, self.saved_copt_perf, self.saved_ema_perf],
                    [gen_path_perf, crit_path_perf, gopt_path_perf, copt_path_perf, gen_ema_path_perf]
                ):
                    if len(dq) == dq.maxlen:
                        old_path = dq.popleft()
                        if os.path.exists(old_path):
                            os.remove(old_path)
                    dq.append(path)
            
            folder_name = self.result_path
            # Folder structure setup
            base_folder = os.path.join(self.result_path, 'Track_Performance')
            folder_names = df.index.unique()

        else:
            print("Sample classes are not represented or not enough samples, so tracking is skipped")

        threshold_range = np.arange(0.5, 1.0, 0.01)
        ks_statistics, p_values, avg_sparsity_trains, avg_sparsity_fakes, high_p_threshold, high_p_value = self.analyze_sparsity(X_train_binary,  X_fake.cpu().numpy(), threshold_range)
        
        if high_p_value > 0.05:  # ✅ Save if p_value > 0.05
            save_by_p_value = True

        if save_by_p_value and not self.stop_saving_models:
            os.makedirs(self.model_path, exist_ok=True)

            # ✅ Define paths for saving
            gen_pv = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_pvalue.pth")
            crit_pv = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_critic_pvalue.pth")
            gen_ema_pv = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_ema_pvalue.pth")
            gopt_pv = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_G_opt_pvalue.pth")
            copt_pv = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_D_opt_pvalue.pth")

            # ✅ Save models and optimizers
            torch.save(self.G.state_dict(), gen_pv)
            torch.save(self.D.state_dict(), crit_pv)
            torch.save(self.ema.ema_model.state_dict(), gen_ema_pv)
            torch.save(self.G_opt.state_dict(), gopt_pv)
            torch.save(self.D_opt.state_dict(), copt_pv)

        if not self.p_value_reached and high_p_value > 0.05:
            self.p_value_reached = True
        
        if self.p_value_reached and not self.stop_saving_models and self.p_value_save_count < self.max_pvalue_saves:

            gen_pvr = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_pvalue_reached.pth")
            crit_pvr = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_critic_pvalue_reached.pth")
            gen_ema_pvr = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_generator_ema_pvalue_reached.pth")
            gopt_pvr = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_G_opt_pvalue_reached.pth")
            copt_pvr = os.path.join(self.model_path, f"{self.model_name}_{epoch:06d}_D_opt_pvalue_reached.pth")

            # ✅ Save models
            torch.save(self.G.state_dict(), gen_pvr)
            torch.save(self.D.state_dict(), crit_pvr)
            torch.save(self.ema.ema_model.state_dict(), gen_ema_pvr)
            torch.save(self.G_opt.state_dict(), gopt_pvr)
            torch.save(self.D_opt.state_dict(), copt_pvr)

            self.p_value_save_count += 1

            if self.p_value_save_count >= self.max_pvalue_saves:
                self.stop_saving_models = True
            
        # Ensure X_fake is a NumPy array on CPU
        X_fake_np = X_fake.cpu().numpy() if isinstance(X_fake, torch.Tensor) else X_fake
        y_fake_np = y_fake.cpu().numpy() if isinstance(y_fake, torch.Tensor) else y_fake

        # Thresholding: binarize or fallback to high_p_threshold
        fake_thresh = 0.5
        X_fake_binary = np.where(X_fake_np > fake_thresh, 1, 0)  # binarize

        # Resize fake data to match shape of real data
        X_fake_xsize = X_fake_binary[0:X_train_binary.shape[0]]
        y_fake_xsize = y_fake_np[0:X_fake_xsize.shape[0]]

        # Calculate sparsity (fraction of zeros) for each sample
        sparsity_train = np.mean(X_train_binary == 0, axis=1)
        sparsity_fake = np.mean(X_fake_xsize == 0, axis=1)

        # Placeholder for high-p-value (passed from previous analysis)
        sparsity_scores = np.array([high_p_value])

        # In case epoch is a PyTorch tensor
        epoch_val = epoch.item() if isinstance(epoch, torch.Tensor) else epoch

        all_scores = np.append(epoch_val, sparsity_scores)
        
        score_cols = ["epoch", "p_value"]
        
        # Check if the file exists
        if not os.path.exists(self.result_path + '/ext_scores.csv'):
            # Open the file in write mode, creating a new file if it does not exist
            with open(self.result_path + '/ext_scores.csv', 'w', newline='') as csv_file:
                # Create a writer object
                writer = csv.writer(csv_file)
                # Write the column names as the first row of the CSV file
                writer.writerow(score_cols)

        with open(self.result_path + '/ext_scores.csv', 'a', newline='') as csv_file:
            # Create a writer object
            writer = csv.writer(csv_file)
            # Write the array as a row in the CSV file
            writer.writerow(all_scores)
        
        dfe = pd.read_csv(self.result_path + '/ext_scores.csv')

        self.store_best_generated_data(dfe, epoch, X_fake.cpu().numpy() if isinstance(X_fake, torch.Tensor) else X_fake,
                                       y_fake.cpu().numpy() if isinstance(y_fake, torch.Tensor) else y_fake, threshold_range,ks_statistics,p_values,sparsity_train,sparsity_fake,avg_sparsity_trains,avg_sparsity_fakes,X_train_binary,self.y_train.cpu().numpy() if isinstance(self.y_train, torch.Tensor) else self.y_train,X_fake_xsize,y_fake_xsize,fake_thresh)
    

        dfep = dfe.set_index("epoch")
        dfepl = dfep.transpose().values.tolist()

        dflp = dfl.set_index("epoch")
        dflpl = dflp.transpose().values.tolist()

        score_cols = ["p_value"]

        fig = plt.figure(figsize=(18, 12), dpi=300)
        
        ax1 = plt.subplot2grid((4, 5), (0, 0), colspan=5)

        x = dfep.index
        lines = dfepl
        labels  = ["p_value"]

        # fig1 = plt.figure()
        for i, l in zip(lines, labels):  
            ax1.plot(np.array(x), np.array(i), label='l')
            ax1.legend(labels, loc="lower left")
            ax1.set_title("sparsity check")

        x = dflp.index
        lines = [dflp[col].values for col in dflp.columns if col != "g_cls_last" and col != "C_real" and col != "C_synth"]
        labels = [col for col in dflp.columns if col != "g_cls_last" and col != "C_real" and col != "C_synth"]

        ax2 = plt.subplot2grid((4, 5), (1, 0), colspan=5)

        for line, label in zip(lines, labels):  
            ax2.plot(np.array(x), np.array(line), label=label)
        ax2.legend(loc="lower left")
        ax2.set_title("Loss")

        plt.savefig(self.result_path + "/track.png")
        plt.clf()
            
        #plt.savefig("images/mnist_%d.png" % epoch)
        #plt.close()

    @timeit
    def _pcoa_manual(self, distance_matrix):
        """Perform PCoA on a given distance matrix."""
        # Number of samples
        n_samples = distance_matrix.shape[0]

        # Center the distance matrix (Gower's centering)
        H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
        B = -0.5 * H.dot(distance_matrix ** 2).dot(H)

        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(B)

        # Sort eigenvectors and eigenvalues in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Select the principal coordinates
        principal_coords = eigvecs * np.sqrt(np.maximum(eigvals, 0))

        return eigvals, principal_coords

    @timeit
    def _project_new_data(self, new_data, original_data, principal_coords_original):
        """Project new data onto existing PCoA space."""
        # Calculate Jaccard distance from new points to original points
        new_distance_matrix = cdist(new_data, original_data, metric='jaccard')

        # Center the new distance matrix using the centering matrix H from the original data
        num_samples = original_data.shape[0]
        H = np.eye(num_samples) - np.ones((num_samples, num_samples)) / num_samples
        new_B = -0.5 * H.dot(new_distance_matrix ** 2).dot(H)

        # Project new points onto the PCoA space using the original eigenvectors
        projected_new_coords = np.linalg.lstsq(principal_coords_original, new_B.T, rcond=None)[0].T

        return projected_new_coords
    
    @timeit
    def store_best_generated_data(self, dfe, epoch, X_fake, y_fake, threshold_range, ks_statistics, p_values, sparsity_train, sparsity_fake, avg_sparsity_trains, avg_sparsity_fakes, X_train_binary, y_train, X_fake_xsize, y_fake_xsize, fake_thresh):
        if epoch in dfe['epoch'].values:
            row = dfe[dfe['epoch'] == epoch].iloc[0]
            current_score = row['p_value']

            # Check if the current score is higher than the highest score
            if current_score > .05:

                self.plot_sparsity(epoch, threshold_range, ks_statistics, p_values, sparsity_train, sparsity_fake, avg_sparsity_trains, avg_sparsity_fakes, X_train_binary, y_train, X_fake_xsize, y_fake_xsize, fake_thresh)

                # # Define random binary matrix and labels
                # def stratified_sample(X, y, n_samples_per_class):
                #     X_sampled, y_sampled = [], []
                #     unique_labels = np.unique(y)
                #     for label in unique_labels:
                #         # Get indices for the current label
                #         label_indices = np.where(y == label)[0]
                        
                #         # Sample from this label's data
                #         sampled_indices = np.random.choice(label_indices, n_samples_per_class, replace=False)
                #         X_sampled.append(X[sampled_indices])
                #         y_sampled.append(y[sampled_indices])
                    
                #     return np.vstack(X_sampled), np.hstack(y_sampled)

                # n_samples_per_class = 50 

                # # Sample real, fake, and random data
                # X_train_sampled, y_train_sampled = stratified_sample(X_train_binary, y_train, n_samples_per_class)
                # X_fake_sampled, y_fake_sampled = stratified_sample(X_fake_xsize, y_fake_xsize, n_samples_per_class)

                # n_dimensions = 2

                # # Define random binary matrix and labels
                # num_random_samples = X_train_sampled.shape[0]  # Adjust this to the desired number of random samples
                # num_features = X_train_binary.shape[1]  # Assuming the number of features is the same

                # X_random = np.random.randint(2, size=(num_random_samples, num_features))
                # y_random = np.random.choice([0, 1, 2], size=num_random_samples)  # Randomly assign labels

                # real_distance_matrix = squareform(pdist(X_train_sampled, metric='jaccard'))

                # # Perform PCoA on real data
                # eigvals_real, principal_coords_real = self._pcoa_manual(real_distance_matrix)

                # # Project fake samples onto the PCoA space of the real data
                # projected_fake_coords = self._project_new_data(X_fake_sampled, X_train_sampled, principal_coords_real)

                # # Project random samples onto the PCoA space of the real data
                # projected_random_coords = self._project_new_data(X_random, X_train_sampled, principal_coords_real)


                # # Select the top principal coordinates
                # principal_coords_real = principal_coords_real[:, :n_dimensions]
                # projected_fake_coords = projected_fake_coords[:, :n_dimensions]
                # projected_random_coords = projected_random_coords[:, :n_dimensions]

                # # Combine data for plotting
                # combined_coords = np.vstack((principal_coords_real, projected_fake_coords, projected_random_coords))
                # combined_labels = np.concatenate((y_train_sampled, y_fake_sampled, y_random))
                # combined_types = ['real'] * len(y_train_sampled) + ['fake'] * len(y_fake_sampled) + ['random'] * len(y_random)

                # pcoa_df = pd.DataFrame(combined_coords, columns=['PC1', 'PC2'])
                # pcoa_df['label'] = combined_labels
                # pcoa_df['type'] = combined_types


                # # Replace numeric labels with textual ones
                # label_mapping = self.label_dict
                # pcoa_df['label'] = pcoa_df['label'].map(label_mapping)

                # pcoa_df_sampled = pcoa_df

                # predefined_colors = [
                #     '#FF6347',  # tomato red
                #     '#4682B4',  # steel blue
                #     '#32CD32',  # lime green
                #     '#FFD700',  # gold
                #     '#FF4500',  # orange red
                #     '#1E90FF',  # dodger blue
                #     '#228B22',  # forest green
                #     '#8A2BE2',  # blue violet
                #     '#D2691E',  # chocolate
                #     '#DC143C'   # crimson
                # ]

                # def generate_palettes(label_mapping):
                #     palette_overall = {}
                #     palette_label = {}
                #     palette_group = {}

                #     for i, (index, label) in enumerate(label_mapping.items()):
                #         color = predefined_colors[i % len(predefined_colors)]
                #         palette_overall[label] = color
                #         palette_label[f'real_{label}'] = color
                #         palette_label[f'fake_{label}'] = predefined_colors[(i + 1) % len(predefined_colors)]
                    
                #     # Add 'random' color
                #     palette_overall['random'] = '#FFD700'  # gold
                #     palette_label['random'] = '#FFD700'    # gold
                #     palette_group['random'] = '#FFD700'    # gold
                    
                #     # Default colors for 'real' and 'fake' groups
                #     palette_group['real'] = '#FF6347'  # tomato red
                #     palette_group['fake'] = '#4682B4'  # steel blue
                    
                #     return palette_overall, palette_label, palette_group

                # palette_overall, palette_label, palette_group = generate_palettes(label_mapping)

                # # Create a combined column to handle both 'label' and 'type'
                # pcoa_df_sampled['combined'] = pcoa_df_sampled.apply(
                #     lambda row: f"{row['type']}_{row['label']}" if row['type'] != 'random' else 'random', axis=1
                # )

                # pcoa_df_sampled['label_rand'] = pcoa_df_sampled.apply(
                #     lambda row: 'random' if row['type'] == 'random' else row['label'], axis=1
                # )

                # # File paths
                # plot_path_overall = f'{self.result_path}/pcoa_plot_overall.png'

                # # Delete previous files if they exist
                # if plot_path_overall and os.path.exists(plot_path_overall):
                #     os.remove(plot_path_overall)

                # # Create and save the overall plot
                # plt.figure(figsize=(10, 8))
                # sns.scatterplot(data=pcoa_df_sampled, x='PC1', y='PC2', hue='label_rand', style='type', s=100, palette=palette_overall,
                #                 markers={"real": "o", "fake": "^", "random": "s"})
                # plt.title('PCoA of Real, Fake, and Random Samples by Label')
                # plt.xlabel('Principal Coordinate 1')
                # plt.ylabel('Principal Coordinate 2')
                # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                # plt.savefig(plot_path_overall)
                # plt.show()

                # # Create and save individual plots for each label
                # for label, group_df in pcoa_df_sampled.groupby('label'):
                #     plot_path_label = f'{self.result_path}/pcoa_plot_{label}.png'
                    
                #     # Delete previous files if they exist
                #     if plot_path_label and os.path.exists(plot_path_label):
                #         os.remove(plot_path_label)
                    
                #     plt.figure(figsize=(10, 8))
                #     sns.scatterplot(data=group_df, x='PC1', y='PC2', hue='type', s=100, palette=palette_group)
                #     plt.title(f'PCoA of Real, Fake, and Random Samples by Label: {label}')
                #     plt.xlabel('Principal Coordinate 1')
                #     plt.ylabel('Principal Coordinate 2')
                #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                #     plt.savefig(plot_path_label)
                #     plt.show()

                #self.file_manager.delete_existing_files()

                self.file_manager.create_paths(epoch)  
    
    @timeit
    def classify(self, X_train, y_train, X_synth, y_synth, X_test, y_test, model='logi'):

        results = []
        results_val = []
        results_rv = []

        X_train_in = np.asarray(X_train).squeeze()
        y_train_in = np.asarray(y_train).squeeze()
        X_test_in = np.asarray(X_test).squeeze()
        y_test_in = np.asarray(y_test).squeeze()
        X_synth_in = np.asarray(X_synth).squeeze()
        y_synth_in = np.asarray(y_synth).squeeze()
            
        unique_labels = np.unique(y_synth_in)
        real_labels = np.zeros_like(y_train_in)
        fake_labels = np.ones_like(y_synth_in)
        
        real_labels_val = np.zeros_like(y_test_in)

        comb_x = np.concatenate((X_train_in, X_synth_in))
        comb_y_dis = np.concatenate((y_train_in, y_synth_in))
        comb_y = np.concatenate((real_labels, fake_labels))

        comb_x_val = np.concatenate((X_test_in, X_synth_in))
        comb_y_dis_val = np.concatenate((y_test_in, y_synth_in))
        comb_y_val = np.concatenate((real_labels_val, fake_labels))

        real_labels_val_1 = np.ones_like(y_test_in)

        comb_x_rv = np.concatenate((X_test_in, X_train_in))
        comb_y_dis_rv = np.concatenate((y_test_in, y_train_in))
        comb_y_rv = np.concatenate((real_labels_val_1, real_labels))

        X_train, X_test, y_train, y_test = train_test_split(comb_x, comb_y, test_size=0.3, random_state=42)
        X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(comb_x_val, comb_y_val, test_size=0.3, random_state=42)
        X_train_rv, X_test_rv, y_train_rv, y_test_rv = train_test_split(comb_x_rv, comb_y_rv, test_size=0.3, random_state=42)

        if len(np.unique(y_train)) > 1: 

            model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
            model.fit(X_train, y_train)
            gen_pred = model.predict_proba(X_test)[:,1]
            overall_score = roc_auc_score(y_test, gen_pred)
            results.append({'Label': 'Overall', 'ROC AUC Score': overall_score})

            model_val = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
            model_val.fit(X_train_val, y_train_val)
            gen_pred_val = model_val.predict_proba(X_test_val)[:,1]
            overall_score_val = roc_auc_score(y_test_val, gen_pred_val)
            results_val.append({'Label': 'Overall', 'ROC AUC Score': overall_score_val})

            model_rv = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
            model_rv.fit(X_train_rv, y_train_rv)
            gen_pred_rv = model_rv.predict_proba(X_test_rv)[:,1]
            overall_score_rv = roc_auc_score(y_test_rv, gen_pred_rv)
            results_rv.append({'Label': 'Overall', 'ROC AUC Score': overall_score_rv})
        else:
            results.append({'Label': 'Overall', 'ROC AUC Score': np.nan})
            results_val.append({'Label': 'Overall', 'ROC AUC Score': np.nan})
            results_rv.append({'Label': 'Overall', 'ROC AUC Score': np.nan})

        for label in unique_labels:
            label_indices = (comb_y_dis == label)

            X_train, X_test, y_train, y_test = train_test_split(comb_x[label_indices], comb_y[label_indices], test_size=0.3, random_state=42)

            label_indices_val = (comb_y_dis_val == label)

            X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(comb_x_val[label_indices_val], comb_y_val[label_indices_val], test_size=0.3, random_state=42)

            label_indices_rv = (comb_y_dis_rv == label)

            X_train_rv, X_test_rv, y_train_rv, y_test_rv = train_test_split(comb_x_rv[label_indices_rv], comb_y_rv[label_indices_rv], test_size=0.3, random_state=42)

            if (len(X_train) > 0 and len(X_test) > 0) and len(np.unique(y_train)) > 1:

                model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                model.fit(X_train, y_train)
                gen_pred = model.predict_proba(X_test)[:,1]
                overall_score = roc_auc_score(y_test, gen_pred)
                results.append({'Label': label, 'ROC AUC Score': overall_score})

                model_val = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                model_val.fit(X_train_val, y_train_val)
                gen_pred_val = model_val.predict_proba(X_test_val)[:,1]
                overall_score_val = roc_auc_score(y_test_val, gen_pred_val)
                results_val.append({'Label': label, 'ROC AUC Score': overall_score_val})

                model_rv = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                model_rv.fit(X_train_rv, y_train_rv)
                gen_pred_rv = model_rv.predict_proba(X_test_rv)[:,1]
                overall_score_rv = roc_auc_score(y_test_rv, gen_pred_rv)
                results_rv.append({'Label': label, 'ROC AUC Score': overall_score})

            else:
                results.append({'Label': label, 'ROC AUC Score': np.nan})
                results_val.append({'Label': label, 'ROC AUC Score': np.nan})
                results_rv.append({'Label': label, 'ROC AUC Score': np.nan})

        bindf = pd.DataFrame(results).set_index('Label', inplace = False)
        bindf.index.name = None
        bindfv = pd.DataFrame(results_val).set_index('Label', inplace = False)
        bindfv.index.name = None
        bindfrv = pd.DataFrame(results_rv).set_index('Label', inplace = False)
        bindfrv.index.name = None
        
        ########

        proportions = [0, .25, 1]
        scores_df = {'Label': ['Overall'] + list(range(len(set(y_train_in))))}

        for prop in proportions:

            nsynth = int(len(X_synth_in) * prop)
            synth_indices = np.random.choice(len(X_synth_in), nsynth, replace=False)
            X_synth_sub = X_synth_in[synth_indices]
            y_synth_sub = y_synth_in[synth_indices]
            X_train_comb = np.concatenate((X_train_in, X_synth_sub))
            y_train_comb = np.concatenate((y_train_in, y_synth_sub))

            if len(np.unique(y_train_comb)) > 1:

                
                y_train_binarized = label_binarize(y_train_comb, classes=np.unique(y_train_comb))
                y_test_binarized = label_binarize(y_test_in, classes=np.unique(y_test_in))
            
                if len(unique_labels) == 2:
                    model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                else:
                    model = LogisticRegression(penalty='l1', solver='saga', multi_class='ovr', max_iter=500)

                model.fit(X_train_comb, y_train_comb)
                y_pred_proba = model.predict_proba(X_test_in)

                # Initialize list to store ROC AUC scores for each class
                class_scores = []

                if len(unique_labels) == 2:
                    for i in range(len(unique_labels)):
                        # Binary classification case
                        roc_auc = roc_auc_score(y_test_in, y_pred_proba[:, 1])
                        class_scores.append(roc_auc)
                    macro_avg_score = roc_auc_score(y_test_in, y_pred_proba[:, 1])
                else:
                    # Multi-class classification case
                    for i in range(y_test_binarized.shape[1]):
                        # Calculate the ROC AUC score for each class
                        roc_auc = roc_auc_score(y_test_binarized[:, i], y_pred_proba[:, i])
                        class_scores.append(roc_auc)
                        
                    macro_avg_score = roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr', average='macro')

                # Combine class scores and macro average into a list with the overall score first
                scores = [macro_avg_score] + class_scores

                # Assuming you want to store these scores in a DataFrame similar to your XGBoost example
                scores_df[f'ROC AUC Score {prop}'] = scores
                
            else:
                scores_df[f'ROC AUC Score {prop}'] = np.nan
        
        scores_df = pd.DataFrame(scores_df).set_index('Label', inplace = False)
        scores_df.index.name = None

        scores_df['Real_Fake'] = bindf
        scores_df['RF_Val'] = bindfv
        scores_df['Real_Val'] = bindfrv

        return(scores_df)

    @timeit
    def plot_sparsity(self, epoch, threshold_range, ks_statistics, p_values, sparsity_train, sparsity_fake, avg_sparsity_trains, avg_sparsity_fakes, X_train, y_train, X_fake, y_fake_xsize, fake_thresh):

        # === Second Figure: Histograms of Real, Fake, and Random Sparsity ===
        num_randm = X_train.shape[0]
        num_features = X_train.shape[1]
        X_random = np.random.randint(2, size=(num_randm, num_features))
        sparsity_random = np.mean(X_random == 0, axis=1)

        fig2, ax2 = plt.subplots(figsize=(10, 7))

        flat_real = sparsity_train.flatten()
        flat_fake = sparsity_fake.flatten()
        flat_random = sparsity_random.flatten()

        ks_stat, ks_pvalue = ks_2samp(flat_real, flat_fake)

        # Plot histograms
        ax2.hist(flat_real, bins=100, alpha=0.5, label='Real Training Data')
        ax2.hist(flat_fake, bins=100, alpha=0.5, label='Synthetic Data')
        ax2.hist(flat_random, bins=100, alpha=0.5, label='Random Sparsity Values')

        # Legend with larger font
        ax2.legend(loc='upper right', fontsize=10)

        # Add p-value at the top-center with larger font
        ax2.text(
            0.5, 0.98,
            f'KS-test p-value (Real vs Synthetic): {ks_pvalue:.2e}',
            transform=ax2.transAxes,
            fontsize=12,
            ha='center',
            va='top',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3')
        )

        # Labels and title with larger font
        ax2.set_title('Histograms of Real, Fake, and Random Sparsity', fontsize=16)
        ax2.set_xlabel('Sparsity', fontsize=14)
        ax2.set_ylabel('Frequency', fontsize=14)

        # Tick label font size
        ax2.tick_params(axis='both', labelsize=12)

        plt.tight_layout()
        file_path2 = os.path.join(self.result_path, f'sparsity_plots_hist.png')
        plt.savefig(file_path2)
        plt.show()

        # Third plot: Histograms by disease type
        unique_diseases = np.unique(y_train)
        label_mapping = self.label_dict
        for disease in unique_diseases:
            fig_disease, ax_disease = plt.subplots(1, 1, figsize=(10, 8))
            
            mask_real = (y_train == disease)
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
            file_path_disease = os.path.join(self.result_path, f'sparsity_plots_disease_{label_mapping[disease]}.png')
            plt.savefig(file_path_disease)
            plt.show()

    @timeit
    def analyze_sparsity(self, X_train, X_fake, threshold_range):
        high_p_value = 0
        high_p_threshold = 0
        ks_statistics = []
        p_values = []
        avg_sparsity_trains = []
        avg_sparsity_fakes = []
        sparsity_diffs = []

        for threshold in threshold_range:
            # Binarize X_fake based on the threshold
            X_fake_binary = np.where(X_fake > threshold, 1, 0)
            fake_xsize = X_fake_binary[:X_train.shape[0]]

            # Calculate sparsity
            sparsity_train = np.mean(X_train == 0, axis=1)
            sparsity_fake = np.mean(fake_xsize == 0, axis=1)

            # Calculate average sparsity
            avg_sparsity_train = np.mean(sparsity_train)
            avg_sparsity_fake = np.mean(sparsity_fake)

            # Calculate sparsity difference
            spars_diff = abs(avg_sparsity_train - avg_sparsity_fake)

            # KS Test
            ks_statistic, p_value = stats.ks_2samp(sparsity_train, sparsity_fake)

            if p_value > high_p_value:
                high_p_value = p_value
                high_p_threshold = threshold

            # Append results to lists
            ks_statistics.append(ks_statistic)
            p_values.append(p_value)
            avg_sparsity_trains.append(avg_sparsity_train)
            avg_sparsity_fakes.append(avg_sparsity_fake)
            sparsity_diffs.append(spars_diff)
        
        return ks_statistics, p_values, avg_sparsity_trains, avg_sparsity_fakes, high_p_threshold, high_p_value
            