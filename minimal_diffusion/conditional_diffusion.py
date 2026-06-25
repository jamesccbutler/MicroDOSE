import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import numpy as np
import os
import torch.optim as optim
from tqdm import tqdm

from diffusers import DDPMScheduler
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
import math

# get minimum of data before reverse normalizion

def min_max_normalize(data, new_min, new_max):
    mask = data != -1  # Mask for values that are NOT -1

    # Extract min and max values from non -1 elements
    if mask.any():  # Ensure there are valid values to normalize
        masked_data = data[mask]
        old_min, old_max = masked_data.min(), masked_data.max()

        # Normalize only non -1 values
        normalized_data = data.clone()  # Clone to avoid modifying the original tensor
        normalized_data[mask] = new_min + (masked_data - old_min) * (new_max - new_min) / (old_max - old_min)

        return normalized_data
    else:
        return data  # If all values are -1, return unchanged
    

def clip_lower(value, decimals):
    factor = 10 ** decimals
    return math.floor(value * factor) / factor

def clip_p10(value):
    exponent = math.floor(math.log10(abs(value)))  # Get exponent of value
    return 10 ** (exponent - 1)  # Move one power lower

def plot_histogram(t_dataset, min_val, max_val, norm_clip=None, raw_clip=None, title="Histogram", save_path=None):
    """
    Memory-efficient histogram plotting with filtered data.
    
    Parameters:
        t_dataset (torch.Tensor or np.ndarray): The dataset to plot.
        min_val (float): Minimum value for filtering.
        max_val (float): Maximum value for filtering.
        norm_clip (float, optional): Additional annotation.
        raw_clip (float, optional): Additional annotation.
        title (str, optional): Title of the histogram.
        save_path (str, optional): Path to save the histogram image.
    """
    # Ensure dataset is in NumPy format efficiently
    if isinstance(t_dataset, torch.Tensor):
        # Use `torch.histc` for efficiency if tensor is large
        if t_dataset.numel() > 1e7:  # Adjust threshold as needed
            hist = torch.histc(t_dataset, bins=50, min=min_val, max=max_val).cpu().numpy()
            bins = np.linspace(min_val, max_val, 51)  # 50 bins, 51 edges
        else:
            flattened_data = t_dataset.cpu().numpy().ravel()
            hist, bins = np.histogram(flattened_data, bins=50, range=(min_val, max_val))
    else:
        hist, bins = np.histogram(t_dataset.ravel(), bins=50, range=(min_val, max_val))

    # Plot histogram efficiently
    plt.figure(figsize=(10, 6))
    plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]), edgecolor='black', alpha=0.7)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)

    # Annotate with min/max and optional values
    annotation_text = f"Min: {min_val}\nMax: {max_val}"
    if norm_clip is not None:
        annotation_text += f"\nNorm Clip: {norm_clip}"
    if raw_clip is not None:
        annotation_text += f"\nRaw Clip: {raw_clip}"
    
    plt.text(0.02, 0.98, annotation_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    # Save or display plot efficiently
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.clf()  # Clear figure to free memory
        plt.close()
        print(f"Histogram saved as: {save_path}")
    else:
        plt.show()
        plt.clf()
        plt.close()


class Dataset1D(Dataset):
    def __init__(self, data_tensor: Tensor, labels_tensor: Tensor, return_data=True, return_labels=True):
        super().__init__()
        self.data_tensor = data_tensor.clone()
        self.labels_tensor = labels_tensor.squeeze().clone()
        self.return_data = return_data
        self.return_labels = return_labels
        
    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        output = []
        if self.return_data:
            output.append(self.data_tensor[idx].clone().float())
        if self.return_labels:
            output.append(self.labels_tensor[idx].clone().long())

        # If only one item is returned, return it directly instead of a tuple
        if len(output) == 1:
            return output[0]
        return tuple(output)
    

# masks input is for masks used for inpainting
class UNet1DTrainer:
    def __init__(self, model, train_dataset, masks = None, val_dataset=None, inpaint = False, batch_size=32, lr=1e-4, weight_decay=1e-5, device=None, timestep_range=(0, 1000), save_path = "./Default"):
        """
        Trainer class for the UNet1DModel.
        
        Args:
            model (nn.Module): The UNet1DModel instance.
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset, optional): The validation dataset.
            batch_size (int, optional): Batch size. Default is 32.
            lr (float, optional): Learning rate. Default is 1e-4.
            weight_decay (float, optional): Weight decay for optimizer. Default is 1e-5.
            device (torch.device, optional): Device to run training on. Default is auto-detected.
            timestep_range (tuple, optional): Range of timesteps for diffusion. Default is (0, 1000).
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.in_channels = self.model.in_channels
        self.feature_size = self.model.sample_size

        self.best_val_loss = float('inf')

        self.inpaint = inpaint
        self.save_path = save_path
        if masks is not None:
            self.masks = masks.to(device)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.model_path = os.path.join(self.save_path, "Models")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.batch_size = batch_size
        self.timestep_range = timestep_range

        # Scheduler Initialization
        #self.scheduler = DDPMScheduler(num_train_timesteps=timestep_range[1])
        self.scheduler = DDPMScheduler(
            num_train_timesteps=timestep_range[1], 
            beta_schedule="squaredcos_cap_v2",  # More data-adaptive noise schedule
            variance_type="learned_range",  # Let model learn variance instead of assuming Gaussian
            clip_sample=False  # Prevent artificial smoothing
        )
        
        self.scheduler.set_timesteps(self.timestep_range[1])

        # DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size) if val_dataset else None

        # Optimizer and Loss
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        # Loss tracking
        self.train_losses = []
        self.val_losses = []

    def train(self, epochs=10, log_interval=10):
        """
        Train the model.
        
        Args:
            epochs (int, optional): Number of training epochs. Default is 10.
            log_interval (int, optional): Steps after which to log training info. Default is 10.
            save_path (str, optional): Path to save the model checkpoint. Default is None.
        """
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, batch in pbar:
                #print(f"Batch {batch_idx} shape: {data.shape}")
                data, class_labels = batch

                if self.in_channels == 2:
                    binary_condition = data[:,1,:].unsqueeze(1)
                    binary_condition = binary_condition.to(self.device)
                    data = data[:,0,:].unsqueeze(1)
                    

                data = data.to(self.device)
                class_labels = class_labels.to(self.device)
                
                # Generate random timesteps within specified range
                timesteps = torch.randint(self.timestep_range[0], self.timestep_range[1], (data.shape[0],), dtype=torch.long, device=self.device)

                # Add noise using the scheduler
                noise = torch.randn_like(data)
                noisy_data = self.scheduler.add_noise(data, noise, timesteps)

                if self.in_channels == 2:
                    noisy_data = torch.cat((noisy_data, binary_condition), dim=1) 
                
                # Forward pass
                self.optimizer.zero_grad()
                noise_pred = self.model(noisy_data, timesteps, class_labels=class_labels).sample
                
                # Compute loss
                loss = self.criterion(noise_pred, noise)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                if batch_idx % log_interval == 0:
                    pbar.set_postfix(loss=loss.item())
            
            avg_train_loss = epoch_loss / len(self.train_loader)
            self.train_losses.append(avg_train_loss)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.6f}")

            # Inside training loop, after validation:
            if self.val_loader:
                val_loss = self.validate()
                self.val_losses.append(val_loss)

                # Save model only if validation loss improves
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss  # Update best loss
                    torch.save(self.model.state_dict(), os.path.join(self.model_path, "best_unet1d.pth"))
                    print(f"New best model saved with val_loss: {val_loss:.6f}")
            
            torch.save(self.model.state_dict(), os.path.join(self.model_path, "unet1d_checkpoint.pth"))

            # Plot training & validation loss
            self.plot_losses()

    def generate_samples(self, num_samples=100, num_steps=20, data_shape=(1, 762), class_labels: Optional[torch.Tensor] = None):
        """
        Generate new samples from the trained diffusion model.

        Args:
            num_samples (int): Number of samples to generate.
            num_steps (int): Number of diffusion steps (matching training setup).
            data_shape (tuple): Shape of each data sample (excluding batch size).
            save_path (str, optional): Path to save the generated samples plot.

        Returns:
            Tensor: Generated samples.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():

            # Diffusion Scheduler
            scheduler = DDPMScheduler(num_train_timesteps=num_steps)
            scheduler.set_timesteps(num_steps)

            if class_labels is not None:
                class_labels = class_labels.to(device)
            else:
                class_labels = None 

            if self.inpaint == True:

                if self.in_channels == 1:
                
                    # The following code assumes masks is 1 channel and model accepts 1 channel
                    image= torch.randn((num_samples, *data_shape))
                    mask = self.masks
                    ref_image = torch.where(mask == 0, torch.tensor(-1.0, dtype=torch.float32), mask.float())

                    # Iteratively denoise from T to 0
                    for t in tqdm(scheduler.timesteps, desc="Generating Samples"):

                        noise = torch.randn_like(ref_image)
                        noisy=self.scheduler.add_noise(ref_image, noise, t)
                        image = image * mask + noisy * (1 - mask)
                    
                        noise_pred = self.model(image.to(device), torch.tensor([t] * num_samples, device=device), class_labels=class_labels).sample
                        image = scheduler.step(noise_pred, t, image.to(device)).prev_sample
                        image = image.cpu()

                else:
                    # inpainting when model is trained using 2 channels, GAN binary masks will be duplicated to 2 channels
                    # one channel is inpainted, the second representing presence/absence only uses time schedule noise addition

                    # The following code assumes masks is 1 channel and model accepts 1 channel
                    
                    # noise to be transformed into final image
                    image= torch.randn((num_samples, *data_shape))

                    # masks should be 0 or 1
                    mask = self.masks.float().round() 

                    # ref_image: the values we want to remain
                    binary_condition = torch.where(mask == 0, torch.tensor(-1), torch.tensor(1))
                    ref_image = binary_condition.to(torch.float32)

                    # Iteratively denoise from T to 0
                    for t in tqdm(scheduler.timesteps, desc="Generating Samples"):

                        noise = torch.randn_like(ref_image)
                        noisy=self.scheduler.add_noise(ref_image, noise, t)

                        # inpaint only applied to first channel, because the binary channel should remain either 1 or -1
                        image = image * mask + noisy * (1 - mask)
                        #second channel contains noise according to noise schedule
                        chan_2_inpaint = torch.cat((image, binary_condition), dim = 1)
                    
                        noise_pred = self.model(chan_2_inpaint.to(device), torch.tensor([t] * num_samples, device=device), class_labels=class_labels).sample
                        image = scheduler.step(noise_pred, t, image.to(device)).prev_sample
                        image = image.cpu()

            else:
                # generate samples, works with 2 channels and 1 channel
                image = torch.randn((num_samples, *data_shape))

                if self.in_channels == 2:

                    mask = self.masks
                    binary_condition = torch.where(mask == 0, torch.tensor(-1), torch.tensor(1))

                    # Iteratively denoise from T to 0
                    for t in tqdm(scheduler.timesteps, desc="Generating Samples"):
                        cond_image = torch.cat((image, binary_condition), dim = 1)
                        noise_pred = self.model(cond_image.to(device), torch.tensor([t] * num_samples, device=device), class_labels=class_labels).sample
                        image = scheduler.step(noise_pred, t, image.to(device)).prev_sample
                        image = image.cpu()
                else:
                    # Iteratively denoise from T to 0
                    for t in tqdm(scheduler.timesteps, desc="Generating Samples"):
                        noise_pred = self.model(image.to(device), torch.tensor([t] * num_samples, device=device), class_labels=class_labels).sample
                        image = scheduler.step(noise_pred, t, image.to(device)).prev_sample
                        image = image.cpu()
        return image

    def validate(self):
        """Perform validation step."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                data, class_labels = batch

                if self.in_channels == 2:
                    binary_condition = data[:,1,:].unsqueeze(1)
                    binary_condition = binary_condition.to(self.device)
                    data = data[:,0,:].unsqueeze(1)
                    

                data = data.to(self.device)
                class_labels = class_labels.to(self.device)

                timesteps = torch.randint(self.timestep_range[0], self.timestep_range[1], (data.shape[0],), dtype=torch.long, device=self.device)

                noise = torch.randn_like(data)
                noisy_data = self.scheduler.add_noise(data, noise, timesteps)

                if self.in_channels == 2:
                    noisy_data = torch.cat((noisy_data, binary_condition), dim=1) 

                noise_pred = self.model(noisy_data, timesteps, class_labels= class_labels).sample

                loss = self.criterion(noise_pred, noise)
                
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_loss:.6f}")
        self.model.train()
        return avg_loss

    def plot_losses(self):
        os.makedirs(self.save_path, exist_ok=True)
        file_path = os.path.join(self.save_path, 'train_val_loss_plot.png')
        """Plot training and validation loss over epochs."""
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label='Training Loss', marker='o')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', marker='s')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training & Validation Loss')
        plt.yscale('log')
        plt.legend()
        plt.grid()
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free up memory
