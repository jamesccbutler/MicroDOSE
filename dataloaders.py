import torch 
from torch.utils.data import Dataset, DataLoader
from torch import nn, einsum, Tensor

#import DiffMicro.Train_GAN_Binary.data_loader as data_loader
from experimentor import Experimentor
import numpy as np

from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.unique_labels = np.unique(labels)
        self.label_to_indices = {label: np.where(labels == label)[0].tolist() for label in self.unique_labels}
        for label in self.unique_labels:
            np.random.shuffle(self.label_to_indices[label])

    def __iter__(self):
        indices = []
        num_batches = len(self.labels) // self.batch_size

        for _ in range(num_batches):
            batch = []
            for label in self.unique_labels:
                label_indices = self.label_to_indices[label][:self.batch_size // len(self.unique_labels)]
                self.label_to_indices[label] = self.label_to_indices[label][len(label_indices):]
                batch.extend(label_indices)
                # Re-shuffle the indices if they are exhausted
                if len(self.label_to_indices[label]) < self.batch_size // len(self.unique_labels):
                    np.random.shuffle(self.label_to_indices[label])
            np.random.shuffle(batch)
            indices.extend(batch)

        # Handle the remaining samples
        remaining_indices = [i for i in range(len(self.labels)) if i not in indices]
        np.random.shuffle(remaining_indices)
        indices.extend(remaining_indices)
        
        return iter(indices)

    def __len__(self):
        return len(self.labels)

def get_micro_dataloaders(exp, batch_size=64):

    X_train = exp.X_train_binary.squeeze()
    y_train = exp.y_train.ravel()
    X_test = exp.X_test_binary.squeeze()
    y_test = exp.y_test.ravel()

    X_train = np.clip(X_train, 0.0005, 0.9995)
    y_train = y_train
    X_test = np.clip(X_test, 0.0005, 0.9995)
    y_test = y_test

    dataset_train = Dataset1D(X_train, y_train)
    dataset_test = Dataset1D(X_test, y_test)

    train_sampler = BalancedBatchSampler(y_train, batch_size)
    test_sampler = BalancedBatchSampler(y_test, batch_size)

    # Create dataloaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle = True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle = True)

    return train_loader, test_loader, X_train, y_train, X_test, y_test


class Dataset1D(Dataset):
    def __init__(self, data, labels):
        # ensure tensors
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# def get_mbddpm_dataloaders(exp, batch_size=64):
#     X_train = exp.X_train_binary.squeeze()
#     y_train = exp.y_train.ravel()
#     X_val = exp.X_val_binary.squeeze()
#     y_val = exp.y_val.ravel()

#     # clip if needed
#     X_train = np.clip(X_train, 0.0005, 0.9995)
#     X_val = np.clip(X_val, 0.0005, 0.9995)

#     dataset_train = Dataset1D(X_train, y_train)
#     dataset_val = Dataset1D(X_val, y_val)

#     train_loader = DataLoader(dataset_train,
#                               batch_size=batch_size,
#                               shuffle=True)

#     val_loader = DataLoader(dataset_val,
#                              batch_size=batch_size,
#                              shuffle=False)

#     return train_loader, val_loader, X_train, y_train, X_val, y_val

def get_mbddpm_dataloaders(exp, batch_size=64):
    """
    Creates dataloaders for MBDDPM.
    Automatically uses validation set if available,
    otherwise uses test set.
    Returns: (train_loader, val_or_test_loader, X_train, y_train, X_val_or_test, y_val_or_test)
    """

    # --- TRAIN DATA ---
    X_train = exp.X_train_binary.squeeze()
    y_train = exp.y_train.ravel()
    X_train = np.clip(X_train, 0.0005, 0.9995)

    dataset_train = Dataset1D(X_train, y_train)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

    # --- VALIDATION OR TEST ---
    val_loader = None
    val_name = None  # for printing + debugging

    # Prefer validation if available
    if getattr(exp, "X_val_binary", None) is not None:
        X_val = exp.X_val_binary.squeeze()
        y_val = exp.y_val.ravel()
        val_name = "validation"

    # Otherwise fall back to test set
    elif getattr(exp, "X_test_binary", None) is not None:
        X_val = exp.X_test_binary.squeeze()
        y_val = exp.y_test.ravel()
        val_name = "test"

    else:
        raise ValueError("Experimentor object contains no validation or test data.")

    # Clip validation/test (if needed)
    X_val = np.clip(X_val, 0.0005, 0.9995)

    dataset_val = Dataset1D(X_val, y_val)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    print(f"Created dataloaders: train + {val_name}")

    return train_loader, val_loader, X_train, y_train, X_val, y_val