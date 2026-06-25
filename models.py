import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch import Tensor

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

class CustomActivation(nn.Module):
    def __init__(self, min_val, max_val):
        super(CustomActivation, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return self.min_val + (self.max_val - self.min_val) * x
    

class Generator_Binary(nn.Module):
    def __init__(self, latent_dim, img_cols, nclasses, min_val=0.0005, max_val=0.9995):
        super(Generator_Binary, self).__init__()
        self.label_emb = nn.Embedding(nclasses, 10)
        self.latent_dim = latent_dim
        self.img_cols = img_cols
        self.nclasses = nclasses
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + 10, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, img_cols),
            nn.Sigmoid()
        )

        self.custom_activation = CustomActivation(min_val, max_val)

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((noise, label_embedding), -1)
        gen_output = self.model(gen_input)
        return self.custom_activation(gen_output)
    
    def sample_latent(self, num_samples, labels = None):
        noise = torch.randn((num_samples, self.latent_dim))
        if labels is None:
            labels = torch.randint(0, self.nclasses, (num_samples,))
        return noise, labels


class Discriminator_Binary(nn.Module):
    def __init__(self, img_cols, nclasses):
        super(Discriminator_Binary, self).__init__()
        self.label_emb = nn.Embedding(nclasses, 10)

        self.model = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Linear(img_cols + 10, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels)
        d_in = torch.cat((img, label_embedding), -1)
        validity = self.model(d_in)
        return validity
    
class Generator(nn.Module):
    def __init__(self, latent_dim, img_cols=1000, nclasses=2):  
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(nclasses, 20)  # Increase embedding size
        self.latent_dim = latent_dim
        self.img_cols = img_cols

        self.model = nn.Sequential(
            nn.Linear(latent_dim + 20, 512),  # Increase model capacity
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, img_cols),  # Output dimension matches feature count
            nn.Tanh()  # Normalized output (-1 to 1)
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((noise, label_embedding), -1)
        gen_output = self.model(gen_input)
        return gen_output  # Already scaled to [-1,1]

    def sample_latent(self, num_samples, labels=None):
        noise = torch.randn((num_samples, self.latent_dim))
        if labels is None:
            labels = torch.randint(0, self.label_emb.num_embeddings, (num_samples,))
        return noise, labels


class Critic(nn.Module):  # WGAN's Discriminator
    def __init__(self, img_cols=1000, nclasses=2):
        super(Critic, self).__init__()
        self.label_emb = nn.Embedding(nclasses, 20)

        self.model = nn.Sequential(
            nn.Linear(img_cols + 20, 512),  # Large capacity to handle 1000 features
            nn.LeakyReLU(0.1),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1)  # Raw score for WGAN
        )

    def forward(self, img, labels):
        label_embedding = self.label_emb(labels)
        d_in = torch.cat((img, label_embedding), -1)
        return self.model(d_in)  # Output raw Wasserstein score
    
class Classifier(nn.Module):
    def __init__(self, input_dim=1000, nclasses=2):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),  # Larger hidden layer
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, nclasses)  # Output class logits
        )

    def forward(self, x):
        return self.model(x)  # Logits for classification
    

class Generator_dbg(nn.Module):
    def __init__(self, latent_dim, img_cols, nclasses, min_val=0.0005, max_val=0.9995):
        super(Generator_dbg, self).__init__()
        self.latent_dim = latent_dim
        self.nclasses = nclasses
        self.img_cols = img_cols
        self.init_width = 64

        def gen_seq(x):
            sequence = []
            power = 6  # Start from 2^6 = 64
            while (2 ** power) <= x:
                sequence.append(2 ** power)
                power += 1
            # Append the next power of 2 greater than x
            sequence.append(2 ** power)
            sequence.reverse()
            return sequence

        self.channel_sequence = gen_seq(self.img_cols)

        # Embedding layer for labels
        self.embedding = nn.Embedding(nclasses, 10)
        
        # Fully connected layers for label processing
        self.label_dense = nn.Linear(10, self.init_width)

        # Fully connected layers for noise processing
        self.noise_dense = nn.Sequential(
            nn.Linear(self.latent_dim, 1 * self.init_width * self.channel_sequence[0]),
            nn.LeakyReLU(0.2)
        )

        # Build convolutional layers dynamically
        conv_layers = []
        initial_channels = self.channel_sequence[0] + 1  # Start with 512 + 1 channels
        current_width = self.init_width # Starting width

        # First layer: Reduce channel size, keep width the same
        conv_layers.append(
            nn.ConvTranspose2d(initial_channels, self.channel_sequence[1], (1, 32), stride=(1, 1), padding=(0, 16), bias=False)
        )
        conv_layers.append(nn.BatchNorm2d(self.channel_sequence[1]))
        conv_layers.append(nn.LeakyReLU(0.2))
        initial_channels = self.channel_sequence[1]

        # Add intermediate layers to adjust channel size and width
        for out_channels in self.channel_sequence[2:]:  # Leave the final layer out for special handling
            conv_layers.append(
                nn.ConvTranspose2d(initial_channels, out_channels, (1, 32), stride=(1, 2), padding=(0, 16), bias=False)
            )
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.LeakyReLU(0.2))
            initial_channels = out_channels
            current_width *= 2  # Double the width each step

        # Add the final layer to ensure a single output channel with exact width
        conv_layers.append(
            nn.ConvTranspose2d(initial_channels, 1, (1, 128), stride=(1, 2), padding=(0, 64), bias=False)
        )

        # Compile the layers into a sequential block
        self.conv_layers = nn.Sequential(*conv_layers)

        self.sigmoid = nn.Sigmoid()

        self.custom_activation = CustomActivation(min_val, max_val)


    def forward(self, noise, labels):
        # Process labels
        li = self.embedding(labels).view(-1, 10)  # Flattened embedding
        li = self.label_dense(li).view(-1, 1, 1, self.init_width)  # Reshape to (1, 64, 1)

        # Process noise
        gen = self.noise_dense(noise).view(-1, self.channel_sequence[0], 1, self.init_width)  # Reshape to (1, 64, 256)

        # Concatenate noise and label information
        merged = torch.cat((gen, li), dim=1)  # Concatenate along channels

        # Pass through convolutional layers
        gen = self.conv_layers(merged)

        gen = gen[:, :, :, :self.img_cols]

        gen = gen.view(-1, self.img_cols)

        gen = self.sigmoid(gen)

        gen = self.custom_activation(gen)

        return gen
    
    def sample_latent(self, num_samples, labels = None):
        noise = torch.randn((num_samples, self.latent_dim))
        if labels is None:
            labels = torch.randint(0, self.nclasses, (num_samples,))
        return noise, labels
    

class Critic_dbg(nn.Module):
    def __init__(self, img_cols, nclasses):
        super(Critic_dbg, self).__init__()
        self.nclasses = nclasses
        self.img_cols = img_cols

        # Embedding layer for label input
        self.embedding = nn.Embedding(nclasses, 10)

        # Define sequence of output channels
        self.seq = self.define_channel_sequence(self.img_cols)
        print("Channel Sequence:", self.seq) 

        # Fully connected layer to scale up to image dimensions
        self.fc_label = nn.Linear(10, self.seq[-1])

        self.fc_features = nn.Linear(self.img_cols, self.seq[-1])

        # Create convolutional layers based on the sequence
        self.convs = nn.ModuleList()
        in_channels = 2  # Start with 2 channels (image + label)
        
        for out_channels in self.seq[:-1]:
            self.convs.append(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 5), stride=(1, 2), padding= (0,2))
            )
            in_channels = out_channels

        # Fully connected layer for final output
        self.fc_output = nn.Linear(out_channels * self.seq[0], 1)

        # Dropout layer
        self.dropout = nn.Dropout(0.3)

    def define_channel_sequence(self, img_cols):
        # Start with the highest power of 2 less than or equal to img_cols
        max_power = 2 ** int(math.log2(img_cols))
        
        # Create a sequence that halves the channel size until it reaches 64
        seq = []
        current = max_power
        while current >= 64:
            seq.append(current)
            current //= 2
        seq.reverse()
        return seq

    def forward(self, img, labels):
        # Process labels
        li = self.embedding(labels).view(-1, 10)  # Flattened embedding
        li = self.fc_label(li)  # Fully connected layer
        li = li.view(-1, 1, 1, self.seq[-1])  # Reshape to (batch_size, 1, img_rows, img_cols)

        img = img.view(-1, 1, 1, self.img_cols)  # Ensure img has appropriate dimensions
        img = self.fc_features(img)
        img = img.view(-1, 1, 1, self.seq[-1])

        # Concatenate label as an additional channel to the image
        merge = torch.cat((img, li), dim=1)  # Concatenate along the channel dimension

        for conv in self.convs:
            merge = conv(merge)
            merge = F.leaky_relu(merge, negative_slope=0.2)
            merge = self.dropout(merge)

        # Flatten and apply the final dense layer
        critic = torch.flatten(merge, start_dim=1)
        critic = self.fc_output(critic)

        return critic