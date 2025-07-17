import math

import torch
import torch.nn as nn

from sae.losses import SAELoss


class TiedLinear(nn.Module):
    """Linear layer with tied weights for encoder-decoder"""
    def __init__(self, input_size, hidden_size, bias=True):
        super(TiedLinear, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.Tensor(hidden_size, input_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)

    def forward_transpose(self, x):
        """Forward pass using transposed weights (for decoder)"""
        return nn.functional.linear(x, self.weight.t(), None)

class SAE(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_weight=0.01, sparsity_target=0.05):
        """
        SAE model with tied encoder and decoder
        Parameters:
            input_size: int, the dimension of the input data
            hidden_size: int, the dimension of the hidden layer
            sparsity_weight: float, weight for sparsity penalty
            sparsity_target: float, target sparsity level
        Returns:
            SAE model
        """
        super(SAE, self).__init__()
        self.dictionary = TiedLinear(input_size, hidden_size, bias=True)
        self.activation = nn.ReLU()

        self.encoder = nn.Sequential(
            self.dictionary,
            self.activation,
        )
        self.decoder = self.dictionary.forward_transpose
        self.loss_fn = SAELoss(sparsity_weight=sparsity_weight, sparsity_target=sparsity_target)

    def forward(self, x):
        # Handle 3D input: (batch_size, sequence_length, embedding_dim)
        original_shape = x.shape
        if len(x.shape) == 3:
            batch_size, seq_length, embed_dim = x.shape
            # Reshape to 2D: (batch_size * sequence_length, embedding_dim)
            x = x.reshape(-1, embed_dim)
            was_3d = True
        else:
            was_3d = False

        encoded = self.encoder(x)
        # Use transposed weights for decoder
        reconstructed = self.decoder(encoded)

        # Reshape back to original dimensions if input was 3D
        if was_3d:
            reconstructed = reconstructed.reshape(batch_size, seq_length, embed_dim)
            encoded = encoded.reshape(batch_size, seq_length, -1)  # encoded has hidden_dim

        return reconstructed, encoded

    def compute_loss(self, x, y=None):
        """
        Compute the combined reconstruction and sparsity loss

        Args:
            x: input data
            y: target data (if None, uses x as target for reconstruction)
        """
        if y is None:
            y = x

        reconstructed, encoded = self.forward(x)
        loss, loss_dict = self.loss_fn(reconstructed, y, encoded)
        return loss, loss_dict

    def train_step(self, x, y=None, optimizer=None):
        """
        Single training step

        Args:
            x: input data
            y: target data (if None, uses x as target)
            optimizer: optimizer instance (if None, creates Adam optimizer)
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        optimizer.zero_grad()
        loss, loss_dict = self.compute_loss(x, y)
        loss.backward()
        optimizer.step()

        return loss, loss_dict
