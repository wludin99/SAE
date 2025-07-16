import torch
import torch.nn as nn
from sae.losses import SAELoss

class AddIntercepts(nn.Module):
    """Custom layer to add learnable intercepts"""
    def __init__(self, size):
        super(AddIntercepts, self).__init__()
        self.intercepts = nn.Parameter(torch.zeros(size))
    
    def forward(self, x):
        return x + self.intercepts

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
        self.dictionary = nn.Linear(input_size, hidden_size, bias=False)
        self.encoder = nn.Sequential(
            self.dictionary,
            AddIntercepts(hidden_size),
            nn.ReLU(),
        )
        self.decoder = self.dictionary.T
        self.loss_fn = SAELoss(sparsity_weight=sparsity_weight, sparsity_target=sparsity_target)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
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