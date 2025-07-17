import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    Combined loss function for Sparse Autoencoders that combines:
    - Reconstruction loss (MSE)
    - Sparsity loss (L1 regularization on activations)

    This is the main loss function used in the SAE training pipeline.
    """

    def __init__(self, sparsity_weight=0.1, reconstruction_weight=1.0):
        """
        Initialize the combined loss function

        Args:
            sparsity_weight (float): Weight for the sparsity penalty
            reconstruction_weight (float): Weight for the reconstruction loss
        """
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.reconstruction_weight = reconstruction_weight

    def forward(self, reconstructed, original, activations):
        """
        Compute the combined loss

        Args:
            reconstructed (torch.Tensor): Output from the autoencoder
            original (torch.Tensor): Original input data
            activations (torch.Tensor): Hidden layer activations (before ReLU)

        Returns:
            tuple: (total_loss, reconstruction_loss, sparsity_loss)
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, original)

        # Sparsity loss (L1 regularization on activations)
        sparsity_loss = torch.mean(torch.abs(activations))

        # Combined loss
        total_loss = (self.reconstruction_weight * reconstruction_loss +
                     self.sparsity_weight * sparsity_loss)

        return total_loss, reconstruction_loss, sparsity_loss


class SAELoss(nn.Module):
    """
    Custom loss function for Sparse Autoencoders that combines:
    - Reconstruction loss (MSE)
    - Sparsity loss (L1 regularization on activations)
    """

    def __init__(self, sparsity_weight=0.01, sparsity_target=0.05):
        """
        Initialize the SAE loss function

        Args:
            sparsity_weight (float): Weight for the sparsity penalty (lambda in the loss)
            sparsity_target (float): Target sparsity level (rho in KL divergence)
        """
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target

    def forward(self, reconstructed, original, activations):
        """
        Compute the combined loss

        Args:
            reconstructed (torch.Tensor): Output from the autoencoder
            original (torch.Tensor): Original input data
            activations (torch.Tensor): Hidden layer activations (before ReLU)

        Returns:
            torch.Tensor: Combined loss value
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, original)

        # Sparsity loss (L1 regularization on activations)
        sparsity_loss = torch.mean(torch.abs(activations))

        # Alternative: KL divergence sparsity loss (more sophisticated)
        # sparsity_loss = self._kl_divergence_sparsity(activations)

        # Combined loss
        total_loss = reconstruction_loss + self.sparsity_weight * sparsity_loss

        return total_loss, {
            "reconstruction_loss": reconstruction_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "total_loss": total_loss.item()
        }

    def _kl_divergence_sparsity(self, activations):
        """
        KL divergence sparsity loss (alternative to L1)
        This encourages activations to be sparse with a target sparsity level
        """
        # Apply sigmoid to get probabilities
        rho_hat = torch.mean(torch.sigmoid(activations), dim=0)

        # KL divergence: rho * log(rho/rho_hat) + (1-rho) * log((1-rho)/(1-rho_hat))
        kl_div = (self.sparsity_target * torch.log(self.sparsity_target / (rho_hat + 1e-8)) +
                 (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target) / (1 - rho_hat + 1e-8)))

        return torch.sum(kl_div)


class L1SparsityLoss(nn.Module):
    """
    Simple L1 sparsity loss for comparison
    """

    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight

    def forward(self, activations):
        return self.weight * torch.mean(torch.abs(activations))


class L2SparsityLoss(nn.Module):
    """
    L2 sparsity loss (weight decay)
    """

    def __init__(self, weight=0.01):
        super().__init__()
        self.weight = weight

    def forward(self, activations):
        return self.weight * torch.mean(activations ** 2)
