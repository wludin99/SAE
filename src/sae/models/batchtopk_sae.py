import torch
import torch.nn as nn


class BatchTopK(nn.Module):
    """
    BatchTopK is a module that zeros all values except the top-k values in the batch.
    """
    def __init__(self, topk):
        super().__init__()
        self.topk = topk

    def forward(self, x):
        original_shape = x.shape
        x_flat = x.view(-1)

        # Get top-k values and their indices
        topk_values, topk_indices = x_flat.topk(min(self.topk * original_shape[0], x_flat.numel()))

        # Create zero tensor
        result = torch.zeros_like(x_flat)

        # Set top-k values
        result[topk_indices] = topk_values

        # Reshape back to original shape
        return result.view(original_shape)


class BatchTopKSAE(nn.Module):
    def __init__(self, input_size, hidden_size, sparsity_weight=0.01, sparsity_target=0.05, topk=10):
        """
        SAE model with tied encoder and decoder
        Parameters:
            input_size: int, the dimension of the input data
            hidden_size: int, the dimension of the hidden layer
            sparsity_weight: float, weight for sparsity penalty
            sparsity_target: float, target sparsity level
            topk: int, number of top values to keep in BatchTopK
        Returns:
            SAE model
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target

        self.encoder = nn.Linear(input_size, hidden_size, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(input_size))
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_size))

        # Initialize BatchTopK with a default batch_size (will be updated during forward pass)
        self.topk_layer = BatchTopK(topk)

        self.decoder = nn.Linear(hidden_size, input_size, bias=False)
        self.loss_fn = nn.MSELoss()

        # Initialize weights
        self._init_weights()

    def forward(self, x):
        original_shape = x.shape

        # Flatten if 3D input
        if x.dim() == 3:
            batch_size, seq_len, embed_dim = x.shape
            x = x.view(-1, embed_dim)


        x = x - self.pre_bias
        x = self.encoder(x)
        x = x + self.encoder_bias
        z = self.topk_layer(x)
        x = self.decoder(z)
        x = x + self.pre_bias

        # Reshape back to original shape if needed
        if len(original_shape) == 3:
            x = x.view(original_shape)
            z = z.view(batch_size, seq_len, -1)  # Reshape encoded to match original

        # Return (reconstructed, encoded) to match regular SAE interface
        return x, z

    def _init_weights(self, sample_embeddings=None):
        """
        Initialize weights for the SAE model:
        1. Encoder and decoder weights are transposes of each other
        2. Decoder dictionary weights have magnitude 1
        3. Pre-bias is initialized to average of sample embeddings

        Args:
            sample_embeddings: Optional tensor of sample embeddings to compute pre-bias average
        """
        # Initialize encoder weights with random values
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        # Make decoder weights the transpose of encoder weights
        self.decoder.weight.data = self.encoder.weight.data.T.clone()

        # Normalize decoder weights to have magnitude 1
        decoder_norms = torch.norm(self.decoder.weight, dim=1, keepdim=True)
        self.decoder.weight.data = self.decoder.weight.data / decoder_norms

        # Update encoder weights to maintain the transpose relationship
        self.encoder.weight.data = self.decoder.weight.data.T.clone()

        # Initialize pre-bias
        if sample_embeddings is not None:
            # Compute average of sample embeddings
            if sample_embeddings.dim() == 3:
                # For 3D input: (batch_size, sequence_length, embedding_dim)
                avg_embedding = sample_embeddings.mean(dim=(0, 1))  # Average across batch and sequence
            else:
                # For 2D input: (batch_size, embedding_dim)
                avg_embedding = sample_embeddings.mean(dim=0)  # Average across batch

            self.pre_bias.data = avg_embedding.clone()
        else:
            # Initialize to zeros if no sample embeddings provided
            self.pre_bias.data.zero_()

        # Initialize encoder bias to zeros
        self.encoder_bias.data.zero_()

    def reinit_with_embeddings(self, sample_embeddings):
        """
        Reinitialize weights using sample embeddings to compute pre-bias average.
        This is useful when you want to initialize the pre-bias after the model is created.

        Args:
            sample_embeddings: Tensor of sample embeddings to compute pre-bias average
        """
        self._init_weights(sample_embeddings=sample_embeddings)

    def orthogonal_weight_update(self, lr=0.01):
        """
        Apply orthogonal weight updates to decoder weights:
        1. Remove component parallel to current weight from gradient
        2. Apply the orthogonal gradient update with learning rate
        3. Clamp magnitudes to 1

        Args:
            lr: learning rate for the weight update
        """
        with torch.no_grad():
            # Get current decoder weights (should already be unit norm)
            current_weights = self.decoder.weight.data.clone()

            # Get the gradient of decoder weights
            decoder_grad = self.decoder.weight.grad

            if decoder_grad is not None:
                # For each dictionary element, remove component parallel to current weight
                for i in range(self.decoder.weight.shape[0]):
                    current_unit = current_weights[i] / torch.norm(current_weights[i])

                    # Compute projection of gradient onto current weight direction
                    grad_projection = torch.dot(decoder_grad[i], current_unit)

                    # Remove parallel component from gradient
                    orthogonal_grad = decoder_grad[i] - grad_projection * current_unit

                    # Apply orthogonal gradient update with learning rate
                    self.decoder.weight.data[i] = current_weights[i] - lr * orthogonal_grad

                # Clamp to unit magnitude
                decoder_norms = torch.norm(self.decoder.weight, dim=1, keepdim=True)
                self.decoder.weight.data = self.decoder.weight.data / decoder_norms

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
        loss = self.loss_fn(reconstructed, y)

        # Create a simple loss dictionary for compatibility
        loss_dict = {
            "reconstruction_loss": loss.item(),
            "total_loss": loss.item()
        }

        return loss, loss_dict

    def train_step(self, x, y=None, optimizer=None, lr=0.01):
        """
        Single training step with orthogonal weight updates for decoder weights

        Args:
            x: input data
            y: target data (if None, uses x as target)
            optimizer: optimizer instance (if None, creates Adam optimizer)
            lr: learning rate for orthogonal updates
        """
        if optimizer is None:
            # Create optimizer excluding decoder weights
            decoder_params = [self.decoder.weight]
            other_params = [p for p in self.parameters() if p is not self.decoder.weight]
            optimizer = torch.optim.Adam(other_params, lr=lr)

        optimizer.zero_grad()
        loss, loss_dict = self.compute_loss(x, y)
        loss.backward()

        # Apply orthogonal weight update for decoder weights
        self.orthogonal_weight_update(lr=lr)

        # Apply regular optimizer step for other parameters
        optimizer.step()

        return loss, loss_dict
