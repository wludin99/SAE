"""
CUDA-based correlation analysis functions for SAE metrics.
"""

import torch
import numpy as np


def cuda_calculate_correlation_matrix(activations, feature_array):
    """
    CUDA tensor-based correlation calculation with memory-efficient batching.
    Leverages GPU parallelism and optimized tensor operations.
    
    Args:
        activations: numpy array of shape (n_sequences, seq_len, n_latents) or (n_sequences * seq_len, n_latents)
        feature_array: numpy array of shape (n_sequences, n_features, max_seq_len) 
                      that should be flattened to (n_sequences * seq_len, n_features)
    
    Returns:
        tuple: (pearson_corr, spearman_corr, pearson_p, spearman_p) as numpy arrays
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("   ⚠️  CUDA not available, falling back to CPU")
        return _cpu_calculate_correlation_matrix(activations, feature_array)
    
    device = torch.device('cuda')
    
    # Handle input shapes properly
    if len(activations.shape) == 3:
        # 3D activations: (n_sequences, seq_len, n_latents)
        n_sequences, seq_len, n_latents = activations.shape
        # Flatten to (n_sequences * seq_len, n_latents)
        activations_flat = activations.reshape(-1, n_latents)
        total_tokens = n_sequences * seq_len
    else:
        # 2D activations: (n_sequences * seq_len, n_latents)
        total_tokens, n_latents = activations.shape
        activations_flat = activations
        n_sequences = feature_array.shape[0]
        seq_len = total_tokens // n_sequences
    
    n_features = feature_array.shape[1]
    max_seq_len = feature_array.shape[2]
    
    print(f"   🔄 Using CUDA tensor operations on {device}")
    print(f"   📊 Processing {n_latents} latents × {n_features} features = {n_latents * n_features} correlations")
    print(f"   📊 Activations shape: {activations_flat.shape}, Features shape: {feature_array.shape}")
    
    # Convert to tensors and move to GPU
    activations_tensor = torch.tensor(activations_flat, dtype=torch.float32, device=device)  # (total_tokens, n_latents)
    feature_tensor = torch.tensor(feature_array, dtype=torch.float32, device=device)  # (n_sequences, n_features, max_seq_len)
    
    # Flatten feature array to match activations: (total_tokens, n_features)
    # Transpose to (n_sequences, max_seq_len, n_features) then reshape to (total_tokens, n_features)
    feature_tensor_flat = feature_tensor.transpose(1, 2).reshape(-1, n_features)  # (n_sequences * max_seq_len, n_features)
    
    # Ensure feature array matches activation array length
    if feature_tensor_flat.shape[0] != total_tokens:
        # Pad or truncate feature array to match
        if feature_tensor_flat.shape[0] < total_tokens:
            # Pad with zeros
            padding = torch.zeros(total_tokens - feature_tensor_flat.shape[0], n_features, device=device)
            feature_tensor_flat = torch.cat([feature_tensor_flat, padding], dim=0)
        else:
            # Truncate
            feature_tensor_flat = feature_tensor_flat[:total_tokens]
    
    print(f"   📊 Flattened features shape: {feature_tensor_flat.shape}")
    
    # Initialize result tensors on GPU
    pearson_corr = torch.zeros(n_latents, n_features, device=device)
    pearson_p = torch.zeros(n_latents, n_features, device=device)
    
    # Calculate memory requirements and adjust batch sizes
    # Each correlation calculation needs: activations + features + intermediate results
    # Estimate memory per latent-feature pair
    estimated_memory_per_pair = total_tokens * 4 * 4  # 4 bytes per float32, 4 tensors
    available_memory = torch.cuda.get_device_properties(device).total_memory * 0.7  # Use 70% of GPU memory
    
    # Calculate optimal batch sizes
    max_pairs_per_batch = max(1, int(available_memory / estimated_memory_per_pair))
    latent_batch_size = min(50, max(1, int(np.sqrt(max_pairs_per_batch))))
    feature_batch_size = min(5, max(1, max_pairs_per_batch // latent_batch_size))
    
    print(f"   🔄 Using batched processing: {latent_batch_size} latents × {feature_batch_size} features per batch")
    
    # Process in batches to avoid memory issues
    for latent_batch_start in range(0, n_latents, latent_batch_size):
        latent_batch_end = min(latent_batch_start + latent_batch_size, n_latents)
        batch_latents = latent_batch_end - latent_batch_start
        
        for feature_batch_start in range(0, n_features, feature_batch_size):
            feature_batch_end = min(feature_batch_start + feature_batch_size, n_features)
            batch_features = feature_batch_end - feature_batch_start
            
            print(f"   🔄 Processing batch: latents {latent_batch_start}-{latent_batch_end-1}, features {feature_batch_start}-{feature_batch_end-1}")
            
            # Process each feature in the batch
            for feature_idx in range(feature_batch_start, feature_batch_end):
                # Get feature values: (total_tokens,)
                feature_vals = feature_tensor_flat[:, feature_idx]  # Shape: (total_tokens,)
                
                # Process each latent in the batch
                for latent_batch_idx in range(batch_latents):
                    latent_idx = latent_batch_start + latent_batch_idx
                    
                    # Get latent activations for this dimension: (total_tokens,)
                    latent_activations = activations_tensor[:, latent_idx]  # Shape: (total_tokens,)
                    
                    # For binary features, we need to include both 0s and 1s
                    # Only remove actual padding (positions beyond sequence length)
                    # Since we're using the activation sequence length, all positions should be valid
                    valid_mask = torch.ones_like(feature_vals, dtype=torch.bool)
                    
                    # Apply mask to both tensors
                    valid_activations = latent_activations[valid_mask]  # (n_valid,)
                    valid_features = feature_vals[valid_mask]  # (n_valid,)
                    
                    # Calculate Pearson correlation using tensor operations
                    n_valid = len(valid_activations)
                    
                    if n_valid > 1:
                        # Mean centering
                        mean_activations = valid_activations.mean()
                        mean_features = valid_features.mean()
                        
                        # Centered values
                        centered_activations = valid_activations - mean_activations
                        centered_features = valid_features - mean_features
                        
                        # Covariance: E[(X-μx)(Y-μy)]
                        covariance = (centered_activations * centered_features).mean()
                        
                        # Standard deviations
                        std_activations = centered_activations.std()
                        std_features = centered_features.std()

                        
                        # Avoid division by zero
                        if std_activations > 1e-8 and std_features > 1e-8:
                            # Calculate Pearson correlation
                            pearson_val = covariance / (std_activations * std_features)
                            
                            # Simplified p-value calculation (approximate)
                            # t-statistic = r * sqrt((n-2)/(1-r²))
                            t_stat_pearson = pearson_val * torch.sqrt((n_valid - 2) / (1 - pearson_val**2 + 1e-8))
                            
                            # Approximate p-value
                            p_pearson_val = 1.0 / (1.0 + torch.abs(t_stat_pearson))
                            
                            # Store results
                            pearson_corr[latent_idx, feature_idx] = pearson_val
                            pearson_p[latent_idx, feature_idx] = p_pearson_val
    
    # Move results back to CPU and convert to numpy
    pearson_corr = pearson_corr.cpu().numpy()
    pearson_p = pearson_p.cpu().numpy()
    
    return pearson_corr, pearson_p


def _cpu_calculate_correlation_matrix(activations, feature_array):
    """
    CPU fallback for correlation calculation when CUDA is not available.
    """
    print("   🔄 Using CPU tensor operations...")
    
    # Convert to tensors on CPU
    activations_tensor = torch.tensor(activations, dtype=torch.float32)
    feature_tensor = torch.tensor(feature_array, dtype=torch.float32)
    
    n_sequences, n_latents = activations_tensor.shape
    n_features = feature_tensor.shape[1]
    max_seq_len = feature_tensor.shape[2]
    
    # Initialize result tensors
    pearson_corr = torch.zeros(n_latents, n_features)
    pearson_p = torch.zeros(n_latents, n_features)
    
    # Process each latent-feature pair
    for latent_idx in range(n_latents):
        for feature_idx in range(n_features):
            # Get data for this pair
            latent_vals = activations_tensor[:, latent_idx]
            feature_vals = feature_tensor[:, feature_idx, :]
            
            # Flatten and create data points
            activation_data = torch.repeat_interleave(latent_vals, max_seq_len)
            feature_data = feature_vals.flatten()
            
            # For binary features, we need to include both 0s and 1s
            # Since we're using the activation sequence length, all positions should be valid
            valid_mask = torch.ones_like(feature_data, dtype=torch.bool)
            
            activation_data = activation_data[valid_mask]
            feature_data = feature_data[valid_mask]
            
            # Calculate correlations using torch operations
            n = len(activation_data)
            if n > 1:
                # Pearson correlation
                mean_x = activation_data.mean()
                mean_y = feature_data.mean()
                
                numerator = ((activation_data - mean_x) * (feature_data - mean_y)).mean()
                denom_x = (activation_data - mean_x).std()
                denom_y = (feature_data - mean_y).std()
                
                if denom_x > 0 and denom_y > 0:
                    pearson_corr[latent_idx, feature_idx] = numerator / (denom_x * denom_y)
                
                # Simplified p-value
                if abs(pearson_corr[latent_idx, feature_idx]) > 0:
                    t_stat = pearson_corr[latent_idx, feature_idx] * torch.sqrt((n - 2) / (1 - pearson_corr[latent_idx, feature_idx] ** 2))
                    pearson_p[latent_idx, feature_idx] = 1.0 / (1.0 + abs(t_stat))
    
    return pearson_corr.numpy(), pearson_p.numpy() 