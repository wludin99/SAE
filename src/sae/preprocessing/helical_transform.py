from helical.models.helix_mrna import HelixmRNA, HelixmRNAConfig
import torch

def helical_transform(seqs: torch.Tensor, device: str = 'cuda') -> torch.Tensor:
    """
    Transform a sequence of DNA sequences into a helical representation.
    """
    helix_mrna_config = HelixmRNAConfig(batch_size=5, device=device, max_length=100)
    helix_mrna = HelixmRNA(configurer=helix_mrna_config)
    # prepare data for input to the model
    processed_input_data = helix_mrna.process_data(seqs)

    # generate the embeddings for the processed data
    embeddings = helix_mrna.get_embeddings(processed_input_data)
    return embeddings