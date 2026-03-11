"""Feature engineering utilities for RNA sequence-level models."""
import torch


def build_residue_features(seq_tokens: torch.Tensor, pad_token: int = 4) -> torch.Tensor:
    """Build per-residue handcrafted features.

    Feature layout per residue:
    - 4 one-hot nucleotide channels (A/C/G/U)
    - 1 normalized position [0, 1]
    - 1 sin(position)
    - 1 cos(position)
    - 1 GC indicator
    - 1 purine indicator (A/G)
    """
    if seq_tokens.ndim != 1:
        raise ValueError("seq_tokens must be a 1D tensor")

    seq_len = int(seq_tokens.shape[0])
    if seq_len == 0:
        return torch.zeros(0, 9, dtype=torch.float32)

    token_float = seq_tokens.to(torch.long)
    valid = token_float != pad_token

    one_hot = torch.zeros(seq_len, 4, dtype=torch.float32)
    for i in range(4):
        one_hot[:, i] = (token_float == i).float()

    pos = torch.linspace(0.0, 1.0, steps=seq_len, dtype=torch.float32)
    angle = pos * (2.0 * torch.pi)
    pos_feats = torch.stack([pos, torch.sin(angle), torch.cos(angle)], dim=-1)

    gc = ((token_float == 1) | (token_float == 2)).float().unsqueeze(-1)
    purine = ((token_float == 0) | (token_float == 2)).float().unsqueeze(-1)

    features = torch.cat([one_hot, pos_feats, gc, purine], dim=-1)
    features = features * valid.float().unsqueeze(-1)
    return features
