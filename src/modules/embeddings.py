"""
Embedding modules for RNA sequences and MSA
"""
import torch
import torch.nn as nn
import math


class RNAEmbedding(nn.Module):
    """Embed RNA sequences with positional encoding"""
    
    def __init__(self, vocab_size=5, embed_dim=256, max_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token embedding (A=0, C=1, G=2, U=3, PAD=4)
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=4)
        
        # Learnable positional embedding
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) - tokenized sequence
        Returns:
            embeddings: (batch, seq_len, embed_dim)
        """
        batch_size, seq_len = x.shape
        
        # Token embeddings
        token_emb = self.token_embed(x)  # (batch, seq_len, embed_dim)
        
        # Positional embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)
        
        # Combine
        embeddings = token_emb + pos_emb
        embeddings = self.norm(embeddings)
        
        return embeddings


class MSAEmbedding(nn.Module):
    """Embed MSA (Multiple Sequence Alignment)"""
    
    def __init__(self, vocab_size=5, embed_dim=256, max_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Token embedding (shared with sequence embedding)
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=4)
        
        # Positional embedding (column-wise)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        
        # MSA row (sequence) embedding
        self.msa_row_embed = nn.Embedding(128, embed_dim)  # Up to 128 MSA sequences
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, msa):
        """
        Args:
            msa: (batch, n_seqs, seq_len) - MSA tokens
        Returns:
            embeddings: (batch, n_seqs, seq_len, embed_dim)
        """
        batch_size, n_seqs, seq_len = msa.shape
        
        # Token embeddings
        token_emb = self.token_embed(msa)  # (batch, n_seqs, seq_len, embed_dim)
        
        # Positional embeddings (column position)
        positions = torch.arange(seq_len, device=msa.device).unsqueeze(0).unsqueeze(0)
        positions = positions.expand(batch_size, n_seqs, -1)
        pos_emb = self.pos_embed(positions)
        
        # MSA row embeddings (which sequence in alignment)
        row_ids = torch.arange(n_seqs, device=msa.device).unsqueeze(0).unsqueeze(-1)
        row_ids = row_ids.expand(batch_size, -1, seq_len)
        row_emb = self.msa_row_embed(row_ids)
        
        # Combine all embeddings
        embeddings = token_emb + pos_emb + row_emb
        embeddings = self.norm(embeddings)
        
        return embeddings


def tokenize_sequence(seq_str):
    """
    Convert RNA sequence string to tokens
    A=0, C=1, G=2, U=3, PAD=4, GAP(-)=4
    """
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3, '-': 4, 'N': 4}
    tokens = [mapping.get(c.upper(), 4) for c in seq_str]
    return torch.tensor(tokens, dtype=torch.long)


def tokenize_msa(msa_sequences):
    """
    Convert list of MSA sequences to token tensor
    
    Args:
        msa_sequences: List of sequence strings
    Returns:
        msa_tokens: (n_seqs, seq_len) tensor
    """
    token_list = [tokenize_sequence(seq) for seq in msa_sequences]
    
    # Pad to same length
    max_len = max(len(t) for t in token_list)
    padded = [torch.cat([t, torch.full((max_len - len(t),), 4, dtype=torch.long)]) 
              for t in token_list]
    
    return torch.stack(padded)
