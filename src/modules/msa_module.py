"""
MSA Transformer Module (Evoformer-style)
Processes Multiple Sequence Alignments to extract evolutionary patterns
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange, einsum


class MSAAttention(nn.Module):
    """Multi-head attention for MSA processing"""
    
    def __init__(self, embed_dim, n_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, ..., seq_len, embed_dim)
            mask: Optional attention mask
        Returns:
            output: (batch, ..., seq_len, embed_dim)
        """
        *leading_dims, seq_len, embed_dim = x.shape
        batch_total = torch.prod(torch.tensor(leading_dims)).item()
        
        # Reshape to (batch_total, seq_len, embed_dim)
        x_flat = x.reshape(batch_total, seq_len, embed_dim)
        
        # QKV projection
        qkv = self.qkv(x_flat)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.n_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_heads)
        
        # Attention scores with numerical stability
        attn = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale
        
        # Clamp to prevent extreme values
        attn = torch.clamp(attn, min=-50.0, max=50.0)
        
        if mask is not None:
            attn = attn.masked_fill(~mask, float('-inf'))
        
        # Safe softmax with numerical stability
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        
        # Reshape back to original leading dimensions
        out = out.reshape(*leading_dims, seq_len, embed_dim)
        
        return out


class OuterProductMean(nn.Module):
    """Outer product mean: MSA -> Pair representation"""
    
    def __init__(self, d_msa=256, d_pair=128, n_outer=32):
        super().__init__()
        self.linear_a = nn.Linear(d_msa, n_outer)
        self.linear_b = nn.Linear(d_msa, n_outer)
        self.linear_out = nn.Linear(n_outer * n_outer, d_pair)
        self.norm = nn.LayerNorm(d_msa)
        
    def forward(self, msa):
        """
        Args:
            msa: (batch, n_seqs, seq_len, d_msa)
        Returns:
            pair: (batch, seq_len, seq_len, d_pair)
        """
        msa = self.norm(msa)
        
        # Project to outer product space
        a = self.linear_a(msa)  # (batch, n_seqs, seq_len, n_outer)
        b = self.linear_b(msa)
        
        # Outer product and mean over sequences
        outer = einsum(a, b, 'b s i c1, b s j c2 -> b i j c1 c2')
        
        # Safe division with epsilon to prevent division by zero/NaN
        n_seqs = max(1, msa.shape[1])  # At least 1
        outer = outer / float(n_seqs)  # Mean over n_seqs
        
        # Flatten outer product dimensions and project
        batch, L, _, c1, c2 = outer.shape
        outer = outer.reshape(batch, L, L, c1 * c2)
        pair = self.linear_out(outer)
        
        return pair


class TriangleAttention(nn.Module):
    """Triangle attention for pair representation updates"""
    
    def __init__(self, d_pair=128, n_heads=4):
        super().__init__()
        self.attention = MSAAttention(d_pair, n_heads)
        self.norm = nn.LayerNorm(d_pair)
        
    def forward(self, pair):
        """
        Args:
            pair: (batch, seq_len, seq_len, d_pair)
        Returns:
            updated_pair: (batch, seq_len, seq_len, d_pair)
        """
        # Triangle update starting from edges
        # Attend along rows (i->j, varying j)
        pair_normed = self.norm(pair)
        updated = self.attention(pair_normed)
        return pair + updated


class MSATransformerBlock(nn.Module):
    """Single MSA Transformer block with row/column attention and pair updates"""
    
    def __init__(self, d_msa=256, d_pair=128, n_heads=8, dropout=0.1):
        super().__init__()
        
        # MSA row attention (within sequence)
        self.row_attn = MSAAttention(d_msa, n_heads, dropout)
        self.row_norm = nn.LayerNorm(d_msa)
        
        # MSA column attention (across sequences)
        self.col_attn = MSAAttention(d_msa, n_heads, dropout)
        self.col_norm = nn.LayerNorm(d_msa)
        
        # Feed-forward for MSA
        self.ff = nn.Sequential(
            nn.LayerNorm(d_msa),
            nn.Linear(d_msa, d_msa * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_msa * 4, d_msa),
            nn.Dropout(dropout)
        )
        
        # Outer product mean: MSA -> Pair
        self.outer_product = OuterProductMean(d_msa, d_pair)
        
        # Triangle attention for pair
        self.triangle_attn = TriangleAttention(d_pair, n_heads=4)
        
    def forward(self, msa, pair):
        """
        Args:
            msa: (batch, n_seqs, seq_len, d_msa)
            pair: (batch, seq_len, seq_len, d_pair)
        Returns:
            msa_out: (batch, n_seqs, seq_len, d_msa)
            pair_out: (batch, seq_len, seq_len, d_pair)
        """
        # MSA row attention (within each sequence)
        msa = msa + self.row_attn(self.row_norm(msa))
        
        # MSA column attention (across sequences at each position)
        # Transpose to (batch, seq_len, n_seqs, d_msa)
        msa_t = rearrange(msa, 'b s n d -> b n s d')
        msa_t = msa_t + self.col_attn(self.col_norm(msa_t))
        msa = rearrange(msa_t, 'b n s d -> b s n d')
        
        # Feed-forward
        msa = msa + self.ff(msa)
        
        # Update pair representation from MSA
        pair = pair + self.outer_product(msa)
        
        # Triangle attention on pair
        pair = self.triangle_attn(pair)
        
        return msa, pair


class MSATransformer(nn.Module):
    """Full MSA Transformer (Evoformer-style)"""
    
    def __init__(self, d_msa=256, d_pair=128, n_blocks=6, n_heads=8, dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Stack of MSA transformer blocks
        self.blocks = nn.ModuleList([
            MSATransformerBlock(d_msa, d_pair, n_heads, dropout)
            for _ in range(n_blocks)
        ])
        
        # Final pooling: MSA -> single sequence representation
        self.msa_to_single = nn.Sequential(
            nn.LayerNorm(d_msa),
            nn.Linear(d_msa, d_msa),
        )
        
    def forward(self, msa_emb, pair_init=None):
        """
        Args:
            msa_emb: (batch, n_seqs, seq_len, d_msa) - embedded MSA
            pair_init: Optional initial pair representation
        Returns:
            single: (batch, seq_len, d_msa) - single sequence representation
            pair: (batch, seq_len, seq_len, d_pair) - pair representation
        """
        batch, n_seqs, seq_len, d_msa = msa_emb.shape
        
        # Initialize pair representation if not provided
        if pair_init is None:
            d_pair = 128
            pair = torch.zeros(batch, seq_len, seq_len, d_pair, 
                             device=msa_emb.device, dtype=msa_emb.dtype)
        else:
            pair = pair_init
        
        msa = msa_emb
        
        # Process through MSA transformer blocks with optional checkpointing
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                msa, pair = checkpoint(block, msa, pair, use_reentrant=False)
            else:
                msa, pair = block(msa, pair)
        
        # Pool MSA to single representation (mean over sequences)
        single = msa.mean(dim=1)  # (batch, seq_len, d_msa)
        single = self.msa_to_single(single)
        
        return single, pair
