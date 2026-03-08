"""
Structure Module (Simplified for Option B)
Uses geometric-aware attention without full IPA implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GeometricAttention(nn.Module):
    """
    Simplified geometric attention
    Attends to both features and relative distances
    """
    
    def __init__(self, d_model=384, n_heads=8, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.distance_proj = nn.Linear(1, n_heads)  # Project distances to attention logits
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, coords=None):
        """
        Args:
            x: (batch, seq_len, d_model) - features
            coords: (batch, seq_len, 3) - current 3D coordinates (optional)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Standard attention
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.n_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.n_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.n_heads)
        
        # Attention scores from features
        attn = torch.einsum('bhid,bhjd->bhij', q, k) / (self.head_dim ** 0.5)
        
        # Add geometric bias from distances
        if coords is not None:
            # Compute pairwise distances
            diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (batch, seq_len, seq_len, 3)
            dist = torch.norm(diff, dim=-1, keepdim=True)  # (batch, seq_len, seq_len, 1)
            
            # Project distances to attention bias
            dist_bias = self.distance_proj(dist)  # (batch, seq_len, seq_len, n_heads)
            dist_bias = rearrange(dist_bias, 'b i j h -> b h i j')
            
            attn = attn + dist_bias
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.out_proj(out)
        
        return out


class StructureUpdateBlock(nn.Module):
    """Single iteration of structure refinement"""
    
    def __init__(self, d_single=384, d_pair=128, n_heads=8, dropout=0.1):
        super().__init__()
        
        # Combine single and pair information
        self.pair_to_single = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, d_single),
        )
        
        # Geometric attention
        self.geo_attn = GeometricAttention(d_single, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_single)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.LayerNorm(d_single),
            nn.Linear(d_single, d_single * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_single * 4, d_single),
            nn.Dropout(dropout)
        )
        
        # Coordinate update head
        self.coord_update = nn.Sequential(
            nn.LayerNorm(d_single),
            nn.Linear(d_single, d_single // 2),
            nn.ReLU(),
            nn.Linear(d_single // 2, 3),
        )
        
    def forward(self, single, pair, coords):
        """
        Args:
            single: (batch, seq_len, d_single)
            pair: (batch, seq_len, seq_len, d_pair)
            coords: (batch, seq_len, 3) - current coordinates
        Returns:
            single_out: Updated features
            coords_out: Updated coordinates
        """
        # Incorporate pair information
        # Average pair features for each residue
        pair_avg = pair.mean(dim=2)  # (batch, seq_len, d_pair)
        single = single + self.pair_to_single(pair_avg)
        
        # Geometric attention with current coordinates
        single = single + self.geo_attn(self.norm1(single), coords)
        
        # Feed-forward
        single = single + self.ff(single)
        
        # Predict coordinate update (residual)
        delta_coords = self.coord_update(single)
        coords = coords + delta_coords
        
        return single, coords


class StructureModule(nn.Module):
    """
    Structure prediction module with iterative refinement
    Simplified version for Option B (no full IPA)
    """
    
    def __init__(self, d_single=384, d_pair=128, n_iterations=3, n_heads=8, dropout=0.1):
        super().__init__()
        self.n_iterations = n_iterations
        
        # Project to structure module hidden dimension
        self.input_proj = nn.Linear(256, d_single)  # From MSA output to structure dim
        
        # Iterative structure update blocks
        self.update_blocks = nn.ModuleList([
            StructureUpdateBlock(d_single, d_pair, n_heads, dropout)
            for _ in range(n_iterations)
        ])
        
        # Initial coordinate prediction
        self.init_coords = nn.Sequential(
            nn.Linear(d_single, d_single // 2),
            nn.ReLU(),
            nn.Linear(d_single // 2, 3),
        )
        
    def forward(self, single, pair):
        """
        Args:
            single: (batch, seq_len, 256) - from MSA transformer
            pair: (batch, seq_len, seq_len, 128) - from MSA transformer
        Returns:
            coords: (batch, seq_len, 3) - predicted C1' coordinates
            all_coords: List of coordinates from each iteration
        """
        # Project to structure dimension
        single = self.input_proj(single)  # (batch, seq_len, d_single)
        
        # Initialize coordinates (rough guess)
        coords = self.init_coords(single)  # (batch, seq_len, 3)
        
        all_coords = [coords]
        
        # Iterative refinement
        for block in self.update_blocks:
            single, coords = block(single, pair, coords)
            all_coords.append(coords)
        
        return coords, all_coords
