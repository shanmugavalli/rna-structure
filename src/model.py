"""
Main RNA Structure Prediction Model (Option B)
Integrates MSA Transformer + Structure Module
"""
import torch
import torch.nn as nn
from modules.embeddings import RNAEmbedding, MSAEmbedding
from modules.msa_module import MSATransformer
from modules.structure_module import StructureModule


class RNALitePredictor(nn.Module):
    """CPU-friendly sequence model for small/limited datasets."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = getattr(config, 'feature_dim', 9)
        hidden_dim = getattr(config, 'lite_hidden_dim', 128)
        embed_dim = getattr(config, 'lite_embed_dim', 64)

        self.seq_embed = nn.Embedding(config.vocab_size, embed_dim, padding_idx=4)
        self.feature_proj = nn.Linear(self.feature_dim, embed_dim)
        self.encoder = nn.LSTM(
            input_size=embed_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, seq_tokens, msa_tokens=None, residue_features=None):
        if residue_features is None:
            batch_size, seq_len = seq_tokens.shape
            residue_features = torch.zeros(
                batch_size,
                seq_len,
                self.feature_dim,
                dtype=torch.float32,
                device=seq_tokens.device,
            )

        seq_emb = self.seq_embed(seq_tokens)
        feat_emb = self.feature_proj(residue_features)
        x = torch.cat([seq_emb, feat_emb], dim=-1)

        x, _ = self.encoder(x)
        x = self.norm(x)
        x = self.dropout(x)
        coords = self.head(x)
        return coords, [coords]


class RNAStructurePredictor(nn.Module):
    """
    Full RNA 3D structure prediction model
    
    Architecture:
    1. Embed sequence and MSA
    2. MSA Transformer (extract evolutionary patterns)
    3. Structure Module (predict 3D coordinates)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.seq_embed = RNAEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            max_len=config.max_seq_length
        )
        
        self.msa_embed = MSAEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.d_single,
            max_len=config.max_seq_length
        )
        
        # MSA Transformer
        self.msa_transformer = MSATransformer(
            d_msa=config.d_single,
            d_pair=config.d_pair,
            n_blocks=config.msa_depth,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_checkpoint=config.use_gradient_checkpointing
        )
        
        # Structure prediction module
        self.structure_module = StructureModule(
            d_single=config.structure_hidden,
            d_pair=config.d_pair,
            n_iterations=config.structure_iterations,
            n_heads=config.n_heads,
            dropout=config.dropout,
            use_checkpoint=config.use_gradient_checkpointing
        )
        
    def forward(self, seq_tokens, msa_tokens):
        """
        Args:
            seq_tokens: (batch, seq_len) - target sequence tokens
            msa_tokens: (batch, n_seqs, seq_len) - MSA tokens
        Returns:
            coords: (batch, seq_len, 3) - predicted C1' coordinates
            all_coords: List of intermediate predictions
        """
        # Embed MSA (includes target sequence as first row)
        msa_emb = self.msa_embed(msa_tokens)  # (batch, n_seqs, seq_len, d_single)
        
        # MSA Transformer: extract evolutionary patterns
        single, pair = self.msa_transformer(msa_emb)
        # single: (batch, seq_len, d_single)
        # pair: (batch, seq_len, seq_len, d_pair)
        
        # Structure Module: predict 3D coordinates
        coords, all_coords = self.structure_module(single, pair)
        # coords: (batch, seq_len, 3)
        
        return coords, all_coords
    
    def predict(self, seq_tokens, msa_tokens, return_intermediates=False):
        """Inference mode prediction"""
        self.eval()
        with torch.no_grad():
            coords, all_coords = self.forward(seq_tokens, msa_tokens)
        
        if return_intermediates:
            return coords, all_coords
        return coords


def build_model(config):
    """Factory that selects a full or CPU-lite model variant."""
    model_variant = getattr(config, 'model_variant', 'full')
    if model_variant == 'cpu_lite':
        return RNALitePredictor(config)
    return RNAStructurePredictor(config)


class EMAModel:
    """Exponential Moving Average of model parameters for stable inference"""
    
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """Update EMA parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model (for inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
