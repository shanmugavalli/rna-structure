"""
Quick test script to validate the implementation
Run this to ensure all modules are working before full training
"""
import torch
import sys
sys.path.append('src')

from config import cfg
from model import RNAStructurePredictor
from modules.embeddings import tokenize_sequence, tokenize_msa
from losses import StructureLoss

def test_model():
    """Test model forward pass"""
    print("=" * 50)
    print("Testing RNA Structure Predictor - Option B")
    print("=" * 50)
    
    # Create dummy data
    batch_size = 2
    seq_len = 50
    n_msa_seqs = 16
    
    print("\n1. Creating dummy input data...")
    print(f"   Batch size: {batch_size}")
    print(f"   Sequence length: {seq_len}")
    print(f"   MSA sequences: {n_msa_seqs}")
    
    # Dummy sequence tokens (A=0, C=1, G=2, U=3)
    seq_tokens = torch.randint(0, 4, (batch_size, seq_len))
    
    # Dummy MSA tokens
    msa_tokens = torch.randint(0, 4, (batch_size, n_msa_seqs, seq_len))
    
    # Ensure first row of MSA matches target sequence
    msa_tokens[:, 0, :] = seq_tokens
    
    print("   [OK] Input data created")
    
    # Create model
    print("\n2. Creating model...")
    model = RNAStructurePredictor(cfg)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {n_params:,}")
    print("   [OK] Model created successfully")
    
    # Test forward pass
    print("\n3. Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            coords, all_coords = model(seq_tokens, msa_tokens)
        
        print(f"   Output shape: {coords.shape}")
        print(f"   Expected: ({batch_size}, {seq_len}, 3)")
        assert coords.shape == (batch_size, seq_len, 3), "Output shape mismatch!"
        print("   [OK] Forward pass successful")
        
        print(f"\n   Intermediate predictions: {len(all_coords)}")
        print(f"   Expected: {cfg.structure_iterations + 1}")
        
    except Exception as e:
        print("   [ERROR] Forward pass failed: {e}")
        return False
    
    # Test loss computation
    print("\n4. Testing loss computation...")
    try:
        criterion = StructureLoss()
        
        # Dummy ground truth
        true_coords = torch.randn(batch_size, seq_len, 3) * 10  # Random coordinates
        
        loss, loss_dict = criterion(coords, true_coords, all_coords)
        
        print(f"   Total loss: {loss.item():.4f}")
        print("   Loss components:")
        for k, v in loss_dict.items():
            print(f"      {k}: {v.item():.4f}")
        
        print("   [OK] Loss computation successful")
        
    except Exception as e:
        print(f"   [ERROR] Loss computation failed: {e}")
        return False
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    try:
        model.train()
        coords, all_coords = model(seq_tokens, msa_tokens)
        loss, _ = criterion(coords, true_coords, all_coords)
        loss.backward()
        
        # Check gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients computed!"
        
        print("   [OK] Backward pass successful")
        
    except Exception as e:
        print(f"   [ERROR] Backward pass failed: {e}")
        return False
    
    # Test embeddings module
    print("\n6. Testing tokenization...")
    try:
        test_seq = "ACGUACGU"
        tokens = tokenize_sequence(test_seq)
        print(f"   Sequence: {test_seq}")
        print(f"   Tokens: {tokens.tolist()}")
        assert len(tokens) == len(test_seq), "Tokenization length mismatch!"
        print("   [OK] Tokenization works")
        
    except Exception as e:
        print(f"   [ERROR] Tokenization failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("[OK] All tests passed!")
    print("=" * 50)
    print("\nThe implementation is ready for training.")
    print("Next steps:")
    print("  1. Place your data in data/raw/")
    print("  2. Run: cd src && python train.py")
    print("=" * 50)
    
    return True


if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
