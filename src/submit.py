"""
Generate submission.csv from predictions
"""
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


def create_submission(predictions_path, test_sequences_path, output_path='submission.csv'):
    """
    Create submission CSV from predictions
    
    Format:
    ID, resname, resid, x_1, y_1, z_1, x_2, y_2, z_2, x_3, y_3, z_3, x_4, y_4, z_4, x_5, y_5, z_5
    """
    print("=" * 50)
    print("Creating Submission File")
    print("=" * 50)
    
    # Load predictions
    print(f"Loading predictions from: {predictions_path}")
    predictions = torch.load(predictions_path, weights_only=False)
    
    # Load test sequences
    print(f"Loading test sequences from: {test_sequences_path}")
    test_df = pd.read_csv(test_sequences_path)
    
    # Prepare submission rows
    submission_rows = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Formatting"):
        target_id = row['target_id']
        sequence = row['sequence']
        
        if target_id not in predictions:
            print(f"Warning: No predictions for {target_id}, skipping...")
            continue
        
        # Get predictions: (5, seq_len, 3)
        coords_all = predictions[target_id]
        
        pred_len = int(coords_all.shape[1])

        # Process each residue
        for res_idx, nucleotide in enumerate(sequence):
            # Create row ID
            row_id = f"{target_id}_{res_idx + 1}"  # 1-based indexing
            
            # Extract coordinates for this residue across 5 predictions
            coords_dict = {
                'ID': row_id,
                'resname': nucleotide,
                'resid': res_idx + 1
            }
            
            # Add coordinates for each of 5 predictions
            for pred_idx in range(5):
                if res_idx < pred_len:
                    x, y, z = coords_all[pred_idx, res_idx, :]
                else:
                    x, y, z = 0.0, 0.0, 0.0
                
                # Clip coordinates to valid PDB range
                x = np.clip(x, -999.999, 9999.999)
                y = np.clip(y, -999.999, 9999.999)
                z = np.clip(z, -999.999, 9999.999)
                
                coords_dict[f'x_{pred_idx + 1}'] = x
                coords_dict[f'y_{pred_idx + 1}'] = y
                coords_dict[f'z_{pred_idx + 1}'] = z
            
            submission_rows.append(coords_dict)
    
    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)
    
    # Reorder columns
    columns = ['ID', 'resname', 'resid']
    for i in range(1, 6):
        columns.extend([f'x_{i}', f'y_{i}', f'z_{i}'])
    submission_df = submission_df[columns]
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Submission file created: {output_path}")
    print(f"  Total rows: {len(submission_df):,}")
    print(f"  Targets: {len(predictions)}")
    print("\nSubmission format validated:")
    print(f"  Columns: {len(submission_df.columns)}")
    print(f"  Required: ID, resname, resid, x_1 through z_5")
    
    # Show sample
    print("\nSample rows:")
    print(submission_df.head(3))
    
    return submission_df


if __name__ == "__main__":
    # Paths
    predictions_path = 'outputs/predictions/test_predictions.pt'
    test_sequences_path = 'data/raw/test_sequences.csv'
    output_path = 'submission.csv'
    
    # Create submission
    submission_df = create_submission(
        predictions_path=predictions_path,
        test_sequences_path=test_sequences_path,
        output_path=output_path
    )
    
    print("\n✓ Ready to submit!")
