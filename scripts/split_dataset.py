"""
Script to split fragments into train/test/validation sets (70/15/15)
"""

import os
import sys
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FRAGMENTS_DIR, TRAIN_DIR, TEST_DIR, VALIDATION_DIR,
    TRAIN_RATIO, TEST_RATIO, VALIDATION_RATIO, RANDOM_SEED
)


def split_dataset(fragments_dir, train_dir, test_dir, validation_dir,
                  train_ratio=TRAIN_RATIO, test_ratio=TEST_RATIO, 
                  validation_ratio=VALIDATION_RATIO, random_seed=RANDOM_SEED,
                  stratify=True):
    """
    Split fragments into train/test/validation sets.
    
    Args:
        fragments_dir: directory containing all fragments
        train_dir: output directory for training set
        test_dir: output directory for test set
        validation_dir: output directory for validation set
        train_ratio: proportion for training (default 0.70)
        test_ratio: proportion for testing (default 0.15)
        validation_ratio: proportion for validation (default 0.15)
        random_seed: random seed for reproducibility
        stratify: whether to maintain class distribution in splits
    """
    # Verify ratios sum to 1.0
    total_ratio = train_ratio + test_ratio + validation_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)
    
    # Load fragment mapping
    mapping_path = os.path.join(fragments_dir, 'fragment_mapping.csv')
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Fragment mapping not found: {mapping_path}")
    
    df = pd.read_csv(mapping_path)
    print(f"Loaded {len(df)} fragments from mapping")
    
    # Group by file type
    file_types = df['file_type'].unique()
    print(f"\nFile types found: {', '.join(file_types)}")
    
    train_fragments = []
    test_fragments = []
    val_fragments = []
    
    # Split each file type separately to maintain distribution
    for file_type in file_types:
        type_df = df[df['file_type'] == file_type]
        print(f"\n{file_type}: {len(type_df)} fragments")
        
        if len(type_df) < 3:
            print(f"  Warning: Not enough fragments for {file_type}, adding all to training")
            train_fragments.append(type_df)
            continue
        
        # First split: separate out test set
        temp_ratio = train_ratio + validation_ratio
        train_val_df, test_df = train_test_split(
            type_df,
            test_size=test_ratio,
            random_state=random_seed,
            shuffle=True
        )
        
        # Second split: separate train and validation
        if len(train_val_df) < 2:
            train_df = train_val_df
            val_df = pd.DataFrame()
        else:
            val_ratio_adjusted = validation_ratio / temp_ratio
            train_df, val_df = train_test_split(
                train_val_df,
                test_size=val_ratio_adjusted,
                random_state=random_seed,
                shuffle=True
            )
        
        print(f"  Train: {len(train_df)}, Test: {len(test_df)}, Val: {len(val_df)}")
        
        train_fragments.append(train_df)
        test_fragments.append(test_df)
        if len(val_df) > 0:
            val_fragments.append(val_df)
    
    # Combine all file types
    train_df = pd.concat(train_fragments, ignore_index=True)
    test_df = pd.concat(test_fragments, ignore_index=True)
    val_df = pd.concat(val_fragments, ignore_index=True) if val_fragments else pd.DataFrame()
    
    print(f"\n{'='*60}")
    print("SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"Training set: {len(train_df)} fragments ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} fragments ({len(test_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} fragments ({len(val_df)/len(df)*100:.1f}%)")
    
    # Copy fragments to respective directories
    print(f"\n{'='*60}")
    print("COPYING FILES")
    print(f"{'='*60}")
    
    def copy_fragments(df, dest_dir, set_name):
        print(f"\nCopying {set_name} set...")
        for _, row in df.iterrows():
            src_path = row['fragment_path']
            file_type = row['file_type']
            
            # Create subdirectory for file type
            type_dest_dir = os.path.join(dest_dir, file_type)
            os.makedirs(type_dest_dir, exist_ok=True)
            
            # Destination path
            dest_path = os.path.join(type_dest_dir, row['fragment_filename'])
            
            # Copy file
            if os.path.exists(src_path):
                shutil.copy2(src_path, dest_path)
            else:
                print(f"  Warning: Source file not found: {src_path}")
        
        # Save mapping for this set
        mapping_dest = os.path.join(dest_dir, 'fragment_mapping.csv')
        df.to_csv(mapping_dest, index=False)
        print(f"  ✓ Saved {len(df)} fragments to {dest_dir}")
        print(f"  ✓ Mapping saved to {mapping_dest}")
    
    copy_fragments(train_df, train_dir, "Training")
    copy_fragments(test_df, test_dir, "Test")
    if len(val_df) > 0:
        copy_fragments(val_df, validation_dir, "Validation")
    
    # Print final statistics
    print(f"\n{'='*60}")
    print("CLASS DISTRIBUTION")
    print(f"{'='*60}")
    
    print("\nTraining Set:")
    print(train_df['file_type'].value_counts())
    
    print("\nTest Set:")
    print(test_df['file_type'].value_counts())
    
    if len(val_df) > 0:
        print("\nValidation Set:")
        print(val_df['file_type'].value_counts())
    
    print(f"\n✓ Data split complete!")
    
    return train_df, test_df, val_df


def main():
    parser = argparse.ArgumentParser(
        description='Split fragments into train/test/validation sets'
    )
    
    parser.add_argument(
        '--fragments-dir',
        type=str,
        default=FRAGMENTS_DIR,
        help='Directory containing fragments'
    )
    
    parser.add_argument(
        '--train-dir',
        type=str,
        default=TRAIN_DIR,
        help='Output directory for training set'
    )
    
    parser.add_argument(
        '--test-dir',
        type=str,
        default=TEST_DIR,
        help='Output directory for test set'
    )
    
    parser.add_argument(
        '--validation-dir',
        type=str,
        default=VALIDATION_DIR,
        help='Output directory for validation set'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=TRAIN_RATIO,
        help='Training set ratio (default: 0.70)'
    )
    
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=TEST_RATIO,
        help='Test set ratio (default: 0.15)'
    )
    
    parser.add_argument(
        '--validation-ratio',
        type=float,
        default=VALIDATION_RATIO,
        help='Validation set ratio (default: 0.15)'
    )
    
    parser.add_argument(
        '--random-seed',
        type=int,
        default=RANDOM_SEED,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DATA SPLIT UTILITY")
    print("=" * 60)
    print(f"Fragments Directory: {args.fragments_dir}")
    print(f"Train Directory: {args.train_dir}")
    print(f"Test Directory: {args.test_dir}")
    print(f"Validation Directory: {args.validation_dir}")
    print(f"Split Ratio: {args.train_ratio:.0%} / {args.test_ratio:.0%} / {args.validation_ratio:.0%}")
    print(f"Random Seed: {args.random_seed}")
    print("=" * 60)
    
    # Perform split
    split_dataset(
        fragments_dir=args.fragments_dir,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        validation_dir=args.validation_dir,
        train_ratio=args.train_ratio,
        test_ratio=args.test_ratio,
        validation_ratio=args.validation_ratio,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    main()
